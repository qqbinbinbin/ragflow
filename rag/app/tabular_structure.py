#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Immutable, no-vector row projections for tabular completeness checks."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import uuid
from io import BytesIO
from typing import Any


TABULAR_STRUCTURE_VERSION = "tabular-row/v1"
PRODUCER_SCHEMA_VERSION = "table-producer/v1"
PROJECTION_VERSION = "tabular-structure-projection/v1"
PROJECTION_PART_VERSION = "tabular-structure-part/v1"
PROJECTION_FIELDS = frozenset(
    {
        "version",
        "producer_schema_version",
        "producer_generation_ref",
        "source_sha256",
        "tables",
        "rows",
    }
)

DEFAULT_CONTEXT_ENTRY_LIMIT = 8
DEFAULT_CONTEXT_VALUE_BYTES = 128
DEFAULT_TABLE_LABEL_BYTES = 128
DEFAULT_ROWS_PER_PART = 3000

PROJECTION_ROW_FIELDS = frozenset(
    {
        "id",
        "tabular_structure_version_kwd",
        "structure_kind_kwd",
        "producer_schema_version_kwd",
        "producer_generation_ref_kwd",
        "table_ref_kwd",
        "table_label_kwd",
        "table_context_list",
        "row_ref_kwd",
        "row_ordinal_int",
        "data_row_index_int",
        "row_role_kwd",
        "source_total_count_int",
        "ordered_fields_list",
    }
)

_ULID_RE = re.compile(r"^[0-7][0-9A-HJKMNP-TV-Z]{25}$")
_UNTRUSTED_CONTROL_RE = re.compile("[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


class StructureGenerationConflict(RuntimeError):
    """The persistent generation state violates its single-active invariant."""


class StructureSnapshotMissing(LookupError):
    """The requested immutable structure generation is not readable."""


class StructureSnapshotChanged(RuntimeError):
    """The requested immutable structure generation no longer matches its digest."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _versioned_digest(kind: str, *parts: object) -> str:
    payload = "\x00".join([kind, *(str(part) for part in parts)]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_generation_ref(producer_generation_ref: str) -> None:
    if not isinstance(producer_generation_ref, str):
        raise ValueError("producer_generation_ref must be an opaque UUID/ULID-compatible value")
    try:
        parsed_uuid = uuid.UUID(producer_generation_ref)
        is_canonical_uuid = str(parsed_uuid) == producer_generation_ref.lower()
    except ValueError:
        is_canonical_uuid = False
    if not is_canonical_uuid and not _ULID_RE.fullmatch(producer_generation_ref):
        raise ValueError("producer_generation_ref must be an opaque UUID/ULID-compatible value")


def _sanitize_untrusted_text(value: object) -> str:
    return _UNTRUSTED_CONTROL_RE.sub("", "" if value is None else str(value)).strip()


def _truncate_utf8(value: str, byte_limit: int) -> str:
    if byte_limit < 1:
        raise ValueError("UTF-8 byte limit must be positive")
    encoded = value.encode("utf-8")
    if len(encoded) <= byte_limit:
        return value
    return encoded[:byte_limit].decode("utf-8", errors="ignore")


def _cell_value(parser, worksheet, row_ordinal: int, column_ordinal: int, merged_ranges):
    value = worksheet.cell(row=row_ordinal, column=column_ordinal).value
    if value is not None:
        return value
    return parser._get_merged_cell_value(worksheet, row_ordinal, column_ordinal, merged_ranges)


def _complete_worksheet_rows(worksheet):
    """Return a bounded header probe plus all physically populated row ordinals."""

    if not worksheet.max_row:
        return [], [], []
    cells = getattr(worksheet, "_cells", None)
    if not isinstance(cells, dict):
        raise ValueError("worksheet backend cannot prove complete sparse row coverage")
    # Snapshot sparse cells before iter_rows materializes empty probe cells.
    physical_cells = list(cells.values())
    populated_rows = sorted(
        {
            cell.row
            for cell in physical_cells
            if cell.value is not None and str(cell.value).strip() != ""
        }
    )
    unresolved_rows = sorted(
        {
            cell.row
            for cell in physical_cells
            if cell.__class__.__name__ != "MergedCell" and cell.value is None
        }
    )
    header_rows = list(worksheet.iter_rows(min_row=1, max_row=min(20, worksheet.max_row)))
    return header_rows, populated_rows, unresolved_rows


def _ordered_fields(headers: list[str], values: list[object], *, note: bool) -> list[dict[str, str]]:
    fields = []
    for name, value in zip(headers, values):
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        rendered = str(value).strip()
        if not rendered:
            continue
        fields.append({"name": str(name), "value": rendered})
        if note:
            break
    return fields


def _is_full_width_merge(row_ordinal: int, width: int, merged_ranges) -> bool:
    return any(
        merged.min_row <= row_ordinal <= merged.max_row
        and merged.min_col == 1
        and merged.max_col >= width
        for merged in merged_ranges
    )


def _has_partial_row_merge(row_ordinal: int, width: int, merged_ranges) -> bool:
    return any(
        merged.min_row <= row_ordinal <= merged.max_row
        and (merged.min_col != 1 or merged.max_col < width)
        for merged in merged_ranges
    )


def _classify_body_row(
    row_ordinal: int,
    values: list[object],
    merged_ranges,
) -> str:
    width = len(values)
    populated = sum(value is not None and str(value).strip() != "" for value in values)
    if _is_full_width_merge(row_ordinal, width, merged_ranges):
        return "unknown"
    if _has_partial_row_merge(row_ordinal, width, merged_ranges):
        return "unknown"
    if populated >= min(2, width):
        return "data"
    return "unknown"


def _row_shape(values: list[object]) -> tuple[str, ...]:
    shape = []
    for value in values:
        if value is None or (isinstance(value, str) and not value.strip()):
            shape.append("empty")
        elif isinstance(value, bool):
            shape.append("boolean")
        elif isinstance(value, (int, float)):
            shape.append("number")
        else:
            shape.append("text")
    return tuple(shape)


def _table_context(
    parser,
    worksheet,
    rows,
    header_start: int,
    *,
    entry_limit: int,
    value_bytes: int,
) -> list[dict[str, str]]:
    if entry_limit < 1:
        raise ValueError("table context entry limit must be positive")
    merged_ranges = list(worksheet.merged_cells.ranges)
    entries = []
    for row_ordinal, row in enumerate(rows[:header_start], start=1):
        values = []
        for column_ordinal in range(1, len(row) + 1):
            value = _cell_value(parser, worksheet, row_ordinal, column_ordinal, merged_ranges)
            value = _sanitize_untrusted_text(value)
            if value and (not values or values[-1] != value):
                values.append(value)
        if not values:
            continue
        pairs = []
        if len(values) == 1:
            pairs.append(("context", values[0]))
        else:
            pairs.extend(zip(values[::2], values[1::2]))
            if len(values) % 2:
                pairs.append(("context", values[-1]))
        for name, value in pairs:
            entries.append(
                {
                    "name": _truncate_utf8(_sanitize_untrusted_text(name), value_bytes),
                    "value": _truncate_utf8(_sanitize_untrusted_text(value), value_bytes),
                }
            )
            if len(entries) == entry_limit:
                return entries
    return entries


def build_tabular_structure_projection(
    filename: str,
    binary: bytes,
    *,
    producer_generation_ref: str | None = None,
    table_context_entry_limit: int = DEFAULT_CONTEXT_ENTRY_LIMIT,
    table_context_value_bytes: int = DEFAULT_CONTEXT_VALUE_BYTES,
    parser=None,
) -> dict[str, Any]:
    """Build one immutable generation from complete workbook bytes.

    The returned records are a derived read model. They contain no vectors and
    are never inserted into the ordinary relevance-chunk index.
    """

    if not isinstance(binary, bytes) or not binary:
        raise ValueError("complete immutable workbook bytes are required")
    producer_generation_ref = producer_generation_ref or str(uuid.uuid4())
    _validate_generation_ref(producer_generation_ref)

    # Reuse RAGFlow's mature xls/xlsx loader and header detector. Importing here
    # keeps this fixed-schema producer independent from the normal chunk path.
    if parser is None:
        from rag.app.table import Excel

        parser = Excel()
    workbook = parser._load_excel_to_workbook(BytesIO(binary))
    source_sha256 = hashlib.sha256(binary).hexdigest()
    records = []
    tables = []

    for sheet_ordinal, sheet_name in enumerate(workbook.sheetnames, start=1):
        worksheet = workbook[sheet_name]
        # Completeness cannot reuse the normal parser's tail-sampling shortcut:
        # a real row after a large blank region must remain visible and fail closed.
        header_rows, populated_row_ordinals, unresolved_row_ordinals = _complete_worksheet_rows(worksheet)
        if not header_rows or not populated_row_ordinals:
            continue
        headers, header_start, data_start = parser._parse_sheet_structure(worksheet, header_rows)
        if not headers:
            continue

        table_ordinal = 1
        table_ref = "tbl_v1_" + _versioned_digest(
            "tabular-table/v1",
            source_sha256,
            sheet_ordinal,
            table_ordinal,
        )
        context = _table_context(
            parser,
            worksheet,
            header_rows,
            header_start,
            entry_limit=table_context_entry_limit,
            value_bytes=table_context_value_bytes,
        )
        merged_ranges = list(worksheet.merged_cells.ranges)
        pending_rows = []
        data_row_index = 0
        has_unknown = False

        body_rows = []
        body_gap_seen = False
        previous_body_ordinal = data_start
        body_ordinals = {ordinal for ordinal in populated_row_ordinals if ordinal > data_start}
        unresolved_body_ordinals = {ordinal for ordinal in unresolved_row_ordinals if ordinal > data_start}
        if unresolved_body_ordinals:
            # One sentinel row is enough to invalidate the denominator without
            # expanding a preformatted blank worksheet into millions of records.
            body_ordinals.add(min(unresolved_body_ordinals))
        for row_ordinal in sorted(body_ordinals):
            values = [
                _cell_value(parser, worksheet, row_ordinal, column_ordinal, merged_ranges)
                for column_ordinal in range(1, len(headers) + 1)
            ]
            if row_ordinal > previous_body_ordinal + 1:
                body_gap_seen = True
            body_rows.append((row_ordinal, values, body_gap_seen))
            previous_body_ordinal = row_ordinal

        established_shape = None
        for body_index, (row_ordinal, values, follows_body_gap) in enumerate(body_rows):
            if follows_body_gap:
                row_role = "unknown"
            else:
                row_role = _classify_body_row(
                    row_ordinal,
                    values,
                    merged_ranges,
                )
                current_shape = _row_shape(values)
                next_shape = _row_shape(body_rows[body_index + 1][1]) if body_index + 1 < len(body_rows) else None
                if row_role == "data" and all(value_type == "text" for value_type in current_shape):
                    row_role = "unknown"
                if (
                    row_role == "data"
                    and established_shape is not None
                    and current_shape != established_shape
                    and current_shape == next_shape
                ):
                    row_role = "unknown"
            if row_role == "data":
                data_row_index += 1
                current_data_index = data_row_index
                established_shape = established_shape or _row_shape(values)
            else:
                current_data_index = None
                has_unknown = has_unknown or row_role == "unknown"
            row_ref = f"{table_ref}:{row_ordinal}"
            record_id = "tsr_v1_" + _versioned_digest(
                "tabular-row-record/v1",
                producer_generation_ref,
                row_ref,
            )
            pending_rows.append(
                {
                    "id": record_id,
                    "tabular_structure_version_kwd": TABULAR_STRUCTURE_VERSION,
                    "structure_kind_kwd": "table_row",
                    "producer_schema_version_kwd": PRODUCER_SCHEMA_VERSION,
                    "producer_generation_ref_kwd": producer_generation_ref,
                    "table_ref_kwd": table_ref,
                    "table_label_kwd": _sanitize_untrusted_text(sheet_name),
                    "table_context_list": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                    "row_ref_kwd": row_ref,
                    "row_ordinal_int": row_ordinal,
                    "data_row_index_int": current_data_index,
                    "row_role_kwd": row_role,
                    "source_total_count_int": None,
                    "ordered_fields_list": json.dumps(
                        _ordered_fields(headers, values, note=row_role == "note"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )

        if not pending_rows:
            continue
        source_total_count = None if has_unknown else data_row_index
        for record in pending_rows:
            record["source_total_count_int"] = source_total_count
        records.extend(pending_rows)
        tables.append(
            {
                "table_ref": table_ref,
                "sheet_ordinal": sheet_ordinal,
                "table_ordinal": table_ordinal,
                "row_count": len(pending_rows),
                "data_row_count": data_row_index,
                "source_total_count": source_total_count,
            }
        )

    projection = {
        "version": PROJECTION_VERSION,
        "producer_schema_version": PRODUCER_SCHEMA_VERSION,
        "producer_generation_ref": producer_generation_ref,
        "source_sha256": source_sha256,
        "tables": tables,
        "rows": records,
    }
    validate_tabular_structure_projection(projection)
    return projection


def validate_tabular_structure_projection(projection: dict[str, Any]) -> None:
    if not isinstance(projection, dict) or set(projection) != PROJECTION_FIELDS:
        raise ValueError("structure projection does not match the fixed top-level schema")
    if projection.get("version") != PROJECTION_VERSION:
        raise ValueError("unsupported tabular structure projection version")
    if projection.get("producer_schema_version") != PRODUCER_SCHEMA_VERSION:
        raise ValueError("unsupported table producer schema version")
    generation_ref = projection.get("producer_generation_ref")
    _validate_generation_ref(generation_ref)
    source_sha256 = projection.get("source_sha256")
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("source SHA-256 is invalid")
    rows = projection.get("rows")
    if not isinstance(rows, list):
        raise ValueError("projection rows must be a list")

    ids = set()
    row_refs = set()
    by_table: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != PROJECTION_ROW_FIELDS:
            raise ValueError("structure projection row fields do not match the fixed schema")
        if row["tabular_structure_version_kwd"] != TABULAR_STRUCTURE_VERSION:
            raise ValueError("unsupported structure row version")
        if row["structure_kind_kwd"] != "table_row":
            raise ValueError("invalid structure kind")
        if row["producer_schema_version_kwd"] != PRODUCER_SCHEMA_VERSION:
            raise ValueError("unsupported row producer schema version")
        if row["producer_generation_ref_kwd"] != generation_ref:
            raise ValueError("mixed producer generations are not allowed")
        for field_name, label in (
            ("id", "record ID"),
            ("table_ref_kwd", "table reference"),
            ("table_label_kwd", "table label"),
            ("row_ref_kwd", "row reference"),
        ):
            if not isinstance(row[field_name], str):
                raise ValueError(f"{label} must be a string")
        if len(row["table_label_kwd"].encode("utf-8")) > DEFAULT_TABLE_LABEL_BYTES:
            raise ValueError("table label exceeds the UTF-8 byte limit")
        if _sanitize_untrusted_text(row["table_label_kwd"]) != row["table_label_kwd"]:
            raise ValueError("table label contains unsafe controls")
        if row["row_role_kwd"] not in {"data", "note", "unknown"}:
            raise ValueError("invalid structure row role")
        if not isinstance(row["row_ordinal_int"], int) or isinstance(row["row_ordinal_int"], bool) or row["row_ordinal_int"] < 1:
            raise ValueError("row ordinal must be a positive integer")
        data_row_index = row["data_row_index_int"]
        if data_row_index is not None and (
            not isinstance(data_row_index, int) or isinstance(data_row_index, bool) or data_row_index < 1
        ):
            raise ValueError("data row index must be a positive integer or null")
        source_total = row["source_total_count_int"]
        if source_total is not None and (
            not isinstance(source_total, int) or isinstance(source_total, bool) or source_total < 0
        ):
            raise ValueError("source total must be a non-negative integer or null")
        expected_id = "tsr_v1_" + _versioned_digest(
            "tabular-row-record/v1",
            generation_ref,
            row["row_ref_kwd"],
        )
        if row["id"] != expected_id:
            raise ValueError("structure record identity does not match its generation and row reference")
        if row["row_ref_kwd"] != f"{row['table_ref_kwd']}:{row['row_ordinal_int']}":
            raise ValueError("row reference does not match table and physical row identity")
        parsed_lists = {}
        for field_name, label in (
            ("table_context_list", "table context"),
            ("ordered_fields_list", "ordered fields"),
        ):
            if not isinstance(row[field_name], str):
                raise ValueError(f"{label} must be JSON text")
            try:
                values = json.loads(row[field_name])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{label} must be valid JSON") from exc
            if not isinstance(values, list) or any(
                not isinstance(item, dict)
                or set(item) != {"name", "value"}
                or not isinstance(item["name"], str)
                or not isinstance(item["value"], str)
                for item in values
            ):
                raise ValueError(f"{label} must use the fixed name/value schema")
            parsed_lists[field_name] = values
        context = parsed_lists["table_context_list"]
        if len(context) > DEFAULT_CONTEXT_ENTRY_LIMIT:
            raise ValueError("table context exceeds the entry limit")
        for item in context:
            if any(len(item[field].encode("utf-8")) > DEFAULT_CONTEXT_VALUE_BYTES for field in ("name", "value")):
                raise ValueError("table context exceeds the UTF-8 byte limit")
            if any(_sanitize_untrusted_text(item[field]) != item[field] for field in ("name", "value")):
                raise ValueError("table context contains unsafe controls")
        if row["id"] in ids or row["row_ref_kwd"] in row_refs:
            raise ValueError("duplicate structure row identity")
        ids.add(row["id"])
        row_refs.add(row["row_ref_kwd"])
        by_table.setdefault(row["table_ref_kwd"], []).append(row)

    for table_rows in by_table.values():
        if len({row["table_label_kwd"] for row in table_rows}) != 1:
            raise ValueError("table label is not generation-wide")
        if len({row["table_context_list"] for row in table_rows}) != 1:
            raise ValueError("table context is not generation-wide")
        ordinals = [row["row_ordinal_int"] for row in table_rows]
        if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
            raise ValueError("structure rows are not in deterministic physical order")
        data_indices = [row["data_row_index_int"] for row in table_rows if row["row_role_kwd"] == "data"]
        if data_indices != list(range(1, len(data_indices) + 1)):
            raise ValueError("data row indices are not globally contiguous")
        if any(row["row_role_kwd"] != "data" and row["data_row_index_int"] is not None for row in table_rows):
            raise ValueError("non-data rows cannot have a data row index")
        totals = {row["source_total_count_int"] for row in table_rows}
        expected_total = None if any(row["row_role_kwd"] == "unknown" for row in table_rows) else len(data_indices)
        if totals != {expected_total}:
            raise ValueError("source total is not generation-wide or fail-closed")

    tables = projection.get("tables")
    if not isinstance(tables, list):
        raise ValueError("table manifest must be a list")
    manifest_refs = set()
    for table in tables:
        expected_keys = {
            "table_ref",
            "sheet_ordinal",
            "table_ordinal",
            "row_count",
            "data_row_count",
            "source_total_count",
        }
        if not isinstance(table, dict) or set(table) != expected_keys:
            raise ValueError("table manifest does not match the fixed schema")
        if not isinstance(table["table_ref"], str):
            raise ValueError("table manifest reference must be a string")
        for field_name in ("sheet_ordinal", "table_ordinal"):
            value = table[field_name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("table manifest ordinals must be positive integers")
        for field_name in ("row_count", "data_row_count"):
            value = table[field_name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("table manifest counts must be non-negative integers")
        source_total = table["source_total_count"]
        if source_total is not None and (
            not isinstance(source_total, int) or isinstance(source_total, bool) or source_total < 0
        ):
            raise ValueError("table manifest source total must be a non-negative integer or null")
        table_ref = table["table_ref"]
        expected_table_ref = "tbl_v1_" + _versioned_digest(
            "tabular-table/v1",
            source_sha256,
            table["sheet_ordinal"],
            table["table_ordinal"],
        )
        if table_ref != expected_table_ref:
            raise ValueError("table reference does not match source and physical table identity")
        if table_ref in manifest_refs or table_ref not in by_table:
            raise ValueError("table manifest does not match projected records")
        manifest_refs.add(table_ref)
        table_rows = by_table[table_ref]
        data_row_count = sum(row["row_role_kwd"] == "data" for row in table_rows)
        source_totals = {row["source_total_count_int"] for row in table_rows}
        if (
            table["row_count"] != len(table_rows)
            or table["data_row_count"] != data_row_count
            or source_totals != {table["source_total_count"]}
        ):
            raise ValueError("table manifest counts do not match projected records")
    if manifest_refs != set(by_table):
        raise ValueError("table manifest is missing projected tables")


def partition_tabular_structure_projection(
    projection: dict[str, Any],
    *,
    rows_per_part: int = DEFAULT_ROWS_PER_PART,
) -> list[dict[str, Any]]:
    validate_tabular_structure_projection(projection)
    if rows_per_part < 1:
        raise ValueError("rows_per_part must be positive")
    rows = projection["rows"]
    parts = []
    for offset in range(0, len(rows), rows_per_part):
        part_rows = rows[offset : offset + rows_per_part]
        parts.append(
            {
                "version": PROJECTION_PART_VERSION,
                "producer_generation_ref": projection["producer_generation_ref"],
                "part_number": len(parts) + 1,
                "row_offset": offset,
                "row_count": len(part_rows),
                "rows": part_rows,
            }
        )
    return parts


def _storage_call(method, *args, tenant_id: str | None = None):
    parameters = inspect.signature(method).parameters.values()
    accepts_tenant = any(parameter.name == "tenant_id" or parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    if tenant_id and accepts_tenant:
        return method(*args, tenant_id=tenant_id)
    return method(*args)


def _put_verified_immutable(storage, bucket: str, object_name: str, payload: bytes, tenant_id: str | None) -> None:
    if _storage_call(storage.obj_exist, bucket, object_name, tenant_id=tenant_id):
        if _storage_call(storage.get, bucket, object_name, tenant_id=tenant_id) != payload:
            raise IOError("refusing to overwrite immutable structure projection object")
        return
    _storage_call(storage.put, bucket, object_name, payload, tenant_id=tenant_id)
    if not _storage_call(storage.obj_exist, bucket, object_name, tenant_id=tenant_id) or _storage_call(storage.get, bucket, object_name, tenant_id=tenant_id) != payload:
        raise IOError("failed to verify structure projection object")


def store_tabular_structure_projection(
    storage,
    *,
    bucket: str,
    document_id: str,
    projection: dict[str, Any],
    rows_per_part: int = DEFAULT_ROWS_PER_PART,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Write immutable parts and publish their generation manifest last.

    The manifest only proves that the generation was fully persisted. It is not
    an active-generation pointer; activation remains a separate API concern.
    """

    if not bucket or not document_id:
        raise ValueError("bucket and document_id are required")
    parts = partition_tabular_structure_projection(projection, rows_per_part=rows_per_part)
    generation_ref = projection["producer_generation_ref"]
    document_ref = _versioned_digest("tabular-structure-document/v1", document_id)
    prefix = f"_fuxi/tabular-structure/v1/{document_ref}/{generation_ref}"
    manifest_parts = []

    for part in parts:
        payload = _canonical_json(part)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        object_name = f"{prefix}/part-{part['part_number']:06d}-{payload_sha256}.json"
        _put_verified_immutable(storage, bucket, object_name, payload, tenant_id)
        manifest_parts.append(
            {
                "part_number": part["part_number"],
                "object_name": object_name,
                "row_offset": part["row_offset"],
                "row_count": part["row_count"],
                "sha256": payload_sha256,
            }
        )

    manifest = {
        "version": PROJECTION_VERSION,
        "producer_schema_version": projection["producer_schema_version"],
        "producer_generation_ref": generation_ref,
        "source_sha256": projection["source_sha256"],
        "row_count": len(projection["rows"]),
        "tables": projection["tables"],
        "parts": manifest_parts,
    }
    manifest_payload = _canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    manifest_object_name = f"{prefix}/manifest-{manifest_sha256}.json"
    _put_verified_immutable(storage, bucket, manifest_object_name, manifest_payload, tenant_id)
    return {
        "producer_generation_ref": generation_ref,
        "manifest_object_name": manifest_object_name,
        "manifest_sha256": manifest_sha256,
        "part_count": len(parts),
        "row_count": len(projection["rows"]),
    }


def _get_immutable_object(storage, bucket: str, object_name: str, tenant_id: str | None) -> bytes:
    payload = _storage_call(storage.get, bucket, object_name, tenant_id=tenant_id)
    if not isinstance(payload, bytes) or not payload:
        raise StructureSnapshotMissing("structure snapshot object is missing")
    return payload


def _decode_snapshot_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructureSnapshotChanged(f"{label} payload is invalid") from exc
    if not isinstance(value, dict):
        raise StructureSnapshotChanged(f"{label} payload is invalid")
    return value


def load_tabular_structure_projection(
    storage,
    *,
    bucket: str,
    document_id: str,
    producer_generation_ref: str,
    manifest_object_name: str,
    manifest_sha256: str,
    expected_part_count: int | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Read and verify one exact immutable projection generation."""

    if not bucket or not document_id or not manifest_object_name:
        raise StructureSnapshotMissing("structure snapshot identity is missing")
    _validate_generation_ref(producer_generation_ref)
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256 or ""):
        raise StructureSnapshotChanged("manifest digest is invalid")
    document_ref = _versioned_digest("tabular-structure-document/v1", document_id)
    expected_prefix = f"_fuxi/tabular-structure/v1/{document_ref}/{producer_generation_ref}/"
    expected_manifest_name = f"{expected_prefix}manifest-{manifest_sha256}.json"
    if manifest_object_name != expected_manifest_name:
        raise StructureSnapshotChanged("manifest document scope changed")

    manifest_payload = _get_immutable_object(storage, bucket, manifest_object_name, tenant_id)
    if hashlib.sha256(manifest_payload).hexdigest() != manifest_sha256:
        raise StructureSnapshotChanged("manifest digest changed")
    manifest = _decode_snapshot_json(manifest_payload, "manifest")
    expected_manifest_fields = {
        "version",
        "producer_schema_version",
        "producer_generation_ref",
        "source_sha256",
        "row_count",
        "tables",
        "parts",
    }
    if set(manifest) != expected_manifest_fields:
        raise StructureSnapshotChanged("manifest schema changed")
    if manifest["version"] != PROJECTION_VERSION or manifest["producer_schema_version"] != PRODUCER_SCHEMA_VERSION:
        raise StructureSnapshotChanged("manifest version changed")
    if manifest["producer_generation_ref"] != producer_generation_ref:
        raise StructureSnapshotChanged("manifest generation changed")
    if not isinstance(manifest["parts"], list):
        raise StructureSnapshotChanged("manifest parts changed")
    if expected_part_count is not None and (
        not isinstance(expected_part_count, int)
        or isinstance(expected_part_count, bool)
        or expected_part_count < 0
        or len(manifest["parts"]) != expected_part_count
    ):
        raise StructureSnapshotChanged("generation part count changed")

    rows: list[dict[str, Any]] = []
    expected_offset = 0
    for expected_part_number, part_manifest in enumerate(manifest["parts"], start=1):
        expected_part_fields = {"part_number", "object_name", "row_offset", "row_count", "sha256"}
        if not isinstance(part_manifest, dict) or set(part_manifest) != expected_part_fields:
            raise StructureSnapshotChanged("part manifest changed")
        if (
            part_manifest["part_number"] != expected_part_number
            or part_manifest["row_offset"] != expected_offset
            or not isinstance(part_manifest["row_count"], int)
            or isinstance(part_manifest["row_count"], bool)
            or part_manifest["row_count"] < 0
            or not isinstance(part_manifest["object_name"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(part_manifest["sha256"]))
        ):
            raise StructureSnapshotChanged("part manifest changed")
        expected_part_name = f"{expected_prefix}part-{expected_part_number:06d}-{part_manifest['sha256']}.json"
        if part_manifest["object_name"] != expected_part_name:
            raise StructureSnapshotChanged("part document scope changed")

        part_payload = _get_immutable_object(storage, bucket, part_manifest["object_name"], tenant_id)
        if hashlib.sha256(part_payload).hexdigest() != part_manifest["sha256"]:
            raise StructureSnapshotChanged("part digest changed")
        part = _decode_snapshot_json(part_payload, "part")
        if set(part) != {"version", "producer_generation_ref", "part_number", "row_offset", "row_count", "rows"}:
            raise StructureSnapshotChanged("part schema changed")
        if (
            part["version"] != PROJECTION_PART_VERSION
            or part["producer_generation_ref"] != producer_generation_ref
            or part["part_number"] != expected_part_number
            or part["row_offset"] != expected_offset
            or part["row_count"] != part_manifest["row_count"]
            or not isinstance(part["rows"], list)
            or len(part["rows"]) != part["row_count"]
        ):
            raise StructureSnapshotChanged("part generation changed")
        rows.extend(part["rows"])
        expected_offset += part["row_count"]

    if expected_offset != manifest["row_count"]:
        raise StructureSnapshotChanged("manifest row count changed")
    projection = {
        "version": manifest["version"],
        "producer_schema_version": manifest["producer_schema_version"],
        "producer_generation_ref": producer_generation_ref,
        "source_sha256": manifest["source_sha256"],
        "tables": manifest["tables"],
        "rows": rows,
    }
    try:
        validate_tabular_structure_projection(projection)
    except ValueError as exc:
        raise StructureSnapshotChanged("structure projection validation changed") from exc
    return projection


def page_tabular_structure_rows(
    projection: dict[str, Any],
    *,
    table_ref: str,
    cursor: int = 0,
    page_size: int = 30,
) -> dict[str, Any]:
    """Return one stable offset page from an already verified generation."""

    validate_tabular_structure_projection(projection)
    if not isinstance(table_ref, str) or not table_ref:
        raise ValueError("table_ref is required")
    if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
        raise ValueError("cursor must be a non-negative integer")
    if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size < 1:
        raise ValueError("page_size must be a positive integer")

    table = next((item for item in projection["tables"] if item["table_ref"] == table_ref), None)
    if table is None:
        raise StructureSnapshotMissing("table snapshot is missing")
    rows = [row for row in projection["rows"] if row["table_ref_kwd"] == table_ref]
    rows.sort(
        key=lambda row: (
            row["data_row_index_int"] if row["data_row_index_int"] is not None else float("inf"),
            row["row_ordinal_int"],
            row["row_ref_kwd"],
        )
    )
    page_rows = rows[cursor : cursor + page_size]
    next_cursor = cursor + len(page_rows)
    has_more = next_cursor < len(rows)
    return {
        "producer_generation_ref": projection["producer_generation_ref"],
        "table_ref": table_ref,
        "rows": page_rows,
        "total": len(rows),
        "source_total_count": table["source_total_count"],
        "has_more": has_more,
        "next_cursor": next_cursor if has_more else None,
    }
