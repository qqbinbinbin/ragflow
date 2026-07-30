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
import struct
import uuid
from io import BytesIO
from typing import Any


TABULAR_STRUCTURE_VERSION = "tabular-row/v1"
PRODUCER_SCHEMA_VERSION = "table-producer/v3"
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
            row_ordinal
            for row_ordinal, _column_ordinal in getattr(
                worksheet,
                "_fuxi_unresolved_coordinates",
                set(),
            )
        }
    )
    header_rows = list(worksheet.iter_rows(min_row=1, max_row=min(20, worksheet.max_row)))
    return header_rows, populated_rows, unresolved_rows


def _ordered_fields(headers: list[str], values: list[object], *, note: bool) -> list[dict[str, str]]:
    fields = []
    for name, value in zip(headers, values):
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        rendered = (
            str(int(value))
            if isinstance(value, float) and value.is_integer()
            else str(value).strip()
        )
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


def _row_merge_signature(row_ordinal: int, merged_ranges) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            (merged.min_col, merged.max_col)
            for merged in merged_ranges
            if merged.min_row <= row_ordinal <= merged.max_row
        )
    )


def _classify_body_row(
    row_ordinal: int,
    values: list[object],
    merged_ranges,
    *,
    repeated_merge_signatures: set[tuple[tuple[int, int], ...]] | None = None,
) -> str:
    width = len(values)
    populated = sum(value is not None and str(value).strip() != "" for value in values)
    if _is_full_width_merge(row_ordinal, width, merged_ranges):
        return "unknown"
    merge_signature = _row_merge_signature(row_ordinal, merged_ranges)
    if (
        _has_partial_row_merge(row_ordinal, width, merged_ranges)
        and merge_signature not in (repeated_merge_signatures or set())
    ):
        return "unknown"
    if populated >= min(2, width):
        return "data"
    return "unknown"


def _row_shape(values: list[object], *, distinguish_text_digits: bool = False) -> tuple[str, ...]:
    shape = []
    for value in values:
        if value is None or (isinstance(value, str) and not value.strip()):
            shape.append("empty")
        elif isinstance(value, bool):
            shape.append("boolean")
        elif isinstance(value, (int, float)):
            shape.append("number")
        else:
            rendered = str(value).strip()
            if distinguish_text_digits and any(char.isdigit() for char in rendered):
                shape.append("text_with_digit")
            else:
                shape.append("text")
    return tuple(shape)


def _is_repeated_header_row(headers: list[str], values: list[object]) -> bool:
    """Treat an exact repeated header as a structural boundary."""

    if len(headers) != len(values):
        return False
    normalized_values = ["" if value is None else str(value).strip() for value in values]
    if not all(normalized_values):
        return False
    return normalized_values == [str(header).strip() for header in headers]


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


def _region_membership_sha256(sheet_ordinal: int, members: set[tuple[int, int]]) -> str:
    payload = "\n".join(
        f"{sheet_ordinal}:{row_ordinal}:{column_ordinal}"
        for row_ordinal, column_ordinal in sorted(members)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _logical_occupied_cells(
    parser,
    worksheet,
    unresolved_formula_coordinates: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    cells = getattr(worksheet, "_cells", None)
    if not isinstance(cells, dict):
        raise ValueError("worksheet backend cannot prove exact cell membership")
    occupied = {
        (cell.row, cell.column)
        for cell in cells.values()
        if cell.value is not None and str(cell.value).strip() != ""
    }
    for merged in worksheet.merged_cells.ranges:
        anchor = worksheet.cell(merged.min_row, merged.min_col).value
        if anchor is None or str(anchor).strip() == "":
            continue
        occupied.update(
            (row_ordinal, column_ordinal)
            for row_ordinal in range(merged.min_row, merged.max_row + 1)
            for column_ordinal in range(merged.min_col, merged.max_col + 1)
        )
    occupied.update(unresolved_formula_coordinates or set())
    return occupied


def _connected_cell_regions(
    occupied: set[tuple[int, int]],
    *,
    tolerance: int = 2,
) -> list[set[tuple[int, int]]]:
    remaining = set(occupied)
    regions = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        members = {start}
        pending = [start]
        while pending:
            row_ordinal, column_ordinal = pending.pop()
            for row_delta in range(-tolerance, tolerance + 1):
                for column_delta in range(-tolerance, tolerance + 1):
                    if row_delta == 0 and column_delta == 0:
                        continue
                    neighbor = (row_ordinal + row_delta, column_ordinal + column_delta)
                    if neighbor not in remaining:
                        continue
                    remaining.remove(neighbor)
                    members.add(neighbor)
                    pending.append(neighbor)
        regions.append(members)
    return regions


def _region_bbox(members: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [row for row, _column in members]
    columns = [column for _row, column in members]
    return min(rows), min(columns), max(rows), max(columns)


def _cell_distance_to_bbox(
    coordinate: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> int:
    row_ordinal, column_ordinal = coordinate
    min_row, min_column, max_row, max_column = bbox
    row_distance = max(min_row - row_ordinal, 0, row_ordinal - max_row)
    column_distance = max(min_column - column_ordinal, 0, column_ordinal - max_column)
    return max(row_distance, column_distance)


def _cell_distance_to_members(
    coordinate: tuple[int, int],
    members: set[tuple[int, int]],
) -> int:
    row_ordinal, column_ordinal = coordinate
    return min(
        max(abs(row_ordinal - member_row), abs(column_ordinal - member_column))
        for member_row, member_column in members
    )


def _worksheet_structure_regions(
    parser,
    worksheet,
    sheet_ordinal: int,
    unresolved_formula_coordinates: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    occupied = _logical_occupied_cells(parser, worksheet, unresolved_formula_coordinates)
    if not occupied:
        return []
    g1_regions = _connected_cell_regions(occupied, tolerance=1)
    g2_regions = _connected_cell_regions(occupied, tolerance=2)
    bboxes = [_region_bbox(members) for members in g2_regions]
    unresolved = set(unresolved_formula_coordinates or set())
    unresolved_by_region = [set() for _region in g2_regions]
    has_unbound_unresolved = False
    for coordinate in unresolved:
        owners = [
            index
            for index, members in enumerate(g2_regions)
            if _cell_distance_to_members(coordinate, members) <= 2
        ]
        if len(owners) == 1:
            unresolved_by_region[owners[0]].add(coordinate)
        else:
            has_unbound_unresolved = True

    result = []
    for members, unresolved_members, bbox in zip(g2_regions, unresolved_by_region, bboxes):
        g1_children = [child for child in g1_regions if child.issubset(members)]
        result.append(
            {
                "members": members,
                "unresolved_members": unresolved_members,
                "bbox": bbox,
                "membership_sha256": _region_membership_sha256(sheet_ordinal, members),
                "has_unbound_unresolved": has_unbound_unresolved,
                "g1_children": sorted(
                    g1_children,
                    key=lambda child: (*_region_bbox(child), _region_membership_sha256(sheet_ordinal, child)),
                ),
            }
        )
    result.sort(key=lambda region: (*region["bbox"], region["membership_sha256"]))
    return result


def _formula_coordinates_from_biff_stream(stream: bytes) -> list[set[tuple[int, int]]]:
    boundsheet_offsets = []
    cursor = 0
    while cursor + 4 <= len(stream):
        record_id, record_length = struct.unpack_from("<HH", stream, cursor)
        payload_start = cursor + 4
        payload_end = payload_start + record_length
        if payload_end > len(stream):
            raise ValueError("BIFF record exceeds workbook stream")
        payload = stream[payload_start:payload_end]
        if record_id == 0x0085 and record_length >= 6 and payload[5] == 0:
            boundsheet_offsets.append(struct.unpack_from("<I", payload, 0)[0])
        cursor = payload_end

    result = []
    for sheet_offset in boundsheet_offsets:
        formulas = set()
        cursor = sheet_offset
        saw_worksheet_bof = False
        saw_eof = False
        while cursor + 4 <= len(stream):
            record_id, record_length = struct.unpack_from("<HH", stream, cursor)
            payload_start = cursor + 4
            payload_end = payload_start + record_length
            if payload_end > len(stream):
                raise ValueError("BIFF sheet record exceeds workbook stream")
            payload = stream[payload_start:payload_end]
            if record_id == 0x0809 and record_length >= 4:
                saw_worksheet_bof = struct.unpack_from("<H", payload, 2)[0] == 0x0010
            elif record_id == 0x0006 and record_length >= 4:
                row_ordinal, column_ordinal = struct.unpack_from("<HH", payload, 0)
                formulas.add((row_ordinal + 1, column_ordinal + 1))
            elif record_id == 0x000A:
                saw_eof = True
                break
            cursor = payload_end
        if not saw_worksheet_bof or not saw_eof:
            raise ValueError("BIFF worksheet substream is incomplete")
        result.append(formulas)
    if not result:
        raise ValueError("BIFF workbook has no worksheet boundsheets")
    return result


def _formula_coordinates_by_sheet(binary: bytes) -> tuple[list[set[tuple[int, int]]], bool]:
    """Recover formula locations without exposing formula text to projections."""

    if binary.startswith(b"\xd0\xcf\x11\xe0"):
        try:
            import olefile

            with olefile.OleFileIO(BytesIO(binary)) as ole:
                stream_name = next(
                    name
                    for name in ("Workbook", "Book")
                    if ole.exists(name)
                )
                stream = ole.openstream(stream_name).read()
            return _formula_coordinates_from_biff_stream(stream), True
        except Exception:
            return [], False
    if not binary.startswith(b"PK\x03\x04"):
        return [], True
    try:
        from openpyxl import load_workbook

        workbook = load_workbook(BytesIO(binary), data_only=False, read_only=False)
    except Exception:
        return [], False
    result = []
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        result.append(
            {
                (cell.row, cell.column)
                for cell in getattr(worksheet, "_cells", {}).values()
                if cell.__class__.__name__ != "MergedCell" and cell.data_type == "f"
            }
        )
    return result, True


def _region_has_unknown_body_candidate(parser, worksheet, region: dict[str, Any]) -> bool:
    if region["unresolved_members"]:
        return True
    region_worksheet, _row_offset = _copy_structure_region(parser, worksheet, region)
    header_rows, populated_rows, unresolved_rows = _complete_worksheet_rows(region_worksheet)
    if unresolved_rows:
        return True
    if not populated_rows:
        return False
    _headers, _header_start, data_start = _parse_region_structure(parser, region_worksheet, header_rows)
    return any(row_ordinal > data_start for row_ordinal in populated_rows)


def _region_fragment(
    sheet_ordinal: int,
    region: dict[str, Any],
    members: set[tuple[int, int]],
) -> dict[str, Any]:
    unresolved_members = region["unresolved_members"] & members
    return {
        "members": members,
        "unresolved_members": unresolved_members,
        "bbox": _region_bbox(members),
        "membership_sha256": _region_membership_sha256(sheet_ordinal, members),
        "has_unbound_unresolved": region["has_unbound_unresolved"],
        "g1_children": [members],
    }


def _copy_structure_region(parser, worksheet, region: dict[str, Any]):
    from openpyxl import Workbook

    region_workbook = Workbook()
    target = region_workbook.active
    min_row, min_column, _max_row, _max_column = region["bbox"]
    members = region["members"]
    merged_ranges = list(worksheet.merged_cells.ranges)
    merged_coordinates = {
        (row_ordinal, column_ordinal)
        for merged in merged_ranges
        for row_ordinal in range(merged.min_row, merged.max_row + 1)
        for column_ordinal in range(merged.min_col, merged.max_col + 1)
    }

    for row_ordinal, column_ordinal in sorted(members):
        if (row_ordinal, column_ordinal) in merged_coordinates:
            continue
        value = worksheet.cell(row_ordinal, column_ordinal).value
        if value is not None and str(value).strip() != "":
            target.cell(
                row=row_ordinal - min_row + 1,
                column=column_ordinal - min_column + 1,
                value=value,
            )

    for merged in merged_ranges:
        coordinates = {
            (row_ordinal, column_ordinal)
            for row_ordinal in range(merged.min_row, merged.max_row + 1)
            for column_ordinal in range(merged.min_col, merged.max_col + 1)
        }
        if not coordinates.issubset(members):
            continue
        value = worksheet.cell(merged.min_row, merged.min_col).value
        target.cell(
            row=merged.min_row - min_row + 1,
            column=merged.min_col - min_column + 1,
            value=value,
        )
        target.merge_cells(
            start_row=merged.min_row - min_row + 1,
            start_column=merged.min_col - min_column + 1,
            end_row=merged.max_row - min_row + 1,
            end_column=merged.max_col - min_column + 1,
        )

    for row_ordinal, column_ordinal in region["unresolved_members"]:
        target.cell(
            row=row_ordinal - min_row + 1,
            column=column_ordinal - min_column + 1,
        )
    target._fuxi_unresolved_coordinates = {
        (row_ordinal - min_row + 1, column_ordinal - min_column + 1)
        for row_ordinal, column_ordinal in region["unresolved_members"]
    }
    return target, min_row - 1


def _parse_region_structure(parser, worksheet, rows):
    fallback = parser._parse_sheet_structure(worksheet, rows)
    max_scan_rows = min(20, len(rows))
    candidates = []
    merged_ranges = list(worksheet.merged_cells.ranges)
    for start in range(max_scan_rows):
        if parser._is_empty_row([cell.value for cell in rows[start]]):
            continue
        for depth in range(1, max_scan_rows - start + 1):
            end = start + depth
            parent_rows = range(start + 1, end)
            if not all(
                any(
                    merged.min_row <= row_ordinal <= merged.max_row
                    and (merged.max_col > merged.min_col or merged.max_row > merged.min_row)
                    for merged in merged_ranges
                )
                for row_ordinal in parent_rows
            ):
                continue

            headers = parser._build_headers_for_region(worksheet, rows, start, end)
            if not headers:
                continue
            following_offset_counts: dict[tuple[int, ...], int] = {}
            for row_index, row in enumerate(rows[end:], start=end + 1):
                if parser._is_empty_row([cell.value for cell in row]):
                    continue
                if _is_full_width_merge(row_index, len(headers), merged_ranges):
                    continue
                values = [
                    _cell_value(parser, worksheet, row_index, column_index, merged_ranges)
                    for column_index in range(1, len(headers) + 1)
                ]
                offsets = _record_field_offsets(values)
                if offsets:
                    following_offset_counts[offsets] = following_offset_counts.get(offsets, 0) + 1
            proven_offsets = [
                offsets for offsets, count in following_offset_counts.items() if count >= 2
            ]
            if not proven_offsets:
                continue
            record_offsets = max(
                proven_offsets,
                key=lambda offsets: (following_offset_counts[offsets], len(offsets), offsets),
            )
            if any(
                offset >= len(headers) or headers[offset].startswith("Column_")
                for offset in record_offsets
            ):
                continue
            distinct_headers = {headers[offset] for offset in record_offsets}
            if len(distinct_headers) < min(2, len(record_offsets)):
                continue
            candidates.append(
                (
                    len(record_offsets),
                    following_offset_counts[record_offsets],
                    -start,
                    depth,
                    headers,
                    start,
                    end,
                )
            )
    if candidates:
        (
            _field_count,
            _record_count,
            _negative_start,
            _depth,
            headers,
            header_start,
            data_start,
        ) = max(candidates)
        return headers, header_start, data_start
    return fallback


def _record_field_offsets(values: list[object]) -> tuple[int, ...]:
    return tuple(
        index
        for index, value in enumerate(values)
        if value is not None and str(value).strip() != ""
    )


def _header_boundary_proven(
    parser,
    worksheet,
    *,
    header_start: int,
    data_start: int,
    headers: list[str],
    body_rows: list[tuple[int, list[object], bool]],
    merged_ranges,
    repeated_merge_signatures: set[tuple[tuple[int, int], ...]],
) -> bool:
    first_header_row = header_start + 1
    if any(
        merged.min_row <= data_start
        and merged.max_row >= first_header_row
        and (merged.max_col > merged.min_col or merged.max_row > merged.min_row)
        for merged in merged_ranges
    ):
        return True

    header_values = [
        _cell_value(parser, worksheet, data_start, column_ordinal, merged_ranges)
        for column_ordinal in range(1, len(headers) + 1)
    ]
    body_values = [
        values
        for row_ordinal, values, _follows_body_gap in body_rows
        if _classify_body_row(
            row_ordinal,
            values,
            merged_ranges,
            repeated_merge_signatures=repeated_merge_signatures,
        ) == "data"
        and not _is_repeated_header_row(headers, values)
    ]
    if len(body_values) < 2:
        return True
    for column_index, header_value in enumerate(header_values):
        header_kind = _row_shape([header_value], distinguish_text_digits=True)[0]
        body_kinds = {
            _row_shape([values[column_index]], distinguish_text_digits=True)[0]
            for values in body_values
            if column_index < len(values)
            and values[column_index] is not None
            and str(values[column_index]).strip() != ""
        }
        if body_kinds and header_kind not in body_kinds:
            return True
    return False


def _text_structure(value: object) -> tuple[str, ...]:
    rendered = str(value).strip()
    result = []
    for char in rendered:
        if char.isdigit():
            kind = "digit"
        elif char.isalpha():
            kind = "alpha"
        elif char.isspace():
            kind = "space"
        else:
            kind = "symbol"
        if not result or result[-1] != kind:
            result.append(kind)
    return tuple(result)


def _single_column_axis_proven(
    headers: list[str],
    body_rows: list[tuple[int, list[object], bool]],
) -> bool:
    if len(headers) != 1:
        return True
    values = [
        values[0]
        for _row_ordinal, values, _follows_body_gap in body_rows
        if values and values[0] is not None and str(values[0]).strip() != ""
    ]
    if len(values) < 2:
        return False
    if all(isinstance(value, (bool, int, float)) for value in values):
        return True
    shapes = {_text_structure(value) for value in values}
    header_shape = _text_structure(headers[0])
    return (
        len(shapes) == 1
        and next(iter(shapes)) != header_shape
        and any(kind in {"digit", "symbol"} for kind in next(iter(shapes)))
    )


def _context_with_headers(
    context: list[dict[str, str]],
    headers: list[str],
    *,
    entry_limit: int,
    value_bytes: int,
) -> list[dict[str, str]]:
    result = list(context[:entry_limit])
    if len(result) >= entry_limit:
        return result
    for header in headers:
        value = _truncate_utf8(_sanitize_untrusted_text(header), value_bytes)
        if not value or any(item["name"] == "field" and item["value"] == value for item in result):
            continue
        result.append({"name": "field", "value": value})
        if len(result) == entry_limit:
            break
    return result


def _project_structure_region(
    *,
    parser,
    worksheet,
    sheet_name: str,
    sheet_ordinal: int,
    table_ordinal: int,
    source_sha256: str,
    producer_generation_ref: str,
    row_offset: int,
    table_context_entry_limit: int,
    table_context_value_bytes: int,
    force_unknown_total: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    header_rows, populated_row_ordinals, unresolved_row_ordinals = _complete_worksheet_rows(worksheet)
    if not header_rows or not populated_row_ordinals:
        return None
    headers, header_start, data_start = _parse_region_structure(parser, worksheet, header_rows)
    if not headers:
        return None

    body_ordinals = {ordinal for ordinal in populated_row_ordinals if ordinal > data_start}
    unresolved_body_ordinals = {ordinal for ordinal in unresolved_row_ordinals if ordinal > data_start}
    body_ordinals.update(unresolved_body_ordinals)
    if not body_ordinals:
        return None

    table_ref = "tbl_v1_" + _versioned_digest(
        "tabular-table/v1",
        source_sha256,
        sheet_ordinal,
        table_ordinal,
    )
    context = _context_with_headers(
        _table_context(
            parser,
            worksheet,
            header_rows,
            header_start,
            entry_limit=table_context_entry_limit,
            value_bytes=table_context_value_bytes,
        ),
        headers,
        entry_limit=table_context_entry_limit,
        value_bytes=table_context_value_bytes,
    )
    merged_ranges = list(worksheet.merged_cells.ranges)
    body_rows = []
    body_gap_seen = False
    previous_body_ordinal = data_start
    for row_ordinal in sorted(body_ordinals):
        values = [
            _cell_value(parser, worksheet, row_ordinal, column_ordinal, merged_ranges)
            for column_ordinal in range(1, len(headers) + 1)
        ]
        if row_ordinal > previous_body_ordinal + 1:
            body_gap_seen = True
        body_rows.append((row_ordinal, values, body_gap_seen))
        previous_body_ordinal = row_ordinal

    merge_signature_counts = {}
    for row_ordinal, _values, _follows_body_gap in body_rows:
        signature = _row_merge_signature(row_ordinal, merged_ranges)
        if signature:
            merge_signature_counts[signature] = merge_signature_counts.get(signature, 0) + 1
    repeated_merge_signatures = {
        signature for signature, count in merge_signature_counts.items() if count >= 3
    }
    if not _header_boundary_proven(
        parser,
        worksheet,
        header_start=header_start,
        data_start=data_start,
        headers=headers,
        body_rows=body_rows,
        merged_ranges=merged_ranges,
        repeated_merge_signatures=repeated_merge_signatures,
    ):
        return None
    distinguish_text_digits = not any(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for _row_ordinal, values, _follows_body_gap in body_rows
        for value in values
    )
    pending_rows = []
    data_row_index = 0
    has_unknown = False
    established_shape = None
    for body_index, (local_row_ordinal, values, follows_body_gap) in enumerate(body_rows):
        row_role = "unknown"
        row_has_unresolved = any(
            row_ordinal == local_row_ordinal
            for row_ordinal, _column_ordinal in getattr(
                worksheet,
                "_fuxi_unresolved_coordinates",
                set(),
            )
        )
        if row_has_unresolved:
            row_role = "unknown"
        else:
            row_role = _classify_body_row(
                local_row_ordinal,
                values,
                merged_ranges,
                repeated_merge_signatures=repeated_merge_signatures,
            )
            current_shape = _row_shape(values, distinguish_text_digits=distinguish_text_digits)
            next_shape = (
                _row_shape(
                    body_rows[body_index + 1][1],
                    distinguish_text_digits=distinguish_text_digits,
                )
                if body_index + 1 < len(body_rows)
                else None
            )
            if _is_repeated_header_row(headers, values):
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
            established_shape = established_shape or _row_shape(
                values,
                distinguish_text_digits=distinguish_text_digits,
            )
        else:
            current_data_index = None
            has_unknown = has_unknown or row_role == "unknown"
        row_ordinal = local_row_ordinal + row_offset
        row_ref = f"{table_ref}:{row_ordinal}"
        pending_rows.append(
            {
                "id": "tsr_v1_" + _versioned_digest(
                    "tabular-row-record/v1",
                    producer_generation_ref,
                    row_ref,
                ),
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

    data_field_offsets = [
        _record_field_offsets(values)
        for _row_ordinal, values, _follows_body_gap in body_rows
        if _classify_body_row(
            _row_ordinal,
            values,
            merged_ranges,
            repeated_merge_signatures=repeated_merge_signatures,
        ) == "data"
        and not _is_repeated_header_row(headers, values)
    ]
    data_value_shapes = [
        tuple(
            (
                "boolean"
                if isinstance(value, bool)
                else "number"
                if isinstance(value, (int, float))
                else ("text", *_text_structure(value))
                if len(headers) == 1
                else "text"
            )
            for value in values
            if value is not None and str(value).strip() != ""
        )
        for _row_ordinal, values, _follows_body_gap in body_rows
        if _classify_body_row(
            _row_ordinal,
            values,
            merged_ranges,
            repeated_merge_signatures=repeated_merge_signatures,
        ) == "data"
        and not _is_repeated_header_row(headers, values)
    ]
    record_axis_proven = (
        len(data_field_offsets) >= 2
        and len(set(data_field_offsets)) == 1
        and len(set(data_value_shapes)) == 1
        and bool(data_field_offsets[0])
        and data_row_index == len(data_field_offsets)
        and _single_column_axis_proven(headers, body_rows)
    )
    source_total_count = (
        data_row_index
        if record_axis_proven and not has_unknown and not force_unknown_total
        else None
    )
    for record in pending_rows:
        record["source_total_count_int"] = source_total_count
    return (
        {
            "table_ref": table_ref,
            "sheet_ordinal": sheet_ordinal,
            "table_ordinal": table_ordinal,
            "row_count": len(pending_rows),
            "data_row_count": data_row_index,
            "source_total_count": source_total_count,
        },
        pending_rows,
    )


def _members_prove_repeated_axis(region: dict[str, Any]) -> bool:
    rows: dict[int, set[int]] = {}
    for row_ordinal, column_ordinal in region["members"]:
        rows.setdefault(row_ordinal, set()).add(column_ordinal)
    signatures = [
        tuple(column - min(columns) for column in sorted(columns))
        for columns in rows.values()
        if columns
    ]
    return len(signatures) >= 2 and len(set(signatures)) == 1


def _region_aligns_with_projected_columns(
    region: dict[str, Any],
    projected: list[dict[str, Any]],
) -> bool:
    region_columns = {column for _row, column in region["members"]}
    return any(region_columns == item["member_columns"] for item in projected)


def _unknown_structure_region(
    *,
    parser,
    worksheet,
    region: dict[str, Any],
    sheet_name: str,
    sheet_ordinal: int,
    table_ordinal: int,
    source_sha256: str,
    producer_generation_ref: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    table_ref = "tbl_v1_" + _versioned_digest(
        "tabular-table/v1",
        source_sha256,
        sheet_ordinal,
        table_ordinal,
    )
    merged_ranges = list(worksheet.merged_cells.ranges)
    rows = []
    for row_ordinal in sorted({row for row, _column in region["members"]}):
        columns = sorted(column for row, column in region["members"] if row == row_ordinal)
        fields = []
        for column_ordinal in columns:
            value = _cell_value(parser, worksheet, row_ordinal, column_ordinal, merged_ranges)
            if value is None or str(value).strip() == "":
                continue
            fields.extend(
                _ordered_fields(
                    [f"Column_{column_ordinal}"],
                    [value],
                    note=False,
                )
            )
        row_ref = f"{table_ref}:{row_ordinal}"
        rows.append(
            {
                "id": "tsr_v1_" + _versioned_digest(
                    "tabular-row-record/v1",
                    producer_generation_ref,
                    row_ref,
                ),
                "tabular_structure_version_kwd": TABULAR_STRUCTURE_VERSION,
                "structure_kind_kwd": "table_row",
                "producer_schema_version_kwd": PRODUCER_SCHEMA_VERSION,
                "producer_generation_ref_kwd": producer_generation_ref,
                "table_ref_kwd": table_ref,
                "table_label_kwd": _sanitize_untrusted_text(sheet_name),
                "table_context_list": "[]",
                "row_ref_kwd": row_ref,
                "row_ordinal_int": row_ordinal,
                "data_row_index_int": None,
                "row_role_kwd": "unknown",
                "source_total_count_int": None,
                "ordered_fields_list": json.dumps(fields, ensure_ascii=False, separators=(",", ":")),
            }
        )
    if not rows:
        return None
    return (
        {
            "table_ref": table_ref,
            "sheet_ordinal": sheet_ordinal,
            "table_ordinal": table_ordinal,
            "row_count": len(rows),
            "data_row_count": 0,
            "source_total_count": None,
        },
        rows,
    )


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
    formula_coordinates, formula_inventory_proven = _formula_coordinates_by_sheet(binary)
    source_sha256 = hashlib.sha256(binary).hexdigest()
    records = []
    tables = []

    for sheet_ordinal, sheet_name in enumerate(workbook.sheetnames, start=1):
        worksheet = workbook[sheet_name]
        sheet_formula_coordinates = (
            formula_coordinates[sheet_ordinal - 1]
            if sheet_ordinal <= len(formula_coordinates)
            else set()
        )
        regions = _worksheet_structure_regions(
            parser,
            worksheet,
            sheet_ordinal,
            sheet_formula_coordinates,
        )
        projected = []
        unprojected = []
        candidates = []
        for region in regions:
            children = region["g1_children"]
            if len(children) <= 1:
                candidates.append((region, False))
                continue
            # G1/G2 disagreement makes the object boundary unresolved. Keep
            # the exact G2 member set as one unknown candidate until a governed
            # adjudicator can decide whether it contains one object or several.
            candidates.append((region, True))

        sheet_has_unresolved = any(
            region["unresolved_members"] or region["has_unbound_unresolved"]
            for region in regions
        )
        for region_index, (region, force_unknown_total) in enumerate(candidates, start=1):
            region_worksheet, row_offset = _copy_structure_region(parser, worksheet, region)
            result = _project_structure_region(
                parser=parser,
                worksheet=region_worksheet,
                sheet_name=sheet_name,
                sheet_ordinal=sheet_ordinal,
                table_ordinal=region_index,
                source_sha256=source_sha256,
                producer_generation_ref=producer_generation_ref,
                row_offset=row_offset,
                table_context_entry_limit=table_context_entry_limit,
                table_context_value_bytes=table_context_value_bytes,
                force_unknown_total=(
                    force_unknown_total
                    or sheet_has_unresolved
                    or not formula_inventory_proven
                ),
            )
            if force_unknown_total and result is not None:
                result = _unknown_structure_region(
                    parser=parser,
                    worksheet=worksheet,
                    region=region,
                    sheet_name=sheet_name,
                    sheet_ordinal=sheet_ordinal,
                    table_ordinal=region_index,
                    source_sha256=source_sha256,
                    producer_generation_ref=producer_generation_ref,
                )
            if result is None:
                if (
                    _members_prove_repeated_axis(region)
                    or region["unresolved_members"]
                    or _region_aligns_with_projected_columns(region, projected)
                ):
                    result = _unknown_structure_region(
                        parser=parser,
                        worksheet=worksheet,
                        region=region,
                        sheet_name=sheet_name,
                        sheet_ordinal=sheet_ordinal,
                        table_ordinal=region_index,
                        source_sha256=source_sha256,
                        producer_generation_ref=producer_generation_ref,
                )
                if result is None:
                    unprojected.append((region_index, region))
                    continue
            table, table_rows = result
            projected.append(
                {
                    "table": table,
                    "rows": table_rows,
                    "bbox": region["bbox"],
                    "members": region["members"],
                    "member_columns": {column for _row, column in region["members"]},
                }
            )

        if unprojected:
            for region_index, region in unprojected:
                result = _unknown_structure_region(
                    parser=parser,
                    worksheet=worksheet,
                    region=region,
                    sheet_name=sheet_name,
                    sheet_ordinal=sheet_ordinal,
                    table_ordinal=region_index,
                    source_sha256=source_sha256,
                    producer_generation_ref=producer_generation_ref,
                )
                if result is None:
                    continue
                table, table_rows = result
                projected.append(
                    {
                        "table": table,
                        "rows": table_rows,
                        "bbox": region["bbox"],
                        "members": region["members"],
                        "member_columns": {column for _row, column in region["members"]},
                    }
                )

        if unprojected and projected:
            for item in projected:
                item["table"]["source_total_count"] = None
                for row in item["rows"]:
                    row["source_total_count_int"] = None

        if projected and any(region["has_unbound_unresolved"] for region in regions):
            for item in projected:
                item["table"]["source_total_count"] = None
                for row in item["rows"]:
                    row["source_total_count_int"] = None

        projected.sort(
            key=lambda item: (
                item["table"]["sheet_ordinal"],
                item["table"]["table_ordinal"],
            )
        )
        for item in projected:
            item["rows"].sort(key=lambda row: row["row_ordinal_int"])
            item["table"]["row_count"] = len(item["rows"])
            tables.append(item["table"])
            records.extend(item["rows"])

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
        if totals == {None}:
            continue
        if (
            any(row["row_role_kwd"] == "unknown" for row in table_rows)
            or len(data_indices) < 2
            or totals != {len(data_indices)}
        ):
            raise ValueError("source total is not generation-wide or fail-closed")

    tables = projection.get("tables")
    if not isinstance(tables, list):
        raise ValueError("table manifest must be a list")
    manifest_refs = set()
    manifest_coordinates = []
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
        manifest_coordinates.append((table["sheet_ordinal"], table["table_ordinal"]))
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
    if manifest_coordinates != sorted(manifest_coordinates):
        raise ValueError("table manifest is not in deterministic physical order")
    ordinals_by_sheet: dict[int, list[int]] = {}
    for sheet_ordinal, table_ordinal in manifest_coordinates:
        ordinals_by_sheet.setdefault(sheet_ordinal, []).append(table_ordinal)
    if any(
        ordinals != list(range(1, len(ordinals) + 1))
        for ordinals in ordinals_by_sheet.values()
    ):
        raise ValueError("table manifest ordinals are not contiguous within a worksheet")


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
