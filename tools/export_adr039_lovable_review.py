#!/usr/bin/env python3
"""Render Producer audit evidence into the approved content-free Review B TSVs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from collections import Counter
from collections.abc import Iterable
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


SOURCE_REGION_FIELDS = (
    "source_region_ref",
    "worksheet_ordinal",
    "bbox",
    "row_count",
    "column_count",
    "membership_sha256",
    "member_count",
    "member_coordinate_set",
    "assigned_object_ref",
    "assignment_count",
)

OUTPUT_OBJECT_FIELDS = (
    "row_kind",
    "object_ref",
    "worksheet_ordinal",
    "component_region_refs",
    "component_membership_sha256_list",
    "union_membership_sha256",
    "union_member_count",
    "emitted_member_coordinate_multiset",
    "emitted_cell_multiset_sha256",
    "emitted_member_occurrence_count",
    "member_max_ingest_count",
    "record_slot_sha256",
    "proven_record_slot_count",
    "record_slot_coordinate_sets",
    "enumeration_rule_version",
    "structure_generation_ref",
    "table_ref",
    "identity_validation_status",
    "matched_rule",
    "decision_chain_stop",
    "enumeration_status",
    "enumeration_reason",
    "covered_count",
    "source_total_count",
)



def _json_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render_rows(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> str:
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(_json_value(row.get(field)) for field in fields))
    return "\n".join(lines) + "\n"


def review_generation_ref_from_bytes(document_id: str, binary: bytes) -> str:
    from rag.app.tabular_structure_runtime import structure_generation_ref

    return structure_generation_ref(document_id, binary)


def _independent_table_ref(
    source_sha256: str,
    worksheet_ordinal: int,
    table_ordinal: int,
    membership_sha256: str,
) -> str:
    from rag.app.tabular_structure import (
        ENUMERATION_RULE_VERSION,
        PRODUCER_SCHEMA_VERSION,
        PROJECTION_VERSION,
        STRUCTURE_PRODUCER_ALGORITHM_VERSION,
    )

    payload = "\x00".join(
        str(value)
        for value in (
            "tabular-table/v2",
            PRODUCER_SCHEMA_VERSION,
            PROJECTION_VERSION,
            STRUCTURE_PRODUCER_ALGORITHM_VERSION,
            ENUMERATION_RULE_VERSION,
            source_sha256,
            worksheet_ordinal,
            table_ordinal,
            membership_sha256,
        )
    ).encode("utf-8")
    identity = hashlib.sha256(payload).hexdigest()
    return f"tbl_v2_{membership_sha256}_{identity}"


def _validated_output_rows(
    audit: dict[str, Any],
    expected_generation_ref: str,
) -> list[dict[str, Any]]:
    if audit["producer_generation_ref"] != expected_generation_ref:
        raise ValueError("Producer audit generation identity failed independent validation")

    table_ordinals: Counter[int] = Counter()
    rows = []
    for source in audit["output_objects"]:
        row = {field: source.get(field) for field in OUTPUT_OBJECT_FIELDS}
        if source["row_kind"] == "object":
            worksheet_ordinal = source["worksheet_ordinal"]
            table_ordinals[worksheet_ordinal] += 1
            expected_table_ref = _independent_table_ref(
                audit["source_sha256"],
                worksheet_ordinal,
                table_ordinals[worksheet_ordinal],
                source["union_membership_sha256"],
            )
            if source["table_ref"] != expected_table_ref or source["object_ref"] != expected_table_ref:
                raise ValueError("Producer audit table identity failed independent validation")
        elif source["row_kind"] != "defect_tombstone":
            raise ValueError("Producer audit output row kind is invalid")
        row["identity_validation_status"] = "validated"
        rows.append(row)
    return rows


def _validate_audit_shape(audit: dict[str, Any]) -> None:
    if not isinstance(audit, dict):
        raise ValueError("Review B input must be a Producer audit object")
    required = {"version", "producer_generation_ref", "enumeration_rule_version", "source_sha256", "source_regions", "output_objects", "defects"}
    if not required.issubset(audit):
        raise ValueError("Review B input is missing Producer audit fields")
    if audit["version"] != "tabular-structure-producer-audit/v1":
        raise ValueError("unsupported Producer audit version")
    if not isinstance(audit["source_regions"], list) or not isinstance(audit["output_objects"], list):
        raise ValueError("Review B audit collections must be lists")


def render_review_b_tsv(
    audit: dict[str, Any],
    *,
    expected_generation_ref: str,
) -> tuple[str, str]:
    """Return `(source_regions.tsv, output_objects.tsv)` with no private fields."""

    _validate_audit_shape(audit)
    from rag.app.tabular_structure import _validate_tabular_structure_producer_audit

    _validate_tabular_structure_producer_audit(audit)
    source_rows = [{field: row.get(field) for field in SOURCE_REGION_FIELDS} for row in audit["source_regions"]]
    output_rows = _validated_output_rows(audit, expected_generation_ref)
    return (
        _render_rows(source_rows, SOURCE_REGION_FIELDS),
        _render_rows(output_rows, OUTPUT_OBJECT_FIELDS),
    )


def write_review_b_tsv(
    audit: dict[str, Any],
    output_directory: str | Path,
    *,
    expected_generation_ref: str,
) -> tuple[Path, Path]:
    source_tsv, output_tsv = render_review_b_tsv(
        audit,
        expected_generation_ref=expected_generation_ref,
    )
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "source_regions.tsv"
    output_path = directory / "output_objects.tsv"
    source_path.write_text(source_tsv, encoding="utf-8", newline="")
    output_path.write_text(output_tsv, encoding="utf-8", newline="")
    return source_path, output_path


def _production_excel_parser():
    """Load the production Excel classes without importing unrelated DB services."""

    replacements: dict[str, types.ModuleType] = {}
    rag_nlp = types.ModuleType("rag.nlp")
    rag_nlp.find_codec = lambda *_args, **_kwargs: "utf-8"
    rag_nlp.rag_tokenizer = types.SimpleNamespace(tokenize=lambda text: text.split())
    rag_nlp.tokenize = lambda *_args, **_kwargs: None
    rag_nlp.tokenize_table = lambda *_args, **_kwargs: []
    replacements["rag.nlp"] = rag_nlp

    lazy_image = types.ModuleType("rag.utils.lazy_image")
    lazy_image.LazyImage = object
    replacements["rag.utils.lazy_image"] = lazy_image

    excel_parser_path = REPOSITORY_ROOT / "deepdoc/parser/excel_parser.py"
    excel_spec = importlib.util.spec_from_file_location(
        "adr039_review_b_excel_parser",
        excel_parser_path,
    )
    if excel_spec is None or excel_spec.loader is None:
        raise RuntimeError("cannot load the production Excel parser")

    original_modules = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        excel_module = importlib.util.module_from_spec(excel_spec)
        excel_spec.loader.exec_module(excel_module)

        knowledgebase_service = types.ModuleType("api.db.services.knowledgebase_service")
        knowledgebase_service.KnowledgebaseService = type("KnowledgebaseService", (), {})
        figure_parser = types.ModuleType("deepdoc.parser.figure_parser")
        figure_parser.vision_figure_parser_figure_xlsx_wrapper = lambda **_kwargs: []
        parser_utils = types.ModuleType("deepdoc.parser.utils")
        parser_utils.get_text = lambda *_args, **_kwargs: ""
        deepdoc_parser = types.ModuleType("deepdoc.parser")
        deepdoc_parser.ExcelParser = excel_module.RAGFlowExcelParser
        common = types.ModuleType("common")
        common.settings = types.SimpleNamespace(
            DOC_ENGINE_INFINITY=False,
            DOC_ENGINE_OCEANBASE=False,
        )
        common_constants = types.ModuleType("common.constants")
        common_constants.MAXIMUM_TASK_PAGE_NUMBER = 100000
        xpinyin = types.ModuleType("xpinyin")
        xpinyin.Pinyin = type("Pinyin", (), {})
        table_replacements = {
            "api.db.services.knowledgebase_service": knowledgebase_service,
            "deepdoc.parser.figure_parser": figure_parser,
            "deepdoc.parser.utils": parser_utils,
            "deepdoc.parser": deepdoc_parser,
            "common": common,
            "common.constants": common_constants,
            "xpinyin": xpinyin,
        }
        for name, module in table_replacements.items():
            if name not in original_modules:
                original_modules[name] = sys.modules.get(name)
            sys.modules[name] = module

        loader = SourceFileLoader(
            "adr039_review_b_table_parser",
            str(REPOSITORY_ROOT / "rag/app/table.py"),
        )
        table_spec = importlib.util.spec_from_loader(loader.name, loader)
        if table_spec is None or table_spec.loader is None:
            raise RuntimeError("cannot load the production table parser")
        table_module = importlib.util.module_from_spec(table_spec)
        table_spec.loader.exec_module(table_module)
        return table_module.Excel()
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _build_audit(binary: bytes, document_id: str) -> dict[str, Any]:
    from rag.app.tabular_structure import _build_tabular_structure_projection_with_audit

    filename = "source.xlsx" if binary.startswith(b"PK\x03\x04") else "source.xls"
    _projection, audit = _build_tabular_structure_projection_with_audit(
        filename,
        binary,
        producer_generation_ref=review_generation_ref_from_bytes(document_id, binary),
        parser=_production_excel_parser(),
    )
    return audit


def export_source_bound_review_b(
    source_path: str | Path,
    expected_source_sha256: str,
    output_directory: str | Path,
    document_id: str,
) -> dict[str, Any]:
    source = Path(source_path)
    binary = source.read_bytes()
    source_sha256 = hashlib.sha256(binary).hexdigest()
    if source_sha256 != expected_source_sha256:
        raise ValueError("source SHA-256 mismatch; refusing to parse")

    expected_generation_ref = review_generation_ref_from_bytes(document_id, binary)
    audit = _build_audit(binary, document_id)
    source_tsv, output_tsv = render_review_b_tsv(
        audit,
        expected_generation_ref=expected_generation_ref,
    )
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source_regions.tsv").write_text(source_tsv, encoding="utf-8", newline="")
    (directory / "output_objects.tsv").write_text(output_tsv, encoding="utf-8", newline="")

    status_counts = Counter(row["enumeration_status"] for row in audit["output_objects"])
    l2_reasons = Counter(
        row["enumeration_reason"]
        for row in audit["output_objects"]
        if row["enumeration_status"] == "not_guaranteed_explained"
    )
    defects = [
        {
            "row_kind": row["row_kind"],
            "matched_rule": row["matched_rule"],
            "enumeration_reason": row["enumeration_reason"],
            "worksheet_ordinal": row["worksheet_ordinal"],
            "component_region_refs": row["component_region_refs"],
        }
        for row in audit["output_objects"]
        if row["enumeration_status"] == "defect"
    ]
    return {
        "source_region_count": len(audit["source_regions"]),
        "output_object_count": len(audit["output_objects"]),
        "supported_complete_count": status_counts["supported_complete"],
        "l2_reason_distribution": dict(sorted(l2_reasons.items())),
        "genuine_defects": defects,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--document-id", required=True)
    args = parser.parse_args()
    summary = export_source_bound_review_b(
        args.source,
        args.expected_source_sha256,
        args.output_directory,
        args.document_id,
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
