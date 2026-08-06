import hashlib
import json
import struct
import uuid
from io import BytesIO

import pytest
from openpyxl import Workbook
from openpyxl.styles import PatternFill

from test.fuxi.test_table_semantic_rows import _load_table_module

import rag.app.tabular_structure as tabular_structure

from rag.app.tabular_structure import (
    PRODUCER_SCHEMA_VERSION,
    PROJECTION_ROW_FIELDS,
    _ordered_fields,
    _formula_coordinates_from_biff_stream,
    build_tabular_structure_projection,
    partition_tabular_structure_projection,
    store_tabular_structure_projection,
    validate_tabular_structure_projection,
)


def test_current_producer_versions_invalidate_pre_enumeration_generations():
    assert tabular_structure.TABULAR_STRUCTURE_VERSION == "tabular-row/v2"
    assert PRODUCER_SCHEMA_VERSION == "table-producer/v6"
    assert tabular_structure.PROJECTION_VERSION == "tabular-structure-projection/v6"
    assert tabular_structure.PROJECTION_PART_VERSION == "tabular-structure-part/v3"
    assert tabular_structure.STRUCTURE_PRODUCER_ALGORITHM_VERSION == "region-producer/v12"
    assert tabular_structure.ENUMERATION_RULE_VERSION == "enumeration-rules/v4"


def test_structurally_closed_header_only_table_proves_an_empty_record_axis(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous empty register"
    sheet.merge_cells("A1:F1")
    sheet["A1"] = "Anonymous register"
    sheet.append(
        [
            "Sequence",
            "Reference",
            "Received",
            "Issuer",
            "Issued",
            "Reason",
        ]
    )

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["rows"] == []
    assert len(projection["tables"]) == 1
    assert projection["tables"][0] == {
        "table_ref": projection["tables"][0]["table_ref"],
        "sheet_ordinal": 1,
        "table_ordinal": 1,
        "row_count": 0,
        "data_row_count": 0,
        "source_total_count": 0,
        "table_label": "Anonymous empty register",
        "table_context": [
            {"name": "context", "value": "Anonymous register"},
            *(
                {"name": "field", "value": name}
                for name in (
                    "Sequence",
                    "Reference",
                    "Received",
                    "Issuer",
                    "Issued",
                    "Reason",
                )
            ),
        ],
        "ordered_columns": [
            {
                "column_id": f"col_v1:1:{ordinal}",
                "column_ordinal": ordinal,
                "header_path": [name],
                "name": name,
            }
            for ordinal, name in enumerate(
                (
                    "Sequence",
                    "Reference",
                    "Received",
                    "Issuer",
                    "Issued",
                    "Reason",
                ),
                start=1,
            )
        ],
        "enumeration_status": "supported_complete",
        "enumeration_reason": "empty_record_axis_proven",
        "matched_rule": "L1-08",
    }


def test_empty_record_axis_ignores_a_disjoint_sidecar_outside_the_table_columns(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous empty register"
    sheet.merge_cells("A1:G1")
    sheet["A1"] = "Anonymous register"
    headers = [
        "Sequence",
        "Reference",
        "Received",
        "Issuer",
        "Issued",
        "Reason",
        "Category",
    ]
    for column, header in enumerate(headers, start=1):
        sheet.cell(row=2, column=column, value=header)
    sheet["H2"] = "Resource type"
    sheet["I2"] = "Related resource"
    sheet["I2"].hyperlink = "https://example.invalid/resource"

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    complete = [
        table
        for table in projection["tables"]
        if table["enumeration_status"] == "supported_complete"
    ]
    assert len(complete) == 1
    assert complete[0]["matched_rule"] == "L1-08"
    assert complete[0]["source_total_count"] == 0
    assert [
        column["column_id"] for column in complete[0]["ordered_columns"]
    ] == [f"col_v1:1:{ordinal}" for ordinal in range(1, 8)]
    assert all(
        column["name"] != "Related resource"
        for column in complete[0]["ordered_columns"]
    )
    expected_membership = tabular_structure._region_membership_sha256(
        1,
        {
            (row, column)
            for row in (1, 2)
            for column in range(1, 8)
        },
    )
    assert complete[0]["table_ref"].startswith(
        f"tbl_v2_{expected_membership}_"
    )
    assert all(
        row["table_ref_kwd"] != complete[0]["table_ref"]
        for row in projection["rows"]
    )


def test_merged_title_with_trailing_blank_header_and_link_sidecar_proves_empty_record_axis(
    table_parser,
):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous header-only register"
    sheet.merge_cells("A1:H1")
    sheet["A1"] = "Anonymous register"
    for column, header in enumerate(
        (
            "Sequence",
            "Reference",
            "Received",
            "Issuer",
            "Issued",
            "Reason",
            "Category",
        ),
        start=1,
    ):
        sheet.cell(row=2, column=column, value=header)
    sheet["I2"] = "Related resource"
    sheet["I2"].hyperlink = "https://example.invalid/resource"

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    complete = [
        table
        for table in projection["tables"]
        if table["enumeration_status"] == "supported_complete"
    ]
    assert len(complete) == 1
    assert complete[0]["matched_rule"] == "L1-08"
    assert complete[0]["source_total_count"] == 0
    assert [column["name"] for column in complete[0]["ordered_columns"]] == [
        "Sequence",
        "Reference",
        "Received",
        "Issuer",
        "Issued",
        "Reason",
        "Category",
    ]


def test_unmerged_wide_single_row_is_not_promoted_to_an_empty_list(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(
        [
            "Prepared by",
            "Approved by",
            "Reviewed by",
            "Issued on",
            "Status",
            "Revision",
            "Comment",
        ]
    )

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert all(table["matched_rule"] != "L1-08" for table in projection["tables"])


def test_trailing_blank_does_not_hide_content_inside_the_title_span(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.merge_cells("A1:I1")
    sheet["A1"] = "Anonymous register"
    for column, header in enumerate(
        ("Sequence", "Reference", "Received", "Issuer", "Issued", "Reason", "Category"),
        start=1,
    ):
        sheet.cell(row=2, column=column, value=header)
    sheet["I2"] = "Ambiguous content"

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert all(table["matched_rule"] != "L1-08" for table in projection["tables"])


def test_empty_record_axis_does_not_ignore_content_below_its_table_columns(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.merge_cells("A1:G1")
    sheet["A1"] = "Anonymous register"
    for column, header in enumerate(
        ("Sequence", "Reference", "Received", "Issuer", "Issued", "Reason", "Category"),
        start=1,
    ):
        sheet.cell(row=2, column=column, value=header)
    sheet["H2"] = "Resource type"
    sheet["I2"] = "Related resource"
    sheet["I2"].hyperlink = "https://example.invalid/resource"
    sheet["C4"] = "Unresolved content"

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert all(table["matched_rule"] != "L1-08" for table in projection["tables"])


def test_two_cell_signoff_row_is_not_promoted_to_an_empty_list(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Prepared by", "Approved by"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert all(table["matched_rule"] != "L1-08" for table in projection["tables"])


def test_header_with_unresolved_body_content_is_not_promoted_to_an_empty_list(
    table_parser,
):
    workbook = Workbook()
    sheet = workbook.active
    sheet.merge_cells("A1:C1")
    sheet["A1"] = "Anonymous register"
    sheet.append(["Sequence", "Reference", "Reason"])
    sheet.merge_cells("A3:C3")
    sheet["A3"] = "Pending note"

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert all(table["matched_rule"] != "L1-08" for table in projection["tables"])


def test_table_ref_identity_binds_all_versions_and_exact_membership(monkeypatch):
    source_sha256 = hashlib.sha256(b"same source").hexdigest()
    first = tabular_structure._table_ref(source_sha256, 1, 1, "a" * 64)
    changed_members = tabular_structure._table_ref(source_sha256, 1, 1, "b" * 64)

    monkeypatch.setattr(tabular_structure, "PRODUCER_SCHEMA_VERSION", "table-producer/test-next")
    changed_schema = tabular_structure._table_ref(source_sha256, 1, 1, "a" * 64)
    monkeypatch.setattr(tabular_structure, "PRODUCER_SCHEMA_VERSION", PRODUCER_SCHEMA_VERSION)
    monkeypatch.setattr(
        tabular_structure,
        "PROJECTION_VERSION",
        "tabular-structure-projection/test-next",
    )
    changed_projection = tabular_structure._table_ref(source_sha256, 1, 1, "a" * 64)
    monkeypatch.setattr(
        tabular_structure,
        "PROJECTION_VERSION",
        "tabular-structure-projection/v4",
    )
    monkeypatch.setattr(
        tabular_structure,
        "STRUCTURE_PRODUCER_ALGORITHM_VERSION",
        "region-producer/test-next",
    )
    changed_algorithm = tabular_structure._table_ref(source_sha256, 1, 1, "a" * 64)
    monkeypatch.setattr(
        tabular_structure,
        "ENUMERATION_RULE_VERSION",
        "enumeration-rules/test-next",
        raising=False,
    )
    changed_enumeration_rules = tabular_structure._table_ref(source_sha256, 1, 1, "a" * 64)

    assert first.startswith("tbl_v2_" + "a" * 64 + "_")
    assert len(
        {
            first,
            changed_members,
            changed_schema,
            changed_projection,
            changed_algorithm,
            changed_enumeration_rules,
        }
    ) == 6


def test_projection_root_requires_the_current_enumeration_rule_version(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )

    assert projection["enumeration_rule_version"] == "enumeration-rules/v4"

    missing = dict(projection)
    missing.pop("enumeration_rule_version")
    with pytest.raises(ValueError, match="fixed top-level schema"):
        validate_tabular_structure_projection(missing)

    changed = dict(projection)
    changed["enumeration_rule_version"] = "enumeration-rules/unknown"
    with pytest.raises(ValueError, match="enumeration rule version"):
        validate_tabular_structure_projection(changed)


def test_table_manifest_requires_one_strict_enumeration_decision(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    table = projection["tables"][0]

    assert {
        "enumeration_status": table["enumeration_status"],
        "enumeration_reason": table["enumeration_reason"],
        "matched_rule": table["matched_rule"],
    } == {
        "enumeration_status": "supported_complete",
        "enumeration_reason": "record_axis_proven",
        "matched_rule": "L1-05",
    }

    for field in ("enumeration_status", "enumeration_reason", "matched_rule"):
        missing = json.loads(json.dumps(projection))
        missing["tables"][0].pop(field)
        with pytest.raises(ValueError, match="table manifest"):
            validate_tabular_structure_projection(missing)

    changed = json.loads(json.dumps(projection))
    changed["tables"][0]["enumeration_reason"] = "record_axis_not_proven"
    with pytest.raises(ValueError, match="enumeration decision"):
        validate_tabular_structure_projection(changed)

    invalid_rule = json.loads(json.dumps(projection))
    invalid_rule["tables"][0]["matched_rule"] = []
    with pytest.raises(ValueError, match="enumeration decision"):
        validate_tabular_structure_projection(invalid_rule)


def test_duplicate_source_member_ingestion_reaches_d2(table_parser, monkeypatch):
    original = tabular_structure._new_projected_item

    def duplicate_member_event(**kwargs):
        item = original(**kwargs)
        item["emitted_member_events"].append(item["emitted_member_events"][0])
        return item

    monkeypatch.setattr(tabular_structure, "_new_projected_item", duplicate_member_event)
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "D2"
    assert projection["tables"][0]["enumeration_reason"] == "membership_not_closed"


def test_unassigned_source_member_ingestion_reaches_d2(table_parser, monkeypatch):
    original = tabular_structure._new_projected_item

    def drop_last_member_event(**kwargs):
        item = original(**kwargs)
        item["emitted_member_events"] = item["emitted_member_events"][:-1]
        return item

    monkeypatch.setattr(tabular_structure, "_new_projected_item", drop_last_member_event)
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )

    assert projection["tables"][0]["matched_rule"] == "D2"
    assert projection["tables"][0]["enumeration_reason"] == "membership_not_closed"


def test_proven_record_slot_count_mismatch_reaches_d3(table_parser, monkeypatch):
    original = tabular_structure._new_projected_item

    def add_unemitted_proven_slot(**kwargs):
        item = original(**kwargs)
        item["proven_record_slots"].append(99)
        return item

    monkeypatch.setattr(tabular_structure, "_new_projected_item", add_unemitted_proven_slot)
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "D3"
    assert projection["tables"][0]["enumeration_reason"] == "record_count_mismatch"


def test_proven_record_slots_must_match_emitted_data_row_ordinals(table_parser, monkeypatch):
    original = tabular_structure._new_projected_item

    def replace_first_slot_with_header(**kwargs):
        item = original(**kwargs)
        item["proven_record_slots"][0] = min(row for row, _column in item["members"])
        return item

    monkeypatch.setattr(
        tabular_structure,
        "_new_projected_item",
        replace_first_slot_with_header,
    )
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "D3"
    assert projection["tables"][0]["enumeration_reason"] == "record_count_mismatch"


def test_missing_unknown_projection_emits_d4_tombstone(table_parser, monkeypatch):
    workbook = Workbook()
    workbook.active["A1"] = "Standalone note"
    monkeypatch.setattr(tabular_structure, "_unknown_structure_region", lambda **_kwargs: None)

    projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"] == []
    assert projection["rows"] == []
    assert set(projection) == tabular_structure.PROJECTION_FIELDS
    assert audit["version"] == "tabular-structure-producer-audit/v1"
    assert audit["producer_generation_ref"] == projection["producer_generation_ref"]
    assert audit["enumeration_rule_version"] == projection["enumeration_rule_version"]
    assert audit["source_sha256"] == projection["source_sha256"]
    assert audit["defects"] == [
        {
            "row_kind": "defect_tombstone",
            "table_ref": None,
            "sheet_ordinal": 1,
            "source_region_ordinal": 1,
            "membership_sha256": tabular_structure._region_membership_sha256(1, {(1, 1)}),
            "enumeration_status": "defect",
            "enumeration_reason": "missing_projection",
            "matched_rule": "D4",
        }
    ]

    fabricated_identity = json.loads(json.dumps(audit))
    fabricated_identity["defects"][0]["table_ref"] = "tbl_v2_" + "0" * 128
    with pytest.raises(ValueError, match="cannot fabricate"):
        tabular_structure._validate_tabular_structure_producer_audit(fabricated_identity)

    wrong_reason = json.loads(json.dumps(audit))
    wrong_reason["defects"][0]["enumeration_reason"] = "membership_not_closed"
    with pytest.raises(ValueError, match="decision is invalid"):
        tabular_structure._validate_tabular_structure_producer_audit(wrong_reason)

    duplicate_region = json.loads(json.dumps(audit))
    duplicate_region["defects"].append(dict(duplicate_region["defects"][0]))
    with pytest.raises(ValueError, match="not unique deterministic"):
        tabular_structure._validate_tabular_structure_producer_audit(duplicate_region)


def test_producer_audit_exports_content_free_closed_object_evidence(table_parser):
    projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )

    assert set(projection) == tabular_structure.PROJECTION_FIELDS
    assert audit["defects"] == []
    assert len(audit["source_regions"]) == 1
    assert len(audit["output_objects"]) == 1
    source = audit["source_regions"][0]
    output = audit["output_objects"][0]
    assert source["assigned_object_ref"] == output["object_ref"] == projection["tables"][0]["table_ref"]
    assert source["assignment_count"] == 1
    assert output["row_kind"] == "object"
    assert output["identity_validation_status"] == "pending_independent_validation"
    assert output["enumeration_status"] == "supported_complete"
    tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_rejects_record_slot_outside_union_members(table_parser):
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    forged = json.loads(json.dumps(audit))
    output = forged["output_objects"][0]
    output["record_slot_coordinate_sets"][0] = ["1:999:999"]
    output["record_slot_sha256"] = tabular_structure._audit_digest(
        "adr039-record-slot/v1",
        ["|".join(slot) for slot in output["record_slot_coordinate_sets"]],
    )

    with pytest.raises(ValueError, match="record slot.*union"):
        tabular_structure._validate_tabular_structure_producer_audit(forged)


def test_producer_audit_rejects_duplicate_complete_record_slots(table_parser):
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    forged = json.loads(json.dumps(audit))
    output = forged["output_objects"][0]
    output["record_slot_coordinate_sets"][1] = list(
        output["record_slot_coordinate_sets"][0]
    )
    output["record_slot_sha256"] = tabular_structure._audit_digest(
        "adr039-record-slot/v1",
        ["|".join(slot) for slot in output["record_slot_coordinate_sets"]],
    )

    with pytest.raises(ValueError, match="record slots.*unique"):
        tabular_structure._validate_tabular_structure_producer_audit(forged)


def test_producer_audit_rejects_header_row_as_a_complete_record_slot(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code"])
    sheet.append(["S-1"])
    sheet.append(["S-2"])
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )
    forged = json.loads(json.dumps(audit))
    output = forged["output_objects"][0]
    assert output.get("emitted_data_row_ordinals") == [2, 3]
    output["record_slot_coordinate_sets"] = [["1:1:1"], ["1:2:1"]]
    output["record_slot_sha256"] = tabular_structure._audit_digest(
        "adr039-record-slot/v1",
        ["1:1:1", "1:2:1"],
    )

    with pytest.raises(ValueError, match="record slots.*data rows"):
        tabular_structure._validate_tabular_structure_producer_audit(forged)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bbox", [1, 1, 999, 999], "source geometry"),
        ("row_count", 999, "source geometry"),
        ("column_count", 999, "source geometry"),
        ("worksheet_ordinal", 2, "source region reference"),
        ("source_region_ref", "1:99", "source region reference"),
    ],
)
def test_producer_audit_recomputes_source_region_geometry(
    table_parser,
    field,
    value,
    message,
):
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    audit["source_regions"][0][field] = value
    if field == "source_region_ref":
        audit["output_objects"][0]["component_region_refs"][0] = value

    with pytest.raises(ValueError, match=message):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_cannot_self_approve_identity_validation(table_parser):
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    audit["output_objects"][0]["identity_validation_status"] = "validated"

    with pytest.raises(ValueError, match="identity validation status"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_requires_each_d4_defect_to_match_its_tombstone(
    table_parser,
    monkeypatch,
):
    workbook = Workbook()
    workbook.active["A1"] = "Standalone note"
    monkeypatch.setattr(tabular_structure, "_unknown_structure_region", lambda **_kwargs: None)
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )
    audit["defects"] = []

    with pytest.raises(ValueError, match="D4 defect.*tombstone"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_rejects_overlapping_complete_components(table_parser):
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _vertical_complete_with_headerless_continuation_bytes(),
        parser=table_parser,
    )
    assert len(audit["source_regions"]) == 2
    first, second = audit["source_regions"]
    second["member_coordinate_set"].append(first["member_coordinate_set"][0])
    second["member_coordinate_set"].sort(key=tabular_structure._audit_coordinate_key)
    second["membership_sha256"] = hashlib.sha256(
        "\n".join(second["member_coordinate_set"]).encode("ascii")
    ).hexdigest()
    second["member_count"] = len(second["member_coordinate_set"])
    parsed_members = [
        tabular_structure._audit_coordinate_key(value)
        for value in second["member_coordinate_set"]
    ]
    member_rows = {row for _sheet, row, _column in parsed_members}
    member_columns = {column for _sheet, _row, column in parsed_members}
    second["bbox"] = [
        min(member_rows),
        min(member_columns),
        max(member_rows),
        max(member_columns),
    ]
    second["row_count"] = len(member_rows)
    second["column_count"] = len(member_columns)
    output = audit["output_objects"][0]
    output["component_membership_sha256_list"][1] = second["membership_sha256"]

    with pytest.raises(ValueError, match="component memberships.*disjoint"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_recomputes_table_identity(table_parser):
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    forged_ref = "tbl_v2_" + audit["output_objects"][0]["union_membership_sha256"] + "_" + "0" * 64
    output = audit["output_objects"][0]
    output["object_ref"] = forged_ref
    output["table_ref"] = forged_ref
    audit["source_regions"][0]["assigned_object_ref"] = forged_ref

    with pytest.raises(ValueError, match="table identity"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_binds_output_worksheet_to_components(table_parser):
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    audit["output_objects"][0]["worksheet_ordinal"] = 2

    with pytest.raises(ValueError, match="output worksheet"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_requires_numeric_component_order(table_parser):
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _vertical_complete_with_headerless_continuation_bytes(),
        parser=table_parser,
    )
    output = audit["output_objects"][0]
    output["component_region_refs"].reverse()
    output["component_membership_sha256_list"].reverse()

    with pytest.raises(ValueError, match="component references.*ordered"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_requires_multiple_assignments_to_reach_d2(table_parser):
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    first = audit["output_objects"][0]
    second = json.loads(json.dumps(first))
    second_ref = tabular_structure._table_ref(
        audit["source_sha256"],
        1,
        2,
        second["union_membership_sha256"],
    )
    second.update(
        {
            "object_ref": second_ref,
            "table_ref": second_ref,
            "matched_rule": "R8",
            "decision_chain_stop": "R8",
            "enumeration_status": "not_guaranteed_explained",
            "enumeration_reason": "record_axis_not_proven",
            "source_total_count": None,
        }
    )
    audit["output_objects"].append(second)
    audit["source_regions"][0]["assigned_object_ref"] = None
    audit["source_regions"][0]["assignment_count"] = 2

    with pytest.raises(ValueError, match="multiple assignments.*D2"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_requires_the_decision_stop_to_match_the_rule(table_parser):
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    audit["output_objects"][0]["decision_chain_stop"] = "R8"

    with pytest.raises(ValueError, match="decision chain stop"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_rejects_a_total_on_a_noncomplete_decision(table_parser):
    workbook = Workbook()
    workbook.active.append(["Code", "State"])
    workbook.active.append(["A-1", "Open"])
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )
    assert audit["output_objects"][0]["enumeration_status"] == "not_guaranteed_explained"
    audit["output_objects"][0]["source_total_count"] = 1

    with pytest.raises(ValueError, match="decision conflicts with source total"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_requires_unclosed_membership_to_be_d2(table_parser):
    workbook = Workbook()
    workbook.active.append(["Code", "State"])
    workbook.active.append(["A-1", "Open"])
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )
    output = audit["output_objects"][0]
    output["emitted_member_coordinate_multiset"] = output[
        "emitted_member_coordinate_multiset"
    ][:-1]
    emitted = output["emitted_member_coordinate_multiset"]
    output["emitted_cell_multiset_sha256"] = tabular_structure._audit_digest(
        "adr039-emitted-cell-multiset/v1",
        emitted,
    )
    output["emitted_member_occurrence_count"] = len(emitted)
    output["member_max_ingest_count"] = 1

    with pytest.raises(ValueError, match="unclosed membership.*D2"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_record_slots_cover_the_complete_source_rows(table_parser):
    workbook = Workbook()
    workbook.active.append(["Code", "State"])
    workbook.active.append(["A-1", "Open"])
    workbook.active.append(["A-2", "Closed"])
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )
    output = audit["output_objects"][0]
    output["record_slot_coordinate_sets"][0] = output[
        "record_slot_coordinate_sets"
    ][0][:-1]
    output["record_slot_sha256"] = tabular_structure._audit_digest(
        "adr039-record-slot/v1",
        ["|".join(slot) for slot in output["record_slot_coordinate_sets"]],
    )

    with pytest.raises(ValueError, match="record slot.*complete source row"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


@pytest.mark.parametrize("defect_rule", ["D2", "D3", "D4"])
def test_producer_audit_rejects_unsupported_defect_labels(table_parser, defect_rule):
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    output = audit["output_objects"][0]
    status, reason = tabular_structure.ENUMERATION_DECISIONS[defect_rule]
    output.update(
        {
            "matched_rule": defect_rule,
            "decision_chain_stop": defect_rule,
            "enumeration_status": status,
            "enumeration_reason": reason,
            "source_total_count": None,
        }
    )

    with pytest.raises(ValueError, match="defect decision.*evidence"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_cannot_hide_d3_evidence_as_l2(table_parser, monkeypatch):
    original = tabular_structure._new_projected_item

    def add_unemitted_proven_slot(**kwargs):
        item = original(**kwargs)
        item["proven_record_slots"].append(99)
        return item

    monkeypatch.setattr(tabular_structure, "_new_projected_item", add_unemitted_proven_slot)
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    output = audit["output_objects"][0]
    status, reason = tabular_structure.ENUMERATION_DECISIONS["R8"]
    output.update(
        {
            "matched_rule": "R8",
            "decision_chain_stop": "R8",
            "enumeration_status": status,
            "enumeration_reason": reason,
        }
    )

    with pytest.raises(ValueError, match="record count evidence.*D3"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)


def test_producer_audit_cannot_hide_d3_evidence_as_d1(table_parser, monkeypatch):
    original = tabular_structure._new_projected_item

    def add_unemitted_proven_slot(**kwargs):
        item = original(**kwargs)
        item["proven_record_slots"].append(99)
        return item

    monkeypatch.setattr(tabular_structure, "_new_projected_item", add_unemitted_proven_slot)
    _projection, audit = tabular_structure._build_tabular_structure_projection_with_audit(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    output = audit["output_objects"][0]
    status, reason = tabular_structure.ENUMERATION_DECISIONS["D1"]
    output.update(
        {
            "matched_rule": "D1",
            "decision_chain_stop": "D1",
            "enumeration_status": status,
            "enumeration_reason": reason,
        }
    )

    with pytest.raises(ValueError, match="record count evidence.*D3"):
        tabular_structure._validate_tabular_structure_producer_audit(audit)



def test_unproven_single_record_uses_the_default_l2_decision(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "State"])
    sheet.append(["A-1", "Open"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert {
        "enumeration_status": projection["tables"][0]["enumeration_status"],
        "enumeration_reason": projection["tables"][0]["enumeration_reason"],
        "matched_rule": projection["tables"][0]["matched_rule"],
    } == {
        "enumeration_status": "not_guaranteed_explained",
        "enumeration_reason": "record_axis_not_proven",
        "matched_rule": "R8",
    }


def _l2_reason_workbook_bytes(rule):
    workbook = Workbook()
    sheet = workbook.active
    if rule == "R1":
        sheet["A1"] = "Standalone note"
    elif rule == "R2":
        sheet.append(["Code", "Value"])
        sheet.append(["A-1", 1])
        sheet.append(["A-2", 2])
        sheet.append(["A-3", 3])
        sheet.row_dimensions[3].hidden = True
    elif rule == "R3":
        sheet.append([None, "Column A", "Column B"])
        sheet.append(["Row A", 1, 2])
        sheet.append(["Row B", 3, 4])
    elif rule == "R4":
        sheet.append(["Code", "Value"])
        sheet.append(["A-1", 1])
        sheet.append(["A-2", 2])
        sheet.append(["Aggregate", "=SUM(B2:B3)"])
    elif rule == "R5":
        sheet.append(["Code", "Value"])
        sheet.append(["A-1", 1])
        sheet.append(["A-2", 2])
        sheet.append(["Code", "Value"])
        sheet.append(["B-1", 3])
        sheet.append(["B-2", 4])
    elif rule == "R6":
        return _complete_table_with_partially_overlapping_unknown_bytes()
    elif rule == "R7":
        sheet.append(["Code", "Value"])
        fills = ("FFF2CC", "FFF2CC", "D9EAD3", "D9EAD3")
        for index, color in enumerate(fills, start=1):
            sheet.append([f"A-{index}", index])
            for cell in sheet[sheet.max_row]:
                cell.fill = PatternFill(fill_type="solid", fgColor=color)
    else:
        raise AssertionError(f"unhandled L2 rule: {rule}")
    return _save_workbook(workbook)


@pytest.mark.parametrize(
    ("rule", "reason"),
    [
        ("R1", "not_a_list"),
        ("R2", "total_unstable"),
        ("R3", "matrix_layout"),
        ("R4", "subtotal_rows_mixed"),
        ("R5", "multi_block_unseparated"),
        ("R6", "partial_overlap_continuation"),
        ("R7", "visual_only_boundary"),
    ],
)
def test_real_workbook_structures_reach_each_ordered_l2_reason(
    table_parser,
    rule,
    reason,
):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _l2_reason_workbook_bytes(rule),
        parser=table_parser,
    )

    assert projection["tables"]
    assert projection["rows"]
    assert all(table["source_total_count"] is None for table in projection["tables"])
    assert {table["matched_rule"] for table in projection["tables"]} == {rule}
    assert {table["enumeration_reason"] for table in projection["tables"]} == {reason}


def _collision_workbook_bytes(fixture_id):
    workbook = Workbook()
    sheet = workbook.active
    if fixture_id == "C12":
        sheet["A1"] = "=UNKNOWN_FUNCTION(1)"
        return _save_workbook(workbook)
    if fixture_id == "C16":
        sheet.cell(1, 1, "Earlier")
        sheet.cell(1, 2, "Detail")
        sheet.cell(4, 2, "Later")
        sheet.cell(4, 3, "Extra")
        return _save_workbook(workbook)
    if fixture_id == "C17":
        for column, color in enumerate(("FFF2CC", "FFF2CC", "D9EAD3", "D9EAD3"), start=1):
            sheet.cell(1, column, f"Text-{column}")
            sheet.cell(1, column).fill = PatternFill(fill_type="solid", fgColor=color)
        return _save_workbook(workbook)

    sheet.append(["Code", "State"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])

    if fixture_id in {"C23", "C34", "C37", "M02_R3_R4_R7"}:
        workbook = Workbook()
        sheet = workbook.active
        row_count = 5 if fixture_id in {"C37", "M02_R3_R4_R7"} else 4
        sheet.cell(1, 2, "Column A")
        sheet.cell(1, 3, "Column B")
        for row in range(2, row_count + 1):
            sheet.cell(row, 1, f"Row-{row}")
            sheet.cell(row, 2, row)
            sheet.cell(row, 3, row * 10)
        if fixture_id == "C23":
            sheet.row_dimensions[3].hidden = True
        if fixture_id in {"C34", "M02_R3_R4_R7"}:
            sheet.cell(row_count, 3, f"=SUM(C2:C{row_count - 1})")
        if fixture_id in {"C37", "M02_R3_R4_R7"}:
            for row, color in zip(range(2, row_count + 1), ("FFF2CC", "FFF2CC", "D9EAD3", "D9EAD3")):
                for column in range(1, 4):
                    sheet.cell(row, column).fill = PatternFill(fill_type="solid", fgColor=color)
        return _save_workbook(workbook)
    if fixture_id in {"C24", "C47", "M01_R2_R4_R7"}:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["Code", "Value"])
        for row in range(2, 6):
            sheet.cell(row, 1, f"A-{row}")
            sheet.cell(row, 2, row)
        sheet.cell(6, 1, "Aggregate")
        sheet.cell(6, 2, "=SUM(B2:B5)")
        if fixture_id in {"C24", "M01_R2_R4_R7"}:
            sheet.row_dimensions[3].hidden = True
        if fixture_id in {"C47", "M01_R2_R4_R7"}:
            for row, color in zip(range(2, 7), ("FFF2CC", "FFF2CC", "D9EAD3", "D9EAD3", "D9EAD3")):
                for column in (1, 2):
                    sheet.cell(row, column).fill = PatternFill(fill_type="solid", fgColor=color)
        return _save_workbook(workbook)
    if fixture_id in {"C25", "C45", "C57"}:
        workbook = Workbook()
        sheet = workbook.active
        rows = (
            (1, ("Code", "Value")),
            (2, ("A-1", 1)),
            (3, ("A-2", 2)),
            (4, ("Code", "Value")),
            (5, ("B-1", 3)),
            (6, ("B-2", 4)),
            (7, ("B-3", 5)),
        )
        for row, values in rows:
            sheet.cell(row, 1, values[0])
            sheet.cell(row, 2, values[1])
        if fixture_id == "C25":
            sheet.row_dimensions[3].hidden = True
        if fixture_id == "C45":
            sheet.cell(7, 2, "=SUM(B5:B6)")
        if fixture_id == "C57":
            for row, color in zip(range(2, 8), ("FFF2CC", "FFF2CC", "FFF2CC", "FFF2CC", "D9EAD3", "D9EAD3")):
                for column in (1, 2):
                    sheet.cell(row, column).fill = PatternFill(fill_type="solid", fgColor=color)
        return _save_workbook(workbook)
    if fixture_id in {"C26", "C27"}:
        if fixture_id == "C26":
            sheet.cell(6, 2, "Later")
            sheet.cell(6, 3, "=UNKNOWN_FUNCTION(1)")
        else:
            for row, color in zip(range(2, 6), ("FFF2CC", "FFF2CC", "D9EAD3", "D9EAD3")):
                if row > 3:
                    sheet.cell(row, 1, f"A-{row}")
                    sheet.cell(row, 2, "Open")
                for column in (1, 2):
                    sheet.cell(row, column).fill = PatternFill(fill_type="solid", fgColor=color)
            sheet.row_dimensions[3].hidden = True
        return _save_workbook(workbook)
    if fixture_id == "C35":
        workbook = Workbook()
        sheet = workbook.active
        sheet.cell(1, 2, "Column A")
        sheet.cell(1, 3, "Column B")
        for row in range(2, 4):
            sheet.cell(row, 1, f"Row-{row}")
            sheet.cell(row, 2, row)
            sheet.cell(row, 3, row * 10)
        sheet.cell(4, 2, "Column A")
        sheet.cell(4, 3, "Column B")
        for row in range(5, 7):
            sheet.cell(row, 1, f"Row-{row}")
            sheet.cell(row, 2, row)
            sheet.cell(row, 3, row * 10)
        return _save_workbook(workbook)
    if fixture_id == "C36":
        sheet.cell(6, 2, None)
        sheet.cell(6, 3, "Column A")
        sheet.cell(6, 4, "Column B")
        sheet.cell(7, 2, "Row A")
        sheet.cell(7, 3, 1)
        sheet.cell(7, 4, 2)
        sheet.cell(8, 2, "Row B")
        sheet.cell(8, 3, 3)
        sheet.cell(8, 4, 4)
    elif fixture_id == "C46":
        sheet.cell(6, 2, "B-1")
        sheet.cell(6, 3, 1)
        sheet.cell(7, 2, "B-2")
        sheet.cell(7, 3, 2)
        sheet.cell(8, 2, "Aggregate")
        sheet.cell(8, 3, "=SUM(C6:C7)")
    elif fixture_id == "C56":
        for row, values in (
            (6, ("Code", "Value")),
            (7, ("B-1", 1)),
            (8, ("B-2", 2)),
            (9, ("Code", "Value")),
            (10, ("C-1", 3)),
            (11, ("C-2", 4)),
        ):
            sheet.cell(row, 2, values[0])
            sheet.cell(row, 3, values[1])
    elif fixture_id == "C67":
        fills = ("FFF2CC", "FFF2CC", "D9EAD3", "D9EAD3")
        sheet.cell(6, 2, "Code")
        sheet.cell(6, 3, "Value")
        for offset, color in enumerate(fills, start=7):
            sheet.cell(offset, 2, f"B-{offset}")
            sheet.cell(offset, 3, offset)
            for column in (2, 3):
                sheet.cell(offset, column).fill = PatternFill(fill_type="solid", fgColor=color)
    else:
        raise AssertionError(f"unhandled collision fixture: {fixture_id}")
    return _save_workbook(workbook)


@pytest.mark.parametrize(
    ("fixture_id", "expected_rules", "expected_stop"),
    [
        ("C12", {"R1", "R2"}, "R1"),
        ("C16", {"R1", "R6"}, "R1"),
        ("C17", {"R1", "R7"}, "R1"),
        ("C23", {"R2", "R3"}, "R2"),
        ("C24", {"R2", "R4"}, "R2"),
        ("C25", {"R2", "R5"}, "R2"),
        ("C26", {"R2", "R6"}, "R2"),
        ("C27", {"R2", "R7"}, "R2"),
        ("C34", {"R3", "R4"}, "R3"),
        ("C35", {"R3", "R5"}, "R3"),
        ("C36", {"R3", "R6"}, "R3"),
        ("C37", {"R3", "R7"}, "R3"),
        ("C45", {"R4", "R5"}, "R4"),
        ("C46", {"R4", "R6"}, "R4"),
        ("C47", {"R4", "R7"}, "R4"),
        ("C56", {"R5", "R6"}, "R5"),
        ("C57", {"R5", "R7"}, "R5"),
        ("C67", {"R6", "R7"}, "R6"),
        ("M01_R2_R4_R7", {"R2", "R4", "R7"}, "R2"),
        ("M02_R3_R4_R7", {"R3", "R4", "R7"}, "R3"),
    ],
)
def test_real_workbook_collision_vectors_use_production_predicates(
    table_parser,
    monkeypatch,
    fixture_id,
    expected_rules,
    expected_stop,
):
    captured = []
    ordered = tabular_structure._ordered_enumeration_rule

    def capture(predicates, l1_rule):
        captured.append(dict(predicates))
        return ordered(predicates, l1_rule)

    monkeypatch.setattr(tabular_structure, "_ordered_enumeration_rule", capture)
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _collision_workbook_bytes(fixture_id),
        parser=table_parser,
    )

    expected_vector = {rule: rule in expected_rules for rule in tabular_structure.NEGATIVE_ENUMERATION_RULES}
    assert expected_vector in captured
    assert expected_stop in {table["matched_rule"] for table in projection["tables"]}


def test_visual_only_boundary_uses_style_clusters_without_formula_or_merge(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    for row, color in ((2, "FFF2CC"), (3, "FFF2CC"), (4, "D9EAD3"), (5, "D9EAD3")):
        sheet.cell(row, 1, f"A-{row}")
        sheet.cell(row, 2, row)
        for column in (1, 2):
            sheet.cell(row, column).fill = PatternFill(fill_type="solid", fgColor=color)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["matched_rule"] == "R7"


def test_dense_matrix_with_occupied_corner_stops_at_matrix_rule(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Dimension", "Column A", "Column B"])
    sheet.append(["Row A", 1, 2])
    sheet.append(["Row B", 3, 4])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "R3"


def test_dense_three_column_text_list_is_not_a_matrix(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status", "Owner"])
    sheet.append(["A-1", "Open", "Team-1"])
    sheet.append(["A-2", "Closed", "Team-2"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] == 2
    assert projection["tables"][0]["matched_rule"] == "L1-07"


def test_dense_text_list_without_digit_markers_is_not_a_matrix(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Name", "State", "Owner"])
    sheet.append(["Alpha", "Open", "East"])
    sheet.append(["Beta", "Closed", "West"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "R8"


def test_dense_numeric_record_list_is_not_a_matrix(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Count", "Owner"])
    sheet.append(["A-1", 1, "East"])
    sheet.append(["A-2", 2, "West"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] == 2
    assert projection["tables"][0]["matched_rule"] == "L1-07"


def test_mixed_value_matrix_cannot_claim_a_complete_record_axis(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Dimension", "Count", "Status"])
    sheet.append(["North", 1, "High"])
    sheet.append(["South", 2, "Low"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "R3"


def test_sparse_matrix_cannot_claim_a_complete_record_axis(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Dimension", "First", "Second"])
    sheet.append(["North", 1, None])
    sheet.append(["South", None, 2])
    sheet.append(["West", 3, 4])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "R3"


def test_matrix_decision_considers_the_complete_candidate_region(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Name", "Count", "Score"])
    sheet.append(["Alice", 1, 10])
    sheet.append(["Bob", 2, 20])
    sheet.append(["Carol", 3, "Pending"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "R8"


def test_single_style_cluster_does_not_trigger_visual_only_boundary(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    for row in range(2, 6):
        sheet.cell(row, 1, f"A-{row}")
        sheet.cell(row, 2, row)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["matched_rule"] == "L1-07"


def test_multicell_non_list_without_isomorphic_slots_reaches_r1(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet["A1"] = "Prepared by"
    sheet["B1"] = "Reviewed by"

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["matched_rule"] == "R1"


def test_hidden_header_does_not_make_a_stable_record_total_unstable(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    sheet.append(["A-1", 1])
    sheet.append(["A-2", 2])
    sheet.row_dimensions[1].hidden = True

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] == 2
    assert projection["tables"][0]["matched_rule"] == "L1-07"


def test_uncached_nonaggregate_formula_reaches_total_unstable(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    sheet.append(["A-1", 1])
    sheet.append(["A-2", "=B2"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "R2"


def test_table_ref_rejects_membership_identity_tampering(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )
    original = projection["tables"][0]["table_ref"]
    prefix, version, membership, identity = original.split("_")
    changed_membership = "0" * 64 if membership != "0" * 64 else "1" * 64
    tampered = f"{prefix}_{version}_{changed_membership}_{identity}"
    projection["tables"][0]["table_ref"] = tampered
    for row in projection["rows"]:
        row["table_ref_kwd"] = tampered
        row["row_ref_kwd"] = f"{tampered}:{row['row_ordinal_int']}"
        row["id"] = "tsr_v1_" + tabular_structure._versioned_digest(
            "tabular-row-record/v1",
            projection["producer_generation_ref"],
            row["row_ref_kwd"],
        )

    with pytest.raises(ValueError, match="table reference"):
        validate_tabular_structure_projection(projection)


def _workbook_bytes(
    *,
    headers=("Code", "Description", "Force"),
    include_unknown=False,
    include_merged_body=False,
    include_note=True,
    include_context=False,
    rows=3,
):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Inspection"
    if include_context:
        sheet.merge_cells("A1:C1")
        sheet["A1"] = "\u9879\u76ee \u00b5\u03a9\u2103\u0085\u202e"
        sheet["A3"] = "Program"
        sheet["B3"] = "Platform-X"
        sheet["A4"], sheet["B4"], sheet["C4"] = headers
    else:
        sheet.append(list(headers))

    for index in range(1, rows + 1):
        sheet.append(["R-DUP" if index <= 2 else f"R-{index:04d}", "Repeated" if index <= 2 else "Item", 7.5])

        if include_merged_body and index == 1:
            body_row = sheet.max_row + 1
            sheet.merge_cells(start_row=body_row, start_column=1, end_row=body_row, end_column=3)
            sheet.cell(body_row, 1, "R-MERGED")

    if include_unknown:
        sheet.append(["Sparse body row", None, None])

    if include_note:
        note_row = sheet.max_row + 1
        sheet.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=3)
        sheet.cell(note_row, 1, "For reference only")

    second = workbook.create_sheet("Secondary")
    second.append(["Part", "Status"])
    second.append(["P-1", "Approved"])

    output = BytesIO()
    workbook.save(output)
    return output.getvalue()
def _generation_ref():
    return str(uuid.uuid4())


def _save_workbook(workbook):
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def _vertical_multi_region_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    sheet.append([None, None])
    sheet.append([None, None])
    sheet.append(["Code", "Status"])
    sheet.append(["B-1", "Open"])
    sheet.append(["B-2", "Closed"])
    return _save_workbook(workbook)


def _horizontal_multi_region_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Left code", "Left status", None, None, "Right code", "Right status"])
    sheet.append(["L-1", "Open", None, None, "R-1", "Closed"])
    sheet.append(["L-2", "Closed", None, None, "R-2", "Open"])
    return _save_workbook(workbook)


def _horizontal_varying_merge_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous horizontal merges"
    sheet.append(["Field A", "Field B", "Field C", "Field D"])
    sheet.merge_cells("A2:B2")
    sheet["A2"] = "A-1"
    sheet["C2"] = "C-1"
    sheet["D2"] = "D-1"
    sheet.merge_cells("B3:C3")
    sheet["A3"] = "A-2"
    sheet["B3"] = "B-2"
    sheet["D3"] = "D-2"
    sheet.merge_cells("C4:D4")
    sheet["A4"] = "A-3"
    sheet["B4"] = "B-3"
    sheet["C4"] = "C-3"
    return _save_workbook(workbook)


def _vertical_varying_merge_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous vertical merges"
    sheet.append(["Field A", "Field B", "Field C", "Field D"])
    sheet.merge_cells("A2:A3")
    sheet["A2"] = "A-group-1"
    sheet["B2"] = "B-1"
    sheet["C2"] = "C-1"
    sheet["D2"] = "D-1"
    sheet["B3"] = "B-2"
    sheet["C3"] = "C-2"
    sheet["D3"] = "D-2"
    sheet.merge_cells("B4:B5")
    sheet["A4"] = "A-3"
    sheet["B4"] = "B-group-2"
    sheet["C4"] = "C-3"
    sheet["D4"] = "D-3"
    sheet["A5"] = "A-4"
    sheet["C5"] = "C-4"
    sheet["D5"] = "D-4"
    return _save_workbook(workbook)


def _mixed_row_merge_supplier_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous mixed merges"
    sheet.append(["Group", "Material", "Supplier", "Evidence", "Note"])

    # Each record remains physically present, but each row has a different
    # combination of vertical and horizontal source merges.
    sheet.merge_cells("A2:A3")
    sheet["A2"] = "G-1"
    sheet.merge_cells("B2:C2")
    sheet["B2"] = "M-1"
    sheet["D2"] = "S-1"
    sheet["E2"] = "Verified"

    sheet.merge_cells("B3:D3")
    sheet["B3"] = "M-2 / S-2"
    sheet["E3"] = "Pending"

    sheet.merge_cells("A4:A5")
    sheet["A4"] = "G-2"
    sheet.merge_cells("C4:E4")
    sheet["C4"] = "S-3"
    sheet["B4"] = "M-3"

    sheet.merge_cells("B5:C5")
    sheet["B5"] = "M-4"
    sheet["D5"] = "S-4"
    sheet["E5"] = "Approved"

    # These columns are a disjoint link sidecar, not table columns.
    sheet["H1"] = "Sidecar label"
    sheet["I1"] = "https://example.invalid/material"
    sheet["I1"].hyperlink = "https://example.invalid/material"
    return _save_workbook(workbook)


def _vertical_complete_with_headerless_continuation_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    sheet.append([None, None])
    sheet.append([None, None])
    sheet.append(["B-1", "Open"])
    return _save_workbook(workbook)


def _vertical_complete_with_subset_headerless_continuation_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Code", "Status", "Owner", "Revision", "Date"])
    sheet.append(["A-1", "Open", "Team-1", "R1", "2026-01-01"])
    sheet.append(["A-2", "Closed", "Team-2", "R2", "2026-01-02"])
    sheet.append([None, None, None, None, None])
    sheet.append([None, None, None, None, None])
    sheet.append(["B-1", "Open", "Team-3", "R3"])
    return _save_workbook(workbook)


def _vertical_complete_with_superset_headerless_continuation_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Code", "Status", "Owner", "Revision"])
    sheet.append(["A-1", "Open", "Team-1", "R1"])
    sheet.append(["A-2", "Closed", "Team-2", "R2"])
    sheet.append([None, None, None, None, None])
    sheet.append([None, None, None, None, None])
    sheet.append(["B-1", "Open", "Team-3", "R3", "2026-01-03"])
    return _save_workbook(workbook)


def _complete_table_with_partially_overlapping_unknown_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    sheet.append([None, None, None])
    sheet.append([None, None, None])
    sheet.append([None, "Unresolved", "Extra"])
    return _save_workbook(workbook)


def _horizontal_complete_with_headerless_sibling_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Left code", "Left status", None, None, "R-1", "Closed"])
    sheet.append(["L-1", "Open", None, None, "R-2", "Open"])
    sheet.append(["L-2", "Closed", None, None, "R-3", "Closed"])
    return _save_workbook(workbook)


def _complete_table_with_independent_repeated_header_table_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Left code", "Left status", None, None, "Right code", "Right status"])
    sheet.append(["L-1", "Open", None, None, "R-1", "Open"])
    sheet.append(["L-2", "Closed", None, None, "Right code", "Right status"])
    sheet.append([None, None, None, None, "R-2", "Closed"])
    return _save_workbook(workbook)


def _complete_table_with_isolated_annotation_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    sheet.cell(row=10, column=10, value="Annotation")
    return _save_workbook(workbook)


def _complete_table_with_axis_aligned_unknown_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    sheet.cell(row=10, column=1, value="Unresolved")
    return _save_workbook(workbook)


def _complete_table_with_g_sensitive_sibling_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Code", "Status", None, None, "R1 code", "R1 status", None, "R2 code", "R2 status"])
    sheet.append(["A-1", "Open", None, None, "B-1", "Open", None, "C-1", "Closed"])
    sheet.append(["A-2", "Closed", None, None, "B-2", "Closed", None, "C-2", "Open"])
    return _save_workbook(workbook)


def _single_column_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Code"])
    sheet.append(["S-1"])
    sheet.append(["S-2"])
    return _save_workbook(workbook)


def _merged_header_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.merge_cells("A1:B1")
    sheet["A1"] = "Items"
    sheet["A2"] = "Code"
    sheet["B2"] = "Status"
    sheet.append(["M-1", "Open"])
    sheet.append(["M-2", "Closed"])
    return _save_workbook(workbook)


def _multilevel_sparse_table_with_context_child_bytes(*, context_row_count=1):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.merge_cells("A2:H3")
    sheet["A2"] = "Anonymous catalogue"
    sheet.merge_cells("A4:B4")
    sheet["A4"] = "Identity"
    sheet.merge_cells("C4:D4")
    sheet["C4"] = "Classification"
    sheet.merge_cells("E4:G4")
    sheet["E4"] = "Attributes"
    for row in range(4, 4 + context_row_count):
        sheet.cell(row=row, column=10, value=f"Anonymous context {row}")
    sheet.merge_cells("A5:B5")
    sheet["A5"] = "Reference"
    sheet.merge_cells("C5:D5")
    sheet["C5"] = "Grouping"
    sheet.merge_cells("E5:G5")
    sheet["E5"] = "Details"
    sheet.merge_cells("A6:H6")
    sheet["A6"] = "Record fields"
    sheet.merge_cells("A7:A8")
    sheet["A7"] = "Number"
    sheet.merge_cells("B7:B8")
    sheet["B7"] = "Code"
    sheet.merge_cells("C7:D7")
    sheet["C7"] = "Group"
    sheet["C8"] = "Type"
    sheet["D8"] = "State"
    sheet.merge_cells("E7:E8")
    sheet["E7"] = "Owner"
    sheet.merge_cells("F7:G7")
    sheet["F7"] = "Limits"
    sheet["F8"] = "Lower"
    sheet["G8"] = "Upper"
    sheet.merge_cells("H7:H8")
    sheet["H7"] = "Optional"
    for index, row in enumerate(range(9, 19), start=1):
        values = [
            index,
            f"R-{index:02d}",
            "A",
            "Open",
            "Team",
            "0",
            "1",
            "present" if index == 8 else None,
        ]
        for column, value in enumerate(values, start=1):
            sheet.cell(row=row, column=column, value=value)
    return _save_workbook(workbook)


def _sparse_region_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Code", None, "Status"])
    sheet.append(["P-1", None, "Open"])
    sheet.append(["P-2", None, "Closed"])
    sheet.append([None, None, None])
    sheet.append(["P-3", None, "Open"])
    sheet.append(["P-4", None, "Closed"])
    return _save_workbook(workbook)


def _g_sensitive_horizontal_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Left code", "Left status", None, "Right code", "Right status"])
    sheet.append(["L-1", "Open", None, "R-1", "Closed"])
    sheet.append(["L-2", "Closed", None, "R-2", "Open"])
    return _save_workbook(workbook)


def _three_level_header_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.merge_cells("A1:B1")
    sheet["A1"] = "Items"
    sheet.merge_cells("A2:B2")
    sheet["A2"] = "Identity"
    sheet["A3"] = "Code"
    sheet["B3"] = "State"
    sheet.append(["T-1", "Open"])
    sheet.append(["T-2", "Closed"])
    return _save_workbook(workbook)


def _four_level_header_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    for row_ordinal, label in enumerate(("Items", "Identity", "Detail"), start=1):
        sheet.merge_cells(
            start_row=row_ordinal,
            start_column=1,
            end_row=row_ordinal,
            end_column=2,
        )
        sheet.cell(row_ordinal, 1, label)
    sheet["A4"] = "Code"
    sheet["B4"] = "State"
    sheet.append(["Q-1", "Open"])
    sheet.append(["Q-2", "Closed"])
    return _save_workbook(workbook)


def _formula_only_workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet["A1"] = '=CONCAT("F-", 1)'
    sheet["A2"] = '=CONCAT("F-", 2)'
    return _save_workbook(workbook)


@pytest.fixture
def table_parser(monkeypatch):
    return _load_table_module(monkeypatch).Excel()


def test_same_bytes_keep_business_identity_but_get_a_new_generation(table_parser):
    source = _workbook_bytes()
    first = build_tabular_structure_projection(
        "anonymous.xlsx",
        source,
        parser=table_parser,
    )
    second = build_tabular_structure_projection(
        "anonymous.xlsx",
        source,
        parser=table_parser,
    )

    assert first["producer_generation_ref"] != second["producer_generation_ref"]
    assert [row["table_ref_kwd"] for row in first["rows"]] == [row["table_ref_kwd"] for row in second["rows"]]
    assert [row["row_ref_kwd"] for row in first["rows"]] == [row["row_ref_kwd"] for row in second["rows"]]
    assert [row["row_role_kwd"] for row in first["rows"]] == [row["row_role_kwd"] for row in second["rows"]]
    assert [row["data_row_index_int"] for row in first["rows"]] == [row["data_row_index_int"] for row in second["rows"]]
    assert [row["source_total_count_int"] for row in first["rows"]] == [row["source_total_count_int"] for row in second["rows"]]
    assert {row["producer_generation_ref_kwd"] for row in first["rows"]} == {first["producer_generation_ref"]}
    assert {row["id"] for row in first["rows"]}.isdisjoint({row["id"] for row in second["rows"]})


def test_vertical_tables_on_one_sheet_get_separate_stable_projections(table_parser):
    source = _vertical_multi_region_workbook_bytes()

    projections = [
        build_tabular_structure_projection(
            "anonymous.xlsx",
            source,
            producer_generation_ref=_generation_ref(),
            parser=table_parser,
        )
        for _ in range(3)
    ]

    expected_identity = [
        (table["table_ordinal"], table["table_ref"])
        for table in projections[0]["tables"]
    ]
    assert [table["table_ordinal"] for table in projections[0]["tables"]] == [1, 2]
    assert [table["source_total_count"] for table in projections[0]["tables"]] == [2, 2]
    assert all(
        [(table["table_ordinal"], table["table_ref"]) for table in projection["tables"]]
        == expected_identity
        for projection in projections[1:]
    )


def test_horizontal_tables_use_exact_columns_without_duplicate_ingestion(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _horizontal_multi_region_workbook_bytes(),
        parser=table_parser,
    )

    assert [table["table_ordinal"] for table in projection["tables"]] == [1, 2]
    rows_by_table = {
        table["table_ref"]: [
            json.loads(row["ordered_fields_list"])
            for row in projection["rows"]
            if row["table_ref_kwd"] == table["table_ref"] and row["row_role_kwd"] == "data"
        ]
        for table in projection["tables"]
    }
    assert rows_by_table[projection["tables"][0]["table_ref"]] == [
        [
            {
                "column_id": "col_v1:1:1",
                "column_ordinal": 1,
                "header_path": ["Left code"],
                "name": "Left code",
                "value": "L-1",
            },
            {
                "column_id": "col_v1:1:2",
                "column_ordinal": 2,
                "header_path": ["Left status"],
                "name": "Left status",
                "value": "Open",
            },
        ],
        [
            {
                "column_id": "col_v1:1:1",
                "column_ordinal": 1,
                "header_path": ["Left code"],
                "name": "Left code",
                "value": "L-2",
            },
            {
                "column_id": "col_v1:1:2",
                "column_ordinal": 2,
                "header_path": ["Left status"],
                "name": "Left status",
                "value": "Closed",
            },
        ],
    ]
    assert rows_by_table[projection["tables"][1]["table_ref"]] == [
        [
            {
                "column_id": "col_v1:1:5",
                "column_ordinal": 1,
                "header_path": ["Right code"],
                "name": "Right code",
                "value": "R-1",
            },
            {
                "column_id": "col_v1:1:6",
                "column_ordinal": 2,
                "header_path": ["Right status"],
                "name": "Right status",
                "value": "Closed",
            },
        ],
        [
            {
                "column_id": "col_v1:1:5",
                "column_ordinal": 1,
                "header_path": ["Right code"],
                "name": "Right code",
                "value": "R-2",
            },
            {
                "column_id": "col_v1:1:6",
                "column_ordinal": 2,
                "header_path": ["Right status"],
                "name": "Right status",
                "value": "Open",
            },
        ],
    ]


def test_vertical_headerless_continuation_downgrades_complete_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _vertical_complete_with_headerless_continuation_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] == 3
    assert projection["tables"][0]["matched_rule"] == "L1-02"


def test_isolated_single_row_annotation_is_not_merged_as_a_continuation(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "State"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    sheet.append([None, None])
    sheet.append([None, None])
    sheet.append(["Prepared by", "Alice"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert projection["tables"][0]["source_total_count"] == 2
    assert projection["tables"][0]["matched_rule"] == "L1-07"
    assert projection["tables"][1]["source_total_count"] is None
    assert projection["tables"][1]["matched_rule"] == "R8"


def test_same_shape_single_row_annotation_is_not_merged_as_a_continuation(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Record Name", "Status", "Owner"])
    sheet.append(["North Unit", "Open", "Team-1"])
    sheet.append(["South Site", "Closed", "Team-2"])
    sheet.append([None, None, None])
    sheet.append([None, None, None])
    sheet.append(["Prepared by", "Alice", "Team-3"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert projection["tables"][0]["source_total_count"] == 2
    assert projection["tables"][0]["matched_rule"] == "L1-07"
    assert projection["tables"][1]["source_total_count"] is None
    assert projection["tables"][1]["matched_rule"] == "R8"


def test_continuation_union_preserves_custom_context_limits(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _vertical_complete_with_headerless_continuation_bytes(),
        table_context_entry_limit=1,
        table_context_value_bytes=5,
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    context = json.loads(projection["rows"][0]["table_context_list"])
    assert len(context) == 1
    assert len(context[0]["value"].encode("utf-8")) <= 5


def test_continuation_with_unknown_row_cannot_drop_it_and_claim_complete(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    sheet.append([None, None])
    sheet.append([None, None])
    sheet.append(["Code", "Status"])
    sheet.append(["B-1", "Open"])
    sheet.append(["B-2", "Closed"])
    sheet.append(["Code", "Status"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert all(table["source_total_count"] is None for table in projection["tables"])
    assert any(
        row["row_ordinal_int"] == 9 and row["row_role_kwd"] == "unknown"
        for row in projection["rows"]
    )


def test_multirow_named_continuation_without_a_proven_axis_cannot_claim_complete(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    sheet.append(["A-1", 1])
    sheet.append(["A-2", 2])
    sheet.append([None, None])
    sheet.append([None, None])
    sheet.append(["Code", "Value"])
    sheet.append(["B-1", 3])
    sheet.append(["Summary", "Current"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert all(table["source_total_count"] is None for table in projection["tables"])
    assert all(table["enumeration_status"] == "not_guaranteed_explained" for table in projection["tables"])


def test_three_equal_continuation_segments_form_one_complete_record_sequence(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    sheet.append([None, None])
    sheet.append([None, None])
    sheet.append(["B-1", "Open"])
    sheet.append([None, None])
    sheet.append([None, None])
    sheet.append(["C-1", "Closed"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] == 4
    assert projection["tables"][0]["matched_rule"] == "L1-02"


def test_multirow_named_superset_continuation_forms_one_complete_record_sequence(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    sheet.append([None, None, None])
    sheet.append([None, None, None])
    sheet.append(["Code", "Status", "Owner"])
    sheet.append(["B-1", "Open", "Team-1"])
    sheet.append(["B-2", "Closed", "Team-2"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] == 4
    assert projection["tables"][0]["matched_rule"] == "L1-02"


def test_named_continuation_left_expansion_rekeys_every_row_to_the_union_manifest(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.cell(row=1, column=2, value="Code")
    sheet.cell(row=1, column=3, value="Status")
    sheet.cell(row=2, column=2, value="A-1")
    sheet.cell(row=2, column=3, value="Open")
    sheet.cell(row=3, column=2, value="A-2")
    sheet.cell(row=3, column=3, value="Closed")
    sheet.cell(row=6, column=1, value="Owner")
    sheet.cell(row=6, column=2, value="Code")
    sheet.cell(row=6, column=3, value="Status")
    sheet.cell(row=7, column=1, value="Team-1")
    sheet.cell(row=7, column=2, value="B-1")
    sheet.cell(row=7, column=3, value="Open")
    sheet.cell(row=8, column=1, value="Team-2")
    sheet.cell(row=8, column=2, value="B-2")
    sheet.cell(row=8, column=3, value="Closed")

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    table = projection["tables"][0]
    assert table["source_total_count"] == 4
    assert table["matched_rule"] == "L1-02"
    assert table["ordered_columns"] == [
        {
            "column_id": "col_v1:1:1",
            "column_ordinal": 1,
            "header_path": ["Owner"],
            "name": "Owner",
        },
        {
            "column_id": "col_v1:1:2",
            "column_ordinal": 2,
            "header_path": ["Code"],
            "name": "Code",
        },
        {
            "column_id": "col_v1:1:3",
            "column_ordinal": 3,
            "header_path": ["Status"],
            "name": "Status",
        },
    ]
    fields_by_row = {
        row["row_ordinal_int"]: json.loads(row["ordered_fields_list"])
        for row in projection["rows"]
    }
    assert [field["column_ordinal"] for field in fields_by_row[2]] == [2, 3]
    assert [field["column_ordinal"] for field in fields_by_row[3]] == [2, 3]
    assert [field["column_ordinal"] for field in fields_by_row[7]] == [1, 2, 3]
    assert [field["column_ordinal"] for field in fields_by_row[8]] == [1, 2, 3]
    validate_tabular_structure_projection(projection)


def test_supported_complete_table_requires_nonempty_ordered_columns(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )
    assert projection["tables"][0]["enumeration_status"] == "supported_complete"
    projection["tables"][0]["ordered_columns"] = []

    with pytest.raises(ValueError, match="supported complete table requires ordered columns"):
        validate_tabular_structure_projection(projection)


def test_hidden_headerless_continuation_row_cannot_claim_complete(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    sheet.append([None, None])
    sheet.append([None, None])
    sheet.append(["B-1", "Open"])
    sheet.row_dimensions[6].hidden = True

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "R2"


def test_subset_headerless_continuation_downgrades_complete_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _vertical_complete_with_subset_headerless_continuation_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] == 3
    assert projection["tables"][0]["matched_rule"] == "L1-02"


def test_superset_headerless_continuation_downgrades_complete_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _vertical_complete_with_superset_headerless_continuation_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "D1"
    assert all("Column_" not in row["ordered_fields_list"] for row in projection["rows"])


def test_unnamed_superset_still_runs_membership_closure(table_parser, monkeypatch):
    original = tabular_structure._merge_continuation_pair

    def duplicate_merged_member(**kwargs):
        merged = original(**kwargs)
        if merged is not None:
            merged["emitted_member_events"].append(merged["emitted_member_events"][0])
        return merged

    monkeypatch.setattr(tabular_structure, "_merge_continuation_pair", duplicate_merged_member)
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _vertical_complete_with_superset_headerless_continuation_bytes(),
        parser=table_parser,
    )

    assert projection["tables"][0]["matched_rule"] == "D2"
    assert projection["tables"][0]["enumeration_reason"] == "membership_not_closed"


def test_partially_overlapping_unknown_downgrades_complete_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _complete_table_with_partially_overlapping_unknown_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert all(table["source_total_count"] is None for table in projection["tables"])
    assert {table["matched_rule"] for table in projection["tables"]} == {"R6"}


def test_partial_overlap_does_not_downgrade_a_disjoint_table_on_the_same_sheet(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status", None, None, None, "Other", "Value"])
    sheet.append(["A-1", "Open", None, None, None, "X-1", 1])
    sheet.append(["A-2", "Closed", None, None, None, "X-2", 2])
    sheet.cell(row=6, column=2, value="Later")
    sheet.cell(row=6, column=3, value="Extra")

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 3
    assert [table["source_total_count"] for table in projection["tables"]] == [None, 2, None]
    assert [table["matched_rule"] for table in projection["tables"]] == ["R6", "L1-03", "R6"]


def test_partial_overlap_before_a_complete_table_is_not_a_continuation(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.cell(row=1, column=2, value="Earlier")
    sheet.cell(row=1, column=3, value="Detail")
    sheet.append([None, None, None])
    sheet.append([None, None, None])
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert [table["source_total_count"] for table in projection["tables"]] == [None, 2]
    assert [table["matched_rule"] for table in projection["tables"]] == ["R8", "L1-07"]


def test_horizontal_headerless_record_axis_does_not_downgrade_disjoint_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _horizontal_complete_with_headerless_sibling_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert [table["source_total_count"] for table in projection["tables"]] == [2, None]
    assert [table["matched_rule"] for table in projection["tables"]] == ["L1-07", "R8"]


def test_projected_table_with_unknown_row_does_not_downgrade_complete_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _complete_table_with_independent_repeated_header_table_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert [table["source_total_count"] for table in projection["tables"]] == [2, None]


def test_axis_aligned_single_unknown_downgrades_complete_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _complete_table_with_axis_aligned_unknown_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert all(table["source_total_count"] is None for table in projection["tables"])


def test_isolated_annotation_does_not_downgrade_complete_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _complete_table_with_isolated_annotation_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert projection["tables"][0]["source_total_count"] == 2
    assert projection["tables"][1]["source_total_count"] is None


def test_g_sensitive_unknown_does_not_downgrade_disjoint_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _complete_table_with_g_sensitive_sibling_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert [table["source_total_count"] for table in projection["tables"]] == [2, None]
    assert [table["matched_rule"] for table in projection["tables"]] == ["L1-07", "R8"]
    assert all(
        row["row_role_kwd"] == "unknown"
        for row in projection["rows"]
        if row["table_ref_kwd"] == projection["tables"][1]["table_ref"]
    )


def test_g_sensitive_horizontal_boundary_cannot_produce_complete(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _g_sensitive_horizontal_workbook_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] is None
    assert all(row["row_role_kwd"] == "unknown" for row in projection["rows"])


def test_single_column_record_axis_gets_a_complete_projection(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _single_column_workbook_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] == 2
    assert [
        json.loads(row["ordered_fields_list"])
        for row in projection["rows"]
        if row["row_role_kwd"] == "data"
    ] == [
        [
            {
                "column_id": "col_v1:1:1",
                "column_ordinal": 1,
                "header_path": ["Code"],
                "name": "Code",
                "value": "S-1",
            }
        ],
        [
            {
                "column_id": "col_v1:1:1",
                "column_ordinal": 1,
                "header_path": ["Code"],
                "name": "Code",
                "value": "S-2",
            }
        ],
    ]
    expected_membership = tabular_structure._region_membership_sha256(
        1,
        {(1, 1), (2, 1), (3, 1)},
    )
    assert projection["tables"][0]["table_ref"].startswith(
        f"tbl_v2_{expected_membership}_"
    )


def test_single_column_free_text_block_cannot_prove_a_complete_record_axis(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Instruction"])
    sheet.append(["Review the source before use"])
    sheet.append(["Confirm the revision before release"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None


def test_single_record_slot_cannot_prove_a_complete_record_axis(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status"])
    sheet.append(["S-1", "Open"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["data_row_count"] == 1
    assert projection["tables"][0]["source_total_count"] is None

    projection["tables"][0]["source_total_count"] = 1
    projection["rows"][0]["source_total_count_int"] = 1
    with pytest.raises(ValueError, match="source total"):
        validate_tabular_structure_projection(projection)


def test_headerless_record_slots_preserve_every_row_and_cannot_claim_complete(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    sheet.append(["A-3", "Open"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert [row["row_ordinal_int"] for row in projection["rows"]] == [1, 2, 3]
    assert all(row["row_role_kwd"] == "unknown" for row in projection["rows"])


def test_unrelated_uncached_formula_does_not_downgrade_complete_sibling(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status"])
    sheet.append(["S-1", "Open"])
    sheet.append(["S-2", "Closed"])
    sheet.cell(row=20_001, column=20, value="=SUM(1, 1)")

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert [table["source_total_count"] for table in projection["tables"]] == [2, None]
    assert [table["matched_rule"] for table in projection["tables"]] == ["L1-07", "R1"]


def test_multilevel_merged_header_stays_with_its_record_axis(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _merged_header_workbook_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] == 2
    assert [row["row_ordinal_int"] for row in projection["rows"]] == [3, 4]


def test_multilevel_sparse_table_ignores_context_only_g1_disagreement(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _multilevel_sparse_table_with_context_child_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    table = projection["tables"][0]
    assert table["matched_rule"] == "L1-04"
    assert table["source_total_count"] == 10
    assert [row["row_ordinal_int"] for row in projection["rows"]] == list(
        range(9, 19)
    )
    assert all(row["row_role_kwd"] == "data" for row in projection["rows"])
    assert [
        column["column_id"] for column in table["ordered_columns"]
    ] == [f"col_v1:1:{ordinal}" for ordinal in range(1, 9)]


def test_repeated_axis_g1_child_before_records_is_not_swallowed_as_context(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _multilevel_sparse_table_with_context_child_bytes(context_row_count=2),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] is None
    assert projection["tables"][0]["matched_rule"] == "R8"


def test_g_sensitive_sparse_table_remains_one_unknown_projection(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _sparse_region_workbook_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] is None
    assert {row["row_ordinal_int"] for row in projection["rows"]} >= {1, 2, 3, 5, 6}
    assert all(row["row_role_kwd"] == "unknown" for row in projection["rows"])


def test_two_non_isomorphic_rows_cannot_prove_a_record_axis(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status", "Owner"])
    sheet.append(["A-1", "Open", None])
    sheet.append(["A-2", None, "North"])

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None


def test_three_level_merged_header_does_not_become_a_data_row(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _three_level_header_workbook_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] == 2
    assert [row["row_ordinal_int"] for row in projection["rows"]] == [4, 5]


def test_four_level_merged_header_has_no_fixed_depth_limit(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _four_level_header_workbook_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] == 2
    assert [row["row_ordinal_int"] for row in projection["rows"]] == [5, 6]


def test_formula_only_region_is_preserved_as_unknown(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _formula_only_workbook_bytes(),
        parser=table_parser,
    )

    assert projection["tables"]
    assert projection["tables"][0]["source_total_count"] is None
    assert len(projection["rows"]) >= 1
    assert all(row["row_role_kwd"] == "unknown" for row in projection["rows"])


def test_duplicate_values_remain_distinct_and_rows_use_only_fixed_fields(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(include_context=False),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )
    target = [row for row in projection["rows"] if row["table_label_kwd"] == "Inspection"]
    data_rows = [row for row in target if row["row_role_kwd"] == "data"]
    uncertain_rows = [row for row in target if row["row_role_kwd"] == "unknown"]

    assert len(data_rows) == 3
    assert len({row["row_ref_kwd"] for row in data_rows}) == 3
    assert [row["data_row_index_int"] for row in data_rows] == [1, 2, 3]
    assert {row["source_total_count_int"] for row in target} == {None}
    assert len(uncertain_rows) == 1
    assert uncertain_rows[0]["data_row_index_int"] is None
    assert all(set(row) == PROJECTION_ROW_FIELDS for row in projection["rows"])
    assert json.loads(data_rows[0]["ordered_fields_list"]) == [
        {
            "column_id": "col_v1:1:1",
            "column_ordinal": 1,
            "header_path": ["Code"],
            "name": "Code",
            "value": "R-DUP",
        },
        {
            "column_id": "col_v1:1:2",
            "column_ordinal": 2,
            "header_path": ["Description"],
            "name": "Description",
            "value": "Repeated",
        },
        {
            "column_id": "col_v1:1:3",
            "column_ordinal": 3,
            "header_path": ["Force"],
            "name": "Force",
            "value": "7.5",
        },
    ]
    assert all("Code" not in set(row) for row in projection["rows"])

    alternate = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(headers=("Component", "Supplier", "Unit")),
        parser=table_parser,
    )
    assert {frozenset(row) for row in alternate["rows"]} == {PROJECTION_ROW_FIELDS}


def test_ordered_fields_remove_only_insignificant_float_zeroes():
    fields = _ordered_fields(
        ["Integral", "Decimal", "Text"],
        [1.0, 1.5, "01-A"],
        note=False,
    )

    assert fields == [
        {"name": "Integral", "value": "1"},
        {"name": "Decimal", "value": "1.5"},
        {"name": "Text", "value": "01-A"},
    ]


def test_stable_plain_text_rows_are_data_and_prove_the_denominator(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "PlainText"
    sheet.append(["Characteristic", "Type", "Requirement"])
    sheet.append(["Rolling resistance", "SC", "<= 7.5 N/KN"])
    sheet.append(["Uniformity", "SC", "RFV <= 170 N"])
    sheet.append(["Stiffness", "SC", "Radial >= 320 N/mm"])
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )
    target = [row for row in projection["rows"] if row["table_label_kwd"] == "PlainText"]

    assert [row["row_role_kwd"] for row in target] == ["data", "data", "data"]
    assert [row["data_row_index_int"] for row in target] == [1, 2, 3]
    assert {row["source_total_count_int"] for row in target} == {3}


def test_repeated_partial_merge_rows_are_data_when_the_shape_is_stable(table_parser, monkeypatch):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "MergedText"
    sheet.append(["Characteristic", "Type", "Requirement"])
    for index, requirement in enumerate(("<= 7.5 N/KN", "RFV <= 170 N", ">= 320 N/mm"), start=2):
        sheet.merge_cells(start_row=index, start_column=1, end_row=index, end_column=2)
        sheet.cell(index, 1, f"Characteristic {index}")
        sheet.cell(index, 3, requirement)
    monkeypatch.setattr(
        table_parser,
        "_parse_sheet_structure",
        lambda _worksheet, _rows: (["Characteristic", "Type", "Requirement"], 0, 1),
    )
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )
    target = [row for row in projection["rows"] if row["table_label_kwd"] == "MergedText"]

    assert [row["row_role_kwd"] for row in target] == ["data", "data", "data"]
    assert {row["source_total_count_int"] for row in target} == {3}


@pytest.mark.parametrize(
    ("workbook_factory", "sheet_name", "column_count", "expected_row_count"),
    [
        (_horizontal_varying_merge_workbook_bytes, "Anonymous horizontal merges", 4, 3),
        (_vertical_varying_merge_workbook_bytes, "Anonymous vertical merges", 4, 4),
        (_mixed_row_merge_supplier_workbook_bytes, "Anonymous mixed merges", 5, 4),
    ],
)
def test_varying_record_merge_shapes_preserve_the_record_axis(
    table_parser,
    monkeypatch,
    workbook_factory,
    sheet_name,
    column_count,
    expected_row_count,
):
    monkeypatch.setattr(
        table_parser,
        "_parse_sheet_structure",
        lambda _worksheet, _rows: (
            [f"Field {chr(65 + index)}" for index in range(column_count)],
            0,
            1,
        ),
    )
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        workbook_factory(),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )

    table = next(
        table
        for table in projection["tables"]
        if table["table_label"] == sheet_name
        and table["row_count"] == expected_row_count
    )
    rows = [
        row
        for row in projection["rows"]
        if row["table_ref_kwd"] == table["table_ref"]
    ]

    assert [row["row_ordinal_int"] for row in rows] == list(
        range(2, expected_row_count + 2)
    )
    assert all(
        "Sidecar label" not in row["ordered_fields_list"]
        and "example.invalid" not in row["ordered_fields_list"]
        for row in rows
    )
    assert [row["row_role_kwd"] for row in rows] == ["data"] * expected_row_count
    assert [row["data_row_index_int"] for row in rows] == list(
        range(1, expected_row_count + 1)
    )
    assert table["source_total_count"] == expected_row_count


def test_varying_record_merge_shapes_are_proven_by_the_real_candidate_stage(
    table_parser,
):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _mixed_row_merge_supplier_workbook_bytes(),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )

    table = next(
        table
        for table in projection["tables"]
        if table["table_label"] == "Anonymous mixed merges"
    )
    rows = [
        row
        for row in projection["rows"]
        if row["table_ref_kwd"] == table["table_ref"]
    ]

    assert [row["row_ordinal_int"] for row in rows] == [2, 3, 4, 5]
    assert [row["row_role_kwd"] for row in rows] == ["data"] * 4
    assert [row["data_row_index_int"] for row in rows] == [1, 2, 3, 4]
    assert table["source_total_count"] == 4


def test_vertical_merge_parent_values_are_inherited_and_emitted_once(
    table_parser,
):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _vertical_varying_merge_workbook_bytes(),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )

    table = next(
        table
        for table in projection["tables"]
        if table["table_label"] == "Anonymous vertical merges"
    )
    rows = [
        row
        for row in projection["rows"]
        if row["table_ref_kwd"] == table["table_ref"]
    ]
    expected_parent_values = {
        2: "A-group-1",
        3: "A-group-1",
        4: "A-3",
        5: "A-4",
    }

    assert [row["row_role_kwd"] for row in rows] == ["data"] * 4
    for row in rows:
        fields = json.loads(row["ordered_fields_list"])
        parent_fields = [field for field in fields if field["column_ordinal"] == 1]
        assert len(parent_fields) == 1
        assert parent_fields[0]["value"] == expected_parent_values[row["row_ordinal_int"]]
    assert table["source_total_count"] == 4


def test_repeated_header_inside_a_table_still_invalidates_completeness(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "RepeatedHeader"
    sheet.append(["Code", "Description"])
    sheet.append(["A-1", "First"])
    sheet.append(["Code", "Description"])
    sheet.append(["A-2", "Second"])
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )
    target = [row for row in projection["rows"] if row["table_label_kwd"] == "RepeatedHeader"]
    repeated_header = next(row for row in target if row["row_ordinal_int"] == 3)

    assert repeated_header["row_role_kwd"] == "unknown"
    assert repeated_header["data_row_index_int"] is None
    assert {row["source_total_count_int"] for row in target} == {None}


def test_table_parser_exposes_the_projection_producer_without_using_chunk_output(monkeypatch):
    table = _load_table_module(monkeypatch)

    projection = table.build_structure_projection("anonymous.xlsx", _workbook_bytes())

    assert projection["version"] == "tabular-structure-projection/v6"
    assert projection["rows"]
    assert all("content_with_weight" not in row for row in projection["rows"])


def test_unknown_body_row_invalidates_the_table_denominator_without_becoming_a_note(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(include_unknown=True),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )
    target = [row for row in projection["rows"] if row["table_label_kwd"] == "Inspection"]
    sparse = next(row for row in target if "Sparse body row" in row["ordered_fields_list"])

    assert sparse["row_role_kwd"] == "unknown"
    assert sparse["data_row_index_int"] is None
    assert {row["source_total_count_int"] for row in target} == {None}


def test_full_width_merged_row_inside_the_body_fails_closed_instead_of_becoming_a_note(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(include_merged_body=True),
        parser=table_parser,
    )
    target = [row for row in projection["rows"] if row["table_label_kwd"] == "Inspection"]
    merged_body = next(row for row in target if "R-MERGED" in row["ordered_fields_list"])

    assert merged_body["row_role_kwd"] == "unknown"
    assert merged_body["data_row_index_int"] is None
    assert {row["source_total_count_int"] for row in target} == {None}


def test_lone_full_width_merged_body_row_cannot_be_silently_excluded_as_a_note(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Description"])
    sheet.merge_cells("A2:B2")
    sheet["A2"] = "R-MERGED"
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert projection["rows"][0]["row_role_kwd"] == "unknown"
    assert projection["rows"][0]["data_row_index_int"] is None


def test_final_full_width_merged_row_after_data_cannot_be_assumed_to_be_a_note(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Description"])
    sheet.append(["A-1", "Ordinary"])
    sheet.merge_cells("A3:B3")
    sheet["A3"] = "A-2"
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert projection["rows"][-1]["row_role_kwd"] == "unknown"


def test_repeated_full_width_body_rows_need_positive_note_evidence(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.append(["A-2", "Closed"])
    for row_ordinal, value in ((4, "A-3"), (5, "A-4")):
        sheet.merge_cells(
            start_row=row_ordinal,
            start_column=1,
            end_row=row_ordinal,
            end_column=2,
        )
        sheet.cell(row_ordinal, 1, value)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert [row["row_ordinal_int"] for row in projection["rows"]] == [2, 3, 4, 5]
    assert [row["row_role_kwd"] for row in projection["rows"][-2:]] == ["unknown", "unknown"]


def test_mixed_generation_and_inconsistent_totals_fail_validation(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )
    projection["rows"][0]["producer_generation_ref_kwd"] = _generation_ref()

    with pytest.raises(ValueError, match="mixed producer generations"):
        validate_tabular_structure_projection(projection)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )
    projection["rows"][0]["source_total_count_int"] = 999

    with pytest.raises(ValueError, match="source total"):
        validate_tabular_structure_projection(projection)


def test_manifest_and_record_identity_tampering_fail_validation(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        parser=table_parser,
    )
    projection["tables"][0]["row_count"] += 1

    with pytest.raises(ValueError, match="table manifest"):
        validate_tabular_structure_projection(projection)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        parser=table_parser,
    )
    projection["rows"][0]["id"] = "tsr_v1_tampered"

    with pytest.raises(ValueError, match="record identity"):
        validate_tabular_structure_projection(projection)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        parser=table_parser,
    )
    projection["tables"][0]["table_ref"] = "tbl_v1_" + "0" * 64

    with pytest.raises(ValueError, match="table reference"):
        validate_tabular_structure_projection(projection)


def test_multi_region_manifest_requires_contiguous_physical_order(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _vertical_multi_region_workbook_bytes(),
        parser=table_parser,
    )
    projection["tables"].reverse()

    with pytest.raises(ValueError, match="physical order"):
        validate_tabular_structure_projection(projection)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _vertical_multi_region_workbook_bytes(),
        parser=table_parser,
    )
    second_table_ref = projection["tables"][1]["table_ref"]
    projection["tables"] = [projection["tables"][1]]
    projection["rows"] = [
        row for row in projection["rows"] if row["table_ref_kwd"] == second_table_ref
    ]

    with pytest.raises(ValueError, match="contiguous"):
        validate_tabular_structure_projection(projection)


def test_malformed_json_evidence_fields_fail_validation(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        parser=table_parser,
    )
    projection["rows"][0]["ordered_fields_list"] = "not-json"

    with pytest.raises(ValueError, match="ordered fields"):
        validate_tabular_structure_projection(projection)


@pytest.mark.parametrize("payload", [5, None, True, {"unexpected": "value"}])
def test_fixed_field_schema_rejects_non_list_json_payloads(table_parser, payload):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        parser=table_parser,
    )
    projection["rows"][0]["ordered_fields_list"] = json.dumps(payload)

    with pytest.raises(ValueError, match="ordered fields must use the fixed field schema"):
        validate_tabular_structure_projection(projection)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tabular_structure_version_kwd", "tabular-row/v0", "row version"),
        ("producer_schema_version_kwd", "table-producer/v0", "producer schema"),
        ("structure_kind_kwd", "ordinary_chunk", "structure kind"),
        ("row_ordinal_int", "5", "row ordinal"),
        ("data_row_index_int", "1", "data row index"),
        ("source_total_count_int", "3", "source total"),
    ],
)
def test_fixed_row_schema_rejects_wrong_versions_kinds_and_integer_types(
    table_parser,
    field,
    value,
    message,
):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        parser=table_parser,
    )
    projection["rows"][0][field] = value

    with pytest.raises(ValueError, match=message):
        validate_tabular_structure_projection(projection)


def test_fixed_row_schema_rejects_non_string_table_label(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        parser=table_parser,
    )
    projection["rows"][0]["table_label_kwd"] = 123

    with pytest.raises(ValueError, match="table label"):
        validate_tabular_structure_projection(projection)


@pytest.mark.parametrize("field", ["sheet_ordinal", "table_ordinal", "row_count", "data_row_count", "source_total_count"])
def test_manifest_rejects_boolean_integer_fields(table_parser, field):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        parser=table_parser,
    )
    projection["tables"][0][field] = True

    with pytest.raises(ValueError, match="table manifest"):
        validate_tabular_structure_projection(projection)


def test_header_only_sheet_produces_no_false_table_manifest(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Description"])
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] is None
    assert all(row["row_role_kwd"] == "unknown" for row in projection["rows"])


def test_second_table_after_an_internal_separator_fails_closed(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    sheet.append(["A-1", 1])
    sheet.append(["A-2", 2])
    sheet.append([None, None])
    sheet.append(["Supplier", "Status"])
    sheet.append(["North", "Approved"])
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] is None
    assert all(row["row_role_kwd"] == "unknown" for row in projection["rows"])


def test_adjacent_second_table_with_a_new_row_shape_fails_closed(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    sheet.append(["A-1", 1])
    sheet.append(["A-2", 2])
    sheet.append(["Supplier", "Status"])
    sheet.append(["North", "Approved"])
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert any(row["row_role_kwd"] == "unknown" for row in projection["rows"])


def test_adjacent_second_table_with_the_same_text_shape_fails_closed(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    sheet.append(["A-1", "One"])
    sheet.append(["A-2", "Two"])
    sheet.append(["Supplier", "Status"])
    sheet.append(["North", "Approved"])
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert any(row["row_role_kwd"] == "unknown" for row in projection["rows"])


def test_complete_projection_does_not_drop_data_after_a_large_blank_region(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    sheet.append(["A-1", 1])
    sheet.cell(row=20_001, column=1, value="A-2")
    sheet.cell(row=20_001, column=2, value=2)
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        parser=table_parser,
    )

    assert any(row["row_ordinal_int"] == 20_001 for row in projection["rows"])
    assert projection["tables"][0]["source_total_count"] is None


def test_large_gap_before_the_first_body_row_also_fails_closed(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    sheet.cell(row=20_001, column=1, value="A-1")
    sheet.cell(row=20_001, column=2, value=1)
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        parser=table_parser,
    )

    assert projection["tables"][0]["source_total_count"] is None
    assert projection["rows"][0]["row_role_kwd"] == "unknown"


def test_complete_projection_only_materializes_a_bounded_header_probe(monkeypatch, table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    sheet.append(["A-1", 1])
    sheet.cell(row=20_001, column=1, value="A-2")
    sheet.cell(row=20_001, column=2, value=2)
    original_iter_rows = sheet.iter_rows

    def bounded_iter_rows(*args, **kwargs):
        assert kwargs.get("max_row", 0) <= 20
        return original_iter_rows(*args, **kwargs)

    monkeypatch.setattr(sheet, "iter_rows", bounded_iter_rows)
    monkeypatch.setattr(table_parser, "_load_excel_to_workbook", lambda _source: workbook)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        b"complete-source-bytes",
        parser=table_parser,
    )

    assert any(row["row_ordinal_int"] == 20_001 for row in projection["rows"])


def test_uncached_formula_only_row_is_preserved_as_unknown(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    sheet.append(["A-1", 1])
    sheet.append(["=CONCAT(\"A-\", 2)", "=SUM(1, 1)"])
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        parser=table_parser,
    )

    formula_row = next(row for row in projection["rows"] if row["row_ordinal_int"] == 3)
    assert formula_row["row_role_kwd"] == "unknown"
    assert projection["tables"][0]["source_total_count"] is None


def test_row_with_value_and_uncached_formula_is_emitted_once_as_unknown(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Value"])
    sheet.append(["A-1", 1])
    sheet.append(["A-2", "=SUM(1, 1)"])
    output = BytesIO()
    workbook.save(output)

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        output.getvalue(),
        parser=table_parser,
    )

    target_rows = [row for row in projection["rows"] if row["row_ordinal_int"] == 3]
    assert len(target_rows) == 1
    assert target_rows[0]["row_role_kwd"] == "unknown"
    assert projection["tables"][0]["source_total_count"] is None


def test_all_uncached_formula_rows_are_preserved_as_unknown(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Code", "Status"])
    sheet.append(["A-1", "Open"])
    sheet.cell(row=3, column=1, value='=CONCAT("A-", 2)')
    sheet.cell(row=3, column=2, value='=CONCAT("Closed")')
    sheet.cell(row=4, column=1, value='=CONCAT("A-", 3)')
    sheet.cell(row=4, column=2, value='=CONCAT("Open")')

    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        parser=table_parser,
    )

    assert {row["row_ordinal_int"] for row in projection["rows"]} >= {2, 3, 4}
    assert {
        row["row_ordinal_int"]: row["row_role_kwd"]
        for row in projection["rows"]
    } == {2: "data", 3: "unknown", 4: "unknown"}
    assert all(table["source_total_count"] is None for table in projection["tables"])


def test_biff_formula_inventory_uses_only_sheet_and_cell_coordinates():
    global_bof = struct.pack("<HHHH", 0x0809, 4, 0x0600, 0x0005)
    worksheet_bof = struct.pack("<HHHH", 0x0809, 4, 0x0600, 0x0010)
    formula = struct.pack("<HHHH", 0x0006, 4, 2, 3)
    eof = struct.pack("<HH", 0x000A, 0)
    sheet_offset = len(global_bof) + 4 + 6 + len(eof)
    boundsheet_payload = struct.pack("<IBB", sheet_offset, 0, 0)
    boundsheet = struct.pack("<HH", 0x0085, len(boundsheet_payload)) + boundsheet_payload
    stream = global_bof + boundsheet + eof + worksheet_bof + formula + eof

    assert _formula_coordinates_from_biff_stream(stream) == [{(3, 4)}]


def test_generation_reference_rejects_customer_text(table_parser):
    with pytest.raises(ValueError, match="UUID/ULID"):
        build_tabular_structure_projection(
            "anonymous.xlsx",
            _workbook_bytes(),
            producer_generation_ref="supplier-run-2026",
            parser=table_parser,
        )


def test_context_is_bounded_and_removes_controls_without_a_language_allowlist(table_parser):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Inspection"
    sheet.merge_cells("A1:C1")
    sheet["A1"] = "\u9879\u76ee \u00b5\u03a9\u2103\u0085\u202e"
    sheet.append(["Code", "Description", "Force"])
    sheet.append(["A-1", "Open", 1])
    sheet.append(["A-2", "Closed", 2])
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _save_workbook(workbook),
        producer_generation_ref=_generation_ref(),
        table_context_entry_limit=1,
        table_context_value_bytes=32,
        parser=table_parser,
    )
    target = next(row for row in projection["rows"] if row["table_label_kwd"] == "Inspection")
    context = json.loads(target["table_context_list"])

    assert len(context) == 1
    assert all(len(item["value"].encode("utf-8")) <= 32 for item in context)
    assert "\u0085" not in json.dumps(context, ensure_ascii=False)
    assert "\u202e" not in json.dumps(context, ensure_ascii=False)
    assert "\u9879\u76ee \u00b5\u03a9\u2103" in json.dumps(context, ensure_ascii=False)


def test_global_indices_and_totals_survive_projection_part_boundaries(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(rows=3002, include_note=False, include_context=False),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )
    parts = partition_tabular_structure_projection(projection, rows_per_part=3000)
    target = [row for row in projection["rows"] if row["table_label_kwd"] == "Inspection"]
    data_rows = [row for row in target if row["row_role_kwd"] == "data"]

    assert len(parts) == 2
    assert [row["data_row_index_int"] for row in data_rows] == list(range(1, 3003))
    assert {row["source_total_count_int"] for row in target} == {3002}
    assert len({row["row_ref_kwd"] for row in target}) == len(target)


def test_projection_build_does_not_change_ordinary_table_chunks(monkeypatch, table_parser):
    table = _load_table_module(monkeypatch)
    source = _workbook_bytes()
    before = table.chunk(
        "anonymous.xlsx",
        binary=source,
        callback=lambda *_args: None,
        kb_id="ordinary-kb",
    )

    build_tabular_structure_projection(
        "anonymous.xlsx",
        source,
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )
    after = table.chunk(
        "anonymous.xlsx",
        binary=source,
        callback=lambda *_args: None,
        kb_id="ordinary-kb",
    )

    assert json.dumps(after, ensure_ascii=False, sort_keys=True) == json.dumps(before, ensure_ascii=False, sort_keys=True)


class _VerifiedStorage:
    def __init__(self, fail_on=None):
        self.objects = {}
        self.fail_on = fail_on

    def put(self, bucket, name, binary, tenant_id=None):
        if self.fail_on and self.fail_on(name):
            return None
        self.objects[(bucket, name)] = bytes(binary)

    def obj_exist(self, bucket, name, tenant_id=None):
        return (bucket, name) in self.objects

    def get(self, bucket, name, tenant_id=None):
        return self.objects.get((bucket, name))


class _StorageWithoutTenantReadArguments(_VerifiedStorage):
    def obj_exist(self, bucket, name):
        return (bucket, name) in self.objects

    def get(self, bucket, name):
        return self.objects.get((bucket, name))


def test_sidecar_storage_writes_verified_parts_before_the_immutable_manifest(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )
    storage = _VerifiedStorage()

    receipt = store_tabular_structure_projection(
        storage,
        bucket="dataset-id",
        document_id="document-id",
        projection=projection,
        rows_per_part=2,
        tenant_id="tenant-id",
    )

    names = [name for bucket, name in storage.objects if bucket == "dataset-id"]
    assert "/manifest-" in names[-1] and names[-1].endswith(".json")
    assert all("/part-" not in name or len(name.rsplit("-", 1)[-1].removesuffix(".json")) == 64 for name in names)
    assert all("anonymous" not in name and "Inspection" not in name for name in names)
    manifest = json.loads(storage.objects[("dataset-id", receipt["manifest_object_name"])])
    assert manifest["producer_generation_ref"] == projection["producer_generation_ref"]
    for part in manifest["parts"]:
        payload = storage.objects[("dataset-id", part["object_name"])]
        assert hashlib.sha256(payload).hexdigest() == part["sha256"]


def test_sidecar_storage_never_publishes_a_manifest_after_a_part_write_failure(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        producer_generation_ref=_generation_ref(),
        parser=table_parser,
    )
    storage = _VerifiedStorage(fail_on=lambda name: "/part-000002-" in name)

    with pytest.raises(IOError, match="verify structure projection object"):
        store_tabular_structure_projection(
            storage,
            bucket="dataset-id",
            document_id="document-id",
            projection=projection,
            rows_per_part=2,
        )

    assert not any("/manifest-" in name for _bucket, name in storage.objects)


def test_sidecar_storage_supports_backends_without_tenant_read_arguments(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
        parser=table_parser,
    )
    storage = _StorageWithoutTenantReadArguments()

    receipt = store_tabular_structure_projection(
        storage,
        bucket="dataset-id",
        document_id="document-id",
        projection=projection,
        tenant_id="tenant-id",
    )

    assert "/manifest-" in receipt["manifest_object_name"]
