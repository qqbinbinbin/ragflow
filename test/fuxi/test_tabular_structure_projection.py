import hashlib
import json
import struct
import uuid
from io import BytesIO

import pytest
from openpyxl import Workbook

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


def test_current_producer_schema_is_v3_for_display_semantics_invalidation():
    assert PRODUCER_SCHEMA_VERSION == "table-producer/v3"
    assert tabular_structure.STRUCTURE_PRODUCER_ALGORITHM_VERSION == "region-producer/v3"


def test_table_ref_identity_binds_algorithm_version_and_exact_membership(monkeypatch):
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
        "tabular-structure-projection/v1",
    )
    monkeypatch.setattr(
        tabular_structure,
        "STRUCTURE_PRODUCER_ALGORITHM_VERSION",
        "region-producer/test-next",
    )
    changed_algorithm = tabular_structure._table_ref(source_sha256, 1, 1, "a" * 64)

    assert first.startswith("tbl_v2_" + "a" * 64 + "_")
    assert len({first, changed_members, changed_schema, changed_projection, changed_algorithm}) == 5


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


def _horizontal_complete_with_headerless_sibling_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Anonymous regions"
    sheet.append(["Left code", "Left status", None, None, "R-1", "Closed"])
    sheet.append(["L-1", "Open", None, None, "R-2", "Open"])
    sheet.append(["L-2", "Closed", None, None, "R-3", "Closed"])
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
        [{"name": "Left code", "value": "L-1"}, {"name": "Left status", "value": "Open"}],
        [{"name": "Left code", "value": "L-2"}, {"name": "Left status", "value": "Closed"}],
    ]
    assert rows_by_table[projection["tables"][1]["table_ref"]] == [
        [{"name": "Right code", "value": "R-1"}, {"name": "Right status", "value": "Closed"}],
        [{"name": "Right code", "value": "R-2"}, {"name": "Right status", "value": "Open"}],
    ]


def test_vertical_headerless_continuation_downgrades_complete_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _vertical_complete_with_headerless_continuation_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert all(table["source_total_count"] is None for table in projection["tables"])


def test_horizontal_headerless_record_axis_downgrades_complete_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _horizontal_complete_with_headerless_sibling_bytes(),
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


def test_g_sensitive_unknown_downgrades_complete_sibling(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _complete_table_with_g_sensitive_sibling_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 2
    assert all(table["source_total_count"] is None for table in projection["tables"])
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
        [{"name": "Code", "value": "S-1"}],
        [{"name": "Code", "value": "S-2"}],
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


def test_unbound_uncached_formula_downgrades_every_table_on_the_sheet(table_parser):
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

    assert projection["tables"]
    assert all(table["source_total_count"] is None for table in projection["tables"])


def test_multilevel_merged_header_stays_with_its_record_axis(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _merged_header_workbook_bytes(),
        parser=table_parser,
    )

    assert len(projection["tables"]) == 1
    assert projection["tables"][0]["source_total_count"] == 2
    assert [row["row_ordinal_int"] for row in projection["rows"]] == [3, 4]


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
        {"name": "Code", "value": "R-DUP"},
        {"name": "Description", "value": "Repeated"},
        {"name": "Force", "value": "7.5"},
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

    assert projection["version"] == "tabular-structure-projection/v1"
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
