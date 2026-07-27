import hashlib
import json
import uuid
from io import BytesIO

import pytest
from openpyxl import Workbook

from test.fuxi.test_table_semantic_rows import _load_table_module

from rag.app.tabular_structure import (
    PRODUCER_SCHEMA_VERSION,
    PROJECTION_ROW_FIELDS,
    build_tabular_structure_projection,
    partition_tabular_structure_projection,
    store_tabular_structure_projection,
    validate_tabular_structure_projection,
)


def test_current_producer_schema_is_v2_for_generation_invalidation():
    assert PRODUCER_SCHEMA_VERSION == "table-producer/v2"


def _workbook_bytes(
    *,
    headers=("Code", "Description", "Force"),
    include_unknown=False,
    include_merged_body=False,
    include_note=True,
    rows=3,
):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Inspection"
    sheet.merge_cells("A1:C1")
    sheet["A1"] = "\u9879\u76ee \u00b5\u03a9\u2103\u0085\u202e"
    sheet["A3"] = "Program"
    sheet["B3"] = "Platform-X"
    sheet["A4"], sheet["B4"], sheet["C4"] = headers

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


def test_duplicate_values_remain_distinct_and_rows_use_only_fixed_fields(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
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

    assert projection["tables"] == []
    assert projection["rows"] == []


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
    assert any(row["row_role_kwd"] == "unknown" for row in projection["rows"])


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


def test_generation_reference_rejects_customer_text(table_parser):
    with pytest.raises(ValueError, match="UUID/ULID"):
        build_tabular_structure_projection(
            "anonymous.xlsx",
            _workbook_bytes(),
            producer_generation_ref="supplier-run-2026",
            parser=table_parser,
        )


def test_context_is_bounded_and_removes_controls_without_a_language_allowlist(table_parser):
    projection = build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(),
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
        _workbook_bytes(rows=3002, include_note=False),
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
