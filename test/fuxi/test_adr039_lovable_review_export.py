import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


BASELINE_PATH = Path(__file__).parent / "fixtures" / "adr039-l1-regression-baseline.json"

FROZEN_L1_EXPECTATIONS = {
    "L1-FIRST-OR-DELAYED-HEADER": (1, (3,), ("supported_complete",), ("L1-07",)),
    "L1-MERGED-MULTILEVEL-10": (1, (10,), ("supported_complete",), ("L1-04",)),
    "L1-SEPARATED-MULTI-LIST": (
        2,
        (2, 2),
        ("supported_complete", "supported_complete"),
        ("L1-03", "L1-03"),
    ),
    "L1-CONTINUATION-EQUAL": (1, (3,), ("supported_complete",), ("L1-02",)),
    "L1-CONTINUATION-SUBSET": (1, (3,), ("supported_complete",), ("L1-02",)),
    "L1-CONTINUATION-SUPERSET-NAMED": (1, (3,), ("supported_complete",), ("L1-02",)),
    "L1-SINGLE-COLUMN-CATALOGUE": (1, (2,), ("supported_complete",), ("L1-05",)),
    "L1-SPARSE-STABLE-SLOTS": (1, (3,), ("supported_complete",), ("L1-06",)),
    "L1-ADR044-XLS-EQUIVALENT": (1, (10,), ("supported_complete",), ("L1-01",)),
}


def _load_baseline():
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _coordinate_digest(coordinates):
    payload = "\n".join(f"1:{row}:{column}" for row, column in sorted(coordinates)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _multiset_digest(coordinates):
    lines = ["adr039-emitted-cell-multiset/v1"]
    lines.extend(f"1:{row}:{column}" for row, column in sorted(coordinates))
    return hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()


def _typed_value_map_digest(cell_values):
    lines = ["adr039-cell-value-map/v1"]
    for row, column, value_type, value in sorted(cell_values):
        normalized = unicodedata.normalize("NFC", value) if value_type == "text" else value
        lines.append(
            json.dumps(
                [1, row, column, value_type, normalized],
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
    return hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()


def _merge_map_digest(merged_ranges):
    lines = ["adr039-merged-ranges/v1"]
    lines.extend(
        json.dumps([1, *merged_range], separators=(",", ":"))
        for merged_range in sorted(merged_ranges)
    )
    return hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()


def _rows_by_columns(rows, columns):
    return {(row, column) for row in rows for column in columns}


def _sample_objects(sample):
    shape = sample["shape"]
    case = sample["builder_case"]
    if case == "delayed_header_two_columns_three_records":
        return [_rows_by_columns(shape["header_rows"] + shape["record_rows"], shape["occupied_columns"])]
    if case in {
        "three_merged_header_levels_two_columns_ten_records",
        "adr044_biff8_conversion_of_three_merged_header_levels_two_columns_ten_records",
    }:
        assert shape["merged_non_anchor_membership"] == "included_when_anchor_nonblank"
        members = {
            (row, column)
            for min_row, min_column, max_row, max_column in shape["merged_ranges"]
            for row in range(min_row, max_row + 1)
            for column in range(min_column, max_column + 1)
        }
        members |= _rows_by_columns([shape["leaf_header_row"]], shape["occupied_columns"])
        members |= _rows_by_columns(shape["record_rows"], shape["occupied_columns"])
        return [members]
    if case == "two_vertical_lists_two_blank_rows_two_records_each":
        return [
            _rows_by_columns([header] + rows, shape["occupied_columns"])
            for header, rows in zip(shape["object_header_rows"], shape["object_record_rows"])
        ]
    if case in {
        "equal_columns_headerless_single_row_continuation",
        "subset_columns_headerless_single_row_continuation",
        "superset_columns_named_header_and_single_record",
    }:
        main = _rows_by_columns(
            [shape["main_header_row"]] + shape["main_record_rows"],
            shape["main_columns"],
        )
        continuation_rows = shape["continuation_record_rows"][:]
        if "continuation_header_row" in shape:
            continuation_rows.insert(0, shape["continuation_header_row"])
        continuation = _rows_by_columns(continuation_rows, shape["continuation_columns"])
        expected_components = sample["component_membership_sha256"]
        assert [_coordinate_digest(main), _coordinate_digest(continuation)] == expected_components
        assert main.isdisjoint(continuation)
        return [main | continuation]
    if case == "single_header_one_column_two_records":
        return [_rows_by_columns(shape["header_rows"] + shape["record_rows"], shape["occupied_columns"])]
    if case == "three_columns_anchor_first_three_sparse_records":
        members = _rows_by_columns(shape["header_rows"] + shape["record_rows"], shape["occupied_columns"])
        return [members - {tuple(cell) for cell in shape["empty_cells"]}]
    raise AssertionError(f"unhandled builder case: {case}")


def test_l1_baseline_freezes_reviewed_object_counts_totals_statuses_and_rules():
    baseline = _load_baseline()

    assert baseline["schema"] == "adr039-l1-regression-baseline/v2"
    assert baseline["runtime_input"] is False
    assert baseline["builder_contract"]["production_import_forbidden"] is True
    actual = {
        sample["sample_code"]: (
            sample["expected_object_count"],
            tuple(sample["expected_source_totals"]),
            tuple(sample["expected_statuses"]),
            tuple(sample["expected_rules"]),
        )
        for sample in baseline["samples"]
    }
    assert actual == FROZEN_L1_EXPECTATIONS


def test_l1_baseline_recomputes_all_membership_and_ingest_evidence():
    baseline = _load_baseline()

    assert baseline["builder_contract"]["membership_hash_input"].endswith("no trailing newline")
    assert "preserving duplicates" in baseline["builder_contract"]["emitted_cell_multiset_hash_input"]
    for sample in baseline["samples"]:
        objects = _sample_objects(sample)
        assert len(objects) == sample["expected_object_count"]
        assert [_coordinate_digest(members) for members in objects] == sample["expected_membership_sha256"]
        assert [
            _multiset_digest(members) for members in objects
        ] == sample["expected_emitted_cell_multiset_sha256"]
        assert sample["expected_emitted_cell_occurrence_count"] == [len(members) for members in objects]
        assert sample["expected_member_max_ingest_count"] == [1] * len(objects)


def test_l1_baseline_binds_duplicate_ingest_and_adr044_content_equivalence():
    baseline = _load_baseline()
    duplicate = baseline["defect_samples"][0]
    emitted = [tuple(cell) for cell in duplicate["emitted_member_coordinate_multiset"]]
    counts = Counter(emitted)

    assert duplicate["expected_check"] == "D2"
    assert duplicate["expected_status"] == "defect"
    assert duplicate["expected_reason"] == "membership_not_closed"
    assert max(counts.values()) == duplicate["expected_member_max_ingest_count"] == 2
    assert _coordinate_digest(set(emitted)) == duplicate["expected_membership_sha256"]
    assert len(emitted) == duplicate["expected_emitted_cell_occurrence_count"]
    assert _multiset_digest(emitted) == duplicate["expected_emitted_cell_multiset_sha256"]
    assert duplicate["expected_emitted_cell_multiset_sha256"] != duplicate["expected_membership_sha256"]

    adr044 = next(sample for sample in baseline["samples"] if sample["sample_code"] == "L1-ADR044-XLS-EQUIVALENT")
    equivalence = adr044["conversion_value_equivalence"]
    assert adr044["converter_version_contract"] == "immutable_converter_build_or_service_version_not_protocol_version"
    value_digest = _typed_value_map_digest(adr044["shape"]["anonymous_typed_cell_values"])
    assert len(adr044["shape"]["anonymous_typed_cell_values"]) == equivalence["expected_value_entry_count"]
    assert value_digest == equivalence["expected_original_value_map_sha256"]
    assert value_digest == equivalence["expected_converted_value_map_sha256"]
    merge_digest = _merge_map_digest(adr044["shape"]["merged_ranges"])
    assert merge_digest == equivalence["expected_original_merge_map_sha256"]
    assert merge_digest == equivalence["expected_converted_merge_map_sha256"]
