"""Immutable layout transport must not imply semantic field interpretation."""
import copy
import importlib
import json

import pytest


def contract():
    return importlib.import_module("rag.app.source_layout")


def example():
    return {
        "version": "source-layout/v1",
        "source_sha256": "a" * 64,
        "producer_generation_ref": "generation-a",
        "sheet_ordinal": 2,
        "object_ref": "object-a",
        "cells": [
            {"anchor": [1, 1], "span": [1, 1, 1, 2], "value": "Label", "state": "literal"},
            {"anchor": [2, 1], "span": [2, 1, 2, 2], "value": None, "state": "blank"},
        ],
    }


def read(value, expected=None):
    expected = expected or example()
    return contract().validate_source_layout(
        value, source_sha256=expected["source_sha256"],
        producer_generation_ref=expected["producer_generation_ref"],
        sheet_ordinal=expected["sheet_ordinal"], object_ref=expected["object_ref"],
    )


def test_layout_survives_json_without_inventing_field_roles():
    value = json.loads(json.dumps(example()))
    assert read(value) == value
    assert "enumeration_status" not in value


@pytest.mark.parametrize("state,content", [
    ("date", "2026-09-18"), ("datetime", "2026-09-18T12:34:56"),
    ("time", "12:34:56.123000"),
    ("date", "2024-02-29"), ("date", "0001-01-01"),
    ("datetime", "2026-09-18T12:00:00.123456+08:00"),
    ("time", "00:00:00"), ("time", "12:00:00+00:00"),
    ("time", "12:00:00-05:30:01.000001"),
])
def test_layout_preserves_typed_temporal_values(state, content):
    value = example()
    value["cells"][0].update(state=state, value=content)
    assert read(json.loads(json.dumps(value))) == value


@pytest.mark.parametrize("state,content", [
    ("date", "2026-02-30"), ("date", "20260918"),
    ("datetime", "2026-09-18"), ("datetime", "2026-09-18 12:34:56"),
    ("time", "25:00:00"), ("time", None), ("date", 20260918),
    ("date", "2026-02-29"), ("date", "0000-01-01"),
    ("datetime", "2026-09-18T25:00:00"), ("datetime", "2026-09-18T12:00:00Z"),
    ("time", "12:00"), ("time", "12:00:60"), ("time", "12:00:00.123"),
    ("time", "12:00:00.000000"), ("time", "12:00:00+24:00"),
    ("time", "12:00:00-00:00"), ("time", "12:00:00+00:00:00.000001"),
    ("time", "12:00:00-00:00:00.000001"),
])
def test_layout_rejects_malformed_temporal_values(state, content):
    value = example()
    value["cells"][0].update(state=state, value=content)
    with pytest.raises(ValueError):
        read(value)


@pytest.mark.parametrize("field,replacement", [
    ("version", "source-layout/unknown"), ("source_sha256", "b" * 64),
    ("producer_generation_ref", "other-generation"), ("sheet_ordinal", 3),
    ("object_ref", "object-b"), ("unreviewed", True),
])
def test_layout_rejects_scope_drift_and_unknown_fields(field, replacement):
    value = example()
    value[field] = replacement
    with pytest.raises(ValueError):
        read(value)


@pytest.mark.parametrize("mutation", ["duplicate", "reversed", "outside", "partial_overlap", "false_blank", "nan", "unknown_state", "bad_coordinate"])
def test_layout_rejects_invalid_cells(mutation):
    value = example()
    cells = value["cells"]
    if mutation == "duplicate":
        cells.append(copy.deepcopy(cells[0]))
    elif mutation == "reversed":
        cells.reverse()
    elif mutation == "outside":
        cells[0]["anchor"] = [3, 1]
    elif mutation == "partial_overlap":
        cells[1]["span"] = [1, 2, 2, 2]
        cells[1]["anchor"] = [2, 2]
    elif mutation == "false_blank":
        cells[0]["state"] = "blank"
    elif mutation == "nan":
        cells[0]["value"] = float("nan")
    elif mutation == "unknown_state":
        cells[0]["state"] = "approved"
    elif mutation == "bad_coordinate":
        cells[0]["anchor"] = [True, 1]
    with pytest.raises(ValueError):
        read(value)


def test_layout_accepts_distinct_physical_values_in_same_merge():
    value = example()
    value["cells"][1] = {"anchor": [1, 2], "span": [1, 1, 1, 2], "value": "Detail", "state": "literal"}
    assert read(value) == value


def test_layout_requires_exact_manifest_membership_even_with_valid_geometry():
    value = example()
    members = {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert contract().validate_layout_membership(value, members) is None
    for bad_members in (members - {(2, 2)}, members | {(3, 1)}):
        with pytest.raises(ValueError, match="membership"):
            contract().validate_layout_membership(value, bad_members)


def test_layout_membership_preserves_merge_detail_without_double_counting():
    value = example()
    value["cells"][1] = {"anchor": [1, 2], "span": [1, 1, 1, 2], "value": "Detail", "state": "literal"}
    assert contract().validate_layout_membership(value, {(1, 1), (1, 2)}) is None


def test_layout_membership_rejects_huge_span_before_expansion():
    value = example()
    value["cells"][0]["span"] = [1, 1, 1000000000, 1000000000]
    with pytest.raises(ValueError, match="membership"):
        contract().validate_layout_membership(value, {(1, 1)})
