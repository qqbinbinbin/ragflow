"""Strict source-layout envelope; carries geometry, not inferred form semantics.

The enclosing immutable projection must authenticate bytes and ownership.
Valid geometry alone never proves an exhaustive form or label/value relation.
"""
from __future__ import annotations

import math
import re
import unicodedata
from copy import deepcopy
from datetime import date, datetime, time


# Exact released tuple for source-layout persistence and readback.
SOURCE_LAYOUT_PROJECTION_CONTRACT = (
    "table-producer/v7", "tabular-structure-projection/v7",
    "region-producer/v29", "enumeration-rules/v9",
)

SOURCE_LAYOUT_TITLE_PROJECTION_CONTRACT = (
    "table-producer/v8", "tabular-structure-projection/v8",
    "region-producer/v30", "enumeration-rules/v9",
)


def source_layout_version_for_contract(contract):
    """Resolve only released exact tuples, never a payload's claimed version."""
    return {
        SOURCE_LAYOUT_PROJECTION_CONTRACT: "source-layout/v1",
        SOURCE_LAYOUT_TITLE_PROJECTION_CONTRACT: "source-layout/v2",
    }.get(contract)


def _coordinates(value, length):
    if not isinstance(value, list) or len(value) != length or any(
        type(number) is not int or number < 1 or number > 2**53 - 1
        for number in value
    ):
        raise ValueError("invalid layout coordinates")
    return tuple(value)


def validate_source_layout(
    value, *, source_sha256, producer_generation_ref, sheet_ordinal, object_ref,
    layout_version="source-layout/v1",
):
    """Return an owned copy after exact schema, scope and geometry validation."""
    if layout_version not in ("source-layout/v1", "source-layout/v2"):
        raise ValueError("unsupported layout version")
    expected = {
        "version": layout_version, "source_sha256": source_sha256,
        "producer_generation_ref": producer_generation_ref,
        "sheet_ordinal": sheet_ordinal, "object_ref": object_ref,
    }
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("invalid layout source digest")
    _coordinates([sheet_ordinal], 1)
    for identity in (producer_generation_ref, object_ref):
        if not isinstance(identity, str) or not identity or any(
            character.isspace() or ord(character) < 32 or 127 <= ord(character) <= 159
            for character in identity
        ):
            raise ValueError("invalid layout identity")
    fields = {*expected, "cells"}
    if layout_version == "source-layout/v2":
        fields.add("sheet_name")
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid layout envelope")
    if layout_version == "source-layout/v2":
        title = value["sheet_name"]
        if not isinstance(title, str) or not title.strip() or any(
            unicodedata.category(character) in ("Cc", "Cf", "Cs")
            for character in title
        ) or len(title.encode("utf-8")) > 1024:
            raise ValueError("invalid layout sheet title")
    if any(type(value[key]) is not type(wanted) or value[key] != wanted for key, wanted in expected.items()):
        raise ValueError("layout scope or version drift")
    if not isinstance(value["cells"], list):
        raise ValueError("invalid layout cells")
    previous_anchor = None
    spans = set()
    for cell in value["cells"]:
        if not isinstance(cell, dict) or set(cell) != {"anchor", "span", "value", "state"}:
            raise ValueError("invalid layout cell")
        anchor = _coordinates(cell["anchor"], 2)
        span = _coordinates(cell["span"], 4)
        top, left, bottom, right = span
        if not (top <= anchor[0] <= bottom and left <= anchor[1] <= right):
            raise ValueError("layout anchor outside span")
        if previous_anchor is not None and anchor <= previous_anchor:
            raise ValueError("duplicate or unordered layout anchor")
        previous_anchor = anchor
        for other in spans:
            if span != other and top <= other[2] and other[0] <= bottom and left <= other[3] and other[1] <= right:
                raise ValueError("conflicting layout spans")
        spans.add(span)
        content = cell["value"]
        if content is not None and type(content) not in (str, bool, int, float):
            raise ValueError("non-JSON layout value")
        if type(content) is float and not math.isfinite(content):
            raise ValueError("non-finite layout value")
        if type(content) is int and abs(content) > 2**53 - 1:
            raise ValueError("unsafe integer layout value")
        state = cell["state"]
        if state not in ("blank", "literal", "formula_unresolved", "date", "datetime", "time"):
            raise ValueError("invalid layout state")
        if state in ("date", "datetime", "time"):
            temporal_type = {"date": date, "datetime": datetime, "time": time}[state]
            if not isinstance(content, str):
                raise ValueError("invalid temporal layout value")
            try:
                temporal = temporal_type.fromisoformat(content)
            except ValueError as exc:
                raise ValueError("invalid temporal layout value") from exc
            if temporal.isoformat() != content:
                raise ValueError("noncanonical temporal layout value")
        if (state == "blank" and content is not None) or (state == "literal" and content is None):
            raise ValueError("layout blank state conflicts with value")
        if state == "formula_unresolved" and content is not None and (not isinstance(content, str) or not content.startswith("=")):
            raise ValueError("invalid unresolved formula")
    return deepcopy(value)


def validate_layout_membership(layout, members: set[tuple[int, int]]) -> None:
    """Bind validated layout spans to the Producer's exact owned source cells.

    Do not allocate a rectangle until its area fits the trusted membership.
    Distinct BIFF values sharing one merge consume that geometry only once.
    """
    covered = set()
    spans = {tuple(cell["span"]) for cell in layout["cells"]}
    for top, left, bottom, right in spans:
        area = (bottom - top + 1) * (right - left + 1)
        if bottom < top or right < left or area > len(members):
            raise ValueError("layout membership span exceeds source")
        for row in range(top, bottom + 1):
            for column in range(left, right + 1):
                coordinate = (row, column)
                if coordinate not in members or coordinate in covered:
                    raise ValueError("layout membership overlaps or escapes source")
                covered.add(coordinate)
    if covered != members:
        raise ValueError("layout membership does not cover source")
