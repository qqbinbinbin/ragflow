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
#  See the License for the specific language governing permissions and limitations under
#  the License.
#

"""Focused tests for rag.app.table.chunk() column roles."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# chunk() removes columns named id, _id, index, idx, so use row_id instead.
TEST_CSV = b"""row_id,title,content,country,category
1,Earthquake hits Turkey,A 5.8 magnitude earthquake struck Konya,Turkey,Disaster
2,Oil prices surge,Brent crude jumped 4.2 percent,Global,Economy
3,AI regulation proposed,EU unveiled a draft regulation,EU,Technology
"""
TEST_DUPLICATE_COLUMNS_CSV = b"""name,name,name_2
Alice,Engineer,Team A
"""

FILENAME = "test.csv"
KB_ID = "test_kb_id"
FIXED_ES_FIELDS = {"docnm_kwd", "title_tks", "content_with_weight", "content_ltks"}


def _noop_callback(*_args, **_kwargs):
    pass


@pytest.fixture(scope="module")
def table_module():
    """Load table parsing without importing unrelated service and model runtimes."""

    class KnowledgebaseService:
        @classmethod
        def delete_field_map(cls, _kb_id):
            pass

        @classmethod
        def update_parser_config(cls, _kb_id, _config):
            pass

    class Pinyin:
        def get_pinyins(self, value, _separator):
            return ["header_%s" % sum(ord(char) for char in value)]

    def tokenize(document, text, _english, **_kwargs):
        document["content_with_weight"] = text
        document["content_ltks"] = text.split()
        return document

    knowledgebase_service = types.ModuleType("api.db.services.knowledgebase_service")
    knowledgebase_service.KnowledgebaseService = KnowledgebaseService
    figure_parser = types.ModuleType("deepdoc.parser.figure_parser")
    figure_parser.vision_figure_parser_figure_xlsx_wrapper = lambda **_kwargs: []
    parser_utils = types.ModuleType("deepdoc.parser.utils")
    parser_utils.get_text = lambda _filename, binary: binary.decode()
    deepdoc_parser = types.ModuleType("deepdoc.parser")
    deepdoc_parser.ExcelParser = type("ExcelParser", (), {})
    rag_nlp = types.ModuleType("rag.nlp")
    rag_nlp.rag_tokenizer = types.SimpleNamespace(tokenize=lambda text: str(text))
    rag_nlp.tokenize = tokenize
    rag_nlp.tokenize_table = lambda *_args, **_kwargs: []
    common = types.ModuleType("common")
    common.settings = types.SimpleNamespace(DOC_ENGINE_INFINITY=False, DOC_ENGINE_OCEANBASE=False)
    common_constants = types.ModuleType("common.constants")
    common_constants.MAXIMUM_TASK_PAGE_NUMBER = 100000
    xpinyin = types.ModuleType("xpinyin")
    xpinyin.Pinyin = Pinyin
    stubs = {
        "api.db.services.knowledgebase_service": knowledgebase_service,
        "deepdoc.parser.figure_parser": figure_parser,
        "deepdoc.parser.utils": parser_utils,
        "deepdoc.parser": deepdoc_parser,
        "rag.nlp": rag_nlp,
        "common": common,
        "common.constants": common_constants,
        "xpinyin": xpinyin,
    }
    originals = {name: sys.modules.get(name) for name in stubs}

    try:
        sys.modules.update(stubs)
        source_path = Path(os.environ.get("FUXI_TABLE_SOURCE", "rag/app/table.py"))
        loader = SourceFileLoader("upstream_table_column_roles", str(source_path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.fixture(autouse=True)
def _es_doc_engine(monkeypatch, table_module):
    monkeypatch.setattr(table_module.settings, "DOC_ENGINE_INFINITY", False)
    monkeypatch.setattr(table_module.settings, "DOC_ENGINE_OCEANBASE", False)


@pytest.fixture
def mock_update_kb(table_module):
    with patch.object(table_module.KnowledgebaseService, "update_parser_config") as mock:
        yield mock


def _run_chunk(table_module, parser_config: dict):
    return table_module.chunk(
        FILENAME,
        binary=TEST_CSV,
        callback=_noop_callback,
        kb_id=KB_ID,
        parser_config=parser_config,
        lang="Chinese",
    )


def test_chunk_deduplicates_repeated_column_names(table_module):
    chunks = table_module.chunk(
        FILENAME,
        binary=TEST_DUPLICATE_COLUMNS_CSV,
        callback=_noop_callback,
        kb_id=KB_ID,
        parser_config={},
        lang="Chinese",
    )
    assert len(chunks) == 1
    content = chunks[0]["content_with_weight"]
    assert "- name: Alice" in content
    assert "- name_3: Engineer" in content
    assert "- name_2: Team A" in content


def test_chunk_auto_mode_all_columns_in_text_and_fixed_es_fields(table_module):
    chunks = _run_chunk(table_module, {})
    assert len(chunks) == 3
    first = chunks[0]
    content = first["content_with_weight"]
    assert "Earthquake hits Turkey" in content
    assert "Konya" in content
    assert "Turkey" in content
    assert "Disaster" in content
    assert "1" in content or "row_id" in content
    assert set(first) == FIXED_ES_FIELDS


def test_chunk_manual_mode_indexing_only(table_module):
    chunks = _run_chunk(
        table_module,
        {
            "table_column_mode": "manual",
            "table_column_roles": {
                "title": "indexing",
                "content": "indexing",
                "row_id": "metadata",
                "country": "metadata",
                "category": "metadata",
            },
        },
    )
    first = chunks[0]
    content = first["content_with_weight"]
    assert "- title:" in content and "Earthquake" in content
    assert "- content:" in content and "Konya" in content
    assert "- country:" not in content
    assert "- category:" not in content
    assert "- row_id:" not in content
    assert set(first) == FIXED_ES_FIELDS


def test_chunk_manual_mode_legacy_vectorize_role(table_module):
    chunks = _run_chunk(
        table_module,
        {
            "table_column_mode": "manual",
            "table_column_roles": {
                "title": "vectorize",
                "content": "indexing",
                "row_id": "metadata",
                "country": "metadata",
                "category": "metadata",
            },
        },
    )
    content = chunks[0]["content_with_weight"]
    assert "- title:" in content and "Earthquake" in content
    assert "- content:" in content and "Konya" in content
    assert "- country:" not in content


def test_chunk_manual_mode_metadata_only_is_not_indexed_on_es(table_module):
    chunks = _run_chunk(
        table_module,
        {
            "table_column_mode": "manual",
            "table_column_roles": {
                column: "metadata"
                for column in ["title", "content", "row_id", "country", "category"]
            },
        },
    )
    assert chunks == []


def test_chunk_manual_mode_both(table_module):
    chunks = _run_chunk(
        table_module,
        {
            "table_column_mode": "manual",
            "table_column_roles": {
                column: "both"
                for column in ["title", "content", "country", "category", "row_id"]
            },
        },
    )
    first = chunks[0]
    content = first["content_with_weight"]
    assert "Earthquake hits Turkey" in content
    assert "Turkey" in content
    assert "Disaster" in content
    assert set(first) == FIXED_ES_FIELDS


def test_chunk_manual_mode_partial_roles_default_to_both(table_module):
    chunks = _run_chunk(
        table_module,
        {
            "table_column_mode": "manual",
            "table_column_roles": {"title": "indexing", "country": "metadata"},
        },
    )
    first = chunks[0]
    content = first["content_with_weight"]
    assert "- title:" in content and "Earthquake" in content
    assert "- country:" not in content
    assert "- row_id:" in content
    assert "- content:" in content
    assert "- category:" in content
    assert set(first) == FIXED_ES_FIELDS


def test_chunk_updates_table_column_names_for_structured_engine(
    table_module, monkeypatch, mock_update_kb: MagicMock
):
    monkeypatch.setattr(table_module.settings, "DOC_ENGINE_INFINITY", True)
    _run_chunk(table_module, {})
    mock_update_kb.assert_called_once()
    args, _kwargs = mock_update_kb.call_args
    assert args[0] == KB_ID
    assert args[1]["table_column_names"] == [
        "row_id",
        "title",
        "content",
        "country",
        "category",
    ]


def test_chunk_count_matches_row_count(table_module):
    assert len(_run_chunk(table_module, {})) == 3
