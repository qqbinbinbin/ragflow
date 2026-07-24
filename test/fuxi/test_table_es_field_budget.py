import importlib.util
from importlib.machinery import SourceFileLoader
import os
import re
import sys
import types
from pathlib import Path

import pandas as pd


def _load_table_module(monkeypatch, dataframe, infinity=False, oceanbase=False):
    calls = {"deleted": [], "updated": []}

    class KnowledgebaseService:
        @classmethod
        def delete_field_map(cls, kb_id):
            calls["deleted"].append(kb_id)

        @classmethod
        def update_parser_config(cls, kb_id, config):
            calls["updated"].append((kb_id, config))

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
    parser_utils.get_text = lambda *_args, **_kwargs: ""
    deepdoc_parser = types.ModuleType("deepdoc.parser")
    deepdoc_parser.ExcelParser = type("ExcelParser", (), {})
    rag_nlp = types.ModuleType("rag.nlp")
    rag_nlp.rag_tokenizer = types.SimpleNamespace(tokenize=lambda text: text.split())
    rag_nlp.tokenize = tokenize
    rag_nlp.tokenize_table = lambda *_args, **_kwargs: []
    common = types.ModuleType("common")
    common.settings = types.SimpleNamespace(
        DOC_ENGINE_INFINITY=infinity,
        DOC_ENGINE_OCEANBASE=oceanbase,
    )
    common_constants = types.ModuleType("common.constants")
    common_constants.MAXIMUM_TASK_PAGE_NUMBER = 100000
    xpinyin = types.ModuleType("xpinyin")
    xpinyin.Pinyin = Pinyin

    for name, module in {
        "api.db.services.knowledgebase_service": knowledgebase_service,
        "deepdoc.parser.figure_parser": figure_parser,
        "deepdoc.parser.utils": parser_utils,
        "deepdoc.parser": deepdoc_parser,
        "rag.nlp": rag_nlp,
        "common": common,
        "common.constants": common_constants,
        "xpinyin": xpinyin,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    source_path = Path(os.environ.get("FUXI_TABLE_SOURCE", "rag/app/table.py"))
    loader = SourceFileLoader("fuxi_table_field_budget", str(source_path))
    spec = importlib.util.spec_from_loader("fuxi_table_field_budget", loader)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    dataframes = dataframe if isinstance(dataframe, list) else [dataframe]
    monkeypatch.setattr(module.Excel, "__call__", lambda *_args, **_kwargs: ([frame.copy() for frame in dataframes], []))
    return module, calls


def test_es_table_chunks_keep_headers_in_content_without_dynamic_mapping_fields(monkeypatch):
    table, calls = _load_table_module(
        monkeypatch,
        pd.DataFrame({"Supplier Name": ["North", "South"], "Quantity": [2, 3]}),
    )

    chunks = table.chunk(
        "supplier.xlsx",
        callback=lambda *_args: None,
        kb_id="kb-es",
    )

    fixed_fields = {"docnm_kwd", "title_tks", "content_with_weight", "content_ltks"}
    assert chunks
    assert all(set(chunk).issubset(fixed_fields) for chunk in chunks)
    assert all("Supplier Name" in chunk["content_with_weight"] for chunk in chunks)
    assert calls == {"deleted": ["kb-es"], "updated": []}


def test_es_workbook_clears_field_map_once_for_multiple_sheets(monkeypatch):
    table, calls = _load_table_module(
        monkeypatch,
        [
            pd.DataFrame({"Supplier Name": ["North"]}),
            pd.DataFrame({"Part Number": ["P-1"]}),
        ],
    )

    table.chunk(
        "supplier.xlsx",
        callback=lambda *_args: None,
        kb_id="kb-es-workbook",
    )

    assert calls == {
        "deleted": ["kb-es-workbook"],
        "updated": [],
    }


def test_table_chunk_allows_calls_without_kb_id(monkeypatch):
    table, calls = _load_table_module(
        monkeypatch,
        pd.DataFrame({"Supplier Name": ["North"]}),
    )

    chunks = table.chunk(
        "supplier.xlsx",
        callback=lambda *_args: None,
    )

    assert chunks
    assert calls == {"deleted": [], "updated": []}


def test_structured_table_engines_keep_chunk_data_and_field_map(monkeypatch):
    table, calls = _load_table_module(
        monkeypatch,
        pd.DataFrame({"Supplier Name": ["North"], "Quantity": [2]}),
        infinity=True,
    )

    chunks = table.chunk(
        "supplier.xlsx",
        callback=lambda *_args: None,
        kb_id="kb-infinity",
    )

    assert chunks[0]["chunk_data"] == {"Supplier Name": "North", "Quantity": 2}
    assert calls["deleted"] == []
    assert len(calls["updated"]) == 1
    assert calls["updated"][0][0] == "kb-infinity"
    assert calls["updated"][0][1]["field_map"]


def test_es_manual_roles_filter_semantic_text_without_dynamic_fields(monkeypatch):
    table, calls = _load_table_module(
        monkeypatch,
        pd.DataFrame({"Supplier Name": ["North"], "Internal Code": ["S-1"]}),
    )

    chunks = table.chunk(
        "supplier.xlsx",
        callback=lambda *_args: None,
        kb_id="kb-es-manual",
        parser_config={
            "table_column_mode": "manual",
            "table_column_roles": {
                "Supplier Name": "indexing",
                "Internal Code": "metadata",
            },
        },
    )

    assert len(chunks) == 1
    assert "Supplier Name" in chunks[0]["content_with_weight"]
    assert "Internal Code" not in chunks[0]["content_with_weight"]
    assert set(chunks[0]) <= {"docnm_kwd", "title_tks", "content_with_weight", "content_ltks"}
    assert calls == {"deleted": ["kb-es-manual"], "updated": []}


def test_structured_workbook_merges_field_map_once(monkeypatch):
    table, calls = _load_table_module(
        monkeypatch,
        [
            pd.DataFrame({"Supplier Name": ["North"]}),
            pd.DataFrame({"Part Number": ["P-1"]}),
        ],
        oceanbase=True,
    )

    table.chunk(
        "supplier.xlsx",
        callback=lambda *_args: None,
        kb_id="kb-oceanbase-workbook",
    )

    assert calls["deleted"] == []
    assert len(calls["updated"]) == 1
    assert calls["updated"][0][1]["table_column_names"] == ["Supplier Name", "Part Number"]
    assert len(calls["updated"][0][1]["field_map"]) == 2
