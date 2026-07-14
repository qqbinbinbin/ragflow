import importlib.util
import sys
import types
from io import BytesIO
from pathlib import Path

import pandas as pd


def _load_excel_parser_class():
    rag_nlp = types.ModuleType("rag.nlp")
    rag_nlp.find_codec = lambda *_args, **_kwargs: "utf-8"
    lazy_image = types.ModuleType("rag.utils.lazy_image")
    lazy_image.LazyImage = object
    sys.modules.setdefault("rag.nlp", rag_nlp)
    sys.modules.setdefault("rag.utils.lazy_image", lazy_image)
    path = Path("deepdoc/parser/excel_parser.py")
    spec = importlib.util.spec_from_file_location("fuxi_excel_parser", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_calamine_fallback_reads_every_legacy_excel_sheet(monkeypatch):
    module = _load_excel_parser_class()
    RAGFlowExcelParser = module.RAGFlowExcelParser
    calls = []
    sheets = {
        "Cover": pd.DataFrame(),
        "Part approval": pd.DataFrame({"part": ["A-1"]}),
        "Inspection": pd.DataFrame({"result": ["pass"]}),
    }

    def fake_load_workbook(*_args, **_kwargs):
        raise ValueError("legacy OLE workbook")

    def fake_read_excel(_file, **kwargs):
        calls.append(kwargs)
        if kwargs.get("engine") != "calamine":
            raise AssertionError("xlrd cannot read workbook formatting")
        return sheets

    monkeypatch.setattr(module, "load_workbook", fake_load_workbook)
    monkeypatch.setattr(module.pd, "read_excel", fake_read_excel)

    workbook = RAGFlowExcelParser._load_excel_to_workbook(
        BytesIO(b"\xd0\xcf\x11\xe0fixture")
    )

    assert calls[-1] == {"sheet_name": None, "engine": "calamine"}
    assert workbook.sheetnames == ["Cover", "Part approval", "Inspection"]
    assert workbook["Part approval"]["A2"].value == "A-1"
    assert workbook["Inspection"]["A2"].value == "pass"
