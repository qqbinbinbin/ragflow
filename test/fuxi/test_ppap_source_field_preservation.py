"""Uploaded-source regression: record counts cannot prove field preservation."""

import hashlib
import json
from pathlib import Path

import pytest

from rag.app.tabular_structure import build_tabular_structure_projection
from test.fuxi.test_table_semantic_rows import _load_table_module


BASELINE = Path("/opt/fuxi/evidence/ppap-v28-previous-field-baseline-20260916t082505058734z-a1c23faf/projections.json")
SOURCES = {
    "front": "/opt/fuxi/evidence/ppap-four-file-current-identity-20260914t051021853198z-6d7431ba/remote-originals/D91-前稳定杆接头总成PH01-江西荣成.xls",
    "tire": "/opt/fuxi/evidence/ppap-four-file-current-identity-20260914t051021853198z-6d7431ba/remote-originals/D91-轮胎总成PH01-中策橡胶.xls",
    "f515": "/opt/fuxi/evidence/f515-new-source-producer-replay-20260915-20260915t020918747155z-f573fe62/F515-new.xls",
    "g91": "/opt/fuxi/evidence/g91-source-download-20260916/G91-PN01.xls",
}


def _complete_fields(projection):
    tables = {table["table_ref"]: table for table in projection["tables"]
              if table["enumeration_status"] == "supported_complete"}
    return {
        (tables[row["table_ref_kwd"]]["sheet_ordinal"], row["row_ordinal_int"],
         field["column_id"], field["value"])
        for row in projection["rows"] if row["table_ref_kwd"] in tables
        for field in json.loads(row["ordered_fields_list"])
    }


@pytest.mark.parametrize("source_key", SOURCES)
def test_current_upload_preserves_every_previously_complete_source_field(monkeypatch, tmp_path, source_key):
    source = Path(SOURCES[source_key])
    if not source.is_file() or not BASELINE.is_file():
        pytest.skip("reviewed uploaded-source evidence is not mounted")
    previous = json.loads(BASELINE.read_text())[source_key]
    binary = source.read_bytes()
    assert hashlib.sha256(binary).hexdigest() == previous["source_sha256"]
    current = build_tabular_structure_projection(
        source.name, binary, parser=_load_table_module(monkeypatch).Excel(),
    )
    (tmp_path / f"{source_key}.projection.json").write_text(json.dumps(current, ensure_ascii=False))
    old_fields, new_fields = _complete_fields(previous), _complete_fields(current)
    missing = sorted(old_fields - new_fields)
    assert not missing, {"source": source_key, "missing_count": len(missing), "missing_coordinates": [item[:3] for item in missing]}
