from pathlib import Path


def test_table_vision_description_preserves_string_return():
    source = Path("rag/app/table.py").read_text(encoding="utf-8")

    assert "_coerce_vision_description_text" in source
    assert '".join(bf[0][1])' not in source
    assert "images[i][\"image_description\"] = _coerce_vision_description_text" in source
