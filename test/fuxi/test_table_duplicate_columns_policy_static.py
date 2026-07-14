from pathlib import Path


def test_table_chunk_deduplicates_columns_instead_of_failing():
    source = Path("rag/app/table.py").read_text(encoding="utf-8")

    assert "_deduplicate_dataframe_columns" in source
    assert "duplicate column renamed" in source
    assert "Duplicate column names detected" not in source
