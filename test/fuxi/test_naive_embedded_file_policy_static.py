from pathlib import Path


def test_naive_chunk_skips_unsupported_embedded_ole_bin_documents():
    source = Path("rag/app/naive.py").read_text(encoding="utf-8")

    assert "_is_supported_embedded_document" in source
    assert "unsupported embedded file skipped" in source
    assert ".bin" in source
    assert ".ole" in source
    assert "Failed to chunk embed" in source


def test_naive_chunk_keeps_supported_embedded_office_pdf_documents():
    source = Path("rag/app/naive.py").read_text(encoding="utf-8")

    for suffix in (
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".txt",
    ):
        assert suffix in source
