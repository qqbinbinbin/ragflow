from pathlib import Path


def test_naive_pdf_page_images_are_enhanced_when_image_context_enabled():
    source = Path("rag/app/naive.py").read_text(encoding="utf-8")

    assert "_enhance_pdf_sections_with_page_vision" in source
    assert "PDF visual model detected" in source
    assert "page_images" in source
    assert "image_context_size > 0" in source
    assert "_enhance_pdf_sections_with_page_vision(" in source
