from pathlib import Path


def test_pptx_parser_extracts_picture_shapes_for_vision_enhancement():
    source = Path("deepdoc/parser/ppt_parser.py").read_text(encoding="utf-8")

    assert "_extract_picture" in source
    assert "shape.image.blob" in source
    assert "picture_describer" in source
    assert "PPTX visual model" in source


def test_presentation_chunk_passes_picture_describer_to_ppt_parser():
    source = Path("rag/app/presentation.py").read_text(encoding="utf-8")

    assert "_build_pptx_picture_describer" in source
    assert "picture_describer=picture_describer" in source
    assert "image_context_size" in source
