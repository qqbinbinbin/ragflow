"""Native XLS geometry must survive the value-only Calamine conversion."""

import struct

import pytest

from test.fuxi.test_table_semantic_rows import _load_excel_parser_module


def record(kind, payload=b""):
    return struct.pack("<HH", kind, len(payload)) + payload


def workbook_stream(cells, *, xf_index_border=0x4321, version=0x0600):
    bof = record(0x0809, struct.pack("<HH", version, 5) + bytes(12))
    xf = record(0x00E0, bytes(10) + struct.pack("<I", xf_index_border) + bytes(6))
    # BOUNDSHEET points to the worksheet BOF, independently of display order.
    offset = len(bof) + len(xf) + 14 + 4
    bound = record(0x0085, struct.pack("<IBBBB", offset, 0, 0, 2, 0) + b"S1")
    return bof + xf + bound + record(0x000A) + record(0x0809, struct.pack("<HH", version, 16) + bytes(12)) + cells + record(0x000A)


@pytest.mark.parametrize("kind,payload", [
    (0x0201, b""), (0x0203, bytes(8)), (0x0205, bytes(2)),
    (0x00FD, bytes(4)), (0x027E, bytes(4)), (0x0006, bytes(16)),
])
def test_biff_cell_border_styles_preserve_coordinates(monkeypatch, kind, payload):
    parser = _load_excel_parser_module(monkeypatch)
    stream = workbook_stream(record(kind, struct.pack("<HHH", 4, 7, 0) + payload))
    assert parser._biff8_cell_borders(stream) == [{(5, 8): (1, 2, 3, 4)}]


@pytest.mark.parametrize("kind,stride", [(0x00BE, 2), (0x00BD, 6)])
def test_biff_multi_cell_records_preserve_blank_grid_cells(monkeypatch, kind, stride):
    parser = _load_excel_parser_module(monkeypatch)
    payload = struct.pack("<HH", 8, 2) + (bytes(stride) * 3) + struct.pack("<H", 4)
    assert parser._biff8_cell_borders(workbook_stream(record(kind, payload))) == [
        {(9, col): (1, 2, 3, 4) for col in (3, 4, 5)}
    ]


@pytest.mark.parametrize("cells", [
    record(0x0201, struct.pack("<HHH", 0, 0, 99)),
    record(0x0201, b"\x00"),
    record(0x00BE, struct.pack("<HHHH", 0, 3, 0, 7)),
    record(0x0201, struct.pack("<HHH", 0, 256, 0)),
])
def test_biff_invalid_geometry_cannot_supply_partial_evidence(monkeypatch, cells):
    parser = _load_excel_parser_module(monkeypatch)
    with pytest.raises(ValueError):
        parser._biff8_cell_borders(workbook_stream(cells))


def test_biff_truncated_substream_is_rejected(monkeypatch):
    parser = _load_excel_parser_module(monkeypatch)
    with pytest.raises(ValueError):
        parser._biff8_cell_borders(workbook_stream(b"")[:-1])


def test_biff_older_version_is_not_interpreted_as_biff8(monkeypatch):
    parser = _load_excel_parser_module(monkeypatch)
    with pytest.raises(ValueError):
        parser._biff8_cell_borders(workbook_stream(b"", version=0x0500))
