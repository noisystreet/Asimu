#!/usr/bin/env python3
"""Generate binary appended VTS fixtures for asimu tests."""

from __future__ import annotations

import base64
import struct
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "mesh"
NX = 2
NY = 2


def build_points(double: bool) -> bytes:
    points: list[float] = []
    for j in range(NY + 1):
        for i in range(NX + 1):
            points.extend([float(i), float(j), 0.0])
    payload = bytearray()
    pack = struct.pack if double else lambda fmt, v: struct.pack(fmt, v)  # noqa: E731
    fmt = "<d" if double else "<f"
    for value in points:
        payload.extend(pack(fmt, value))
    block = struct.pack("<I", len(payload)) + payload
    return block


def write_vts(path: Path, *, double: bool) -> None:
    block_b64 = base64.standard_b64encode(build_points(double)).decode("ascii")
    scalar = "Float64" if double else "Float32"
    xml = f"""<?xml version="1.0"?>
<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">
  <StructuredGrid WholeExtent="0 {NX} 0 {NY} 0 0">
    <Piece Extent="0 {NX} 0 {NY} 0 0">
      <Points>
        <DataArray type="{scalar}" Name="Points" NumberOfComponents="3" format="appended" offset="0"/>
      </Points>
    </Piece>
  </StructuredGrid>
  <AppendedData encoding="base64">
_{block_b64}</AppendedData>
</VTKFile>
"""
    path.write_text(xml, encoding="utf-8")


def write_ascii_reject(path: Path) -> None:
    xml = """<?xml version="1.0"?>
<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">
  <StructuredGrid WholeExtent="0 2 0 2 0 0">
    <Piece Extent="0 2 0 2 0 0">
      <Points>
        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">
          0 0 0 1 0 0
        </DataArray>
      </Points>
    </Piece>
  </StructuredGrid>
</VTKFile>
"""
    path.write_text(xml, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_vts(OUT / "unit_square_2x2_binary.vts", double=True)
    write_vts(OUT / "unit_square_2x2_binary_f32.vts", double=False)
    write_ascii_reject(OUT / "ascii_reject.vts")
    print(f"Wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
