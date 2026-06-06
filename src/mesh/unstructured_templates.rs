//! VTK 线性单元局部面模板（节点顺序与 ParaView/VTU 一致）。
//!
//! VTK 部分面 winding 的法向指向单元内侧（如四面体底面指向顶点）；构造拓扑时
//! 会按单元中心翻转为外向（见 `unstructured.rs`）。

use super::CellKind;

/// 单元局部面：三角或四边形，节点为单元局部索引。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LocalFaceSpec {
    Tri([usize; 3]),
    Quad([usize; 4]),
}

/// 返回单元所有面（局部节点索引，遵循 VTK 原始 winding）。
pub(super) fn local_faces(kind: CellKind) -> &'static [LocalFaceSpec] {
    match kind {
        CellKind::Tet => &TET_FACES,
        CellKind::Hex => &HEX_FACES,
        CellKind::Pyramid => &PYRAMID_FACES,
        CellKind::Prism => &PRISM_FACES,
    }
}

// VTK_LINEAR_TET (10)
const TET_FACES: [LocalFaceSpec; 4] = [
    LocalFaceSpec::Tri([0, 1, 2]),
    LocalFaceSpec::Tri([0, 3, 1]),
    LocalFaceSpec::Tri([1, 3, 2]),
    LocalFaceSpec::Tri([2, 3, 0]),
];

// VTK_HEXAHEDRON (12): 0-1-2-3 底面 z-，4-5-6-7 顶面 z+
const HEX_FACES: [LocalFaceSpec; 6] = [
    LocalFaceSpec::Quad([0, 1, 2, 3]),
    LocalFaceSpec::Quad([4, 5, 6, 7]),
    LocalFaceSpec::Quad([0, 4, 5, 1]),
    LocalFaceSpec::Quad([3, 2, 6, 7]),
    LocalFaceSpec::Quad([0, 3, 7, 4]),
    LocalFaceSpec::Quad([1, 5, 6, 2]),
];

// VTK_PYRAMID (14): 0-1-2-3 底面四边形，4 为顶点
const PYRAMID_FACES: [LocalFaceSpec; 5] = [
    LocalFaceSpec::Quad([0, 1, 2, 3]),
    LocalFaceSpec::Tri([0, 4, 1]),
    LocalFaceSpec::Tri([1, 4, 2]),
    LocalFaceSpec::Tri([2, 4, 3]),
    LocalFaceSpec::Tri([3, 4, 0]),
];

// VTK_WEDGE (13): 0-1-2 / 3-4-5 为两端三角形
const PRISM_FACES: [LocalFaceSpec; 5] = [
    LocalFaceSpec::Tri([0, 1, 2]),
    LocalFaceSpec::Tri([3, 4, 5]),
    LocalFaceSpec::Quad([0, 3, 4, 1]),
    LocalFaceSpec::Quad([1, 4, 5, 2]),
    LocalFaceSpec::Quad([2, 5, 3, 0]),
];
