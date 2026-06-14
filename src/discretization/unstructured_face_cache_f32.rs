//! 非结构面几何 f32 预打包缓存（节点坐标仍 f64；ADR 0016 §4 进入 kernel 前转换）。

use crate::core::FaceId;
use crate::core::{Real, Vector3};
use crate::discretization::unstructured_face_cache::{
    GradientLimiterSampleKind, LsqPrecomputedCell, UnstructuredBoundaryFace,
    UnstructuredBoundaryViscousKind, UnstructuredFaceTopology, UnstructuredInteriorFace,
};

/// f32 IDWLS 正规方程矩阵 \(A\)（与 [`LsqPrecomputedCell`] 单元索引对齐）。
#[derive(Debug, Clone, Copy)]
pub struct LsqPrecomputedCellF32 {
    pub a_xx: f32,
    pub a_xy: f32,
    pub a_xz: f32,
    pub a_yy: f32,
    pub a_yz: f32,
    pub a_zz: f32,
}

impl LsqPrecomputedCellF32 {
    #[must_use]
    pub fn from_f64(cell: &LsqPrecomputedCell) -> Self {
        Self {
            a_xx: cell.a_xx as f32,
            a_xy: cell.a_xy as f32,
            a_xz: cell.a_xz as f32,
            a_yy: cell.a_yy as f32,
            a_yz: cell.a_yz as f32,
            a_zz: cell.a_zz as f32,
        }
    }
}

/// f32 梯度限制器样本（与 f64 `GradientLimiterSample` 索引对齐）。
#[derive(Debug, Clone, Copy)]
pub struct GradientLimiterSampleF32 {
    pub dr: [f32; 3],
    pub kind: GradientLimiterSampleKind,
}

/// LU-SGS 面耦合几何（f32 预打包）。
#[derive(Debug, Clone, Copy)]
pub struct LuSgsCellCouplingF32 {
    pub neighbor: usize,
    pub area: f32,
    pub normal: [f32; 3],
}

/// 非结构 LU-SGS 拓扑邻接 f32 缓存。
#[derive(Debug, Clone)]
pub struct LuSgsUnstructuredCouplingsF32 {
    cells: Vec<Vec<LuSgsCellCouplingF32>>,
}

impl LuSgsUnstructuredCouplingsF32 {
    #[must_use]
    pub fn from_topology_f32(num_cells: usize, topology: &UnstructuredFaceTopologyF32) -> Self {
        let mut cells = vec![Vec::new(); num_cells];
        for face in &topology.interior {
            cells[face.owner].push(LuSgsCellCouplingF32 {
                neighbor: face.neighbor,
                area: face.area,
                normal: face.normal,
            });
            cells[face.neighbor].push(LuSgsCellCouplingF32 {
                neighbor: face.owner,
                area: face.area,
                normal: face.normal,
            });
        }
        Self { cells }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub(crate) fn cells(&self) -> &[Vec<LuSgsCellCouplingF32>] {
        &self.cells
    }
}

/// f32 内面几何（与 [`UnstructuredInteriorFace`] 索引对齐）。
#[derive(Debug, Clone)]
pub struct UnstructuredInteriorFaceF32 {
    pub owner: usize,
    pub neighbor: usize,
    pub area: f32,
    pub normal: [f32; 3],
    pub owner_volume: f32,
    pub neighbor_volume: f32,
    pub inv_owner_volume: f32,
    pub inv_neighbor_volume: f32,
    pub owner_rhs_scale: f32,
    pub neighbor_rhs_scale: f32,
    pub lsq_dr: [f32; 3],
    pub lsq_w: f32,
    pub dr_owner_to_face: [f32; 3],
    pub dr_neighbor_to_face: [f32; 3],
}

/// f32 边界面几何（与 [`UnstructuredBoundaryFace`] 索引对齐）。
#[derive(Debug, Clone, Copy)]
pub struct UnstructuredBoundaryFaceF32 {
    pub face: FaceId,
    pub owner: usize,
    pub area: f32,
    pub normal: [f32; 3],
    pub owner_volume: f32,
    pub owner_rhs_scale: f32,
    pub spacing: f32,
    pub viscous: UnstructuredBoundaryViscousKind,
    pub lsq_dr: [f32; 3],
    pub lsq_w: f32,
    pub dr_owner_to_face: [f32; 3],
}

/// f32 面拓扑视图（着色桶仍用 f64 `InteriorFaceColoring` 的面索引）。
#[derive(Debug, Clone)]
pub struct UnstructuredFaceTopologyF32 {
    pub interior: Vec<UnstructuredInteriorFaceF32>,
    pub boundary: Vec<UnstructuredBoundaryFaceF32>,
}

impl UnstructuredFaceTopologyF32 {
    /// 由 f64 面拓扑一次性预打包（`mesh_cache` 初始化时调用）。
    #[must_use]
    pub fn from_face_topology(topology: &UnstructuredFaceTopology) -> Self {
        Self {
            interior: topology
                .interior
                .iter()
                .map(interior_face_f32_from_f64)
                .collect(),
            boundary: topology
                .boundary
                .iter()
                .map(boundary_face_f32_from_f64)
                .collect(),
        }
    }
}

/// 由 f64 IDWLS 几何一次性预打包。
#[must_use]
pub fn lsq_geometry_f32_from_f64(geometry: &[LsqPrecomputedCell]) -> Vec<LsqPrecomputedCellF32> {
    geometry
        .iter()
        .map(LsqPrecomputedCellF32::from_f64)
        .collect()
}

/// 由 f32 面拓扑构建限制器样本列表。
#[must_use]
pub fn build_cell_gradient_samples_f32(
    num_cells: usize,
    topology: &UnstructuredFaceTopologyF32,
) -> Vec<Vec<GradientLimiterSampleF32>> {
    let mut samples = vec![Vec::new(); num_cells];
    for face in &topology.interior {
        samples[face.owner].push(GradientLimiterSampleF32 {
            dr: face.lsq_dr,
            kind: GradientLimiterSampleKind::NeighborCell(face.neighbor),
        });
        samples[face.neighbor].push(GradientLimiterSampleF32 {
            dr: neg_dr(face.lsq_dr),
            kind: GradientLimiterSampleKind::NeighborCell(face.owner),
        });
    }
    for (idx, face) in topology.boundary.iter().enumerate() {
        samples[face.owner].push(GradientLimiterSampleF32 {
            dr: face.lsq_dr,
            kind: GradientLimiterSampleKind::Boundary(idx),
        });
    }
    samples
}

#[inline]
#[must_use]
pub fn neg_dr(dr: [f32; 3]) -> [f32; 3] {
    [-dr[0], -dr[1], -dr[2]]
}

#[inline]
#[must_use]
pub fn vec3_to_f32(v: Vector3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

#[inline]
#[must_use]
pub fn vec3_from_f32(n: [f32; 3]) -> Vector3 {
    Vector3::new(n[0] as Real, n[1] as Real, n[2] as Real)
}

fn interior_face_f32_from_f64(face: &UnstructuredInteriorFace) -> UnstructuredInteriorFaceF32 {
    UnstructuredInteriorFaceF32 {
        owner: face.owner,
        neighbor: face.neighbor,
        area: face.area as f32,
        normal: vec3_to_f32(face.normal),
        owner_volume: face.owner_volume as f32,
        neighbor_volume: face.neighbor_volume as f32,
        inv_owner_volume: face.inv_owner_volume as f32,
        inv_neighbor_volume: face.inv_neighbor_volume as f32,
        owner_rhs_scale: face.owner_rhs_scale as f32,
        neighbor_rhs_scale: face.neighbor_rhs_scale as f32,
        lsq_dr: vec3_to_f32(face.lsq_dr),
        lsq_w: face.lsq_w as f32,
        dr_owner_to_face: vec3_to_f32(face.dr_owner_to_face),
        dr_neighbor_to_face: vec3_to_f32(face.dr_neighbor_to_face),
    }
}

fn boundary_face_f32_from_f64(face: &UnstructuredBoundaryFace) -> UnstructuredBoundaryFaceF32 {
    UnstructuredBoundaryFaceF32 {
        face: face.face,
        owner: face.owner,
        area: face.area as f32,
        normal: vec3_to_f32(face.normal),
        owner_volume: face.owner_volume as f32,
        owner_rhs_scale: face.owner_rhs_scale as f32,
        spacing: face.spacing as f32,
        viscous: face.viscous,
        lsq_dr: vec3_to_f32(face.lsq_dr),
        lsq_w: face.lsq_w as f32,
        dr_owner_to_face: vec3_to_f32(face.dr_owner_to_face),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet, WallHeat};
    use crate::core::approx_eq;
    use crate::discretization::unstructured_face_cache::UnstructuredSolverMeshCache;
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

    fn single_tet_mesh_and_boundary() -> (UnstructuredMesh3d, BoundarySet) {
        let mesh = UnstructuredMesh3d::new(
            "tet",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell")],
        )
        .expect("mesh");
        let faces = (0..mesh.num_faces())
            .map(|face| crate::core::FaceId(face as u32))
            .collect();
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "wall",
            faces,
            BoundaryKind::Wall {
                no_slip: true,
                heat: WallHeat::Adiabatic,
            },
        )]);
        (mesh, boundary)
    }

    #[test]
    fn face_topology_f32_matches_f64_reference() {
        let (mesh, boundary) = single_tet_mesh_and_boundary();
        let cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let f64_topo = &cache.face_topology;
        let f32_topo = &cache.face_topology_f32;
        assert_eq!(f32_topo.interior.len(), f64_topo.interior.len());
        assert_eq!(f32_topo.boundary.len(), f64_topo.boundary.len());
        assert_eq!(cache.lsq_geometry_f32.len(), cache.lsq_geometry.len());
        for (f64_face, f32_face) in f64_topo.interior.iter().zip(&f32_topo.interior) {
            assert_eq!(f32_face.owner, f64_face.owner);
            assert_eq!(f32_face.neighbor, f64_face.neighbor);
            assert!(approx_eq(f32_face.area as Real, f64_face.area, 1.0e-5));
            assert!(approx_eq(
                f32_face.owner_rhs_scale as Real,
                f64_face.owner_rhs_scale,
                1.0e-5
            ));
            assert!(approx_eq(f32_face.lsq_w as Real, f64_face.lsq_w, 1.0e-5));
            for (i, component) in [f64_face.normal.x, f64_face.normal.y, f64_face.normal.z]
                .into_iter()
                .enumerate()
            {
                assert!(approx_eq(f32_face.normal[i] as Real, component, 1.0e-5));
            }
        }
        for (f64_cell, f32_cell) in cache.lsq_geometry.iter().zip(&cache.lsq_geometry_f32) {
            assert!(approx_eq(f32_cell.a_xx as Real, f64_cell.a_xx, 1.0e-5));
            assert!(approx_eq(f32_cell.a_yy as Real, f64_cell.a_yy, 1.0e-5));
            assert!(approx_eq(f32_cell.a_zz as Real, f64_cell.a_zz, 1.0e-5));
        }
    }
}
