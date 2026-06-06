//! 非结构网格面拓扑与 IDWLS 几何预计算缓存。
//!
//! 面列表与 LSQ 正规方程矩阵 \(A\) 仅依赖网格几何，在求解器 work 区初始化一次；
//! 每步 RHS 只累加右端项 \(b\) 并求解梯度。

use crate::boundary::{BoundaryKind, BoundarySet, WallHeat};
use crate::core::{CellId, FaceId, Real, Vector3};
use crate::error::Result;
use crate::mesh::UnstructuredMesh3d;

/// 非结构内部面拓扑（粘性通量 + IDWLS 邻接样本）。
#[derive(Debug, Clone)]
pub struct UnstructuredInteriorFace {
    pub owner: usize,
    pub neighbor: usize,
    pub area: Real,
    pub normal: Vector3,
    pub owner_volume: Real,
    pub neighbor_volume: Real,
    pub inv_owner_volume: Real,
    pub inv_neighbor_volume: Real,
    /// \(-A_f / V_{\mathrm{owner}}\)，装配 owner 残差用。
    pub owner_rhs_scale: Real,
    /// \(A_f / V_{\mathrm{neighbor}}\)，装配 neighbor 残差用。
    pub neighbor_rhs_scale: Real,
    pub lsq_dr: Vector3,
    pub lsq_w: Real,
}

/// 非结构边界面拓扑。
#[derive(Debug, Clone, Copy)]
pub struct UnstructuredBoundaryFace {
    pub face: FaceId,
    pub owner: usize,
    pub area: Real,
    pub normal: Vector3,
    pub owner_volume: Real,
    pub spacing: Real,
    pub viscous: UnstructuredBoundaryViscousKind,
    pub lsq_dr: Vector3,
    pub lsq_w: Real,
}

/// 边界面粘性类别（与粘性装配语义一致）。
#[derive(Debug, Clone, Copy)]
pub struct UnstructuredBoundaryViscousKind {
    pub wall_heat: Option<WallHeat>,
    pub no_slip: bool,
    pub is_wall: bool,
}

/// 非结构面拓扑缓存。
#[derive(Debug, Clone)]
pub struct UnstructuredFaceTopology {
    pub interior: Vec<UnstructuredInteriorFace>,
    pub boundary: Vec<UnstructuredBoundaryFace>,
}

/// 单元 IDWLS 正规方程矩阵 \(A\)（几何固定部分）。
#[derive(Debug, Clone, Copy, Default)]
pub struct LsqPrecomputedCell {
    pub a_xx: Real,
    pub a_xy: Real,
    pub a_xz: Real,
    pub a_yy: Real,
    pub a_yz: Real,
    pub a_zz: Real,
}

impl LsqPrecomputedCell {
    fn add_geometry(&mut self, dr: Vector3, w: Real) {
        self.a_xx += w * dr.x * dr.x;
        self.a_xy += w * dr.x * dr.y;
        self.a_xz += w * dr.x * dr.z;
        self.a_yy += w * dr.y * dr.y;
        self.a_yz += w * dr.y * dr.z;
        self.a_zz += w * dr.z * dr.z;
    }
}

/// 非结构求解器网格缓存：面拓扑 + IDWLS 几何矩阵。
#[derive(Debug, Clone)]
pub struct UnstructuredSolverMeshCache {
    pub face_topology: UnstructuredFaceTopology,
    pub lsq_geometry: Vec<LsqPrecomputedCell>,
}

impl UnstructuredSolverMeshCache {
    /// 由网格与边界 patch 构建面拓扑，并预计算 IDWLS 矩阵 \(A\)。
    pub fn from_mesh(mesh: &UnstructuredMesh3d, boundaries: &BoundarySet) -> Result<Self> {
        let face_topology = build_face_topology(mesh, boundaries)?;
        let lsq_geometry = precompute_lsq_geometry(mesh.num_cells(), &face_topology);
        Ok(Self {
            face_topology,
            lsq_geometry,
        })
    }
}

fn build_face_topology(
    mesh: &UnstructuredMesh3d,
    boundaries: &BoundarySet,
) -> Result<UnstructuredFaceTopology> {
    let mut interior = Vec::new();
    for face in 0..mesh.num_faces() {
        let face_id = FaceId(face as u32);
        let Some(neighbor_id) = mesh.face_neighbor(face_id)? else {
            continue;
        };
        let owner_id = mesh.face_owner(face_id)?;
        let owner = owner_id.index() as usize;
        let neighbor = neighbor_id.index() as usize;
        let metric = mesh.face_metric(face_id);
        let owner_volume = mesh.cell_metric(owner_id).volume;
        let neighbor_volume = mesh.cell_metric(neighbor_id).volume;
        let (lsq_dr, lsq_w) = interior_lsq_weight(mesh, owner_id, neighbor_id);
        interior.push(UnstructuredInteriorFace {
            owner,
            neighbor,
            area: metric.area,
            normal: metric.normal,
            owner_volume,
            neighbor_volume,
            inv_owner_volume: inv_volume(owner_volume),
            inv_neighbor_volume: inv_volume(neighbor_volume),
            owner_rhs_scale: -metric.area * inv_volume(owner_volume),
            neighbor_rhs_scale: metric.area * inv_volume(neighbor_volume),
            lsq_dr,
            lsq_w,
        });
    }

    let mut boundary = Vec::new();
    for patch in boundaries.patches() {
        if matches!(patch.kind, BoundaryKind::Periodic { .. }) {
            continue;
        }
        let viscous = boundary_viscous_kind(&patch.kind);
        for &face in &patch.face_ids {
            let owner_id = mesh.face_owner(face)?;
            let owner = owner_id.index() as usize;
            let metric = mesh.face_metric(face);
            let owner_volume = mesh.cell_metric(owner_id).volume;
            let (lsq_dr, lsq_w) = boundary_lsq_weight(mesh, owner_id, face);
            boundary.push(UnstructuredBoundaryFace {
                face,
                owner,
                area: metric.area,
                normal: metric.normal,
                owner_volume,
                spacing: boundary_spacing(mesh, owner_id, face),
                viscous,
                lsq_dr,
                lsq_w,
            });
        }
    }

    Ok(UnstructuredFaceTopology { interior, boundary })
}

fn precompute_lsq_geometry(
    num_cells: usize,
    topology: &UnstructuredFaceTopology,
) -> Vec<LsqPrecomputedCell> {
    let mut geometry = vec![LsqPrecomputedCell::default(); num_cells];
    for face in &topology.interior {
        geometry[face.owner].add_geometry(face.lsq_dr, face.lsq_w);
        let dr_neighbor = neg_vector(face.lsq_dr);
        geometry[face.neighbor].add_geometry(dr_neighbor, face.lsq_w);
    }
    for face in &topology.boundary {
        geometry[face.owner].add_geometry(face.lsq_dr, face.lsq_w);
    }
    geometry
}

fn interior_lsq_weight(
    mesh: &UnstructuredMesh3d,
    owner_id: CellId,
    neighbor_id: CellId,
) -> (Vector3, Real) {
    let owner_center = mesh.cell_metric(owner_id).center;
    let neighbor_center = mesh.cell_metric(neighbor_id).center;
    lsq_dr_weight(owner_center, neighbor_center)
}

fn boundary_lsq_weight(
    mesh: &UnstructuredMesh3d,
    owner_id: CellId,
    face: FaceId,
) -> (Vector3, Real) {
    let owner_center = mesh.cell_metric(owner_id).center;
    let mirror = mirrored_face_sample_point(owner_center, mesh.face_metric(face).center);
    lsq_dr_weight(owner_center, mirror)
}

fn lsq_dr_weight(from: Vector3, to: Vector3) -> (Vector3, Real) {
    let dr = vec_sub(to, from);
    let dist = dr.magnitude();
    if dist <= Real::EPSILON {
        (dr, 0.0)
    } else {
        (dr, 1.0 / dist)
    }
}

const DEGENERATE_VOLUME: Real = 1.0e-30;

fn inv_volume(volume: Real) -> Real {
    if volume <= DEGENERATE_VOLUME {
        0.0
    } else {
        1.0 / volume
    }
}

fn boundary_spacing(mesh: &UnstructuredMesh3d, owner: CellId, face: FaceId) -> Real {
    let cell = mesh.cell_metric(owner).center;
    let face_center = mesh.face_metric(face).center;
    vec_sub(cell, face_center).magnitude()
}

fn boundary_viscous_kind(kind: &BoundaryKind) -> UnstructuredBoundaryViscousKind {
    match kind {
        BoundaryKind::Wall { heat, no_slip, .. } => UnstructuredBoundaryViscousKind {
            wall_heat: Some(*heat),
            no_slip: *no_slip,
            is_wall: true,
        },
        _ => UnstructuredBoundaryViscousKind {
            wall_heat: None,
            no_slip: false,
            is_wall: false,
        },
    }
}

pub(crate) fn mirrored_face_sample_point(owner_center: Vector3, face_center: Vector3) -> Vector3 {
    Vector3::new(
        2.0 * face_center.x - owner_center.x,
        2.0 * face_center.y - owner_center.y,
        2.0 * face_center.z - owner_center.z,
    )
}

pub(crate) fn accumulate_lsq_rhs_component(rhs: &mut Vector3, dr: Vector3, w: Real, delta: Real) {
    if w <= 0.0 {
        return;
    }
    *rhs = vec_add_scaled(*rhs, dr, w * delta);
}

pub(crate) fn solve_lsq_gradient(geometry: &LsqPrecomputedCell, rhs: Vector3) -> Option<Vector3> {
    solve_symmetric_3x3(geometry, rhs)
}

fn solve_symmetric_3x3(a: &LsqPrecomputedCell, rhs: Vector3) -> Option<Vector3> {
    let c_xx = a.a_yy * a.a_zz - a.a_yz * a.a_yz;
    let c_xy = a.a_xz * a.a_yz - a.a_xy * a.a_zz;
    let c_xz = a.a_xy * a.a_yz - a.a_xz * a.a_yy;
    let c_yy = a.a_xx * a.a_zz - a.a_xz * a.a_xz;
    let c_yz = a.a_xy * a.a_xz - a.a_xx * a.a_yz;
    let c_zz = a.a_xx * a.a_yy - a.a_xy * a.a_xy;
    let det = a.a_xx * c_xx + a.a_xy * c_xy + a.a_xz * c_xz;
    if det.abs() <= Real::EPSILON {
        return None;
    }
    let inv_det = 1.0 / det;
    Some(Vector3::new(
        (c_xx * rhs.x + c_xy * rhs.y + c_xz * rhs.z) * inv_det,
        (c_xy * rhs.x + c_yy * rhs.y + c_yz * rhs.z) * inv_det,
        (c_xz * rhs.x + c_yz * rhs.y + c_zz * rhs.z) * inv_det,
    ))
}

fn vec_sub(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn vec_add_scaled(a: Vector3, b: Vector3, scale: Real) -> Vector3 {
    Vector3::new(a.x + scale * b.x, a.y + scale * b.y, a.z + scale * b.z)
}

fn neg_vector(v: Vector3) -> Vector3 {
    Vector3::new(-v.x, -v.y, -v.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryPatch, BoundarySet};
    use crate::mesh::{CellKind, UnstructuredCell};

    fn unit_hex_mesh() -> UnstructuredMesh3d {
        UnstructuredMesh3d::new(
            "hex",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Hex, vec![0, 1, 2, 3, 4, 5, 6, 7]).expect("cell")],
        )
        .expect("mesh")
    }

    #[test]
    fn face_topology_counts_match_closed_hex() {
        let mesh = unit_hex_mesh();
        let faces = (0..mesh.num_faces())
            .map(|face| FaceId(face as u32))
            .collect::<Vec<_>>();
        let boundaries = BoundarySet::new(vec![BoundaryPatch::new(
            "all",
            faces,
            BoundaryKind::Farfield {
                mach: 0.0,
                pressure: 101_325.0,
                temperature: 300.0,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        let cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundaries).expect("cache");
        assert!(cache.face_topology.interior.is_empty());
        assert_eq!(cache.face_topology.boundary.len(), mesh.num_faces());
        assert_eq!(cache.lsq_geometry.len(), mesh.num_cells());
    }

    #[test]
    fn precomputed_lsq_geometry_is_positive_definite_on_hex_samples() {
        let mesh = unit_hex_mesh();
        let faces = (0..mesh.num_faces())
            .map(|face| FaceId(face as u32))
            .collect::<Vec<_>>();
        let boundaries = BoundarySet::new(vec![BoundaryPatch::new(
            "all",
            faces,
            BoundaryKind::Farfield {
                mach: 0.0,
                pressure: 101_325.0,
                temperature: 300.0,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        let cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundaries).expect("cache");
        let g = &cache.lsq_geometry[0];
        assert!(g.a_xx > 0.0);
        assert!(g.a_yy > 0.0);
        assert!(g.a_zz > 0.0);
    }
}
