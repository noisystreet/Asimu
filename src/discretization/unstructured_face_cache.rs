//! 非结构网格面拓扑与 IDWLS 几何预计算缓存。
//!
//! 面列表与 LSQ 正规方程矩阵 \(A\) 仅依赖网格几何，在求解器 work 区初始化一次；
//! 每步 RHS 只累加右端项 \(b\) 并求解梯度。

#[path = "interior_face_batch_layout.rs"]
mod interior_face_batch_layout;

use interior_face_batch_layout::build_bucket_batch_layouts;
pub use interior_face_batch_layout::{InteriorFaceBatchStatic4, InteriorFaceBucketBatchLayout};

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
    /// 单元中心 → 面心（owner 侧外推用）。
    pub dr_owner_to_face: Vector3,
    /// 单元中心 → 面心（neighbor 侧外推用）。
    pub dr_neighbor_to_face: Vector3,
}

/// 非结构边界面拓扑。
#[derive(Debug, Clone, Copy)]
pub struct UnstructuredBoundaryFace {
    pub face: FaceId,
    pub owner: usize,
    pub area: Real,
    pub normal: Vector3,
    pub owner_volume: Real,
    /// \(-A_f / V_{\mathrm{owner}}\)，边界面 owner 残差装配用。
    pub owner_rhs_scale: Real,
    pub spacing: Real,
    pub viscous: UnstructuredBoundaryViscousKind,
    pub lsq_dr: Vector3,
    pub lsq_w: Real,
    /// 单元中心 → 面心（边界面 owner 外推用）。
    pub dr_owner_to_face: Vector3,
}

/// 边界面粘性类别（与粘性装配语义一致）。
#[derive(Debug, Clone, Copy)]
pub struct UnstructuredBoundaryViscousKind {
    pub wall_heat: Option<WallHeat>,
    pub no_slip: bool,
    pub is_wall: bool,
}

/// LSQ / 梯度限制器邻接样本。
#[derive(Debug, Clone, Copy)]
pub enum GradientLimiterSampleKind {
    NeighborCell(usize),
    /// `UnstructuredFaceTopology::boundary` 中的索引。
    Boundary(usize),
}

#[derive(Debug, Clone, Copy)]
pub struct GradientLimiterSample {
    pub dr: Vector3,
    pub kind: GradientLimiterSampleKind,
}

/// 非结构面拓扑缓存。
#[derive(Debug, Clone)]
pub struct UnstructuredFaceTopology {
    pub interior: Vec<UnstructuredInteriorFace>,
    pub boundary: Vec<UnstructuredBoundaryFace>,
    /// 内面并行 scatter 用着色桶：同色面不共享单元，可安全并行 `+=`。
    pub interior_coloring: InteriorFaceColoring,
}

/// 内面贪心着色结果：`buckets[c]` 为颜色 `c` 上的面索引列表。
#[derive(Debug, Clone)]
pub struct InteriorFaceColoring {
    pub buckets: Vec<Vec<usize>>,
    pub num_colors: usize,
    /// 每桶四路对齐的静态几何 SoA（init-time；μ/λ 与场变量每步 gather）。
    pub bucket_batch_layouts: Vec<InteriorFaceBucketBatchLayout>,
}

impl InteriorFaceColoring {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// 按颜色桶顺序串行遍历内面索引。
    pub fn for_each_face_index<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        for bucket in &self.buckets {
            for &face_idx in bucket {
                f(face_idx);
            }
        }
    }

    /// 按面索引升序串行遍历（golden / 对照用）。
    pub fn for_each_face_index_linear<F>(&self, num_faces: usize, mut f: F)
    where
        F: FnMut(usize),
    {
        for face_idx in 0..num_faces {
            f(face_idx);
        }
    }

    /// 桶内并行 map、桶间串行（`parallel-fvm`）：适用于 compute/scatter 分离路径。
    /// `simd-fvm` batch 路径同样遵循「各色 bucket 串行、桶内 `par_iter`」；勿改为 bucket 间 `par_iter`（dual_ellipsoid 实测约 26% 回归，见 CHANGELOG）。
    #[cfg(feature = "parallel-fvm")]
    pub fn par_map_buckets<T, F>(&self, f: F) -> Vec<Vec<T>>
    where
        T: Send,
        F: Fn(usize) -> T + Sync,
    {
        use rayon::prelude::*;
        self.buckets
            .iter()
            .map(|bucket| {
                bucket
                    .par_iter()
                    .with_min_len(1024)
                    .map(|&face_idx| f(face_idx))
                    .collect()
            })
            .collect()
    }
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

/// IDWLS RHS 累加用的单元–面关联（单元并行路径：每单元只写自身 \(b_i\)）。
#[derive(Debug, Clone)]
pub struct LsqRhsCellIncidence {
    /// 单元作为 owner 的内面索引。
    pub interior_as_owner: Vec<Vec<usize>>,
    /// 单元作为 neighbor 的内面索引。
    pub interior_as_neighbor: Vec<Vec<usize>>,
    /// 单元拥有的边界面在 `face_topology.boundary` 中的索引。
    pub boundary_faces: Vec<Vec<usize>>,
}

/// 非结构求解器网格缓存：面拓扑 + IDWLS 几何矩阵。
#[derive(Debug, Clone)]
pub struct UnstructuredSolverMeshCache {
    pub face_topology: UnstructuredFaceTopology,
    pub lsq_geometry: Vec<LsqPrecomputedCell>,
    /// 每单元 IDWLS / 限制器样本（内部邻单元 + 边界 ghost 镜像点）。
    pub cell_gradient_samples: Vec<Vec<GradientLimiterSample>>,
    /// IDWLS RHS 单元–面关联（`parallel-fvm` 单元并行累加用）。
    pub lsq_rhs_incidence: LsqRhsCellIncidence,
}

impl UnstructuredSolverMeshCache {
    /// 由网格与边界 patch 构建面拓扑，并预计算 IDWLS 矩阵 \(A\)。
    pub fn from_mesh(mesh: &UnstructuredMesh3d, boundaries: &BoundarySet) -> Result<Self> {
        let face_topology = build_face_topology(mesh, boundaries)?;
        let num_cells = mesh.num_cells();
        let lsq_geometry = precompute_lsq_geometry(num_cells, &face_topology);
        let cell_gradient_samples = build_cell_gradient_samples(num_cells, &face_topology);
        let lsq_rhs_incidence = build_lsq_rhs_cell_incidence(num_cells, &face_topology);
        Ok(Self {
            face_topology,
            lsq_geometry,
            cell_gradient_samples,
            lsq_rhs_incidence,
        })
    }
}

fn build_lsq_rhs_cell_incidence(
    num_cells: usize,
    topology: &UnstructuredFaceTopology,
) -> LsqRhsCellIncidence {
    let mut interior_as_owner = vec![Vec::new(); num_cells];
    let mut interior_as_neighbor = vec![Vec::new(); num_cells];
    let mut boundary_faces = vec![Vec::new(); num_cells];
    for (face_idx, face) in topology.interior.iter().enumerate() {
        interior_as_owner[face.owner].push(face_idx);
        interior_as_neighbor[face.neighbor].push(face_idx);
    }
    for (boundary_idx, face) in topology.boundary.iter().enumerate() {
        boundary_faces[face.owner].push(boundary_idx);
    }
    LsqRhsCellIncidence {
        interior_as_owner,
        interior_as_neighbor,
        boundary_faces,
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
        let owner_center = mesh.cell_metric(owner_id).center;
        let neighbor_center = mesh.cell_metric(neighbor_id).center;
        let face_center = metric.center;
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
            dr_owner_to_face: vec_sub(face_center, owner_center),
            dr_neighbor_to_face: vec_sub(face_center, neighbor_center),
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
            let owner_center = mesh.cell_metric(owner_id).center;
            boundary.push(UnstructuredBoundaryFace {
                face,
                owner,
                area: metric.area,
                normal: metric.normal,
                owner_volume,
                owner_rhs_scale: -metric.area * inv_volume(owner_volume),
                spacing: boundary_spacing(mesh, owner_id, face),
                viscous,
                lsq_dr,
                lsq_w,
                dr_owner_to_face: vec_sub(metric.center, owner_center),
            });
        }
    }

    let interior_coloring = color_interior_faces(&interior, mesh.num_cells());
    Ok(UnstructuredFaceTopology {
        interior,
        boundary,
        interior_coloring,
    })
}

/// 贪心面着色：同色内面不共享 owner/neighbor 单元（FVM scatter 并行前提）。
fn color_interior_faces(
    interior: &[UnstructuredInteriorFace],
    num_cells: usize,
) -> InteriorFaceColoring {
    if interior.is_empty() {
        return InteriorFaceColoring {
            buckets: Vec::new(),
            num_colors: 0,
            bucket_batch_layouts: Vec::new(),
        };
    }
    let mut cell_incident_colors = vec![Vec::<u8>::new(); num_cells];
    let mut face_colors = vec![0u8; interior.len()];

    for (face_idx, face) in interior.iter().enumerate() {
        let mut used = Vec::new();
        for &c in &cell_incident_colors[face.owner] {
            push_unique(&mut used, c);
        }
        for &c in &cell_incident_colors[face.neighbor] {
            push_unique(&mut used, c);
        }
        used.sort_unstable();
        let color = first_available_color(&used);
        face_colors[face_idx] = color;
        cell_incident_colors[face.owner].push(color);
        cell_incident_colors[face.neighbor].push(color);
    }

    let num_colors = face_colors
        .iter()
        .copied()
        .max()
        .map(|c| c as usize + 1)
        .unwrap_or(0);
    let mut buckets = vec![Vec::new(); num_colors];
    for (face_idx, &color) in face_colors.iter().enumerate() {
        buckets[color as usize].push(face_idx);
    }
    let bucket_batch_layouts = build_bucket_batch_layouts(&buckets, interior);
    InteriorFaceColoring {
        buckets,
        num_colors,
        bucket_batch_layouts,
    }
}

fn build_cell_gradient_samples(
    num_cells: usize,
    topology: &UnstructuredFaceTopology,
) -> Vec<Vec<GradientLimiterSample>> {
    let mut samples = vec![Vec::new(); num_cells];
    for face in &topology.interior {
        samples[face.owner].push(GradientLimiterSample {
            dr: face.lsq_dr,
            kind: GradientLimiterSampleKind::NeighborCell(face.neighbor),
        });
        samples[face.neighbor].push(GradientLimiterSample {
            dr: neg_vector(face.lsq_dr),
            kind: GradientLimiterSampleKind::NeighborCell(face.owner),
        });
    }
    for (idx, face) in topology.boundary.iter().enumerate() {
        samples[face.owner].push(GradientLimiterSample {
            dr: face.lsq_dr,
            kind: GradientLimiterSampleKind::Boundary(idx),
        });
    }
    samples
}

fn push_unique(values: &mut Vec<u8>, value: u8) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn first_available_color(used_sorted: &[u8]) -> u8 {
    let mut candidate = 0u8;
    for &used in used_sorted {
        if used > candidate {
            break;
        }
        if used == candidate {
            candidate = candidate.saturating_add(1);
        }
    }
    candidate
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
    crate::exec::cpu::accumulate_lsq_rhs_component(rhs, dr, w, delta);
}

fn sym3_from_lsq(a: &LsqPrecomputedCell) -> crate::exec::cpu::Symmetric3x3 {
    crate::exec::cpu::Symmetric3x3 {
        a_xx: a.a_xx,
        a_xy: a.a_xy,
        a_xz: a.a_xz,
        a_yy: a.a_yy,
        a_yz: a.a_yz,
        a_zz: a.a_zz,
    }
}

#[cfg(feature = "simd-fvm")]
pub(crate) fn sym3_from_lsq_for_exec(a: &LsqPrecomputedCell) -> crate::exec::cpu::Symmetric3x3 {
    sym3_from_lsq(a)
}

pub(crate) fn solve_lsq_gradient(geometry: &LsqPrecomputedCell, rhs: Vector3) -> Option<Vector3> {
    crate::exec::cpu::solve_symmetric_3x3(&sym3_from_lsq(geometry), rhs)
}

fn vec_sub(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
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
        assert!(cache.face_topology.interior_coloring.is_empty());
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

    fn two_tet_mesh() -> UnstructuredMesh3d {
        UnstructuredMesh3d::new(
            "two_tets",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            vec![
                UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell"),
                UnstructuredCell::new(CellKind::Tet, vec![1, 2, 3, 4]).expect("cell"),
            ],
        )
        .expect("mesh")
    }

    #[test]
    fn lsq_rhs_incidence_covers_all_interior_faces() {
        let mesh = two_tet_mesh();
        let faces = (0..mesh.num_faces())
            .map(|face| FaceId(face as u32))
            .collect::<Vec<_>>();
        let boundaries = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
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
        let topology = &cache.face_topology;
        let inc = &cache.lsq_rhs_incidence;
        assert_eq!(inc.interior_as_owner.len(), mesh.num_cells());
        let owner_count: usize = inc.interior_as_owner.iter().map(Vec::len).sum();
        let neighbor_count: usize = inc.interior_as_neighbor.iter().map(Vec::len).sum();
        assert_eq!(owner_count, topology.interior.len());
        assert_eq!(neighbor_count, topology.interior.len());
    }

    #[test]
    fn interior_face_coloring_has_no_same_color_cell_conflicts() {
        let mesh = two_tet_mesh();
        let faces = (0..mesh.num_faces())
            .map(|face| FaceId(face as u32))
            .collect::<Vec<_>>();
        let boundaries = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
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
        let topology = &cache.face_topology;
        assert!(!topology.interior.is_empty());
        assert_eq!(
            topology.interior_coloring.num_colors,
            topology.interior_coloring.buckets.len()
        );
        for bucket in &topology.interior_coloring.buckets {
            let mut cells = std::collections::HashSet::new();
            for &face_idx in bucket {
                let face = &topology.interior[face_idx];
                assert!(cells.insert(face.owner));
                assert!(cells.insert(face.neighbor));
            }
        }
        assert_eq!(
            topology.interior_coloring.buckets.len(),
            topology.interior_coloring.bucket_batch_layouts.len()
        );
        for (bucket, layout) in topology
            .interior_coloring
            .buckets
            .iter()
            .zip(&topology.interior_coloring.bucket_batch_layouts)
        {
            assert_eq!(layout.num_faces(), bucket.len());
            let mut recovered = Vec::with_capacity(bucket.len());
            for batch in &layout.full_batches {
                assert_eq!(batch.face_indices.len(), 4);
                recovered.extend_from_slice(&batch.face_indices);
            }
            recovered.extend_from_slice(&layout.remainder);
            assert_eq!(recovered, *bucket);
        }
    }
}
