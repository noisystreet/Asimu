//! 非结构网格面拓扑与 IDWLS 几何预计算缓存。
//!
//! 面列表与 LSQ 正规方程矩阵 \(A\) 仅依赖网格几何，在求解器 work 区初始化一次；
//! 每步 RHS 只累加右端项 \(b\) 并求解梯度。

#[path = "interior_face_batch_layout.rs"]
mod interior_face_batch_layout;

use interior_face_batch_layout::build_bucket_batch_layouts;
pub use interior_face_batch_layout::{InteriorFaceBatchStatic4, InteriorFaceBucketBatchLayout};

use super::block_lusgs_preconditioner_topology::BlockLusgsPreconditionerTopology;
use crate::boundary::{BoundaryKind, BoundarySet, WallHeat};
use crate::core::{CellId, FaceId, Real, Vector3};
use crate::discretization::unstructured_face_cache_f32::{
    LuSgsUnstructuredCouplingsF32, UnstructuredFaceTopologyF32, build_cell_gradient_samples_f32,
    lsq_geometry_f32_from_f64,
};
use crate::discretization::unstructured_idwls_exec_topo::build_idwls_viscous_host_topology;
use crate::discretization::unstructured_spectral_exec_topo::build_spectral_radius_host_topology;
use crate::error::Result;
use crate::mesh::UnstructuredMesh3d;
use crate::physics::FreestreamParams;

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

/// 边界面无粘 BC 类别（block_lusgs 解析 Jacobian 用）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnstructuredBoundaryInviscidKind {
    Wall {
        no_slip: bool,
        heat: WallHeat,
    },
    Symmetry,
    Farfield(FreestreamParams),
    Inlet {
        supersonic: bool,
        total_pressure: Real,
        total_temperature: Real,
        velocity_direction: [Real; 3],
        freestream: FreestreamParams,
    },
    Outlet {
        supersonic: bool,
        static_pressure: Real,
    },
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
    pub inviscid: UnstructuredBoundaryInviscidKind,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    /// 最大着色桶内面数（exec scatter `Auto` 解析用）。
    #[must_use]
    pub fn max_bucket_faces(&self) -> usize {
        self.buckets.iter().map(Vec::len).max().unwrap_or(0)
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
        crate::exec::parallel::par_map_colored_buckets(&self.buckets, 1024, f)
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
    /// f32 热路径预打包面几何（与 `face_topology` 索引对齐）。
    pub face_topology_f32: UnstructuredFaceTopologyF32,
    pub lsq_geometry: Vec<LsqPrecomputedCell>,
    /// f32 热路径 IDWLS 矩阵 \(A\)（与 `lsq_geometry` 单元索引对齐）。
    pub lsq_geometry_f32:
        Vec<crate::discretization::unstructured_face_cache_f32::LsqPrecomputedCellF32>,
    /// f32 梯度限制器样本（MUSCL 外推用）。
    pub cell_gradient_samples_f32:
        Vec<Vec<crate::discretization::unstructured_face_cache_f32::GradientLimiterSampleF32>>,
    /// f32 LU-SGS 面耦合邻接。
    pub lusgs_couplings_f32: LuSgsUnstructuredCouplingsF32,
    /// 每单元 IDWLS / 限制器样本（内部邻单元 + 边界面心）。
    pub cell_gradient_samples: Vec<Vec<GradientLimiterSample>>,
    /// IDWLS RHS 单元–面关联（`parallel-fvm` 单元并行累加用）。
    pub lsq_rhs_incidence: LsqRhsCellIncidence,
    pub block_lusgs_topology: BlockLusgsPreconditionerTopology,
    /// LU-SGS / block_lusgs 扫掠顺序；默认 CellId。
    pub solver_order: Vec<usize>,
    /// `solver_order` 的逆映射：cell id → sweep rank。
    pub solver_rank: Vec<usize>,
    /// IDWLS 粘性 RHS 静态拓扑（init 一次；CUDA 路径 H2D）。
    pub idwls_viscous_topo:
        crate::discretization::unstructured_idwls_exec_topo::IdwlsViscousHostTopology,
    /// 谱半径静态拓扑（init 一次；CUDA 单元并行 kernel）。
    pub spectral_radius_topo:
        crate::discretization::unstructured_spectral_exec_topo::SpectralRadiusHostTopology,
    /// LU-SGS 扫掠 CSR 拓扑（init 一次；CUDA 串行扫掠 kernel）。
    #[cfg(feature = "cuda")]
    pub lusgs_sweep_topo:
        crate::discretization::unstructured_lusgs_sweep_exec_topo::LuSgsSweepHostTopology,
    /// 无粘内面 CUDA 拓扑（init 一次；P0 消除每步 host collect）。
    #[cfg(feature = "cuda")]
    pub cuda_inviscid_interior_topo: crate::exec::gpu::cuda::ExecInteriorFaceTopology,
    /// 粘性内面 CUDA 拓扑模板（\(\mu,\lambda\) 每步在 scratch 副本上刷新）。
    #[cfg(feature = "cuda")]
    pub cuda_viscous_interior_topo: crate::exec::gpu::cuda::ExecViscousInteriorTopology,
    /// 无粘边界面 CUDA 拓扑（init 一次 H2D）。
    #[cfg(feature = "cuda")]
    pub cuda_inviscid_boundary_topo: crate::exec::gpu::cuda::ExecInviscidBoundaryTopology,
    /// 粘性边界面 CUDA 拓扑（init 一次 H2D）。
    #[cfg(feature = "cuda")]
    pub cuda_viscous_boundary_topo: crate::exec::gpu::cuda::ExecViscousBoundaryTopology,
}

impl UnstructuredSolverMeshCache {
    /// 由网格与边界 patch 构建面拓扑，并预计算 IDWLS 矩阵 \(A\)。
    pub fn from_mesh(mesh: &UnstructuredMesh3d, boundaries: &BoundarySet) -> Result<Self> {
        Self::from_mesh_with_order(mesh, boundaries, None, FreestreamParams::default())
    }

    pub fn from_mesh_with_freestream(
        mesh: &UnstructuredMesh3d,
        boundaries: &BoundarySet,
        freestream: FreestreamParams,
    ) -> Result<Self> {
        Self::from_mesh_with_order(mesh, boundaries, None, freestream)
    }

    pub fn from_mesh_with_order(
        mesh: &UnstructuredMesh3d,
        boundaries: &BoundarySet,
        solver_order: Option<&[usize]>,
        freestream_default: FreestreamParams,
    ) -> Result<Self> {
        let face_topology = build_face_topology(mesh, boundaries, freestream_default)?;
        let face_topology_f32 = UnstructuredFaceTopologyF32::from_face_topology(&face_topology);
        let num_cells = mesh.num_cells();
        let lsq_geometry = precompute_lsq_geometry(num_cells, &face_topology);
        let lsq_geometry_f32 = lsq_geometry_f32_from_f64(&lsq_geometry);
        let cell_gradient_samples_f32 =
            build_cell_gradient_samples_f32(num_cells, &face_topology_f32);
        let lusgs_couplings_f32 =
            LuSgsUnstructuredCouplingsF32::from_topology_f32(num_cells, &face_topology_f32);
        let cell_gradient_samples = build_cell_gradient_samples(num_cells, &face_topology);
        let lsq_rhs_incidence = build_lsq_rhs_cell_incidence(num_cells, &face_topology);
        let solver_order = solver_order
            .map(|order| {
                crate::mesh_order::validate_cell_order(order, num_cells)?;
                Ok::<Vec<usize>, crate::error::AsimuError>(order.to_vec())
            })
            .transpose()?
            .unwrap_or_else(|| crate::mesh_order::identity_order(num_cells));
        let solver_rank = crate::mesh_order::cell_order_rank(&solver_order)?;
        let exec_idwls_viscous_topo =
            build_idwls_viscous_host_topology(&face_topology_f32, &lsq_rhs_incidence, num_cells);
        let spectral_radius_topo =
            build_spectral_radius_host_topology(&face_topology_f32, &lsq_rhs_incidence, num_cells);
        #[cfg(feature = "cuda")]
        let lusgs_sweep_topo =
            crate::discretization::unstructured_lusgs_sweep_exec_topo::LuSgsSweepHostTopology::from_mesh_and_couplings(
                mesh,
                &lusgs_couplings_f32,
            );
        #[cfg(feature = "cuda")]
        let cuda_inviscid_interior_topo =
            crate::discretization::unstructured_interior_exec_topo::build_cuda_inviscid_interior_topology(
                &face_topology_f32,
                &face_topology,
            );
        #[cfg(feature = "cuda")]
        let cuda_viscous_interior_topo =
            crate::discretization::unstructured_interior_exec_topo::build_cuda_viscous_interior_topology(
                &face_topology_f32,
                &face_topology,
            );
        #[cfg(feature = "cuda")]
        let cuda_inviscid_boundary_topo =
            crate::discretization::unstructured_boundary_exec_topo::build_cuda_inviscid_boundary_topology(
                &face_topology_f32,
            );
        #[cfg(feature = "cuda")]
        let cuda_viscous_boundary_topo =
            crate::discretization::unstructured_boundary_exec_topo::build_cuda_viscous_boundary_topology(
                &face_topology_f32,
            );
        let block_lusgs_topology = BlockLusgsPreconditionerTopology::from_interior_faces(
            num_cells,
            &face_topology.interior,
        );
        Ok(Self {
            face_topology,
            face_topology_f32,
            lsq_geometry,
            lsq_geometry_f32,
            cell_gradient_samples_f32,
            lusgs_couplings_f32,
            cell_gradient_samples,
            lsq_rhs_incidence,
            block_lusgs_topology,
            solver_order,
            solver_rank,
            idwls_viscous_topo: exec_idwls_viscous_topo,
            spectral_radius_topo,
            #[cfg(feature = "cuda")]
            lusgs_sweep_topo,
            #[cfg(feature = "cuda")]
            cuda_inviscid_interior_topo,
            #[cfg(feature = "cuda")]
            cuda_viscous_interior_topo,
            #[cfg(feature = "cuda")]
            cuda_inviscid_boundary_topo,
            #[cfg(feature = "cuda")]
            cuda_viscous_boundary_topo,
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
    freestream_default: FreestreamParams,
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
        let inviscid = boundary_inviscid_kind(&patch.kind, freestream_default);
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
                inviscid,
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
    let face_center = mesh.face_metric(face).center;
    lsq_dr_weight(owner_center, face_center)
}

fn lsq_dr_weight(from: Vector3, to: Vector3) -> (Vector3, Real) {
    let dr = vec_sub(to, from);
    let dist = dr.magnitude();
    if dist <= Real::EPSILON {
        (dr, 0.0)
    } else {
        // SU2 `WEIGHTED_LEAST_SQUARES` 与 Blazek 惯例：w = 1/|Δx|²
        (dr, 1.0 / (dist * dist))
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

fn boundary_inviscid_kind(
    kind: &BoundaryKind,
    freestream_default: FreestreamParams,
) -> UnstructuredBoundaryInviscidKind {
    match kind {
        BoundaryKind::Wall { no_slip, heat, .. } => UnstructuredBoundaryInviscidKind::Wall {
            no_slip: *no_slip,
            heat: *heat,
        },
        BoundaryKind::Symmetry => UnstructuredBoundaryInviscidKind::Symmetry,
        BoundaryKind::Farfield {
            mach,
            pressure,
            temperature,
            alpha,
            beta,
        } => UnstructuredBoundaryInviscidKind::Farfield(FreestreamParams {
            mach: *mach,
            pressure: *pressure,
            temperature: *temperature,
            alpha: *alpha,
            beta: *beta,
            velocity_direction: [1.0, 0.0, 0.0],
        }),
        BoundaryKind::Inlet {
            total_pressure,
            total_temperature,
            velocity_direction,
            supersonic,
            ..
        } => UnstructuredBoundaryInviscidKind::Inlet {
            supersonic: *supersonic,
            total_pressure: *total_pressure,
            total_temperature: *total_temperature,
            velocity_direction: *velocity_direction,
            freestream: freestream_default,
        },
        BoundaryKind::TurbulentInlet {
            total_pressure,
            total_temperature,
            velocity_direction,
            ..
        } => UnstructuredBoundaryInviscidKind::Inlet {
            supersonic: false,
            total_pressure: *total_pressure,
            total_temperature: *total_temperature,
            velocity_direction: *velocity_direction,
            freestream: freestream_default,
        },
        BoundaryKind::Outlet {
            static_pressure,
            supersonic,
            ..
        } => UnstructuredBoundaryInviscidKind::Outlet {
            supersonic: *supersonic,
            static_pressure: *static_pressure,
        },
        _ => UnstructuredBoundaryInviscidKind::Farfield(freestream_default),
    }
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

/// f32 梯度路径：RHS 为 f32，正规方程矩阵用 f64 预计算（避免缩尺网格 cast 后 det 下溢）。
pub(crate) fn solve_lsq_gradient_f32_rhs(
    geometry: &LsqPrecomputedCell,
    rhs: [f32; 3],
) -> Option<[f32; 3]> {
    let grad = solve_lsq_gradient(
        geometry,
        Vector3::new(rhs[0] as Real, rhs[1] as Real, rhs[2] as Real),
    )?;
    Some([grad.x as f32, grad.y as f32, grad.z as f32])
}

fn vec_sub(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn neg_vector(v: Vector3) -> Vector3 {
    Vector3::new(-v.x, -v.y, -v.z)
}

#[cfg(test)]
#[path = "unstructured_face_cache_tests.rs"]
mod tests;
