//! 非结构 3D 网格无粘残差装配（typed 场；一阶与二阶 MUSCL 均支持 f32/f64）。

#[cfg(feature = "cuda")]
#[path = "assembly_unstructured_typed_cuda.rs"]
mod assembly_unstructured_typed_cuda;

#[path = "assembly_unstructured_inviscid_f32.rs"]
mod inviscid_f32;

#[path = "assembly_unstructured_first_order_typed.rs"]
mod first_order_typed;

use first_order_typed::{InviscidFirstOrderFaceFlux, first_order_interior_flux};

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::{ComputeFloat, Real};
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::inviscid::{
    InteriorInviscidScatterGeom, scatter_fused_boundary_inviscid_face_typed,
};
use crate::discretization::unstructured_face_cache::UnstructuredFaceTopology;
use crate::discretization::{
    BoundaryGhostBuffer, InviscidFluxConfig, ReconstructionKind,
    UnstructuredLinearReconstructionCtx, UnstructuredSolverMeshCache,
    face_inviscid_flux_from_interface, reconstruct_unstructured_boundary_face,
};
use crate::error::{AsimuError, Result};
use crate::exec::ExecutionContext;
use crate::exec::scatter::{
    InviscidPairScatter, InviscidPairScatterF32, InviscidResidualMut, InviscidResidualMutF32,
    InviscidScatterOp, scatter_inviscid_pairs, scatter_inviscid_pairs_f32,
};
use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::IdealGasEoS;

use super::assembly_unstructured::{
    InviscidAssemblyUnstructuredParams, compute_interior_inviscid_face_contribution,
};
use super::{accumulate_boundary_face_typed, is_degenerate_volume};

/// scatter 精度 dispatch（`ComputeFloat` 密封子集；ADR 0016 P5）。
pub trait InviscidTypedScatterBackend: ComputeFloat {
    fn scatter_inviscid_interior_pairs(
        residual: &mut ConservedResidualT<Self>,
        ctx: &ExecutionContext,
        bucket_len: usize,
        pairs: &[(
            InteriorInviscidScatterGeom,
            crate::discretization::InviscidFlux,
        )],
    );

    fn scatter_fused_interior_face(
        residual: &mut ConservedResidualT<Self>,
        geom: &InteriorInviscidScatterGeom,
        flux: &crate::discretization::InviscidFlux,
    );

    /// CUDA 一阶内面装配（默认 `false`；`f32` + feature `cuda` 可返回 `true`）。
    fn try_cuda_first_order_interior(
        _residual: &mut ConservedResidualT<Self>,
        _params: &mut InviscidAssemblyUnstructuredTypedParams<'_, Self>,
        _topology: &UnstructuredFaceTopology,
    ) -> Result<bool> {
        Ok(false)
    }
}

#[cfg_attr(feature = "parallel-fvm", allow(dead_code))]
impl InviscidTypedScatterBackend for f64 {
    fn scatter_inviscid_interior_pairs(
        residual: &mut ConservedResidualT<f64>,
        ctx: &ExecutionContext,
        bucket_len: usize,
        pairs: &[(
            InteriorInviscidScatterGeom,
            crate::discretization::InviscidFlux,
        )],
    ) {
        scatter_inviscid_interior_pairs_f64(residual, ctx, bucket_len, pairs);
    }

    fn scatter_fused_interior_face(
        residual: &mut ConservedResidualT<f64>,
        geom: &InteriorInviscidScatterGeom,
        flux: &crate::discretization::InviscidFlux,
    ) {
        scatter_fused_interior_face_f64(residual, geom, flux);
    }
}

#[cfg_attr(feature = "parallel-fvm", allow(dead_code))]
impl InviscidTypedScatterBackend for f32 {
    fn scatter_inviscid_interior_pairs(
        residual: &mut ConservedResidualT<f32>,
        ctx: &ExecutionContext,
        bucket_len: usize,
        pairs: &[(
            InteriorInviscidScatterGeom,
            crate::discretization::InviscidFlux,
        )],
    ) {
        scatter_inviscid_interior_pairs_f32(residual, ctx, bucket_len, pairs);
    }

    fn scatter_fused_interior_face(
        residual: &mut ConservedResidualT<f32>,
        geom: &InteriorInviscidScatterGeom,
        flux: &crate::discretization::InviscidFlux,
    ) {
        scatter_fused_interior_face_f32(residual, geom, flux);
    }

    fn try_cuda_first_order_interior(
        residual: &mut ConservedResidualT<f32>,
        params: &mut InviscidAssemblyUnstructuredTypedParams<'_, f32>,
        topology: &UnstructuredFaceTopology,
    ) -> Result<bool> {
        #[cfg(feature = "cuda")]
        {
            assembly_unstructured_typed_cuda::cuda_first_order_f32_interior(
                residual, params, topology,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (residual, params, topology);
            Ok(false)
        }
    }
}

fn scatter_inviscid_interior_pairs_f64(
    residual: &mut ConservedResidualT<f64>,
    ctx: &ExecutionContext,
    bucket_len: usize,
    pairs: &[(
        InteriorInviscidScatterGeom,
        crate::discretization::InviscidFlux,
    )],
) {
    scatter_inviscid_pairs(
        InviscidPairScatter {
            ctx,
            bucket_len,
            pairs,
            residual: InviscidResidualMut {
                density: residual.density.values_mut(),
                mx: residual.momentum_x.values_mut(),
                my: residual.momentum_y.values_mut(),
                mz: residual.momentum_z.values_mut(),
                energy: residual.total_energy.values_mut(),
            },
        },
        inviscid_scatter_extract,
    );
}

fn scatter_inviscid_interior_pairs_f32(
    residual: &mut ConservedResidualT<f32>,
    ctx: &ExecutionContext,
    bucket_len: usize,
    pairs: &[(
        InteriorInviscidScatterGeom,
        crate::discretization::InviscidFlux,
    )],
) {
    scatter_inviscid_pairs_f32(
        InviscidPairScatterF32 {
            ctx,
            bucket_len,
            pairs,
            residual: InviscidResidualMutF32 {
                density: residual.density.values_mut(),
                mx: residual.momentum_x.values_mut(),
                my: residual.momentum_y.values_mut(),
                mz: residual.momentum_z.values_mut(),
                energy: residual.total_energy.values_mut(),
            },
        },
        inviscid_scatter_extract,
    );
}

#[cfg_attr(feature = "parallel-fvm", allow(dead_code))]
fn scatter_fused_interior_face_f64(
    residual: &mut ConservedResidualT<f64>,
    geom: &InteriorInviscidScatterGeom,
    flux: &crate::discretization::InviscidFlux,
) {
    crate::discretization::inviscid::scatter_fused_interior_inviscid_face(
        &mut crate::discretization::inviscid::InteriorInviscidResidualMut {
            density: residual.density.values_mut(),
            mx: residual.momentum_x.values_mut(),
            my: residual.momentum_y.values_mut(),
            mz: residual.momentum_z.values_mut(),
            energy: residual.total_energy.values_mut(),
        },
        geom,
        flux,
    );
}

#[cfg_attr(feature = "parallel-fvm", allow(dead_code))]
fn scatter_fused_interior_face_f32(
    residual: &mut ConservedResidualT<f32>,
    geom: &InteriorInviscidScatterGeom,
    flux: &crate::discretization::InviscidFlux,
) {
    crate::discretization::inviscid::scatter_fused_interior_inviscid_face_typed(
        residual, geom, flux,
    );
}

fn inviscid_scatter_extract(
    g: &InteriorInviscidScatterGeom,
    f: &crate::discretization::InviscidFlux,
) -> InviscidScatterOp {
    InviscidScatterOp {
        owner: g.owner,
        neighbor: g.neighbor,
        owner_scale: g.owner_scale,
        neighbor_scale: g.neighbor_scale,
        mass: f.mass,
        momentum: f.momentum,
        energy: f.energy,
    }
}

/// typed 非结构无粘残差装配上下文。
pub struct InviscidAssemblyUnstructuredTypedParams<'a, T: ComputeFloat> {
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFieldsT<T>,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub gradients: Option<&'a GradientFieldsT<T>>,
    pub min_pressure: Real,
    pub exec: &'a mut ExecutionContext,
}

/// 装配非结构 3D 无粘 Euler 残差（`T=f32`/`f64`）。
#[allow(private_bounds)]
pub fn assemble_inviscid_residual_unstructured_typed<
    T: InviscidTypedScatterBackend + InviscidMusclAssembly + InviscidFirstOrderInterior,
>(
    fields: &ConservedFieldsT<T>,
    residual: &mut ConservedResidualT<T>,
    params: &mut InviscidAssemblyUnstructuredTypedParams<'_, T>,
) -> Result<()> {
    let n = params.mesh.num_cells();
    if fields.num_cells() != n || residual.num_cells() != n || params.primitives.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "非结构 typed 场/残差/primitive 长度须等于网格单元数 {n}"
        )));
    }
    residual.clear();
    let topology = &params.mesh_cache.face_topology;
    match params.config.reconstruction {
        ReconstructionKind::FirstOrder => {
            assemble_first_order_typed(residual, params, topology)?;
        }
        ReconstructionKind::Muscl => {
            T::assemble_muscl_unstructured_typed(residual, params, topology)?;
        }
    }
    Ok(())
}

fn assemble_first_order_typed<T: InviscidTypedScatterBackend + InviscidFirstOrderInterior>(
    residual: &mut ConservedResidualT<T>,
    params: &mut InviscidAssemblyUnstructuredTypedParams<'_, T>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    {
        let _span = info_span!(
            "unstructured_inviscid_interior_faces_typed",
            faces = topology.interior.len(),
            precision = T::PRECISION.label(),
        )
        .entered();
        let interior_on_cuda = T::try_cuda_first_order_interior(residual, params, topology)?;
        if !interior_on_cuda {
            T::assemble_first_order_interior_faces(residual, params, topology)?;
        }
    }
    {
        let _span = info_span!(
            "unstructured_inviscid_boundary_faces_typed",
            faces = topology.boundary.len(),
            precision = T::PRECISION.label(),
        )
        .entered();
        assemble_boundary_faces_first_order_typed(residual, params, topology)?;
    }
    Ok(())
}

/// MUSCL 装配分发（f32 原生重构 / f64 既有路径）。
trait InviscidMusclAssembly: InviscidTypedScatterBackend {
    fn assemble_muscl_unstructured_typed(
        residual: &mut ConservedResidualT<Self>,
        params: &mut InviscidAssemblyUnstructuredTypedParams<'_, Self>,
        topology: &UnstructuredFaceTopology,
    ) -> Result<()>;
}

impl InviscidMusclAssembly for f32 {
    fn assemble_muscl_unstructured_typed(
        residual: &mut ConservedResidualT<f32>,
        params: &mut InviscidAssemblyUnstructuredTypedParams<'_, f32>,
        topology: &UnstructuredFaceTopology,
    ) -> Result<()> {
        inviscid_f32::assemble_inviscid_muscl_unstructured_f32(residual, params, topology)
    }
}

impl InviscidMusclAssembly for f64 {
    fn assemble_muscl_unstructured_typed(
        residual: &mut ConservedResidualT<f64>,
        params: &mut InviscidAssemblyUnstructuredTypedParams<'_, f64>,
        topology: &UnstructuredFaceTopology,
    ) -> Result<()> {
        let f64_params = muscl_f64_params(params)?;
        {
            let _span = info_span!(
                "unstructured_inviscid_interior_faces_typed",
                path = "muscl",
                faces = topology.interior.len(),
                precision = "f64",
            )
            .entered();
            assemble_interior_faces_colored_typed(residual, &f64_params, topology)?;
        }
        {
            let _span = info_span!(
                "unstructured_inviscid_boundary_faces_typed",
                path = "muscl",
                faces = topology.boundary.len(),
                precision = "f64",
            )
            .entered();
            assemble_boundary_faces_muscl_typed(residual, &f64_params, topology)?;
        }
        Ok(())
    }
}

fn spectral_f64_params<'a>(
    params: &'a InviscidAssemblyUnstructuredTypedParams<'a, f64>,
) -> InviscidAssemblyUnstructuredParams<'a> {
    InviscidAssemblyUnstructuredParams {
        mesh: params.mesh,
        eos: params.eos,
        config: params.config,
        boundaries: params.boundaries,
        ghosts: params.ghosts,
        primitives: params.primitives,
        face_topology: Some(&params.mesh_cache.face_topology),
        mesh_cache: Some(params.mesh_cache),
        gradients: params.gradients,
        min_pressure: params.min_pressure,
        exec: params.exec,
    }
}

fn assemble_interior_faces_colored_typed<T: InviscidTypedScatterBackend>(
    residual: &mut ConservedResidualT<T>,
    f64_params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    #[cfg(not(feature = "parallel-fvm"))]
    {
        for bucket in &topology.interior_coloring.buckets {
            for &face_idx in bucket {
                if let Some((geom, flux)) =
                    compute_interior_inviscid_face_contribution(face_idx, f64_params, topology)?
                {
                    T::scatter_fused_interior_face(residual, &geom, &flux);
                }
            }
        }
        return Ok(());
    }

    #[cfg(feature = "parallel-fvm")]
    {
        for bucket in &topology.interior_coloring.buckets {
            let contributions =
                crate::exec::parallel::par_try_map_face_indices(bucket, 1024, |face_idx| {
                    compute_interior_inviscid_face_contribution(face_idx, f64_params, topology)
                })?;
            let pairs: Vec<_> = contributions.into_iter().flatten().collect();
            T::scatter_inviscid_interior_pairs(residual, f64_params.exec, bucket.len(), &pairs);
        }
    }
    Ok(())
}

pub(crate) fn muscl_f64_params<'a>(
    params: &'a InviscidAssemblyUnstructuredTypedParams<'a, f64>,
) -> Result<InviscidAssemblyUnstructuredParams<'a>> {
    if params.gradients.is_none() {
        return Err(AsimuError::Config(
            "非结构 typed MUSCL 须先计算 inviscid linear reconstruction gradients".to_string(),
        ));
    }
    if params.config.unstructured_gradient_limiter.is_none() {
        return Err(AsimuError::Config(
            "非结构 typed MUSCL 须设置 unstructured_limiter（barth_jespersen 或 venkatakrishnan）"
                .to_string(),
        ));
    }
    Ok(InviscidAssemblyUnstructuredParams {
        mesh: params.mesh,
        eos: params.eos,
        config: params.config,
        boundaries: params.boundaries,
        ghosts: params.ghosts,
        primitives: params.primitives,
        face_topology: Some(&params.mesh_cache.face_topology),
        mesh_cache: Some(params.mesh_cache),
        gradients: params.gradients,
        min_pressure: params.min_pressure,
        exec: params.exec,
    })
}

fn assemble_boundary_faces_muscl_typed<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    let mesh_cache = params.mesh_cache.expect("linear reconstruction cache");
    let gradients = params.gradients.expect("linear reconstruction gradients");
    let limiter = params
        .config
        .unstructured_gradient_limiter
        .expect("unstructured limiter");
    let ctx = UnstructuredLinearReconstructionCtx {
        mesh_cache,
        primitives: params.primitives,
        ghosts: params.ghosts,
        eos: params.eos,
        min_pressure: params.min_pressure,
        limiter,
    };
    for bface in &topology.boundary {
        if bface.owner_rhs_scale == 0.0 || is_degenerate_volume(bface.owner_volume) {
            continue;
        }
        let iface = reconstruct_unstructured_boundary_face(
            bface,
            ctx,
            gradients.inviscid_primitive_grad_at(bface.owner),
        )?;
        let flux =
            face_inviscid_flux_from_interface(iface, bface.normal, params.eos, params.config)?;
        scatter_fused_boundary_inviscid_face_typed(
            residual,
            bface.owner,
            bface.owner_rhs_scale,
            &flux,
        );
    }
    Ok(())
}

/// 一阶内面装配分发（f32 串行 typed；f64 可走 parallel 桶）。
trait InviscidFirstOrderInterior: InviscidTypedScatterBackend + InviscidFirstOrderFaceFlux {
    fn assemble_first_order_interior_faces(
        residual: &mut ConservedResidualT<Self>,
        params: &InviscidAssemblyUnstructuredTypedParams<'_, Self>,
        topology: &UnstructuredFaceTopology,
    ) -> Result<()>;
}

impl InviscidFirstOrderInterior for f32 {
    fn assemble_first_order_interior_faces(
        residual: &mut ConservedResidualT<f32>,
        params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
        topology: &UnstructuredFaceTopology,
    ) -> Result<()> {
        assemble_first_order_interior_faces_serial(residual, params, topology)
    }
}

impl InviscidFirstOrderInterior for f64 {
    fn assemble_first_order_interior_faces(
        residual: &mut ConservedResidualT<f64>,
        params: &InviscidAssemblyUnstructuredTypedParams<'_, f64>,
        topology: &UnstructuredFaceTopology,
    ) -> Result<()> {
        #[cfg(feature = "parallel-fvm")]
        {
            let f64_params = spectral_f64_params(params);
            assemble_interior_faces_colored_typed(residual, &f64_params, topology)
        }
        #[cfg(not(feature = "parallel-fvm"))]
        {
            assemble_first_order_interior_faces_serial(residual, params, topology)
        }
    }
}

fn assemble_first_order_interior_faces_serial<T: InviscidFirstOrderFaceFlux>(
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, T>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    for face in &topology.interior {
        if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
            continue;
        }
        if is_degenerate_volume(face.owner_volume) || is_degenerate_volume(face.neighbor_volume) {
            continue;
        }
        let flux = first_order_interior_flux(
            params.primitives,
            face.owner,
            face.neighbor,
            face.normal,
            params.eos,
            params.config,
        )?;
        super::accumulate_interior_face_typed(
            residual,
            face.owner,
            face.neighbor,
            &flux,
            face.area,
            face.owner_volume,
            face.neighbor_volume,
        )?;
    }
    Ok(())
}

fn assemble_boundary_faces_first_order_typed<T: InviscidFirstOrderFaceFlux>(
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, T>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    for bface in &topology.boundary {
        if bface.owner_rhs_scale == 0.0 {
            continue;
        }
        if is_degenerate_volume(bface.owner_volume) {
            continue;
        }
        let ghost = params.ghosts.get_face(bface.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "边界面 FaceId({}) 缺少 ghost 状态",
                bface.face.index()
            ))
        })?;
        let flux = T::first_order_boundary_flux(
            params.primitives,
            bface.owner,
            &ghost,
            bface.normal,
            params.eos,
            params.config,
            params.min_pressure,
        )?;
        accumulate_boundary_face_typed(
            residual,
            bface.owner,
            &flux,
            bface.area,
            bface.owner_volume,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::discretization::freestream_pair::FreestreamPairFixture;
    use crate::discretization::{
        BoundaryGhostBuffer, UnstructuredGradientLimiter, UnstructuredGradientLsqInputF32,
        apply_compressible_boundary_conditions_typed,
        compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq_f32,
    };
    use crate::discretization::{GradientFieldsT, InviscidFluxConfig};
    use crate::exec::ExecutionContext;
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

    fn single_tet_fixture(
        side: &crate::discretization::freestream_pair::UniformFarfieldSide<'_>,
    ) -> (
        UnstructuredMesh3d,
        BoundarySet,
        ConservedFieldsT<f32>,
        BoundaryGhostBuffer,
        UnstructuredSolverMeshCache,
        PrimitiveFieldsT<f32>,
    ) {
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
            .collect::<Vec<_>>();
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces,
            BoundaryKind::Farfield {
                mach: side.fs.mach,
                pressure: side.fs.pressure,
                temperature: side.fs.temperature,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        let fields = ConservedFieldsT::<f32>::from_real_fields(
            &crate::field::ConservedFields::from_freestream_context(
                mesh.num_cells(),
                &side.ctx,
                side.fs,
            )
            .expect("fields"),
        )
        .expect("typed");
        let mut ghosts = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
        apply_compressible_boundary_conditions_typed(
            &mesh,
            &boundary,
            &fields,
            &mut ghosts,
            &side.ctx,
            side.fs,
            None,
        )
        .expect("bc");
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let mut primitives = PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, side.eos, side.min_pressure)
            .expect("fill");
        (mesh, boundary, fields, ghosts, mesh_cache, primitives)
    }

    #[test]
    fn f32_single_tet_uniform_freestream_has_near_zero_rhs() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_side();
        let (mesh, boundary, fields, ghosts, mesh_cache, primitives) = single_tet_fixture(&side);
        let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
        let config = InviscidFluxConfig::default();
        let mut exec = ExecutionContext::for_unit_test();
        let mut params = InviscidAssemblyUnstructuredTypedParams {
            mesh: &mesh,
            eos: side.eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            mesh_cache: &mesh_cache,
            gradients: None,
            min_pressure: side.min_pressure,
            exec: &mut exec,
        };
        assemble_inviscid_residual_unstructured_typed(&fields, &mut rhs, &mut params)
            .expect("assemble");
        assert!(
            rhs.density
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-5),
            "f32 tet density rhs"
        );
    }

    #[test]
    fn f32_single_tet_muscl_uniform_freestream_has_near_zero_rhs() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_side();
        let (mesh, boundary, fields, ghosts, mesh_cache, primitives) = single_tet_fixture(&side);
        let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
        let config = InviscidFluxConfig {
            unstructured_gradient_limiter: Some(UnstructuredGradientLimiter::BarthJespersen),
            ..InviscidFluxConfig::muscl_hllc()
        };
        let mut gradients = GradientFieldsT::<f32>::zeros(mesh.num_cells()).expect("grad");
        let mut exec = ExecutionContext::for_unit_test();
        compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq_f32(
            UnstructuredGradientLsqInputF32 {
                mesh: &mesh,
                mesh_cache: &mesh_cache,
                primitives: &primitives,
                eos: side.eos,
                ghosts: &ghosts,
                min_pressure: side.min_pressure,
                viscous: None,
            },
            &mut gradients,
            &mut exec,
        )
        .expect("gradients");
        let mut params = InviscidAssemblyUnstructuredTypedParams {
            mesh: &mesh,
            eos: side.eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            mesh_cache: &mesh_cache,
            gradients: Some(&gradients),
            min_pressure: side.min_pressure,
            exec: &mut exec,
        };
        assemble_inviscid_residual_unstructured_typed(&fields, &mut rhs, &mut params)
            .expect("assemble");
        assert!(
            rhs.density
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-5),
            "f32 muscl tet density rhs"
        );
    }
}
