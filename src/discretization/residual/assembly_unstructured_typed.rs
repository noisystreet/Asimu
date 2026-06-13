//! 非结构 3D 网格无粘残差装配（typed 场；一阶用 typed primitive，二阶 MUSCL 经 f64 重构）。

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::{ComputeFloat, Real};
use crate::discretization::inviscid::{
    scatter_fused_boundary_inviscid_face_typed, scatter_fused_interior_inviscid_face_typed,
};
use crate::discretization::unstructured_face_cache::UnstructuredFaceTopology;
use crate::discretization::{
    BoundaryGhostBuffer, FaceFluxInput, GradientFields, InviscidFluxConfig, ReconstructionKind,
    UnstructuredLinearReconstructionCtx, UnstructuredSolverMeshCache, face_inviscid_flux,
    face_inviscid_flux_from_interface, reconstruct_unstructured_boundary_face,
};
use crate::error::{AsimuError, Result};
use crate::exec::ExecutionContext;
use crate::field::{
    ConservedFieldsT, ConservedResidualT, PrimitiveFields, PrimitiveFieldsT,
    primitive_from_conserved_relaxed,
};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::IdealGasEoS;

use super::assembly_unstructured::{
    InviscidAssemblyUnstructuredParams, compute_interior_inviscid_face_contribution,
};
use super::{accumulate_boundary_face_typed, accumulate_interior_face_typed, is_degenerate_volume};

/// typed 非结构无粘残差装配上下文。
pub struct InviscidAssemblyUnstructuredTypedParams<'a, T: ComputeFloat> {
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFieldsT<T>,
    /// 二阶 MUSCL：BC/重构/限制器样本用 f64 原始变量（ADR 0016 §4）。
    pub spectral_primitives: &'a PrimitiveFields,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub gradients: Option<&'a GradientFields>,
    pub min_pressure: Real,
    pub exec: &'a ExecutionContext,
}

/// 装配非结构 3D 无粘 Euler 残差（`T=f32`/`f64`）。
pub fn assemble_inviscid_residual_unstructured_typed<T: ComputeFloat>(
    fields: &ConservedFieldsT<T>,
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, T>,
) -> Result<()> {
    let n = params.mesh.num_cells();
    if fields.num_cells() != n || residual.num_cells() != n || params.primitives.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "非结构 typed 场/残差/primitive 长度须等于网格单元数 {n}"
        )));
    }
    if params.spectral_primitives.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "非结构 typed spectral primitive 长度须等于网格单元数 {n}"
        )));
    }
    residual.clear();
    let topology = &params.mesh_cache.face_topology;
    match params.config.reconstruction {
        ReconstructionKind::FirstOrder => {
            assemble_first_order_typed(residual, params, topology)?;
        }
        ReconstructionKind::Muscl => {
            assemble_muscl_typed(residual, params, topology)?;
        }
    }
    Ok(())
}

fn assemble_first_order_typed<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, T>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    {
        let _span = info_span!(
            "unstructured_inviscid_interior_faces_typed",
            faces = topology.interior.len(),
            precision = T::PRECISION.label(),
        )
        .entered();
        assemble_interior_faces_first_order_typed(residual, params, topology)?;
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

fn assemble_muscl_typed<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, T>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    let f64_params = muscl_f64_params(params)?;
    {
        let _span = info_span!(
            "unstructured_inviscid_interior_faces_typed",
            path = "muscl",
            faces = topology.interior.len(),
            precision = T::PRECISION.label(),
        )
        .entered();
        for bucket in &topology.interior_coloring.buckets {
            for &face_idx in bucket {
                if let Some((geom, flux)) =
                    compute_interior_inviscid_face_contribution(face_idx, &f64_params, topology)?
                {
                    scatter_fused_interior_inviscid_face_typed(residual, &geom, &flux);
                }
            }
        }
    }
    {
        let _span = info_span!(
            "unstructured_inviscid_boundary_faces_typed",
            path = "muscl",
            faces = topology.boundary.len(),
            precision = T::PRECISION.label(),
        )
        .entered();
        assemble_boundary_faces_muscl_typed(residual, &f64_params, topology)?;
    }
    Ok(())
}

fn muscl_f64_params<'a, T: ComputeFloat>(
    params: &'a InviscidAssemblyUnstructuredTypedParams<'a, T>,
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
        primitives: params.spectral_primitives,
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

fn first_order_interior_flux<T: ComputeFloat>(
    primitives: &PrimitiveFieldsT<T>,
    owner: usize,
    neighbor: usize,
    normal: crate::core::Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<crate::discretization::InviscidFlux> {
    let owner_prim = primitives.cell_primitive(owner);
    let neighbor_prim = primitives.cell_primitive(neighbor);
    face_inviscid_flux(
        FaceFluxInput::first_order(&owner_prim, &neighbor_prim),
        normal,
        eos,
        config,
    )
}

fn assemble_interior_faces_first_order_typed<T: ComputeFloat>(
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
        accumulate_interior_face_typed(
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

fn assemble_boundary_faces_first_order_typed<T: ComputeFloat>(
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
        let owner_prim = params.primitives.cell_primitive(bface.owner);
        let ghost_prim =
            primitive_from_conserved_relaxed(params.eos, &ghost.conserved, params.min_pressure)?;
        let flux = face_inviscid_flux(
            FaceFluxInput::first_order(&owner_prim, &ghost_prim),
            bface.normal,
            params.eos,
            params.config,
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
        BoundaryGhostBuffer, UnstructuredGradientLimiter, UnstructuredGradientLsqInput,
        UnstructuredGradientScratch, apply_compressible_boundary_conditions,
        compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq,
    };
    use crate::discretization::{GradientFields, InviscidFluxConfig};
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
        PrimitiveFields,
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
        apply_compressible_boundary_conditions(
            &mesh,
            &boundary,
            &fields.cast_real().expect("real"),
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
        let mut spectral = PrimitiveFields::zeros(mesh.num_cells()).expect("spectral");
        spectral
            .fill_from_conserved(
                &fields.cast_real().expect("real"),
                side.eos,
                side.min_pressure,
            )
            .expect("fill spectral");
        (
            mesh, boundary, fields, ghosts, mesh_cache, primitives, spectral,
        )
    }

    #[test]
    fn f32_single_tet_uniform_freestream_has_near_zero_rhs() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_side();
        let (mesh, boundary, fields, ghosts, mesh_cache, primitives, spectral) =
            single_tet_fixture(&side);
        let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
        let config = InviscidFluxConfig::default();
        let exec = ExecutionContext::for_unit_test();
        let params = InviscidAssemblyUnstructuredTypedParams {
            mesh: &mesh,
            eos: side.eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            spectral_primitives: &spectral,
            mesh_cache: &mesh_cache,
            gradients: None,
            min_pressure: side.min_pressure,
            exec: &exec,
        };
        assemble_inviscid_residual_unstructured_typed(&fields, &mut rhs, &params)
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
        let (mesh, boundary, fields, ghosts, mesh_cache, primitives, spectral) =
            single_tet_fixture(&side);
        let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
        let config = InviscidFluxConfig {
            unstructured_gradient_limiter: Some(UnstructuredGradientLimiter::BarthJespersen),
            ..InviscidFluxConfig::muscl_hllc()
        };
        let mut gradients = GradientFields::zeros(mesh.num_cells()).expect("grad");
        let mut grad_scratch = UnstructuredGradientScratch::new(mesh.num_cells());
        let mut exec = ExecutionContext::for_unit_test();
        compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq(
            UnstructuredGradientLsqInput {
                mesh: &mesh,
                mesh_cache: &mesh_cache,
                primitives: &spectral,
                eos: side.eos,
                ghosts: &ghosts,
                min_pressure: side.min_pressure,
                viscous: None,
            },
            &mut gradients,
            &mut grad_scratch,
            &mut exec,
        )
        .expect("gradients");
        let params = InviscidAssemblyUnstructuredTypedParams {
            mesh: &mesh,
            eos: side.eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            spectral_primitives: &spectral,
            mesh_cache: &mesh_cache,
            gradients: Some(&gradients),
            min_pressure: side.min_pressure,
            exec: &exec,
        };
        assemble_inviscid_residual_unstructured_typed(&fields, &mut rhs, &params)
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
