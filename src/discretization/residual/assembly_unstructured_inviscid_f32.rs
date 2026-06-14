//! 非结构无粘 MUSCL 残差 f32 装配。

use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::inviscid::{
    InteriorInviscidScatterGeom, scatter_fused_boundary_inviscid_face_typed,
};
use crate::discretization::reconstruction_unstructured_f32::{
    UnstructuredLinearReconstructionCtxF32, interface_primitive_states_f32_to_f64,
    reconstruct_unstructured_boundary_face_f32, reconstruct_unstructured_interior_face_f32,
};
use crate::discretization::unstructured_face_cache::UnstructuredFaceTopology;
use crate::discretization::{UnstructuredGradientLimiter, face_inviscid_flux_from_interface};
use crate::error::{AsimuError, Result};
use crate::field::ConservedResidualT;

use super::{
    InviscidAssemblyUnstructuredTypedParams, InviscidTypedScatterBackend, is_degenerate_volume,
};

/// f32 非结构 MUSCL 无粘残差装配（重构 f32，Riemann 仍 f64）。
pub(super) fn assemble_inviscid_muscl_unstructured_f32(
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    validate_f32_muscl_params(params)?;
    assemble_interior_faces_muscl_f32(residual, params, topology)?;
    assemble_boundary_faces_muscl_f32(residual, params, topology)?;
    Ok(())
}

fn validate_f32_muscl_params(
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
) -> Result<()> {
    if params.gradients.is_none() {
        return Err(AsimuError::Config(
            "非结构 f32 MUSCL 须先计算 inviscid linear reconstruction gradients".to_string(),
        ));
    }
    if params.config.unstructured_gradient_limiter.is_none() {
        return Err(AsimuError::Config(
            "非结构 f32 MUSCL 须设置 unstructured_limiter（barth_jespersen 或 venkatakrishnan）"
                .to_string(),
        ));
    }
    Ok(())
}

fn assemble_interior_faces_muscl_f32(
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    let gradients = params.gradients.expect("gradients");
    let limiter = muscl_limiter(params);
    let ctx = UnstructuredLinearReconstructionCtxF32 {
        mesh_cache: params.mesh_cache,
        primitives: params.primitives,
        ghosts: params.ghosts,
        eos: params.eos,
        min_pressure: params.min_pressure,
        limiter,
    };
    for bucket in &topology.interior_coloring.buckets {
        for &face_idx in bucket {
            if let Some((geom, flux)) = compute_interior_inviscid_face_contribution_f32(
                face_idx, params, topology, gradients, ctx,
            )? {
                f32::scatter_fused_interior_face(residual, &geom, &flux);
            }
        }
    }
    Ok(())
}

fn compute_interior_inviscid_face_contribution_f32(
    face_idx: usize,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    topology: &UnstructuredFaceTopology,
    gradients: &GradientFieldsT<f32>,
    ctx: UnstructuredLinearReconstructionCtxF32<'_>,
) -> Result<
    Option<(
        InteriorInviscidScatterGeom,
        crate::discretization::InviscidFlux,
    )>,
> {
    let face = &topology.interior[face_idx];
    if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
        return Ok(None);
    }
    let iface_f32 = reconstruct_unstructured_interior_face_f32(
        face,
        ctx,
        gradients.inviscid_primitive_grad_at(face.owner),
        gradients.inviscid_primitive_grad_at(face.neighbor),
    )?;
    let flux = face_inviscid_flux_from_interface(
        interface_primitive_states_f32_to_f64(iface_f32),
        face.normal,
        params.eos,
        params.config,
    )?;
    Ok(Some((
        InteriorInviscidScatterGeom {
            owner: face.owner,
            neighbor: face.neighbor,
            owner_scale: face.owner_rhs_scale,
            neighbor_scale: face.neighbor_rhs_scale,
        },
        flux,
    )))
}

fn assemble_boundary_faces_muscl_f32(
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    let gradients = params.gradients.expect("gradients");
    let limiter = muscl_limiter(params);
    let ctx = UnstructuredLinearReconstructionCtxF32 {
        mesh_cache: params.mesh_cache,
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
        let iface_f32 = reconstruct_unstructured_boundary_face_f32(
            bface,
            ctx,
            gradients.inviscid_primitive_grad_at(bface.owner),
        )?;
        let flux = face_inviscid_flux_from_interface(
            interface_primitive_states_f32_to_f64(iface_f32),
            bface.normal,
            params.eos,
            params.config,
        )?;
        scatter_fused_boundary_inviscid_face_typed(
            residual,
            bface.owner,
            bface.owner_rhs_scale,
            &flux,
        );
    }
    Ok(())
}

fn muscl_limiter(
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
) -> UnstructuredGradientLimiter {
    params
        .config
        .unstructured_gradient_limiter
        .expect("unstructured limiter")
}
