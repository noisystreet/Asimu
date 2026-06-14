//! 非结构无粘 MUSCL 残差 f64 typed 装配（直接读 `GradientFieldsT<f64>`，不经 legacy params 桥接）。

use crate::discretization::UnstructuredGradientLimiter;
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::inviscid::{
    InteriorInviscidScatterGeom, scatter_fused_boundary_inviscid_face_typed,
};
use crate::discretization::unstructured_face_cache::{
    UnstructuredBoundaryFace, UnstructuredFaceTopology, UnstructuredInteriorFace,
};
use crate::discretization::{
    InviscidFlux, UnstructuredLinearReconstructionCtx, face_inviscid_flux_from_interface,
    reconstruct_unstructured_boundary_face, reconstruct_unstructured_interior_face,
};
use crate::error::{AsimuError, Result};
use crate::field::ConservedResidualT;

use super::is_degenerate_volume;
#[cfg(not(feature = "parallel-fvm"))]
use super::scatter_fused_interior_face_f64;
use super::{InviscidAssemblyUnstructuredTypedParams, scatter_inviscid_interior_pairs_f64};

/// f64 非结构 MUSCL 无粘残差装配（typed 参数直达重构与 Riemann）。
pub(super) fn assemble_inviscid_muscl_unstructured_f64(
    residual: &mut ConservedResidualT<f64>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f64>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    validate_f64_muscl_params(params)?;
    assemble_interior_faces_muscl_f64(residual, params, topology)?;
    assemble_boundary_faces_muscl_f64(residual, params, topology)?;
    Ok(())
}

fn validate_f64_muscl_params(
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f64>,
) -> Result<()> {
    if params.gradients.is_none() {
        return Err(AsimuError::Config(
            "非结构 f64 MUSCL 须先计算 inviscid linear reconstruction gradients".to_string(),
        ));
    }
    if params.config.unstructured_gradient_limiter.is_none() {
        return Err(AsimuError::Config(
            "非结构 f64 MUSCL 须设置 unstructured_limiter（barth_jespersen 或 venkatakrishnan）"
                .to_string(),
        ));
    }
    Ok(())
}

fn assemble_interior_faces_muscl_f64(
    residual: &mut ConservedResidualT<f64>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f64>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    let gradients = params.gradients.expect("gradients");
    let limiter = muscl_limiter(params);
    let ctx = UnstructuredLinearReconstructionCtx {
        mesh_cache: params.mesh_cache,
        primitives: params.primitives,
        ghosts: params.ghosts,
        eos: params.eos,
        min_pressure: params.min_pressure,
        limiter,
    };

    #[cfg(not(feature = "parallel-fvm"))]
    {
        for bucket in &topology.interior_coloring.buckets {
            for &face_idx in bucket {
                if let Some((geom, flux)) = compute_interior_inviscid_face_contribution_f64(
                    face_idx,
                    params,
                    topology.interior.as_slice(),
                    gradients,
                    ctx,
                )? {
                    scatter_fused_interior_face_f64(residual, &geom, &flux);
                }
            }
        }
        return Ok(());
    }

    #[cfg(feature = "parallel-fvm")]
    {
        use crate::exec::parallel::par_try_map_face_indices;

        for bucket in &topology.interior_coloring.buckets {
            let contributions = par_try_map_face_indices(bucket, 1024, |face_idx| {
                compute_interior_inviscid_face_contribution_f64(
                    face_idx,
                    params,
                    topology.interior.as_slice(),
                    gradients,
                    ctx,
                )
            })?;
            let pairs: Vec<_> = contributions.into_iter().flatten().collect();
            scatter_inviscid_interior_pairs_f64(residual, params.exec, bucket.len(), &pairs);
        }
        Ok(())
    }
}

fn compute_interior_inviscid_face_contribution_f64(
    face_idx: usize,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f64>,
    topology: &[UnstructuredInteriorFace],
    gradients: &GradientFieldsT<f64>,
    ctx: UnstructuredLinearReconstructionCtx<'_>,
) -> Result<Option<(InteriorInviscidScatterGeom, InviscidFlux)>> {
    let face = &topology[face_idx];
    if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
        return Ok(None);
    }
    let iface = reconstruct_unstructured_interior_face(
        face,
        ctx,
        gradients.inviscid_primitive_grad_at(face.owner),
        gradients.inviscid_primitive_grad_at(face.neighbor),
    )?;
    let flux = face_inviscid_flux_from_interface(iface, face.normal, params.eos, params.config)?;
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

fn assemble_boundary_faces_muscl_f64(
    residual: &mut ConservedResidualT<f64>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f64>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    let gradients = params.gradients.expect("gradients");
    let limiter = muscl_limiter(params);
    let ctx = UnstructuredLinearReconstructionCtx {
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
        assemble_one_boundary_muscl_f64(residual, params, bface, gradients, ctx)?;
    }
    Ok(())
}

fn assemble_one_boundary_muscl_f64(
    residual: &mut ConservedResidualT<f64>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f64>,
    bface: &UnstructuredBoundaryFace,
    gradients: &GradientFieldsT<f64>,
    ctx: UnstructuredLinearReconstructionCtx<'_>,
) -> Result<()> {
    let iface = reconstruct_unstructured_boundary_face(
        bface,
        ctx,
        gradients.inviscid_primitive_grad_at(bface.owner),
    )?;
    let flux = face_inviscid_flux_from_interface(iface, bface.normal, params.eos, params.config)?;
    scatter_fused_boundary_inviscid_face_typed(residual, bface.owner, bface.owner_rhs_scale, &flux);
    Ok(())
}

fn muscl_limiter(
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f64>,
) -> UnstructuredGradientLimiter {
    params
        .config
        .unstructured_gradient_limiter
        .expect("unstructured limiter")
}
