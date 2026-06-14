//! 非结构无粘 MUSCL 残差 f32 装配。

use crate::discretization::UnstructuredGradientLimiter;
use crate::discretization::face_flux_typed::face_inviscid_flux_from_interface_f32;
use crate::discretization::gradient_typed::GradientFieldsT;
#[cfg(not(feature = "parallel-fvm"))]
use crate::discretization::inviscid_f32::scatter_fused_interior_inviscid_face_f32;
use crate::discretization::inviscid_f32::{
    InteriorInviscidScatterGeomF32, InviscidFluxF32, scatter_fused_boundary_inviscid_face_f32,
};
use crate::discretization::reconstruction_unstructured_f32::{
    UnstructuredLinearReconstructionCtxF32, reconstruct_unstructured_boundary_face_f32,
    reconstruct_unstructured_interior_face_f32,
};
use crate::discretization::unstructured_face_cache::UnstructuredFaceTopology;
use crate::discretization::unstructured_face_cache_f32::UnstructuredInteriorFaceF32;
use crate::error::{AsimuError, Result};
use crate::exec::scatter::{
    InviscidPairScatterF32, InviscidResidualMutF32, InviscidScatterOpF32,
    scatter_inviscid_pairs_f32,
};
use crate::field::ConservedResidualT;

use super::super::is_degenerate_volume_f32;
use super::InviscidAssemblyUnstructuredTypedParams;

/// f32 非结构 MUSCL 无粘残差装配（重构与 Riemann 均为原生 f32）。
pub(super) fn assemble_inviscid_muscl_unstructured_f32(
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    _topology: &UnstructuredFaceTopology,
) -> Result<()> {
    validate_f32_muscl_params(params)?;
    assemble_interior_faces_muscl_f32(residual, params)?;
    assemble_boundary_faces_muscl_f32(residual, params)?;
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
) -> Result<()> {
    let topology = &params.mesh_cache.face_topology;
    let topology_f32 = &params.mesh_cache.face_topology_f32;
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

    #[cfg(not(feature = "parallel-fvm"))]
    {
        for bucket in &topology.interior_coloring.buckets {
            for &face_idx in bucket {
                if let Some((geom, flux)) = compute_interior_inviscid_face_contribution_f32(
                    face_idx,
                    params,
                    topology_f32.interior.as_slice(),
                    gradients,
                    ctx,
                )? {
                    scatter_fused_interior_inviscid_face_f32(residual, &geom, &flux);
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
                compute_interior_inviscid_face_contribution_f32(
                    face_idx,
                    params,
                    topology_f32.interior.as_slice(),
                    gradients,
                    ctx,
                )
            })?;
            let pairs: Vec<_> = contributions.into_iter().flatten().collect();
            scatter_inviscid_pairs_f32(
                InviscidPairScatterF32 {
                    ctx: &*params.exec,
                    bucket_len: bucket.len(),
                    pairs: &pairs,
                    residual: InviscidResidualMutF32 {
                        density: residual.density.values_mut(),
                        mx: residual.momentum_x.values_mut(),
                        my: residual.momentum_y.values_mut(),
                        mz: residual.momentum_z.values_mut(),
                        energy: residual.total_energy.values_mut(),
                    },
                },
                inviscid_scatter_extract_f32,
            );
        }
        Ok(())
    }
}

fn inviscid_scatter_extract_f32(
    g: &InteriorInviscidScatterGeomF32,
    f: &InviscidFluxF32,
) -> InviscidScatterOpF32 {
    InviscidScatterOpF32 {
        owner: g.owner,
        neighbor: g.neighbor,
        owner_scale: g.owner_scale,
        neighbor_scale: g.neighbor_scale,
        mass: f.mass,
        momentum: f.momentum,
        energy: f.energy,
    }
}

fn compute_interior_inviscid_face_contribution_f32(
    face_idx: usize,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    topology: &[UnstructuredInteriorFaceF32],
    gradients: &GradientFieldsT<f32>,
    ctx: UnstructuredLinearReconstructionCtxF32<'_>,
) -> Result<Option<(InteriorInviscidScatterGeomF32, InviscidFluxF32)>> {
    let face = &topology[face_idx];
    if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
        return Ok(None);
    }
    let iface_f32 = reconstruct_unstructured_interior_face_f32(
        face,
        ctx,
        gradients.inviscid_primitive_grad_at(face.owner),
        gradients.inviscid_primitive_grad_at(face.neighbor),
    )?;
    let flux =
        face_inviscid_flux_from_interface_f32(iface_f32, face.normal, params.eos, params.config)?;
    Ok(Some((
        InteriorInviscidScatterGeomF32 {
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
) -> Result<()> {
    let topology_f32 = &params.mesh_cache.face_topology_f32;
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
    for bface in &topology_f32.boundary {
        if bface.owner_rhs_scale == 0.0 || is_degenerate_volume_f32(bface.owner_volume) {
            continue;
        }
        let iface_f32 = reconstruct_unstructured_boundary_face_f32(
            bface,
            ctx,
            gradients.inviscid_primitive_grad_at(bface.owner),
        )?;
        let flux = face_inviscid_flux_from_interface_f32(
            iface_f32,
            bface.normal,
            params.eos,
            params.config,
        )?;
        scatter_fused_boundary_inviscid_face_f32(
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
