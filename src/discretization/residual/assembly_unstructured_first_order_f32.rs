//! 非结构一阶无粘残差 f32 装配（读取 `face_topology_f32` 预打包几何；scatter 全 f32）。

use tracing::info_span;

use crate::discretization::face_flux_typed::{
    face_inviscid_flux_first_order_boundary_soa_f32,
    face_inviscid_flux_first_order_interior_soa_f32,
};
#[cfg(not(feature = "parallel-fvm"))]
use crate::discretization::inviscid_f32::scatter_fused_interior_inviscid_face_f32;
use crate::discretization::inviscid_f32::{
    InteriorInviscidScatterGeomF32, InviscidFluxF32, scatter_fused_boundary_inviscid_face_f32,
};
use crate::discretization::residual::is_degenerate_volume_f32;
use crate::discretization::unstructured_face_cache::UnstructuredFaceTopology;
use crate::discretization::unstructured_face_cache_f32::UnstructuredInteriorFaceF32;
use crate::error::{AsimuError, Result};
use crate::exec::scatter::{
    InviscidPairScatterF32, InviscidResidualMutF32, InviscidScatterOpF32,
    scatter_inviscid_pairs_f32,
};
use crate::field::ConservedResidualT;

use super::InviscidAssemblyUnstructuredTypedParams;

#[cfg(not(feature = "parallel-fvm"))]
pub(super) fn assemble_first_order_interior_faces_serial_f32(
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
) -> Result<()> {
    let topology = &params.mesh_cache.face_topology;
    let topology_f32 = &params.mesh_cache.face_topology_f32;
    for bucket in &topology.interior_coloring.buckets {
        for &face_idx in bucket {
            if let Some((geom, flux)) = compute_interior_first_order_face_contribution_f32(
                face_idx,
                topology_f32.interior.as_slice(),
                params,
            )? {
                scatter_fused_interior_inviscid_face_f32(residual, &geom, &flux);
            }
        }
    }
    Ok(())
}

#[cfg(feature = "parallel-fvm")]
pub(super) fn assemble_first_order_interior_faces_parallel_f32(
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    use crate::exec::parallel::par_try_map_face_indices;

    let topology_f32 = &params.mesh_cache.face_topology_f32;
    let _span = info_span!(
        "unstructured_inviscid_first_order_interior_f32",
        path = "parallel_bucket",
        faces = topology.interior.len(),
        colors = topology.interior_coloring.num_colors,
    )
    .entered();
    for bucket in &topology.interior_coloring.buckets {
        let contributions = par_try_map_face_indices(bucket, 1024, |face_idx| {
            compute_interior_first_order_face_contribution_f32(
                face_idx,
                topology_f32.interior.as_slice(),
                params,
            )
        })?;
        let pairs: Vec<_> = contributions.into_iter().flatten().collect();
        scatter_inviscid_pairs_f32(
            InviscidPairScatterF32 {
                ctx: params.exec,
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
            first_order_inviscid_scatter_extract_f32,
        );
    }
    Ok(())
}

fn first_order_inviscid_scatter_extract_f32(
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

fn compute_interior_first_order_face_contribution_f32(
    face_idx: usize,
    interior: &[UnstructuredInteriorFaceF32],
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
) -> Result<Option<(InteriorInviscidScatterGeomF32, InviscidFluxF32)>> {
    let face = &interior[face_idx];
    if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
        return Ok(None);
    }
    if is_degenerate_volume_f32(face.owner_volume) || is_degenerate_volume_f32(face.neighbor_volume)
    {
        return Ok(None);
    }
    let flux = face_inviscid_flux_first_order_interior_soa_f32(
        face.owner,
        face.neighbor,
        params.primitives,
        face.normal,
        params.eos,
        params.config,
    )?;
    Ok(Some((
        InteriorInviscidScatterGeomF32 {
            owner: face.owner,
            neighbor: face.neighbor,
            owner_scale: -face.area / face.owner_volume,
            neighbor_scale: face.area / face.neighbor_volume,
        },
        flux,
    )))
}

pub(super) fn assemble_boundary_faces_first_order_f32(
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
) -> Result<()> {
    for bface in &params.mesh_cache.face_topology_f32.boundary {
        if bface.owner_rhs_scale == 0.0 {
            continue;
        }
        if is_degenerate_volume_f32(bface.owner_volume) {
            continue;
        }
        let ghost = params.ghosts.get_face(bface.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "边界面 FaceId({}) 缺少 ghost 状态",
                bface.face.index()
            ))
        })?;
        let flux = face_inviscid_flux_first_order_boundary_soa_f32(
            params.primitives,
            bface.owner,
            &ghost,
            bface.normal,
            params.eos,
            params.config,
            params.min_pressure,
        )?;
        scatter_fused_boundary_inviscid_face_f32(
            residual,
            bface.owner,
            -bface.area / bface.owner_volume,
            &flux,
        );
    }
    Ok(())
}
