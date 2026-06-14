//! 非结构无粘内面一阶 SIMD 批处理 f32（Roe / Hanel–Van Leer，`simd-fvm`）。

use crate::discretization::flux_config::FluxScheme;
#[cfg(all(feature = "simd-fvm", not(feature = "parallel-fvm")))]
use crate::discretization::inviscid_f32::scatter_fused_interior_inviscid_face_f32;
use crate::discretization::inviscid_f32::{InteriorInviscidScatterGeomF32, InviscidFluxF32};
use crate::discretization::unstructured_face_cache::UnstructuredFaceTopology;
use crate::discretization::unstructured_face_cache_f32::UnstructuredInteriorFaceF32;
use crate::discretization::viscous_boundary_f32::PrimitiveStateF32;
use crate::discretization::{
    InteriorFaceBatchStatic4, InteriorFaceBucketBatchLayout, InviscidFluxConfig, ReconstructionKind,
};
use crate::error::Result;
use crate::exec::scatter::{
    InviscidPairScatterF32, InviscidResidualMutF32, InviscidScatterOpF32,
    scatter_inviscid_pairs_f32,
};
use crate::field::{ConservedResidualT, PrimitiveFieldsT};

use super::InviscidAssemblyUnstructuredTypedParams;
use super::first_order_f32::compute_interior_first_order_face_contribution_f32;

/// 一阶 SIMD 批处理支持的通量格式。
#[derive(Clone, Copy)]
enum FirstOrderSimdSchemeF32 {
    Roe { entropy_fix: bool },
    HanelVanLeer,
}

/// 若配置为一阶 Roe / HVL 且 SIMD 路径已完整处理内面，返回 `Ok(true)`。
pub(super) fn try_assemble_interior_faces_f32(
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    topology: &UnstructuredFaceTopology,
) -> Result<bool> {
    let Some(scheme) = first_order_simd_scheme_f32(params.config) else {
        return Ok(false);
    };

    #[cfg(not(feature = "parallel-fvm"))]
    {
        use tracing::info_span;

        let _span = info_span!(
            "unstructured_inviscid_interior_flux_fused",
            path = "simd_batch4_f32",
            faces = topology.interior.len(),
        )
        .entered();
        for layout in &topology.interior_coloring.bucket_batch_layouts {
            accumulate_inviscid_bucket_batch4_f32_serial(
                layout, residual, params, topology, scheme,
            )?;
        }
        return Ok(true);
    }

    #[cfg(feature = "parallel-fvm")]
    {
        use tracing::info_span;

        let _span = info_span!(
            "unstructured_inviscid_interior_flux_fused",
            path = "simd_batch4_f32",
            faces = topology.interior.len(),
            colors = topology.interior_coloring.num_colors,
        )
        .entered();
        for layout in &topology.interior_coloring.bucket_batch_layouts {
            accumulate_inviscid_bucket_batch4_f32_parallel(
                layout, residual, params, topology, scheme,
            )?;
        }
        Ok(true)
    }
}

fn first_order_simd_scheme_f32(config: &InviscidFluxConfig) -> Option<FirstOrderSimdSchemeF32> {
    if config.reconstruction != ReconstructionKind::FirstOrder {
        return None;
    }
    match config.scheme {
        FluxScheme::Roe(roe_cfg) => Some(FirstOrderSimdSchemeF32::Roe {
            entropy_fix: roe_cfg.entropy_fix,
        }),
        FluxScheme::HanelVanLeer => Some(FirstOrderSimdSchemeF32::HanelVanLeer),
        _ => None,
    }
}

fn primitive_lane_f32(primitives: &PrimitiveFieldsT<f32>, cell: usize) -> PrimitiveStateF32 {
    PrimitiveStateF32 {
        density: primitives.density.values()[cell],
        velocity: [
            primitives.velocity_x.values()[cell],
            primitives.velocity_y.values()[cell],
            primitives.velocity_z.values()[cell],
        ],
        pressure: primitives.pressure.values()[cell],
        temperature: 0.0,
    }
}

fn interior_inviscid_batch4_f32(
    batch: &InteriorFaceBatchStatic4,
    primitives: &PrimitiveFieldsT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    scheme: FirstOrderSimdSchemeF32,
) -> Option<Vec<(InteriorInviscidScatterGeomF32, InviscidFluxF32)>> {
    if !batch.simd_eligible() {
        return None;
    }
    let left = [
        primitive_lane_f32(primitives, batch.owners[0]),
        primitive_lane_f32(primitives, batch.owners[1]),
        primitive_lane_f32(primitives, batch.owners[2]),
        primitive_lane_f32(primitives, batch.owners[3]),
    ];
    let right = [
        primitive_lane_f32(primitives, batch.neighbors[0]),
        primitive_lane_f32(primitives, batch.neighbors[1]),
        primitive_lane_f32(primitives, batch.neighbors[2]),
        primitive_lane_f32(primitives, batch.neighbors[3]),
    ];
    let left_ref = [&left[0], &left[1], &left[2], &left[3]];
    let right_ref = [&right[0], &right[1], &right[2], &right[3]];
    let normals = face_normals_f32(batch);
    let fluxes = match scheme {
        FirstOrderSimdSchemeF32::Roe { entropy_fix } => {
            use crate::discretization::roe::RoeFluxConfig;
            use crate::exec::cpu::face_inviscid_flux_first_order_roe_batch4_f32;

            let roe_cfg = RoeFluxConfig {
                entropy_fix,
                ..RoeFluxConfig::default()
            };
            face_inviscid_flux_first_order_roe_batch4_f32(
                left_ref, right_ref, normals, params.eos, &roe_cfg,
            )
        }
        FirstOrderSimdSchemeF32::HanelVanLeer => {
            use crate::exec::cpu::face_inviscid_flux_first_order_hanel_batch4_f32;

            face_inviscid_flux_first_order_hanel_batch4_f32(
                left_ref, right_ref, normals, params.eos,
            )
        }
    };
    let mut out = Vec::with_capacity(4);
    for (lane, flux) in fluxes.into_iter().enumerate() {
        let f = flux?;
        out.push((
            InteriorInviscidScatterGeomF32 {
                owner: batch.owners[lane],
                neighbor: batch.neighbors[lane],
                owner_scale: batch.owner_rhs_scale[lane] as f32,
                neighbor_scale: batch.neighbor_rhs_scale[lane] as f32,
            },
            f,
        ));
    }
    Some(out)
}

fn face_normals_f32(batch: &InteriorFaceBatchStatic4) -> [[f32; 3]; 4] {
    [
        [batch.nx[0] as f32, batch.ny[0] as f32, batch.nz[0] as f32],
        [batch.nx[1] as f32, batch.ny[1] as f32, batch.nz[1] as f32],
        [batch.nx[2] as f32, batch.ny[2] as f32, batch.nz[2] as f32],
        [batch.nx[3] as f32, batch.ny[3] as f32, batch.nz[3] as f32],
    ]
}

#[cfg(all(feature = "simd-fvm", not(feature = "parallel-fvm")))]
fn accumulate_inviscid_bucket_batch4_f32_serial(
    layout: &InteriorFaceBucketBatchLayout,
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    topology: &UnstructuredFaceTopology,
    scheme: FirstOrderSimdSchemeF32,
) -> Result<()> {
    let interior_f32 = &params.mesh_cache.face_topology_f32.interior;
    for batch in &layout.full_batches {
        if let Some(items) = interior_inviscid_batch4_f32(batch, params.primitives, params, scheme)
        {
            for (geom, flux) in items {
                scatter_fused_interior_inviscid_face_f32(residual, &geom, &flux);
            }
            continue;
        }
        for &face_idx in &batch.face_indices {
            if let Some((geom, flux)) = compute_interior_first_order_face_contribution_f32(
                face_idx,
                interior_f32.as_slice(),
                params,
            )? {
                scatter_fused_interior_inviscid_face_f32(residual, &geom, &flux);
            }
        }
    }
    for &face_idx in &layout.remainder {
        if let Some((geom, flux)) = compute_interior_first_order_face_contribution_f32(
            face_idx,
            interior_f32.as_slice(),
            params,
        )? {
            scatter_fused_interior_inviscid_face_f32(residual, &geom, &flux);
        }
    }
    let _ = topology;
    Ok(())
}

#[cfg(all(feature = "simd-fvm", feature = "parallel-fvm"))]
fn accumulate_inviscid_bucket_batch4_f32_parallel(
    layout: &InteriorFaceBucketBatchLayout,
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    topology: &UnstructuredFaceTopology,
    scheme: FirstOrderSimdSchemeF32,
) -> Result<()> {
    let interior_f32 = &params.mesh_cache.face_topology_f32.interior;
    let batch_parts =
        crate::exec::parallel::par_try_map_batches(&layout.full_batches, 128, |batch| {
            compute_inviscid_full_batch_to_vec_f32(batch, params, interior_f32, topology, scheme)
        })?;
    let remainder_contributions =
        crate::exec::parallel::par_try_map_face_indices(&layout.remainder, 1024, |face_idx| {
            compute_interior_first_order_face_contribution_f32(
                face_idx,
                interior_f32.as_slice(),
                params,
            )
        })?;
    let mut pairs = Vec::new();
    for part in batch_parts {
        pairs.extend(part.into_iter().flatten());
    }
    pairs.extend(remainder_contributions.into_iter().flatten());
    scatter_inviscid_pairs_f32(
        InviscidPairScatterF32 {
            ctx: params.exec,
            bucket_len: layout.num_faces(),
            pairs: &pairs,
            residual: InviscidResidualMutF32 {
                density: residual.density.values_mut(),
                mx: residual.momentum_x.values_mut(),
                my: residual.momentum_y.values_mut(),
                mz: residual.momentum_z.values_mut(),
                energy: residual.total_energy.values_mut(),
            },
        },
        |g, f| InviscidScatterOpF32 {
            owner: g.owner,
            neighbor: g.neighbor,
            owner_scale: g.owner_scale,
            neighbor_scale: g.neighbor_scale,
            mass: f.mass,
            momentum: f.momentum,
            energy: f.energy,
        },
    );
    Ok(())
}

#[cfg(all(feature = "simd-fvm", feature = "parallel-fvm"))]
fn compute_inviscid_full_batch_to_vec_f32(
    batch: &InteriorFaceBatchStatic4,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    interior_f32: &[UnstructuredInteriorFaceF32],
    topology: &UnstructuredFaceTopology,
    scheme: FirstOrderSimdSchemeF32,
) -> Result<Vec<Option<(InteriorInviscidScatterGeomF32, InviscidFluxF32)>>> {
    let _ = topology;
    if let Some(items) = interior_inviscid_batch4_f32(batch, params.primitives, params, scheme) {
        return Ok(items.into_iter().map(Some).collect());
    }
    batch
        .face_indices
        .iter()
        .map(|&face_idx| {
            compute_interior_first_order_face_contribution_f32(face_idx, interior_f32, params)
        })
        .collect()
}
