//! 非结构无粘内面一阶 SIMD 批处理（Roe / Hanel–Van Leer，`simd-fvm`）。

use crate::discretization::flux_config::FluxScheme;
use crate::discretization::{
    InteriorFaceBatchStatic4, InteriorFaceBucketBatchLayout, InviscidFlux, InviscidFluxConfig,
    ReconstructionKind, UnstructuredFaceTopology,
};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual};
use tracing::info_span;

use crate::discretization::inviscid::{
    interior_inviscid_residual_mut, scatter_fused_interior_inviscid_face,
};

#[cfg(not(feature = "parallel-fvm"))]
use super::accumulate_one_interior_inviscid_face_fused;
use super::{
    InteriorInviscidScatterGeom, InviscidAssemblyUnstructuredParams,
    compute_interior_inviscid_face_contribution,
};

type InviscidBatchContribution = Option<(InteriorInviscidScatterGeom, InviscidFlux)>;
type InviscidBatchPartsVec = Vec<Vec<InviscidBatchContribution>>;
type InviscidBatchPartsResult = Result<InviscidBatchPartsVec>;

/// 一阶 SIMD 批处理支持的通量格式。
#[derive(Clone, Copy)]
enum FirstOrderSimdScheme {
    Roe { entropy_fix: bool },
    HanelVanLeer,
}

/// 若配置为一阶 Roe / HVL 且 SIMD 路径已完整处理内面，返回 `Ok(true)`。
pub(super) fn try_assemble_interior_faces_cached(
    residual: &mut ConservedResidual,
    fields: &ConservedFields,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<bool> {
    let Some(scheme) = first_order_simd_scheme(params.config) else {
        return Ok(false);
    };

    #[cfg(not(feature = "parallel-fvm"))]
    {
        let _span = info_span!(
            "unstructured_inviscid_interior_flux_fused",
            path = "simd_batch4",
            faces = topology.interior.len(),
        )
        .entered();
        let mut residual_mut = interior_inviscid_residual_mut(residual);
        for layout in &topology.interior_coloring.bucket_batch_layouts {
            accumulate_inviscid_bucket_batch4(
                layout,
                fields,
                &mut residual_mut,
                params,
                topology,
                scheme,
            )?;
        }
        return Ok(true);
    }

    #[cfg(feature = "parallel-fvm")]
    {
        let bucket_results = {
            let _span = info_span!(
                "unstructured_inviscid_interior_flux_compute",
                path = "simd_batch4",
                faces = topology.interior.len(),
                colors = topology.interior_coloring.num_colors,
            )
            .entered();
            // 与 `InteriorFaceColoring::par_map_buckets` 一致：各色 bucket 串行、桶内并行。
            topology
                .interior_coloring
                .bucket_batch_layouts
                .iter()
                .map(|layout| {
                    compute_inviscid_bucket_batch4_to_vec(layout, fields, params, topology, scheme)
                })
                .collect::<Result<Vec<_>>>()?
        };
        {
            let _span = info_span!(
                "unstructured_inviscid_interior_flux_scatter",
                path = "simd_batch4",
                buckets = topology.interior_coloring.bucket_batch_layouts.len(),
            )
            .entered();
            let mut residual_mut = interior_inviscid_residual_mut(residual);
            for bucket in bucket_results {
                for (geom, flux) in bucket.into_iter().flatten() {
                    scatter_fused_interior_inviscid_face(&mut residual_mut, &geom, &flux);
                }
            }
        }
        Ok(true)
    }
}

fn first_order_simd_scheme(config: &InviscidFluxConfig) -> Option<FirstOrderSimdScheme> {
    if config.reconstruction != ReconstructionKind::FirstOrder {
        return None;
    }
    match config.scheme {
        FluxScheme::Roe(roe_cfg) => Some(FirstOrderSimdScheme::Roe {
            entropy_fix: roe_cfg.entropy_fix,
        }),
        FluxScheme::HanelVanLeer => Some(FirstOrderSimdScheme::HanelVanLeer),
        _ => None,
    }
}

fn flux5_as_inviscid(f: crate::exec::cpu::InviscidFlux5) -> InviscidFlux {
    InviscidFlux {
        mass: f.mass,
        momentum: f.momentum,
        energy: f.energy,
    }
}

fn interior_inviscid_batch4(
    batch: &InteriorFaceBatchStatic4,
    fields: &ConservedFields,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    scheme: FirstOrderSimdScheme,
) -> Result<Option<Vec<(InteriorInviscidScatterGeom, InviscidFlux)>>> {
    if !batch.simd_eligible() {
        return Ok(None);
    }

    let left_cons = [
        fields.cell_state(batch.owners[0])?,
        fields.cell_state(batch.owners[1])?,
        fields.cell_state(batch.owners[2])?,
        fields.cell_state(batch.owners[3])?,
    ];
    let right_cons = [
        fields.cell_state(batch.neighbors[0])?,
        fields.cell_state(batch.neighbors[1])?,
        fields.cell_state(batch.neighbors[2])?,
        fields.cell_state(batch.neighbors[3])?,
    ];
    let normals = batch.normals();

    let flux5 = match scheme {
        FirstOrderSimdScheme::Roe { entropy_fix } => {
            use crate::exec::cpu::face_inviscid_flux_first_order_roe_batch4;

            let left_prim = [
                params.primitives.cell_primitive(batch.owners[0]),
                params.primitives.cell_primitive(batch.owners[1]),
                params.primitives.cell_primitive(batch.owners[2]),
                params.primitives.cell_primitive(batch.owners[3]),
            ];
            let right_prim = [
                params.primitives.cell_primitive(batch.neighbors[0]),
                params.primitives.cell_primitive(batch.neighbors[1]),
                params.primitives.cell_primitive(batch.neighbors[2]),
                params.primitives.cell_primitive(batch.neighbors[3]),
            ];
            face_inviscid_flux_first_order_roe_batch4(
                [&left_prim[0], &left_prim[1], &left_prim[2], &left_prim[3]],
                [
                    &right_prim[0],
                    &right_prim[1],
                    &right_prim[2],
                    &right_prim[3],
                ],
                [&left_cons[0], &left_cons[1], &left_cons[2], &left_cons[3]],
                [
                    &right_cons[0],
                    &right_cons[1],
                    &right_cons[2],
                    &right_cons[3],
                ],
                normals,
                params.eos,
                entropy_fix,
            )
        }
        FirstOrderSimdScheme::HanelVanLeer => {
            use crate::exec::cpu::face_inviscid_flux_first_order_hanel_batch4;

            face_inviscid_flux_first_order_hanel_batch4(
                [&left_cons[0], &left_cons[1], &left_cons[2], &left_cons[3]],
                [
                    &right_cons[0],
                    &right_cons[1],
                    &right_cons[2],
                    &right_cons[3],
                ],
                normals,
                params.eos.gamma,
            )
        }
    };

    let mut out = Vec::with_capacity(4);
    for (lane, f5) in flux5.into_iter().enumerate() {
        let Some(f5) = f5 else {
            return Ok(None);
        };
        out.push((
            InteriorInviscidScatterGeom {
                owner: batch.owners[lane],
                neighbor: batch.neighbors[lane],
                owner_scale: batch.owner_rhs_scale[lane],
                neighbor_scale: batch.neighbor_rhs_scale[lane],
            },
            flux5_as_inviscid(f5),
        ));
    }
    Ok(Some(out))
}

#[cfg(all(feature = "simd-fvm", not(feature = "parallel-fvm")))]
fn accumulate_inviscid_bucket_batch4(
    layout: &InteriorFaceBucketBatchLayout,
    fields: &ConservedFields,
    residual_mut: &mut InteriorInviscidResidualMut<'_>,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
    scheme: FirstOrderSimdScheme,
) -> Result<()> {
    for batch in &layout.full_batches {
        if let Some(items) = interior_inviscid_batch4(batch, fields, params, scheme)? {
            for (geom, flux) in items {
                scatter_fused_interior_inviscid_face(residual_mut, &geom, &flux);
            }
            continue;
        }
        for &face_idx in &batch.face_indices {
            accumulate_one_interior_inviscid_face_fused(face_idx, residual_mut, params, topology)?;
        }
    }
    for &face_idx in &layout.remainder {
        accumulate_one_interior_inviscid_face_fused(face_idx, residual_mut, params, topology)?;
    }
    Ok(())
}

#[cfg(all(feature = "simd-fvm", feature = "parallel-fvm"))]
fn compute_inviscid_bucket_batch4_to_vec(
    layout: &InteriorFaceBucketBatchLayout,
    fields: &ConservedFields,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
    scheme: FirstOrderSimdScheme,
) -> Result<Vec<Option<(InteriorInviscidScatterGeom, InviscidFlux)>>> {
    use rayon::prelude::*;

    let mut out = Vec::with_capacity(layout.num_faces());
    let batch_parts: InviscidBatchPartsResult = layout
        .full_batches
        .par_iter()
        .with_min_len(128)
        .map(|batch| compute_inviscid_full_batch_to_vec(batch, fields, params, topology, scheme))
        .collect();
    for part in batch_parts? {
        out.extend(part);
    }
    let remainder: Result<Vec<_>> = layout
        .remainder
        .par_iter()
        .with_min_len(1024)
        .map(|&face_idx| compute_interior_inviscid_face_contribution(face_idx, params, topology))
        .collect();
    out.extend(remainder?);
    Ok(out)
}

#[cfg(all(feature = "simd-fvm", feature = "parallel-fvm"))]
fn compute_inviscid_full_batch_to_vec(
    batch: &InteriorFaceBatchStatic4,
    fields: &ConservedFields,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
    scheme: FirstOrderSimdScheme,
) -> Result<Vec<Option<(InteriorInviscidScatterGeom, InviscidFlux)>>> {
    if let Some(items) = interior_inviscid_batch4(batch, fields, params, scheme)? {
        return Ok(items.into_iter().map(Some).collect());
    }
    batch
        .face_indices
        .iter()
        .map(|&face_idx| compute_interior_inviscid_face_contribution(face_idx, params, topology))
        .collect()
}
