//! 非结构无粘内面 Roe 一阶 SIMD 批处理（`simd-fvm`）。

use crate::discretization::flux_config::FluxScheme;
use crate::discretization::{
    InteriorFaceBatchStatic4, InteriorFaceBucketBatchLayout, InviscidFlux, InviscidFluxConfig,
    ReconstructionKind, UnstructuredFaceTopology,
};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual};

#[cfg(not(feature = "parallel-fvm"))]
use super::accumulate_one_interior_inviscid_face;
use super::{
    InteriorInviscidScatterGeom, InviscidAssemblyUnstructuredParams,
    compute_interior_inviscid_face_contribution, scatter_interior_inviscid_face,
};

/// 若配置为 Roe 一阶且 SIMD 路径已完整处理内面，返回 `Ok(true)`。
pub(super) fn try_assemble_interior_faces_cached(
    residual: &mut ConservedResidual,
    fields: &ConservedFields,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<bool> {
    let Some(entropy_fix) = roe_first_order_simd_entropy_fix(params.config) else {
        return Ok(false);
    };

    #[cfg(not(feature = "parallel-fvm"))]
    {
        for layout in &topology.interior_coloring.bucket_batch_layouts {
            accumulate_inviscid_bucket_roe_batch4(
                layout,
                fields,
                residual,
                params,
                topology,
                entropy_fix,
            )?;
        }
        return Ok(true);
    }

    #[cfg(feature = "parallel-fvm")]
    {
        use rayon::prelude::*;
        let bucket_results = topology
            .interior_coloring
            .bucket_batch_layouts
            .par_iter()
            .map(|layout| {
                compute_inviscid_bucket_roe_batch4_to_vec(
                    layout,
                    fields,
                    params,
                    topology,
                    entropy_fix,
                )
            })
            .collect::<Vec<_>>();
        for bucket in bucket_results {
            for item in bucket? {
                if let Some((geom, flux)) = item {
                    scatter_interior_inviscid_face(residual, &geom, &flux)?;
                }
            }
        }
        Ok(true)
    }
}

fn roe_first_order_simd_entropy_fix(config: &InviscidFluxConfig) -> Option<bool> {
    if config.reconstruction != ReconstructionKind::FirstOrder {
        return None;
    }
    match config.scheme {
        FluxScheme::Roe(roe_cfg) => Some(roe_cfg.entropy_fix),
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

fn interior_inviscid_roe_batch4(
    batch: &InteriorFaceBatchStatic4,
    fields: &ConservedFields,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    entropy_fix: bool,
) -> Result<Option<Vec<(InteriorInviscidScatterGeom, InviscidFlux)>>> {
    use crate::exec::cpu::face_inviscid_flux_first_order_roe_batch4;

    if !batch.simd_eligible() {
        return Ok(None);
    }

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

    let flux5 = face_inviscid_flux_first_order_roe_batch4(
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
    );

    let mut out = Vec::with_capacity(4);
    for lane in 0..4 {
        let Some(f5) = flux5[lane] else {
            return Ok(None);
        };
        out.push((
            InteriorInviscidScatterGeom {
                owner: batch.owners[lane],
                neighbor: batch.neighbors[lane],
                area: batch.area[lane],
                owner_volume: batch.owner_volume[lane],
                neighbor_volume: batch.neighbor_volume[lane],
            },
            flux5_as_inviscid(f5),
        ));
    }
    Ok(Some(out))
}

#[cfg(all(feature = "simd-fvm", not(feature = "parallel-fvm")))]
fn accumulate_inviscid_bucket_roe_batch4(
    layout: &InteriorFaceBucketBatchLayout,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
    entropy_fix: bool,
) -> Result<()> {
    for batch in &layout.full_batches {
        if let Some(items) = interior_inviscid_roe_batch4(batch, fields, params, entropy_fix)? {
            for (geom, flux) in items {
                scatter_interior_inviscid_face(residual, &geom, &flux)?;
            }
            continue;
        }
        for &face_idx in &batch.face_indices {
            accumulate_one_interior_inviscid_face(face_idx, residual, params, topology)?;
        }
    }
    for &face_idx in &layout.remainder {
        accumulate_one_interior_inviscid_face(face_idx, residual, params, topology)?;
    }
    Ok(())
}

fn compute_inviscid_bucket_roe_batch4_to_vec(
    layout: &InteriorFaceBucketBatchLayout,
    fields: &ConservedFields,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
    entropy_fix: bool,
) -> Result<Vec<Option<(InteriorInviscidScatterGeom, InviscidFlux)>>> {
    let mut out = Vec::with_capacity(layout.num_faces());
    for batch in &layout.full_batches {
        if let Some(items) = interior_inviscid_roe_batch4(batch, fields, params, entropy_fix)? {
            out.extend(items.into_iter().map(Some));
            continue;
        }
        for &face_idx in &batch.face_indices {
            out.push(compute_interior_inviscid_face_contribution(
                face_idx, params, topology,
            )?);
        }
    }
    for &face_idx in &layout.remainder {
        out.push(compute_interior_inviscid_face_contribution(
            face_idx, params, topology,
        )?);
    }
    Ok(out)
}
