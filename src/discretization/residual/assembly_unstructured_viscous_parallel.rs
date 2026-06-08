//! 非结构粘性内面并行桶 flat buffer（P8′ → E2：`ExecScratch`）。

use crate::discretization::viscous::{InteriorViscousFaceFlux, InteriorViscousResidualMut};
use crate::exec::scatter::{
    ViscousResidualMut, ViscousScatterOp, ViscousValidSlotScatter, scatter_viscous_valid_slots,
};

use super::{
    ViscousAssemblyUnstructuredParams, ViscousAssemblyUnstructuredScratch,
    interior_face_flux_contribution,
};

#[cfg(feature = "simd-fvm")]
use super::compute_viscous_batch4_into;

#[inline]
fn viscous_scatter_extract(
    g: &crate::discretization::viscous::InteriorViscousFaceGeom,
    f: &InteriorViscousFaceFlux,
) -> ViscousScatterOp {
    ViscousScatterOp {
        owner: g.owner,
        neighbor: g.neighbor,
        owner_scale: g.owner_scale,
        neighbor_scale: g.neighbor_scale,
        flux_mx: f.mx,
        flux_my: f.my,
        flux_mz: f.mz,
        flux_energy: f.energy,
    }
}

fn viscous_residual_mut_slices<'a>(
    residual_mut: &'a mut InteriorViscousResidualMut<'_>,
) -> ViscousResidualMut<'a> {
    ViscousResidualMut {
        mx: residual_mut.mx,
        my: residual_mut.my,
        mz: residual_mut.mz,
        energy: residual_mut.energy,
    }
}

#[cfg(all(feature = "simd-fvm", feature = "parallel-fvm"))]
pub(super) fn accumulate_viscous_bucket_batch4_fused(
    residual_mut: &mut InteriorViscousResidualMut<'_>,
    layout: &crate::discretization::InteriorFaceBucketBatchLayout,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(crate::core::Real, crate::core::Real)>,
    exec: &mut crate::exec::ExecutionContext,
) {
    let num_batches = layout.full_batches.len();
    let num_remainder = layout.remainder.len();
    let mut ws = exec.scratch_mut().colored_viscous_mut().take_working_set();
    ws.ensure_bucket_layout(num_batches, num_remainder);

    crate::exec::parallel::par_for_each_viscous_batch4_chunks(
        &mut ws.geoms,
        &mut ws.fluxes,
        &mut ws.batch_counts,
        &layout.full_batches,
        128,
        |geom_chunk, flux_chunk, batch| {
            compute_viscous_batch4_into(batch, params, scratch, constant, geom_chunk, flux_chunk)
        },
    );

    let remainder_base = num_batches * 4;
    crate::exec::parallel::par_for_each_viscous_remainder(
        &mut ws.geoms[remainder_base..],
        &mut ws.fluxes[remainder_base..],
        &mut ws.slot_valid[remainder_base..],
        &layout.remainder,
        1024,
        |face_idx, geom, flux, valid| {
            if let Some((g, f)) =
                interior_face_flux_contribution(face_idx, params, scratch, constant)
            {
                *geom = g;
                *flux = f;
                *valid = true;
            } else {
                *valid = false;
            }
        },
    );

    ws.fill_batch_slot_valid();
    scatter_viscous_valid_slots(
        ViscousValidSlotScatter {
            ctx: exec,
            bucket_len: layout.num_faces(),
            geoms: &ws.geoms,
            fluxes: &ws.fluxes,
            valid: &ws.slot_valid,
            residual: viscous_residual_mut_slices(residual_mut),
        },
        viscous_scatter_extract,
    );

    exec.scratch_mut()
        .colored_viscous_mut()
        .restore_working_set(ws);
}

#[cfg(all(feature = "parallel-fvm", not(feature = "simd-fvm")))]
pub(super) fn accumulate_viscous_color_bucket_fused(
    residual_mut: &mut InteriorViscousResidualMut<'_>,
    bucket: &[usize],
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(crate::core::Real, crate::core::Real)>,
    exec: &mut crate::exec::ExecutionContext,
) {
    let n = bucket.len();
    let mut ws = exec.scratch_mut().colored_viscous_mut().take_working_set();
    ws.ensure_face_slots(n);

    crate::exec::parallel::par_for_each_viscous_face_slots(
        &mut ws.geoms,
        &mut ws.fluxes,
        &mut ws.slot_valid,
        bucket,
        1024,
        |face_idx, geom, flux, valid| {
            if let Some((g, f)) =
                interior_face_flux_contribution(face_idx, params, scratch, constant)
            {
                *geom = g;
                *flux = f;
                *valid = true;
            } else {
                *valid = false;
            }
        },
    );

    scatter_viscous_valid_slots(
        ViscousValidSlotScatter {
            ctx: exec,
            bucket_len: n,
            geoms: &ws.geoms,
            fluxes: &ws.fluxes,
            valid: &ws.slot_valid,
            residual: viscous_residual_mut_slices(residual_mut),
        },
        viscous_scatter_extract,
    );

    exec.scratch_mut()
        .colored_viscous_mut()
        .restore_working_set(ws);
}
