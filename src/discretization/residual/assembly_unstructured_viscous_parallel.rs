//! 非结构粘性内面并行桶 flat buffer（P8′）。

use std::mem;

use crate::discretization::viscous::{
    InteriorViscousResidualMut, scatter_fused_interior_viscous_face,
};

#[cfg(feature = "simd-fvm")]
use crate::discretization::viscous::{InteriorViscousFaceFlux, InteriorViscousFaceGeom};

use super::{
    ViscousAssemblyUnstructuredParams, ViscousAssemblyUnstructuredScratch,
    interior_face_flux_contribution,
};

#[cfg(feature = "simd-fvm")]
use super::compute_viscous_batch4_into;

#[cfg(all(feature = "simd-fvm", feature = "parallel-fvm"))]
pub(super) fn scatter_parallel_bucket_slots(
    residual_mut: &mut InteriorViscousResidualMut<'_>,
    geoms: &[InteriorViscousFaceGeom],
    fluxes: &[InteriorViscousFaceFlux],
    batch_counts: &[u8],
    remainder_valid: &[bool],
    num_batches: usize,
    remainder_base: usize,
) {
    for (batch_idx, &count) in batch_counts.iter().enumerate().take(num_batches) {
        let base = batch_idx * 4;
        for lane in 0..count as usize {
            scatter_fused_interior_viscous_face(
                residual_mut,
                &geoms[base + lane],
                &fluxes[base + lane],
            );
        }
    }
    for (offset, valid) in remainder_valid.iter().enumerate() {
        if *valid {
            scatter_fused_interior_viscous_face(
                residual_mut,
                &geoms[remainder_base + offset],
                &fluxes[remainder_base + offset],
            );
        }
    }
}

#[cfg(all(feature = "simd-fvm", feature = "parallel-fvm"))]
pub(super) fn accumulate_viscous_bucket_batch4_fused(
    residual_mut: &mut InteriorViscousResidualMut<'_>,
    layout: &crate::discretization::InteriorFaceBucketBatchLayout,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
    constant: Option<(crate::core::Real, crate::core::Real)>,
) {
    use rayon::prelude::*;

    let num_batches = layout.full_batches.len();
    let num_remainder = layout.remainder.len();
    scratch.ensure_parallel_bucket_buffer(num_batches, num_remainder);

    let mut geoms = mem::take(&mut scratch.parallel_bucket_geoms);
    let mut fluxes = mem::take(&mut scratch.parallel_bucket_fluxes);
    let mut batch_counts = mem::take(&mut scratch.parallel_batch_counts);
    let mut remainder_valid = mem::take(&mut scratch.parallel_slot_valid);

    geoms
        .par_chunks_mut(4)
        .zip(fluxes.par_chunks_mut(4))
        .zip(batch_counts.par_iter_mut())
        .zip(layout.full_batches.par_iter())
        .with_min_len(128)
        .for_each(|(((geom_chunk, flux_chunk), count), batch)| {
            *count = compute_viscous_batch4_into(
                batch, params, scratch, constant, geom_chunk, flux_chunk,
            );
        });

    let remainder_base = num_batches * 4;
    geoms[remainder_base..]
        .par_iter_mut()
        .zip(fluxes[remainder_base..].par_iter_mut())
        .zip(remainder_valid.par_iter_mut())
        .zip(layout.remainder.par_iter())
        .with_min_len(1024)
        .for_each(|(((geom, flux), valid), &face_idx)| {
            if let Some((g, f)) =
                interior_face_flux_contribution(face_idx, params, scratch, constant)
            {
                *geom = g;
                *flux = f;
                *valid = true;
            } else {
                *valid = false;
            }
        });

    scatter_parallel_bucket_slots(
        residual_mut,
        &geoms,
        &fluxes,
        &batch_counts,
        &remainder_valid,
        num_batches,
        remainder_base,
    );

    scratch.parallel_bucket_geoms = geoms;
    scratch.parallel_bucket_fluxes = fluxes;
    scratch.parallel_batch_counts = batch_counts;
    scratch.parallel_slot_valid = remainder_valid;
}

#[cfg(all(feature = "parallel-fvm", not(feature = "simd-fvm")))]
pub(super) fn accumulate_viscous_color_bucket_fused(
    residual_mut: &mut InteriorViscousResidualMut<'_>,
    bucket: &[usize],
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
    constant: Option<(crate::core::Real, crate::core::Real)>,
) {
    use rayon::prelude::*;

    let n = bucket.len();
    scratch.ensure_parallel_bucket_buffer(0, n);
    let mut geoms = mem::take(&mut scratch.parallel_bucket_geoms);
    let mut fluxes = mem::take(&mut scratch.parallel_bucket_fluxes);
    let mut slot_valid = mem::take(&mut scratch.parallel_slot_valid);
    let batch_counts = mem::take(&mut scratch.parallel_batch_counts);

    geoms
        .par_iter_mut()
        .zip(fluxes.par_iter_mut())
        .zip(slot_valid.par_iter_mut())
        .zip(bucket.par_iter())
        .with_min_len(1024)
        .for_each(|(((geom, flux), valid), &face_idx)| {
            if let Some((g, f)) =
                interior_face_flux_contribution(face_idx, params, scratch, constant)
            {
                *geom = g;
                *flux = f;
                *valid = true;
            } else {
                *valid = false;
            }
        });

    for offset in 0..n {
        if slot_valid[offset] {
            scatter_fused_interior_viscous_face(residual_mut, &geoms[offset], &fluxes[offset]);
        }
    }

    scratch.parallel_bucket_geoms = geoms;
    scratch.parallel_bucket_fluxes = fluxes;
    scratch.parallel_slot_valid = slot_valid;
    scratch.parallel_batch_counts = batch_counts;
}
