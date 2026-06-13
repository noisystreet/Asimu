//! 粘性内面 `f32` scatter（串行 / 并行 atomic；ADR 0016 P5）。

use crate::exec::context::{ExecutionContext, ResolvedScatterMode};

use super::atomic::{ViscousResidualPtrsF32, scatter_viscous_op_atomic_f32};
use super::contribution::{ViscousResidualMutF32, ViscousScatterOp, ViscousValidSlotScatterF32};
use super::span::enter_scatter_span;

#[inline]
fn scatter_viscous_op_serial_f32(op: ViscousScatterOp, residual: &mut ViscousResidualMutF32<'_>) {
    let owner_scale = op.owner_scale as f32;
    let neighbor_scale = op.neighbor_scale as f32;
    residual.mx[op.owner] += owner_scale * op.flux_mx as f32;
    residual.my[op.owner] += owner_scale * op.flux_my as f32;
    residual.mz[op.owner] += owner_scale * op.flux_mz as f32;
    residual.energy[op.owner] += owner_scale * op.flux_energy as f32;
    residual.mx[op.neighbor] += neighbor_scale * op.flux_mx as f32;
    residual.my[op.neighbor] += neighbor_scale * op.flux_my as f32;
    residual.mz[op.neighbor] += neighbor_scale * op.flux_mz as f32;
    residual.energy[op.neighbor] += neighbor_scale * op.flux_energy as f32;
}

fn bucket_uses_atomic_scatter(ctx: &ExecutionContext, bucket_len: usize) -> bool {
    matches!(
        ctx.resolved_scatter_mode(),
        ResolvedScatterMode::ParallelUnsafeAtomics
    ) && !ctx.bucket_uses_serial_scatter(bucket_len)
}

/// 按 `valid` 掩码 scatter 粘性内面至 `f32` 残差。
pub fn scatter_viscous_valid_slots_f32<G, F>(
    scatter: ViscousValidSlotScatterF32<'_, G, F>,
    extract: impl Fn(&G, &F) -> ViscousScatterOp + Sync,
) where
    G: Sync,
    F: Sync,
{
    enter_scatter_span(scatter.ctx, scatter.bucket_len);
    let ViscousValidSlotScatterF32 {
        ctx,
        bucket_len,
        geoms,
        fluxes,
        valid,
        mut residual,
    } = scatter;

    if !bucket_uses_atomic_scatter(ctx, bucket_len) {
        for (i, &is_valid) in valid.iter().enumerate() {
            if is_valid {
                scatter_viscous_op_serial_f32(extract(&geoms[i], &fluxes[i]), &mut residual);
            }
        }
        return;
    }

    #[cfg(not(feature = "parallel-fvm"))]
    {
        for (i, &is_valid) in valid.iter().enumerate() {
            if is_valid {
                scatter_viscous_op_serial_f32(extract(&geoms[i], &fluxes[i]), &mut residual);
            }
        }
        return;
    }

    #[cfg(feature = "parallel-fvm")]
    {
        use rayon::prelude::*;

        let ptrs = ViscousResidualPtrsF32::from_slices(
            residual.mx,
            residual.my,
            residual.mz,
            residual.energy,
        );
        valid.par_iter().enumerate().for_each(|(i, &is_valid)| {
            if !is_valid {
                return;
            }
            let op = extract(&geoms[i], &fluxes[i]);
            // SAFETY: 着色桶内面无共享单元。
            unsafe {
                scatter_viscous_op_atomic_f32(op, ptrs);
            }
        });
    }
}
