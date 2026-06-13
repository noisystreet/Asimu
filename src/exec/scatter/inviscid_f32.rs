//! 无粘内面 `f32` scatter（串行 / 并行 atomic；ADR 0016 P5）。

use crate::exec::context::{ExecutionContext, ResolvedScatterMode};

use super::atomic::{InviscidResidualPtrsF32, scatter_inviscid_op_atomic_f32};
use super::contribution::{InviscidPairScatterF32, InviscidResidualMutF32, InviscidScatterOp};
use super::span::enter_scatter_span;

#[inline]
fn scatter_inviscid_op_serial_f32(
    op: InviscidScatterOp,
    residual: &mut InviscidResidualMutF32<'_>,
) {
    let owner_scale = op.owner_scale as f32;
    let neighbor_scale = op.neighbor_scale as f32;
    residual.density[op.owner] += owner_scale * op.mass as f32;
    residual.mx[op.owner] += owner_scale * op.momentum[0] as f32;
    residual.my[op.owner] += owner_scale * op.momentum[1] as f32;
    residual.mz[op.owner] += owner_scale * op.momentum[2] as f32;
    residual.energy[op.owner] += owner_scale * op.energy as f32;
    residual.density[op.neighbor] += neighbor_scale * op.mass as f32;
    residual.mx[op.neighbor] += neighbor_scale * op.momentum[0] as f32;
    residual.my[op.neighbor] += neighbor_scale * op.momentum[1] as f32;
    residual.mz[op.neighbor] += neighbor_scale * op.momentum[2] as f32;
    residual.energy[op.neighbor] += neighbor_scale * op.energy as f32;
}

fn bucket_uses_atomic_scatter(ctx: &ExecutionContext, bucket_len: usize) -> bool {
    matches!(
        ctx.resolved_scatter_mode(),
        ResolvedScatterMode::ParallelUnsafeAtomics
    ) && !ctx.bucket_uses_serial_scatter(bucket_len)
}

/// 按 `(geom, flux)` 对 scatter 无粘内面通量至 `f32` 残差。
pub fn scatter_inviscid_pairs_f32<G, F>(
    scatter: InviscidPairScatterF32<'_, G, F>,
    extract: impl Fn(&G, &F) -> InviscidScatterOp + Sync,
) where
    G: Sync,
    F: Sync,
{
    enter_scatter_span(scatter.ctx, scatter.bucket_len);
    let InviscidPairScatterF32 {
        ctx,
        bucket_len,
        pairs,
        mut residual,
    } = scatter;

    if !bucket_uses_atomic_scatter(ctx, bucket_len) {
        for (geom, flux) in pairs {
            scatter_inviscid_op_serial_f32(extract(geom, flux), &mut residual);
        }
        return;
    }

    #[cfg(not(feature = "parallel-fvm"))]
    {
        for (geom, flux) in pairs {
            scatter_inviscid_op_serial_f32(extract(geom, flux), &mut residual);
        }
        return;
    }

    #[cfg(feature = "parallel-fvm")]
    {
        use rayon::prelude::*;

        let ptrs = InviscidResidualPtrsF32::from_slices(
            residual.density,
            residual.mx,
            residual.my,
            residual.mz,
            residual.energy,
        );
        pairs.par_iter().for_each(|(geom, flux)| {
            let op = extract(geom, flux);
            // SAFETY: 着色桶内面无共享单元。
            unsafe {
                scatter_inviscid_op_atomic_f32(op, ptrs);
            }
        });
    }
}
