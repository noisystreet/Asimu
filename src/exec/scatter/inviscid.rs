//! 无粘内面 scatter（串行 / 并行 atomic）。

use crate::exec::context::{ExecutionContext, ResolvedScatterMode};

use super::atomic::{InviscidResidualPtrs, scatter_inviscid_op_atomic};
use super::contribution::{InviscidPairScatter, InviscidResidualMut, InviscidScatterOp};

#[inline]
fn scatter_inviscid_op_serial(op: InviscidScatterOp, residual: &mut InviscidResidualMut<'_>) {
    residual.density[op.owner] += op.owner_scale * op.mass;
    residual.mx[op.owner] += op.owner_scale * op.momentum[0];
    residual.my[op.owner] += op.owner_scale * op.momentum[1];
    residual.mz[op.owner] += op.owner_scale * op.momentum[2];
    residual.energy[op.owner] += op.owner_scale * op.energy;
    residual.density[op.neighbor] += op.neighbor_scale * op.mass;
    residual.mx[op.neighbor] += op.neighbor_scale * op.momentum[0];
    residual.my[op.neighbor] += op.neighbor_scale * op.momentum[1];
    residual.mz[op.neighbor] += op.neighbor_scale * op.momentum[2];
    residual.energy[op.neighbor] += op.neighbor_scale * op.energy;
}

fn bucket_uses_atomic_scatter(ctx: &ExecutionContext, bucket_len: usize) -> bool {
    matches!(
        ctx.resolved_scatter_mode(),
        ResolvedScatterMode::ParallelUnsafeAtomics
    ) && !ctx.bucket_uses_serial_scatter(bucket_len)
}

fn enter_scatter_span(_ctx: &ExecutionContext, _bucket_len: usize) {
    // 暂禁：每着色桶一次 info span；LU-SGS × 色数 × 步数会使 Chrome trace 体积过大。
    // let bucket_serial = ctx.bucket_uses_serial_scatter(bucket_len);
    // let mode = ctx.effective_scatter_mode_label(bucket_len);
    // let _span = info_span!(
    //     "exec_colored_bucket_scatter",
    //     mode,
    //     bucket_faces = bucket_len,
    //     bucket_serial,
    //     resolved = ?ctx.resolved_scatter_mode(),
    // )
    // .entered();
}

/// 按 `(geom, flux)` 对 scatter 无粘内面通量。
pub fn scatter_inviscid_pairs<G, F>(
    scatter: InviscidPairScatter<'_, G, F>,
    extract: impl Fn(&G, &F) -> InviscidScatterOp + Sync,
) where
    G: Sync,
    F: Sync,
{
    enter_scatter_span(scatter.ctx, scatter.bucket_len);
    let InviscidPairScatter {
        ctx,
        bucket_len,
        pairs,
        mut residual,
    } = scatter;

    if !bucket_uses_atomic_scatter(ctx, bucket_len) {
        for (geom, flux) in pairs {
            scatter_inviscid_op_serial(extract(geom, flux), &mut residual);
        }
        return;
    }

    #[cfg(not(feature = "parallel-fvm"))]
    {
        for (geom, flux) in pairs {
            scatter_inviscid_op_serial(extract(geom, flux), &mut residual);
        }
        return;
    }

    #[cfg(feature = "parallel-fvm")]
    {
        use rayon::prelude::*;

        let ptrs = InviscidResidualPtrs::from_slices(
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
                scatter_inviscid_op_atomic(op, ptrs);
            }
        });
    }
}
