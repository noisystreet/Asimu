//! 粘性内面 scatter（串行 / 并行 atomic）。

use crate::exec::context::{ExecutionContext, ResolvedScatterMode};

use super::atomic::{ViscousResidualPtrs, scatter_viscous_op_atomic};
use super::contribution::{
    ViscousRangeScatter, ViscousResidualMut, ViscousScatterOp, ViscousValidSlotScatter,
};

#[inline]
fn scatter_viscous_op_serial(op: ViscousScatterOp, residual: &mut ViscousResidualMut<'_>) {
    residual.mx[op.owner] += op.owner_scale * op.flux_mx;
    residual.my[op.owner] += op.owner_scale * op.flux_my;
    residual.mz[op.owner] += op.owner_scale * op.flux_mz;
    residual.energy[op.owner] += op.owner_scale * op.flux_energy;
    residual.mx[op.neighbor] += op.neighbor_scale * op.flux_mx;
    residual.my[op.neighbor] += op.neighbor_scale * op.flux_my;
    residual.mz[op.neighbor] += op.neighbor_scale * op.flux_mz;
    residual.energy[op.neighbor] += op.neighbor_scale * op.flux_energy;
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

/// 按 `valid` 掩码 scatter 粘性内面（单 span）。
pub fn scatter_viscous_valid_slots<G, F>(
    scatter: ViscousValidSlotScatter<'_, G, F>,
    extract: impl Fn(&G, &F) -> ViscousScatterOp + Sync,
) where
    G: Sync,
    F: Sync,
{
    enter_scatter_span(scatter.ctx, scatter.bucket_len);
    let ViscousValidSlotScatter {
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
                scatter_viscous_op_serial(extract(&geoms[i], &fluxes[i]), &mut residual);
            }
        }
        return;
    }

    #[cfg(not(feature = "parallel-fvm"))]
    {
        for (i, &is_valid) in valid.iter().enumerate() {
            if is_valid {
                scatter_viscous_op_serial(extract(&geoms[i], &fluxes[i]), &mut residual);
            }
        }
        return;
    }

    #[cfg(feature = "parallel-fvm")]
    {
        use rayon::prelude::*;

        let ptrs = ViscousResidualPtrs::from_slices(
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
                scatter_viscous_op_atomic(op, ptrs);
            }
        });
    }
}

/// 按索引范围 scatter 粘性内面通量。
pub fn scatter_viscous_bucket_range<G, F>(
    scatter: ViscousRangeScatter<'_, G, F>,
    extract: impl Fn(&G, &F) -> ViscousScatterOp + Sync,
) where
    G: Sync,
    F: Sync,
{
    enter_scatter_span(scatter.ctx, scatter.bucket_len);
    let ViscousRangeScatter {
        ctx,
        bucket_len,
        geoms,
        fluxes,
        range,
        mut residual,
    } = scatter;

    if !bucket_uses_atomic_scatter(ctx, bucket_len) {
        for i in range.clone() {
            scatter_viscous_op_serial(extract(&geoms[i], &fluxes[i]), &mut residual);
        }
        return;
    }

    #[cfg(not(feature = "parallel-fvm"))]
    {
        for i in range {
            scatter_viscous_op_serial(extract(&geoms[i], &fluxes[i]), &mut residual);
        }
        return;
    }

    #[cfg(feature = "parallel-fvm")]
    {
        use rayon::prelude::*;

        let ptrs = ViscousResidualPtrs::from_slices(
            residual.mx,
            residual.my,
            residual.mz,
            residual.energy,
        );
        range.into_par_iter().for_each(|i| {
            let op = extract(&geoms[i], &fluxes[i]);
            // SAFETY: 着色桶内面无共享单元。
            unsafe {
                scatter_viscous_op_atomic(op, ptrs);
            }
        });
    }
}
