//! 着色桶 scatter 可观测性（ADR 0013 §9）。

use tracing::trace_span;

use crate::exec::context::ExecutionContext;

/// 每着色桶 scatter 一次 `trace` span（Chrome trace 用 `asimu::exec::scatter=trace` 启用）。
pub(super) fn enter_scatter_span(ctx: &ExecutionContext, bucket_len: usize) {
    #[cfg(test)]
    ctx.record_scatter_invocation();
    let bucket_serial = ctx.bucket_uses_serial_scatter(bucket_len);
    let mode = ctx.effective_scatter_mode_label(bucket_len);
    let _span = trace_span!(
        "exec_colored_bucket_scatter",
        mode,
        bucket_faces = bucket_len,
        bucket_serial,
        resolved = ?ctx.resolved_scatter_mode(),
    )
    .entered();
}
