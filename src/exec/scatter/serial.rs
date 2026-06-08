//! 串行 scatter 回退（无 `parallel-fvm` 或显式 `Serial` / 桶级降级）。

use crate::exec::context::ExecutionContext;

/// 在 exec scatter span 内执行单着色桶的串行 scatter（边界 / 非结构化路径回退）。
pub fn run_bucket_scatter(_ctx: &ExecutionContext, _bucket_len: usize, scatter: impl FnOnce()) {
    // 暂禁 exec_colored_bucket_scatter span（见 inviscid/viscous `enter_scatter_span`）。
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
    scatter();
}
