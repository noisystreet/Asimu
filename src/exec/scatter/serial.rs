//! 串行 scatter 回退（无 `parallel-fvm` 或显式 `Serial` / 桶级降级）。

use crate::exec::context::ExecutionContext;

use super::span::enter_scatter_span;

/// 在 exec scatter span 内执行单着色桶的串行 scatter（边界 / 非结构化路径回退）。
pub fn run_bucket_scatter(ctx: &ExecutionContext, bucket_len: usize, scatter: impl FnOnce()) {
    enter_scatter_span(ctx, bucket_len);
    scatter();
}
