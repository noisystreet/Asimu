//! 网格规模指标：构造 [`ExecutionContext`](super::context::ExecutionContext) 时解析 scatter 模式。

/// 与 scatter `Auto` 解析相关的网格度量（init-time 一次计算）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshExecMetrics {
    pub num_cells: usize,
    pub interior_faces: usize,
    pub max_bucket_faces: usize,
}

impl MeshExecMetrics {
    #[must_use]
    pub const fn new(num_cells: usize, interior_faces: usize, max_bucket_faces: usize) -> Self {
        Self {
            num_cells,
            interior_faces,
            max_bucket_faces,
        }
    }

    /// 无内面（单元测试 / 占位）。
    #[must_use]
    pub const fn empty() -> Self {
        Self::new(0, 0, 0)
    }
}
