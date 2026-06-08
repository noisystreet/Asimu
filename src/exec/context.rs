//! [`ExecutionContext`](ExecutionContext) 与 scatter 模式解析（ADR 0013）。

use tracing::info;

use crate::error::{AsimuError, Result};

use super::metrics::MeshExecMetrics;
use super::scratch::ExecScratch;

/// `Auto` 解析为并行 atomic scatter 的内面数下限（§2.4）。
pub const EXEC_SCATTER_PARALLEL_MIN_FACES: usize = 65_536;

/// 执行后端；E0 仅 CPU 路径。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecBackend {
    CpuScalar,
    CpuParallel,
}

/// 用户配置的 scatter 策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterMode {
    Auto,
    Serial,
    ParallelUnsafeAtomics,
}

/// 构造 [`ExecutionContext`] 后实际使用的 scatter 模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedScatterMode {
    Serial,
    ParallelUnsafeAtomics,
}

/// 算例级 exec 配置（Parse → Validate 后只读）。
#[derive(Debug, Clone, PartialEq)]
pub struct ExecConfig {
    pub backend: ExecBackend,
    pub parallel_min_len: usize,
    pub scatter_mode: ScatterMode,
    pub scatter_parallel_min_faces: usize,
}

impl Default for ExecConfig {
    fn default() -> Self {
        Self {
            backend: if cfg!(feature = "parallel-fvm") {
                ExecBackend::CpuParallel
            } else {
                ExecBackend::CpuScalar
            },
            parallel_min_len: 1024,
            scatter_mode: ScatterMode::Auto,
            scatter_parallel_min_faces: EXEC_SCATTER_PARALLEL_MIN_FACES,
        }
    }
}

/// 执行上下文：backend、已解析 scatter 模式与步间 scratch。
pub struct ExecutionContext {
    backend: ExecBackend,
    requested_scatter_mode: ScatterMode,
    resolved_scatter_mode: ResolvedScatterMode,
    parallel_min_len: usize,
    metrics: MeshExecMetrics,
    scratch: ExecScratch,
}

impl ExecutionContext {
    #[must_use]
    pub fn new(config: ExecConfig, metrics: MeshExecMetrics) -> Self {
        let resolved_scatter_mode = resolve_scatter_mode(&config, &metrics, config.backend);
        info!(
            mode = scatter_mode_label(resolved_scatter_mode),
            reason = scatter_resolve_reason(config.scatter_mode),
            interior_faces = metrics.interior_faces,
            max_bucket_faces = metrics.max_bucket_faces,
            parallel_min_len = config.parallel_min_len,
            scatter_parallel_min_faces = config.scatter_parallel_min_faces,
            backend = ?config.backend,
            "exec_scatter_mode_resolved"
        );
        Self {
            backend: config.backend,
            requested_scatter_mode: config.scatter_mode,
            resolved_scatter_mode,
            parallel_min_len: config.parallel_min_len,
            metrics,
            scratch: ExecScratch::with_metrics(metrics),
        }
    }

    #[must_use]
    pub fn backend(&self) -> ExecBackend {
        self.backend
    }

    #[must_use]
    pub fn requested_scatter_mode(&self) -> ScatterMode {
        self.requested_scatter_mode
    }

    #[must_use]
    pub fn resolved_scatter_mode(&self) -> ResolvedScatterMode {
        self.resolved_scatter_mode
    }

    #[must_use]
    pub fn parallel_min_len(&self) -> usize {
        self.parallel_min_len
    }

    #[must_use]
    pub fn metrics(&self) -> MeshExecMetrics {
        self.metrics
    }

    #[must_use]
    pub fn scratch(&self) -> &ExecScratch {
        &self.scratch
    }

    #[must_use]
    pub fn scratch_mut(&mut self) -> &mut ExecScratch {
        &mut self.scratch
    }

    /// 单桶是否因面数不足而强制串行 scatter（§2.4 桶级降级）。
    #[must_use]
    pub fn bucket_uses_serial_scatter(&self, bucket_len: usize) -> bool {
        bucket_len < self.parallel_min_len
    }

    /// 构造 scatter span 用标签。
    #[must_use]
    pub fn effective_scatter_mode_label(&self, bucket_len: usize) -> &'static str {
        match self.resolved_scatter_mode {
            ResolvedScatterMode::Serial => "serial",
            ResolvedScatterMode::ParallelUnsafeAtomics
                if self.bucket_uses_serial_scatter(bucket_len) =>
            {
                "serial"
            }
            ResolvedScatterMode::ParallelUnsafeAtomics => "atomic",
        }
    }
}

impl ExecConfig {
    pub fn validate(&self) -> Result<()> {
        if self.parallel_min_len == 0 {
            return Err(AsimuError::Config(
                "exec.parallel_min_len 必须大于 0".to_string(),
            ));
        }
        if self.scatter_parallel_min_faces == 0 {
            return Err(AsimuError::Config(
                "exec.scatter_parallel_min_faces 必须大于 0".to_string(),
            ));
        }
        Ok(())
    }
}

fn resolve_scatter_mode(
    config: &ExecConfig,
    metrics: &MeshExecMetrics,
    backend: ExecBackend,
) -> ResolvedScatterMode {
    match config.scatter_mode {
        ScatterMode::Serial => ResolvedScatterMode::Serial,
        ScatterMode::ParallelUnsafeAtomics => ResolvedScatterMode::ParallelUnsafeAtomics,
        ScatterMode::Auto => {
            if backend != ExecBackend::CpuParallel {
                return ResolvedScatterMode::Serial;
            }
            if metrics.interior_faces < config.scatter_parallel_min_faces {
                return ResolvedScatterMode::Serial;
            }
            if metrics.max_bucket_faces < config.parallel_min_len {
                return ResolvedScatterMode::Serial;
            }
            ResolvedScatterMode::ParallelUnsafeAtomics
        }
    }
}

fn scatter_mode_label(mode: ResolvedScatterMode) -> &'static str {
    match mode {
        ResolvedScatterMode::Serial => "serial",
        ResolvedScatterMode::ParallelUnsafeAtomics => "atomic",
    }
}

fn scatter_resolve_reason(requested: ScatterMode) -> &'static str {
    match requested {
        ScatterMode::Auto => "auto",
        ScatterMode::Serial => "explicit",
        ScatterMode::ParallelUnsafeAtomics => "explicit",
    }
}

#[cfg(test)]
impl ExecutionContext {
    /// 单元测试占位 context（小网格 → 串行 scatter）。
    #[must_use]
    pub fn for_unit_test() -> Self {
        Self::new(ExecConfig::default(), MeshExecMetrics::empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_resolves_serial_on_small_mesh() {
        let config = ExecConfig::default();
        let metrics = MeshExecMetrics::new(100, 100, 50);
        let ctx = ExecutionContext::new(config, metrics);
        assert_eq!(ctx.resolved_scatter_mode(), ResolvedScatterMode::Serial);
    }

    #[test]
    fn auto_resolves_atomic_on_large_mesh_with_parallel_fvm() {
        if !cfg!(feature = "parallel-fvm") {
            return;
        }
        let config = ExecConfig::default();
        let metrics = MeshExecMetrics::new(100_000, 100_000, 2048);
        let ctx = ExecutionContext::new(config, metrics);
        assert_eq!(
            ctx.resolved_scatter_mode(),
            ResolvedScatterMode::ParallelUnsafeAtomics
        );
    }

    #[test]
    fn explicit_atomic_not_downgraded_on_small_mesh() {
        let config = ExecConfig {
            scatter_mode: ScatterMode::ParallelUnsafeAtomics,
            ..ExecConfig::default()
        };
        let metrics = MeshExecMetrics::new(10, 10, 4);
        let ctx = ExecutionContext::new(config, metrics);
        assert_eq!(
            ctx.resolved_scatter_mode(),
            ResolvedScatterMode::ParallelUnsafeAtomics
        );
    }

    #[test]
    fn bucket_serial_when_below_parallel_min_len() {
        let config = ExecConfig::default();
        let metrics = MeshExecMetrics::new(100_000, 100_000, 2048);
        let ctx = ExecutionContext::new(config, metrics);
        assert!(ctx.bucket_uses_serial_scatter(512));
        assert!(!ctx.bucket_uses_serial_scatter(2048));
    }
}
