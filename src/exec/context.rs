//! [`ExecutionContext`](ExecutionContext) 与 scatter 模式解析（ADR 0013 / 0017）。

use tracing::info;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::ComputePrecision;
use crate::error::{AsimuError, Result};
use crate::io::CaseNumericsConfig;

use super::backend_state::BackendState;
use super::device::{
    ExecBackend, ExecCpuPolicy, ExecDevice, cpu_policy_for_device, default_cpu_policy,
    exec_backend_view, legacy_backend_to_parts,
};
use super::metrics::MeshExecMetrics;
use super::scratch::ExecScratch;

/// `Auto` 解析为并行 atomic scatter 的内面数下限（§2.4）。
pub const EXEC_SCATTER_PARALLEL_MIN_FACES: usize = 65_536;

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
    pub device: ExecDevice,
    pub cpu_policy: ExecCpuPolicy,
    pub compute_precision: ComputePrecision,
    pub parallel_min_len: usize,
    pub scatter_mode: ScatterMode,
    pub scatter_parallel_min_faces: usize,
}

impl Default for ExecConfig {
    fn default() -> Self {
        let device = ExecDevice::Cpu;
        Self {
            device,
            cpu_policy: default_cpu_policy(),
            compute_precision: ComputePrecision::F64,
            parallel_min_len: 1024,
            scatter_mode: ScatterMode::Auto,
            scatter_parallel_min_faces: EXEC_SCATTER_PARALLEL_MIN_FACES,
        }
    }
}

impl ExecConfig {
    /// 由 case `[numerics]` 构造 exec 配置（ADR 0017）。
    #[must_use]
    pub fn from_numerics(numerics: &CaseNumericsConfig) -> Self {
        let device = numerics.exec_device;
        Self {
            device,
            cpu_policy: cpu_policy_for_device(device),
            compute_precision: numerics.compute_precision,
            ..Self::default()
        }
    }

    /// 单元测试：由扁平 [`ExecBackend`] 构造。
    #[must_use]
    pub fn for_test_backend(backend: ExecBackend) -> Self {
        let (device, cpu_policy) = legacy_backend_to_parts(backend);
        Self {
            device,
            cpu_policy,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn backend(&self) -> ExecBackend {
        exec_backend_view(self.device, self.cpu_policy)
    }

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
        if self.device == ExecDevice::Cpu
            && self.cpu_policy == ExecCpuPolicy::Parallel
            && !cfg!(feature = "parallel-fvm")
        {
            return Err(AsimuError::Config(
                "CPU 并行策略需要启用 Cargo feature parallel-fvm".to_string(),
            ));
        }
        Ok(())
    }
}

/// 执行上下文：设备、scatter 模式与步间 scratch。
pub struct ExecutionContext {
    device: ExecDevice,
    cpu_policy: ExecCpuPolicy,
    compute_precision: ComputePrecision,
    requested_scatter_mode: ScatterMode,
    resolved_scatter_mode: ResolvedScatterMode,
    parallel_min_len: usize,
    metrics: MeshExecMetrics,
    scratch: ExecScratch,
    backend_state: BackendState,
    /// 单元测试：本 context 内着色桶 scatter API 调用次数（避免并行测试污染全局计数）。
    #[cfg(test)]
    scatter_invocation_count: AtomicUsize,
}

impl ExecutionContext {
    pub fn new(config: ExecConfig, metrics: MeshExecMetrics) -> Result<Self> {
        config.validate()?;
        let backend = config.backend();
        let resolved_scatter_mode = resolve_scatter_mode(&config, &metrics, backend);
        let backend_state = BackendState::try_new(&config)?;
        info!(
            mode = scatter_mode_label(resolved_scatter_mode),
            reason = scatter_resolve_reason(config.scatter_mode),
            interior_faces = metrics.interior_faces,
            max_bucket_faces = metrics.max_bucket_faces,
            parallel_min_len = config.parallel_min_len,
            scatter_parallel_min_faces = config.scatter_parallel_min_faces,
            exec_device = config.device.label(),
            exec_backend = ?backend,
            "exec_scatter_mode_resolved"
        );
        if config.device == ExecDevice::GpuCuda {
            info!("cuda_backend_g1: 一阶无粘内面走 device kernel（边界仍 CPU）");
        }
        Ok(Self {
            device: config.device,
            cpu_policy: config.cpu_policy,
            compute_precision: config.compute_precision,
            requested_scatter_mode: config.scatter_mode,
            resolved_scatter_mode,
            parallel_min_len: config.parallel_min_len,
            metrics,
            scratch: ExecScratch::with_metrics(metrics),
            backend_state,
            #[cfg(test)]
            scatter_invocation_count: AtomicUsize::new(0),
        })
    }

    /// GPU 路径：将相关场同步至 host；CPU 为零开销。
    pub fn sync_to_host(&mut self) -> Result<()> {
        self.backend_state.sync_to_host()
    }

    /// GPU 路径：BC 更新后写回 device；CPU 为零开销。
    pub fn sync_to_device(&mut self) -> Result<()> {
        self.backend_state.sync_to_device()
    }

    /// CUDA：守恒场 / BC 刷新后标记 device primitive 过期。
    pub fn mark_cuda_primitives_stale(&mut self) {
        self.backend_state.mark_cuda_primitives_stale();
    }

    /// CUDA：将 host primitive 上传 device（仅当已标记过期）。
    pub fn sync_cuda_primitives_to_device(
        &mut self,
        primitives: &crate::field::PrimitiveFieldsT<f32>,
    ) -> Result<()> {
        self.backend_state
            .sync_cuda_primitives_to_device(primitives)
    }

    /// CUDA G1：一阶无粘内面着色桶 flux + scatter（Roe / HVL）。
    #[cfg(feature = "cuda")]
    pub fn cuda_assemble_first_order_inviscid_interior(
        &mut self,
        residual: &mut crate::field::ConservedResidualT<f32>,
        primitives: &crate::field::PrimitiveFieldsT<f32>,
        topo: &crate::exec::gpu::cuda::ExecInteriorFaceTopology,
        topo_key: usize,
        params: crate::exec::gpu::cuda::CudaFirstOrderInviscidParams,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .assemble_first_order_inviscid_interior(residual, primitives, topo, topo_key, params)
    }

    /// 单元测试：重置本 context 的 scatter 调用计数。
    #[cfg(test)]
    pub fn reset_scatter_invocation_count(&self) {
        self.scatter_invocation_count.store(0, Ordering::Relaxed);
    }

    /// 单元测试：本 context 内 `enter_scatter_span` 调用次数。
    #[cfg(test)]
    #[must_use]
    pub fn scatter_invocation_count(&self) -> usize {
        self.scatter_invocation_count.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(super) fn record_scatter_invocation(&self) {
        self.scatter_invocation_count
            .fetch_add(1, Ordering::Relaxed);
    }

    #[must_use]
    pub fn compute_precision(&self) -> ComputePrecision {
        self.compute_precision
    }

    #[must_use]
    pub fn device(&self) -> ExecDevice {
        self.device
    }

    #[must_use]
    pub fn cpu_policy(&self) -> ExecCpuPolicy {
        self.cpu_policy
    }

    #[must_use]
    pub fn backend(&self) -> ExecBackend {
        exec_backend_view(self.device, self.cpu_policy)
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
        Self::new(ExecConfig::default(), MeshExecMetrics::empty()).expect("unit test exec")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_context_records_compute_precision() {
        let ctx = ExecutionContext::new(
            ExecConfig {
                compute_precision: ComputePrecision::F32,
                ..ExecConfig::default()
            },
            MeshExecMetrics::empty(),
        )
        .expect("ctx");
        assert_eq!(ctx.compute_precision(), ComputePrecision::F32);
    }

    #[test]
    fn auto_resolves_serial_on_small_mesh() {
        let config = ExecConfig::default();
        let metrics = MeshExecMetrics::new(100, 100, 50);
        let ctx = ExecutionContext::new(config, metrics).expect("ctx");
        assert_eq!(ctx.resolved_scatter_mode(), ResolvedScatterMode::Serial);
    }

    #[test]
    fn auto_resolves_atomic_on_large_mesh_with_parallel_fvm() {
        if !cfg!(feature = "parallel-fvm") {
            return;
        }
        let config = ExecConfig::default();
        let metrics = MeshExecMetrics::new(100_000, 100_000, 2048);
        let ctx = ExecutionContext::new(config, metrics).expect("ctx");
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
        let ctx = ExecutionContext::new(config, metrics).expect("ctx");
        assert_eq!(
            ctx.resolved_scatter_mode(),
            ResolvedScatterMode::ParallelUnsafeAtomics
        );
    }

    #[test]
    fn cpu_scalar_disables_parallel_cell_loops() {
        let ctx = ExecutionContext::new(
            ExecConfig::for_test_backend(ExecBackend::CpuScalar),
            MeshExecMetrics::new(100_000, 100_000, 2048),
        )
        .expect("ctx");
        assert!(!ctx.uses_parallel_cell_loops());
    }

    #[test]
    fn exec_context_cpu_scalar_matches_legacy_serial_scatter() {
        let scalar = ExecutionContext::new(
            ExecConfig {
                scatter_mode: ScatterMode::Serial,
                ..ExecConfig::for_test_backend(ExecBackend::CpuScalar)
            },
            MeshExecMetrics::new(100_000, 100_000, 2048),
        )
        .expect("ctx");
        let unit = ExecutionContext::for_unit_test();
        assert_eq!(scalar.resolved_scatter_mode(), unit.resolved_scatter_mode());
    }

    #[test]
    fn from_numerics_maps_gpu_cuda_device() {
        let config = ExecConfig::from_numerics(&CaseNumericsConfig {
            compute_precision: ComputePrecision::F32,
            exec_device: ExecDevice::GpuCuda,
        });
        assert_eq!(config.device, ExecDevice::GpuCuda);
        assert_eq!(config.backend(), ExecBackend::GpuCuda);
        assert_eq!(config.compute_precision, ComputePrecision::F32);
    }
}
