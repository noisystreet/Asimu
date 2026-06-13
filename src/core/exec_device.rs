//! 执行设备配置（ADR 0017 G0）。

use crate::error::{AsimuError, Result};

/// 算例级执行设备族；热路径内不切换。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ExecDevice {
    #[default]
    Cpu,
    GpuCuda,
}

/// 仅当 [`ExecDevice::Cpu`] 时生效的 CPU 并行策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ExecCpuPolicy {
    #[default]
    Scalar,
    Parallel,
}

/// 日志与过渡期兼容用的扁平 backend 视图。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecBackend {
    CpuScalar,
    CpuParallel,
    GpuCuda,
}

impl ExecDevice {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::GpuCuda => "cuda",
        }
    }
}

/// 解析 `[numerics].backend`。
pub fn parse_exec_backend(raw: &str) -> Result<ExecDevice> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "cpu" => Ok(ExecDevice::Cpu),
        "cuda" => Ok(ExecDevice::GpuCuda),
        other => Err(AsimuError::Config(format!(
            "未知 [numerics].backend \"{other}\"；可选 cpu | cuda"
        ))),
    }
}

#[must_use]
pub fn default_cpu_policy() -> ExecCpuPolicy {
    if cfg!(feature = "parallel-fvm") {
        ExecCpuPolicy::Parallel
    } else {
        ExecCpuPolicy::Scalar
    }
}

#[must_use]
pub fn cpu_policy_for_device(device: ExecDevice) -> ExecCpuPolicy {
    match device {
        ExecDevice::Cpu => default_cpu_policy(),
        ExecDevice::GpuCuda => ExecCpuPolicy::Scalar,
    }
}

#[must_use]
pub fn exec_backend_view(device: ExecDevice, cpu_policy: ExecCpuPolicy) -> ExecBackend {
    match device {
        ExecDevice::Cpu => match cpu_policy {
            ExecCpuPolicy::Scalar => ExecBackend::CpuScalar,
            ExecCpuPolicy::Parallel => ExecBackend::CpuParallel,
        },
        ExecDevice::GpuCuda => ExecBackend::GpuCuda,
    }
}

#[must_use]
pub fn legacy_backend_to_parts(backend: ExecBackend) -> (ExecDevice, ExecCpuPolicy) {
    match backend {
        ExecBackend::CpuScalar => (ExecDevice::Cpu, ExecCpuPolicy::Scalar),
        ExecBackend::CpuParallel => (ExecDevice::Cpu, ExecCpuPolicy::Parallel),
        ExecBackend::GpuCuda => (ExecDevice::GpuCuda, ExecCpuPolicy::Scalar),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_exec_backend_accepts_cpu_and_gpu_cuda() {
        assert_eq!(parse_exec_backend("cpu").expect("cpu"), ExecDevice::Cpu);
        assert_eq!(
            parse_exec_backend(" CUDA ").expect("cuda"),
            ExecDevice::GpuCuda
        );
    }

    #[test]
    fn parse_exec_backend_rejects_unknown() {
        let err = parse_exec_backend("gpu-wgpu").expect_err("unknown");
        assert!(err.to_string().contains("backend"));
    }
}
