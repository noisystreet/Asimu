//! [`ExecutionContext`](super::context::ExecutionContext) 设备状态（ADR 0017 G0）。

#[cfg(not(feature = "cuda"))]
use crate::error::AsimuError;
use crate::error::Result;

use super::context::ExecConfig;
use super::device::ExecDevice;

/// 算例级后端状态；构造后不在步间切换设备族。
pub(crate) enum BackendState {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(super::gpu::cuda::CudaBackendState),
}

impl BackendState {
    pub(crate) fn try_new(config: &ExecConfig) -> Result<Self> {
        match config.device {
            ExecDevice::Cpu => Ok(Self::Cpu),
            ExecDevice::GpuCuda => Self::try_new_cuda(),
        }
    }

    fn try_new_cuda() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            Ok(Self::Cuda(super::gpu::cuda::CudaBackendState::try_new()?))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(AsimuError::Exec(
                "backend = \"cuda\" 需要启用 Cargo feature cuda".to_string(),
            ))
        }
    }

    pub(crate) fn sync_to_host(&mut self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            #[cfg(feature = "cuda")]
            Self::Cuda(state) => {
                let _ = state.context();
                Ok(())
            }
        }
    }

    pub(crate) fn sync_to_device(&mut self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            #[cfg(feature = "cuda")]
            Self::Cuda(state) => {
                let _ = state.context();
                Ok(())
            }
        }
    }
}
