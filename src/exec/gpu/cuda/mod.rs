//! NVIDIA CUDA 后端（`cudarc`；ADR 0017 G0+）。

use std::sync::Arc;

use cudarc::driver::CudaContext;
use tracing::info;

use crate::error::{AsimuError, Result};

/// CUDA 设备上下文（G0：初始化占位；G1+ 持有 kernel 模块与缓冲池）。
pub struct CudaBackendState {
    context: Arc<CudaContext>,
}

impl CudaBackendState {
    /// 绑定默认设备（index 0）；无可用 GPU 时返回 [`AsimuError::Exec`]。
    pub fn try_new() -> Result<Self> {
        let context = CudaContext::new(0)
            .map_err(|e| AsimuError::Exec(format!("CUDA 设备初始化失败: {e:?}")))?;
        info!(device_index = 0, "cuda_backend_initialized");
        Ok(Self { context })
    }

    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }
}
