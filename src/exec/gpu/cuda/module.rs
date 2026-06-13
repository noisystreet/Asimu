//! PTX 模块加载（build.rs `nvcc` 预编译）。

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use tracing::info;

use crate::error::{AsimuError, Result};

/// 已加载的一阶无粘 kernel（Roe / HVL）。
pub struct CudaInviscidModule {
    pub(crate) function: CudaFunction,
}

impl CudaInviscidModule {
    pub fn try_load(ctx: &Arc<CudaContext>) -> Result<Self> {
        #[cfg(cuda_kernels_built)]
        {
            let ptx_src = include_str!(env!("CUDA_PTX_INVISCID_F32"));
            let module: Arc<CudaModule> = ctx
                .load_module(Ptx::from_src(ptx_src))
                .map_err(|e| AsimuError::Exec(format!("CUDA 模块加载失败: {e:?}")))?;
            let function = module
                .load_function("inviscid_first_order_bucket_f32")
                .map_err(|e| AsimuError::Exec(format!("CUDA kernel 符号未找到: {e:?}")))?;
            info!("cuda_inviscid_module_loaded");
            Ok(Self { function })
        }
        #[cfg(cuda_kernels_disabled)]
        {
            let _ = ctx;
            Err(AsimuError::Exec(
                "CUDA kernel 未编译（缺少 nvcc）；请安装 CUDA toolkit 后重新构建".to_string(),
            ))
        }
    }
}
