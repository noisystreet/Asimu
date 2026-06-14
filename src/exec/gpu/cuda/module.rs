//! PTX 模块加载（build.rs `nvcc` 预编译）。

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use tracing::info;

use crate::error::{AsimuError, Result};

/// 已加载的一阶无粘 kernel（Roe / HVL）。
pub struct CudaInviscidModule {
    pub(crate) function: CudaFunction,
    pub(crate) boundary_function: CudaFunction,
}

/// 已加载的粘性内面 kernel。
pub struct CudaViscousModule {
    pub(crate) function: CudaFunction,
    pub(crate) face_transport: CudaFunction,
    pub(crate) boundary_function: CudaFunction,
}

/// 已加载的 IDWLS 粘性 RHS kernel。
pub struct CudaIdwlsModule {
    pub(crate) accumulate: CudaFunction,
    pub(crate) solve_gradient: CudaFunction,
}

/// 已加载的非结构谱半径 kernel。
pub struct CudaSpectralRadiusModule {
    pub(crate) accumulate: CudaFunction,
    pub(crate) finalize_dts: CudaFunction,
    pub(crate) min_positive_dt: CudaFunction,
}

/// 已加载的 LU-SGS 对角 kernel。
pub struct CudaLusgsModule {
    pub(crate) diagonal_update: CudaFunction,
    pub(crate) residual_density_sum_sq: CudaFunction,
}

impl CudaInviscidModule {
    pub fn try_load(ctx: &Arc<CudaContext>) -> Result<Self> {
        #[cfg(cuda_kernels_built)]
        {
            let ptx_src = include_str!(env!("CUDA_PTX_INVISCID_F32"));
            let module: Arc<CudaModule> = ctx
                .load_module(Ptx::from_src(ptx_src))
                .map_err(|e| AsimuError::Exec(format!("CUDA 无粘模块加载失败: {e:?}")))?;
            let function = module
                .load_function("inviscid_first_order_bucket_f32")
                .map_err(|e| AsimuError::Exec(format!("CUDA 无粘 kernel 符号未找到: {e:?}")))?;
            let boundary_function = module
                .load_function("inviscid_first_order_boundary_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!("CUDA 无粘边界面 kernel 符号未找到: {e:?}"))
                })?;
            info!("cuda_inviscid_module_loaded");
            Ok(Self {
                function,
                boundary_function,
            })
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

impl CudaViscousModule {
    pub fn try_load(ctx: &Arc<CudaContext>) -> Result<Self> {
        #[cfg(cuda_kernels_built)]
        {
            let ptx_src = include_str!(env!("CUDA_PTX_VISCOUS_F32"));
            let module: Arc<CudaModule> = ctx
                .load_module(Ptx::from_src(ptx_src))
                .map_err(|e| AsimuError::Exec(format!("CUDA 粘性模块加载失败: {e:?}")))?;
            let function = module
                .load_function("viscous_interior_bucket_f32")
                .map_err(|e| AsimuError::Exec(format!("CUDA 粘性 kernel 符号未找到: {e:?}")))?;
            let face_transport = module
                .load_function("viscous_face_transport_f32")
                .map_err(|e| AsimuError::Exec(format!("CUDA 粘性输运 kernel 符号未找到: {e:?}")))?;
            let boundary_function = module.load_function("viscous_boundary_f32").map_err(|e| {
                AsimuError::Exec(format!("CUDA 粘性边界面 kernel 符号未找到: {e:?}"))
            })?;
            info!("cuda_viscous_module_loaded");
            Ok(Self {
                function,
                face_transport,
                boundary_function,
            })
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

impl CudaIdwlsModule {
    pub fn try_load(ctx: &Arc<CudaContext>) -> Result<Self> {
        #[cfg(cuda_kernels_built)]
        {
            let ptx_src = include_str!(env!("CUDA_PTX_IDWLS_F32"));
            let module: Arc<CudaModule> = ctx
                .load_module(Ptx::from_src(ptx_src))
                .map_err(|e| AsimuError::Exec(format!("CUDA IDWLS 模块加载失败: {e:?}")))?;
            let accumulate = module
                .load_function("idwls_viscous_accumulate_cell_f32")
                .map_err(|e| AsimuError::Exec(format!("CUDA IDWLS kernel 符号未找到: {e:?}")))?;
            let solve_gradient = module
                .load_function("idwls_solve_gradient_cell_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!("CUDA IDWLS solve kernel 符号未找到: {e:?}"))
                })?;
            info!("cuda_idwls_module_loaded");
            Ok(Self {
                accumulate,
                solve_gradient,
            })
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

impl CudaSpectralRadiusModule {
    pub fn try_load(ctx: &Arc<CudaContext>) -> Result<Self> {
        #[cfg(cuda_kernels_built)]
        {
            let ptx_src = include_str!(env!("CUDA_PTX_SPECTRAL_RADIUS_F32"));
            let module: Arc<CudaModule> = ctx
                .load_module(Ptx::from_src(ptx_src))
                .map_err(|e| AsimuError::Exec(format!("CUDA 谱半径模块加载失败: {e:?}")))?;
            let accumulate = module
                .load_function("spectral_radius_accumulate_cell_f32")
                .map_err(|e| AsimuError::Exec(format!("CUDA 谱半径 kernel 符号未找到: {e:?}")))?;
            let finalize_dts = module.load_function("finalize_cell_dts_f32").map_err(|e| {
                AsimuError::Exec(format!("CUDA finalize_cell_dts 符号未找到: {e:?}"))
            })?;
            let min_positive_dt =
                module
                    .load_function("min_positive_cell_dt_f32")
                    .map_err(|e| {
                        AsimuError::Exec(format!("CUDA min_positive_cell_dt 符号未找到: {e:?}"))
                    })?;
            info!("cuda_spectral_radius_module_loaded");
            Ok(Self {
                accumulate,
                finalize_dts,
                min_positive_dt,
            })
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

impl CudaLusgsModule {
    pub fn try_load(ctx: &Arc<CudaContext>) -> Result<Self> {
        #[cfg(cuda_kernels_built)]
        {
            let ptx_src = include_str!(env!("CUDA_PTX_LUSGS_DIAGONAL_F32"));
            let module: Arc<CudaModule> = ctx
                .load_module(Ptx::from_src(ptx_src))
                .map_err(|e| AsimuError::Exec(format!("CUDA LU-SGS 模块加载失败: {e:?}")))?;
            let diagonal_update =
                module
                    .load_function("lusgs_diagonal_update_f32")
                    .map_err(|e| {
                        AsimuError::Exec(format!("CUDA LU-SGS 对角 kernel 符号未找到: {e:?}"))
                    })?;
            let residual_density_sum_sq = module
                .load_function("residual_density_sum_sq_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!("CUDA 密度残差 RMS kernel 符号未找到: {e:?}"))
                })?;
            info!("cuda_lusgs_module_loaded");
            Ok(Self {
                diagonal_update,
                residual_density_sum_sq,
            })
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
