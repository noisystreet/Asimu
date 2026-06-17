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
    pub(crate) init_min_positive_scratch: CudaFunction,
}

/// 已加载的 LU-SGS 对角 kernel。
pub struct CudaLusgsModule {
    pub(crate) diagonal_update: CudaFunction,
    pub(crate) residual_density_sum_sq: CudaFunction,
    #[allow(dead_code)] // 对照 kernel；`lusgs_sweep` 单测读取
    pub(crate) sweep_unstructured_serial: CudaFunction,
    pub(crate) sweep_forward_color: CudaFunction,
    pub(crate) sweep_backward_color: CudaFunction,
    pub(crate) sweep_any_nonphysical: CudaFunction,
}

/// 已加载的双时间步 kernel。
pub struct CudaDualTimeModule {
    pub(crate) storage: CudaFunction,
}

/// 已加载的可压缩 BC kernel。
pub struct CudaBcModule {
    pub(crate) apply_boundary_ghosts: CudaFunction,
}

/// 已加载的场恢复 / 扩散系数 kernel（P5）。
pub struct CudaFieldModule {
    pub(crate) fill_primitives: CudaFunction,
    pub(crate) cell_static_temperature: CudaFunction,
    pub(crate) viscous_diffusivity_max: CudaFunction,
    pub(crate) fill_boundary_ghost_buffers: CudaFunction,
    pub(crate) enforce_conserved_positivity: CudaFunction,
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
            let init_min_positive_scratch = module
                .load_function("init_min_positive_scratch_f32")
                .map_err(|e| {
                AsimuError::Exec(format!("CUDA init_min_positive_scratch 符号未找到: {e:?}"))
            })?;
            info!("cuda_spectral_radius_module_loaded");
            Ok(Self {
                accumulate,
                finalize_dts,
                min_positive_dt,
                init_min_positive_scratch,
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
            let sweep_ptx = include_str!(env!("CUDA_PTX_LUSGS_SWEEP_F32"));
            let sweep_module: Arc<CudaModule> = ctx
                .load_module(Ptx::from_src(sweep_ptx))
                .map_err(|e| AsimuError::Exec(format!("CUDA LU-SGS 扫掠模块加载失败: {e:?}")))?;
            let sweep_unstructured_serial = sweep_module
                .load_function("lusgs_sweep_unstructured_serial_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!("CUDA LU-SGS 扫掠 kernel 符号未找到: {e:?}"))
                })?;
            let sweep_forward_color = sweep_module
                .load_function("lusgs_sweep_forward_color_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!(
                        "CUDA LU-SGS 前扫 wavefront kernel 符号未找到: {e:?}"
                    ))
                })?;
            let sweep_backward_color = sweep_module
                .load_function("lusgs_sweep_backward_color_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!(
                        "CUDA LU-SGS 后扫 wavefront kernel 符号未找到: {e:?}"
                    ))
                })?;
            let sweep_any_nonphysical = sweep_module
                .load_function("lusgs_any_nonphysical_conserved_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!("CUDA LU-SGS 正性检查 kernel 符号未找到: {e:?}"))
                })?;
            info!("cuda_lusgs_module_loaded");
            Ok(Self {
                diagonal_update,
                residual_density_sum_sq,
                sweep_unstructured_serial,
                sweep_forward_color,
                sweep_backward_color,
                sweep_any_nonphysical,
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

impl CudaDualTimeModule {
    pub fn try_load(ctx: &Arc<CudaContext>) -> Result<Self> {
        #[cfg(cuda_kernels_built)]
        {
            let ptx_src = include_str!(env!("CUDA_PTX_DUAL_TIME_STORAGE_F32"));
            let module: Arc<CudaModule> = ctx
                .load_module(Ptx::from_src(ptx_src))
                .map_err(|e| AsimuError::Exec(format!("CUDA 双时间步模块加载失败: {e:?}")))?;
            let storage = module.load_function("dual_time_storage_f32").map_err(|e| {
                AsimuError::Exec(format!("CUDA dual_time_storage kernel 符号未找到: {e:?}"))
            })?;
            info!("cuda_dual_time_module_loaded");
            Ok(Self { storage })
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

impl CudaBcModule {
    pub fn try_load(ctx: &Arc<CudaContext>) -> Result<Self> {
        #[cfg(cuda_kernels_built)]
        {
            let ptx_src = include_str!(env!("CUDA_PTX_BOUNDARY_BC_F32"));
            let module: Arc<CudaModule> = ctx
                .load_module(Ptx::from_src(ptx_src))
                .map_err(|e| AsimuError::Exec(format!("CUDA BC 模块加载失败: {e:?}")))?;
            let apply_boundary_ghosts = module
                .load_function("apply_compressible_boundary_ghosts_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!("CUDA apply_boundary_ghosts 符号未找到: {e:?}"))
                })?;
            info!("cuda_bc_module_loaded");
            Ok(Self {
                apply_boundary_ghosts,
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

impl CudaFieldModule {
    pub fn try_load(ctx: &Arc<CudaContext>) -> Result<Self> {
        #[cfg(cuda_kernels_built)]
        {
            let ptx_src = include_str!(env!("CUDA_PTX_FIELD_F32"));
            let module: Arc<CudaModule> = ctx
                .load_module(Ptx::from_src(ptx_src))
                .map_err(|e| AsimuError::Exec(format!("CUDA 场模块加载失败: {e:?}")))?;
            let fill_primitives = module
                .load_function("fill_primitives_from_conserved_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!("CUDA fill_primitives kernel 符号未找到: {e:?}"))
                })?;
            let viscous_diffusivity_max = module
                .load_function("cell_viscous_diffusivity_max_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!("CUDA diffusivity_max kernel 符号未找到: {e:?}"))
                })?;
            let cell_static_temperature = module
                .load_function("cell_static_temperature_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!(
                        "CUDA cell_static_temperature kernel 符号未找到: {e:?}"
                    ))
                })?;
            let fill_boundary_ghost_buffers = module
                .load_function("fill_boundary_ghost_buffers_from_conserved_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!(
                        "CUDA fill_boundary_ghost_buffers kernel 符号未找到: {e:?}"
                    ))
                })?;
            let enforce_conserved_positivity = module
                .load_function("enforce_conserved_positivity_f32")
                .map_err(|e| {
                    AsimuError::Exec(format!(
                        "CUDA enforce_conserved_positivity kernel 符号未找到: {e:?}"
                    ))
                })?;
            info!("cuda_field_module_loaded");
            Ok(Self {
                fill_primitives,
                cell_static_temperature,
                viscous_diffusivity_max,
                fill_boundary_ghost_buffers,
                enforce_conserved_positivity,
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
