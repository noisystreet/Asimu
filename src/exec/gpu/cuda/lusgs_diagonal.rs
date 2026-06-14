//! LU-SGS 对角更新 CUDA launch。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use tracing::info_span;

use super::buffers::CudaFieldBuffers;
use crate::error::{AsimuError, Result};

const BLOCK_THREADS: u32 = 256;

pub fn launch_lusgs_diagonal_update(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    fields: &CudaFieldBuffers,
    sigma: &CudaSlice<f32>,
    cell_dts: &CudaSlice<f32>,
    omega: f32,
) -> Result<()> {
    let num_cells = fields.num_cells() as u32;
    let _span = info_span!("cuda_lusgs_diagonal_update", cells = num_cells).entered();
    let num_blocks = num_cells.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(&omega);
    builder.arg(&fields.cons_rho);
    builder.arg(&fields.cons_mx);
    builder.arg(&fields.cons_my);
    builder.arg(&fields.cons_mz);
    builder.arg(&fields.cons_e);
    builder.arg(&fields.res_rho);
    builder.arg(&fields.res_mx);
    builder.arg(&fields.res_my);
    builder.arg(&fields.res_mz);
    builder.arg(&fields.res_e);
    builder.arg(sigma);
    builder.arg(cell_dts);
    builder.arg(&fields.cons_rho);
    builder.arg(&fields.cons_mx);
    builder.arg(&fields.cons_my);
    builder.arg(&fields.cons_mz);
    builder.arg(&fields.cons_e);
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA LU-SGS 对角 kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}
