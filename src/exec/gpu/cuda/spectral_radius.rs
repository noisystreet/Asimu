//! 非结构谱半径 CUDA launch。

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tracing::info_span;

use super::buffers::CudaFieldBuffers;
use super::spectral_radius_mesh_cache::CudaSpectralMeshDeviceCache;
use crate::error::{AsimuError, Result};

const BLOCK_THREADS: u32 = 256;

pub fn launch_spectral_radius_accumulate(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    mesh: &mut CudaSpectralMeshDeviceCache,
    fields: &CudaFieldBuffers,
    gamma: f32,
    viscous_enabled: bool,
) -> Result<()> {
    let num_cells = mesh.num_cells() as u32;
    let _span = info_span!(
        "cuda_spectral_radius_accumulate",
        cells = num_cells,
        viscous = viscous_enabled,
    )
    .entered();

    stream
        .memset_zeros(mesh.sigma_mut())
        .map_err(|e| AsimuError::Exec(format!("CUDA 谱半径 sigma 清零失败: {e:?}")))?;

    let num_blocks = num_cells.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let viscous_flag = u32::from(viscous_enabled);
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(&gamma);
    builder.arg(&viscous_flag);
    builder.arg(mesh.owner_offsets());
    builder.arg(mesh.owner_indices());
    builder.arg(mesh.neighbor_offsets());
    builder.arg(mesh.neighbor_indices());
    builder.arg(mesh.boundary_offsets());
    builder.arg(mesh.boundary_indices());
    builder.arg(mesh.interior());
    builder.arg(mesh.boundary());
    builder.arg(mesh.boundary_ghosts());
    builder.arg(&fields.prim_rho);
    builder.arg(&fields.prim_p);
    builder.arg(&fields.prim_ux);
    builder.arg(&fields.prim_uy);
    builder.arg(&fields.prim_uz);
    builder.arg(mesh.diffusivity());
    builder.arg(mesh.sigma());
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA 谱半径 kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}

pub fn launch_finalize_cell_dts(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    mesh: &CudaSpectralMeshDeviceCache,
    cfl: f32,
    fixed_dt: Option<f32>,
) -> Result<()> {
    let num_cells = mesh.num_cells() as u32;
    let _span = info_span!(
        "cuda_finalize_cell_dts",
        cells = num_cells,
        fixed_dt = fixed_dt.is_some(),
    )
    .entered();
    let use_fixed = u32::from(fixed_dt.is_some_and(|d| d > 0.0 && d.is_finite()));
    let fixed_val = fixed_dt.unwrap_or(0.0);
    let num_blocks = num_cells.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(&cfl);
    builder.arg(&fixed_val);
    builder.arg(&use_fixed);
    builder.arg(mesh.sigma());
    builder.arg(mesh.cell_dts());
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA finalize_cell_dts launch 失败: {e:?}")))?;
    }
    Ok(())
}

pub fn launch_init_min_positive_scratch(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    min_out: &mut cudarc::driver::CudaSlice<f32>,
) -> Result<()> {
    let _span = info_span!("cuda_init_min_positive_scratch").entered();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(min_out);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!("CUDA init_min_positive_scratch launch 失败: {e:?}"))
        })?;
    }
    Ok(())
}

pub fn launch_min_positive_cell_dt(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    num_cells: u32,
    cell_dts: &cudarc::driver::CudaSlice<f32>,
    min_out: &mut cudarc::driver::CudaSlice<f32>,
) -> Result<()> {
    let _span = info_span!("cuda_min_positive_cell_dt", cells = num_cells).entered();
    let num_blocks = num_cells.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(cell_dts);
    builder.arg(min_out);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!("CUDA min_positive_cell_dt launch 失败: {e:?}"))
        })?;
    }
    Ok(())
}
