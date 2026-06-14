//! IDWLS 粘性 RHS CUDA launch。

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tracing::info_span;

use super::buffers::CudaFieldBuffers;
use super::gradient_buffers::CudaGradientBuffers;
use super::idwls_mesh_cache::{CudaIdwlsMeshDeviceCache, CudaIdwlsRhsDeviceBuffers};
use crate::discretization::unstructured_face_cache_f32::LsqPrecomputedCellF32;
use crate::error::{AsimuError, Result};

const BLOCK_THREADS: u32 = 256;

pub fn launch_idwls_viscous_accumulate(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    mesh: &CudaIdwlsMeshDeviceCache,
    fields: &CudaFieldBuffers,
    rhs: &mut CudaIdwlsRhsDeviceBuffers,
) -> Result<()> {
    let num_cells = mesh.num_cells() as u32;
    let _span = info_span!(
        "cuda_idwls_viscous_accumulate",
        cells = num_cells,
        interior_faces = mesh.interior().len(),
        boundary_faces = mesh.boundary().len(),
    )
    .entered();

    zero_rhs(stream, rhs.bu_mut())?;
    zero_rhs(stream, rhs.bv_mut())?;
    zero_rhs(stream, rhs.bw_mut())?;
    zero_rhs(stream, rhs.bt_mut())?;

    let num_blocks = num_cells.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(mesh.owner_offsets());
    builder.arg(mesh.owner_indices());
    builder.arg(mesh.neighbor_offsets());
    builder.arg(mesh.neighbor_indices());
    builder.arg(mesh.boundary_offsets());
    builder.arg(mesh.boundary_indices());
    builder.arg(mesh.interior());
    builder.arg(mesh.boundary());
    builder.arg(mesh.boundary_ghosts());
    builder.arg(&fields.prim_ux);
    builder.arg(&fields.prim_uy);
    builder.arg(&fields.prim_uz);
    builder.arg(mesh.temperature());
    builder.arg(rhs.bu_slice());
    builder.arg(rhs.bv_slice());
    builder.arg(rhs.bw_slice());
    builder.arg(rhs.bt_slice());
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA IDWLS kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}

pub fn launch_idwls_solve_gradient(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    lsq_geometry: &cudarc::driver::CudaSlice<LsqPrecomputedCellF32>,
    rhs: &CudaIdwlsRhsDeviceBuffers,
    gradients: &mut CudaGradientBuffers,
) -> Result<()> {
    let num_cells = rhs.num_cells() as u32;
    let _span = info_span!("cuda_idwls_solve_gradient", cells = num_cells).entered();
    let num_blocks = num_cells.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(lsq_geometry);
    builder.arg(rhs.bu_slice());
    builder.arg(rhs.bv_slice());
    builder.arg(rhs.bw_slice());
    builder.arg(rhs.bt_slice());
    builder.arg(&mut gradients.du_dx);
    builder.arg(&mut gradients.du_dy);
    builder.arg(&mut gradients.du_dz);
    builder.arg(&mut gradients.dv_dx);
    builder.arg(&mut gradients.dv_dy);
    builder.arg(&mut gradients.dv_dz);
    builder.arg(&mut gradients.dw_dx);
    builder.arg(&mut gradients.dw_dy);
    builder.arg(&mut gradients.dw_dz);
    builder.arg(&mut gradients.dt_dx);
    builder.arg(&mut gradients.dt_dy);
    builder.arg(&mut gradients.dt_dz);
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA IDWLS solve kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}

fn zero_rhs(stream: &Arc<CudaStream>, buf: &mut cudarc::driver::CudaSlice<f32>) -> Result<()> {
    stream
        .memset_zeros(buf)
        .map_err(|e| AsimuError::Exec(format!("CUDA IDWLS memset 失败: {e:?}")))
}
