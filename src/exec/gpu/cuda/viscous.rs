//! 粘性内面 CUDA kernel launch（`viscous_interior_bucket_f32`）。

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use super::buffers::CudaFieldBuffers;
use super::gradient_buffers::CudaGradientBuffers;
use crate::error::{AsimuError, Result};

const BLOCK_THREADS: u32 = 256;

pub(super) fn launch_viscous_bucket(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    bucket_faces: &cudarc::driver::CudaSlice<u32>,
    num_faces: u32,
    face_geom: &cudarc::driver::CudaSlice<super::viscous_face_geom::DeviceViscousFaceGeom>,
    fields: &mut CudaFieldBuffers,
    gradients: &CudaGradientBuffers,
) -> Result<()> {
    let num_blocks = num_faces.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(bucket_faces);
    builder.arg(&num_faces);
    builder.arg(face_geom);
    builder.arg(&fields.prim_ux);
    builder.arg(&fields.prim_uy);
    builder.arg(&fields.prim_uz);
    builder.arg(&gradients.du_dx);
    builder.arg(&gradients.du_dy);
    builder.arg(&gradients.du_dz);
    builder.arg(&gradients.dv_dx);
    builder.arg(&gradients.dv_dy);
    builder.arg(&gradients.dv_dz);
    builder.arg(&gradients.dw_dx);
    builder.arg(&gradients.dw_dy);
    builder.arg(&gradients.dw_dz);
    builder.arg(&gradients.dt_dx);
    builder.arg(&gradients.dt_dy);
    builder.arg(&gradients.dt_dz);
    builder.arg(&mut fields.res_mx);
    builder.arg(&mut fields.res_my);
    builder.arg(&mut fields.res_mz);
    builder.arg(&mut fields.res_e);
    // SAFETY: 着色桶内面无共享单元；参数布局与 `viscous_interior_bucket_f32` 一致。
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA 粘性 kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}
