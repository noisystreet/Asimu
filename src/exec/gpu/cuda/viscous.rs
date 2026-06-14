//! 粘性内面 CUDA kernel launch（`viscous_interior_bucket_f32`）。

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use super::buffers::CudaFieldBuffers;
use super::gradient_buffers::CudaGradientBuffers;
use crate::error::{AsimuError, Result};

const BLOCK_THREADS: u32 = 256;

pub(super) struct ViscousBoundaryLaunch<'a> {
    pub faces:
        &'a cudarc::driver::CudaSlice<super::boundary_face_geom::ExecViscousBoundaryFaceStatic>,
    pub num_faces: u32,
    pub ghosts: &'a cudarc::driver::CudaSlice<super::boundary_face_geom::ViscousBoundaryGhostHost>,
    pub temperatures: &'a cudarc::driver::CudaSlice<f32>,
    pub fields: &'a mut CudaFieldBuffers,
    pub gradients: &'a CudaGradientBuffers,
    pub transport: super::viscous_transport_params::DeviceViscousTransportParams,
}

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

pub(super) fn launch_viscous_boundary(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    launch: ViscousBoundaryLaunch<'_>,
) -> Result<()> {
    if launch.num_faces == 0 {
        return Ok(());
    }
    let num_blocks = launch.num_faces.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(launch.faces);
    builder.arg(&launch.num_faces);
    builder.arg(launch.ghosts);
    builder.arg(&launch.fields.prim_ux);
    builder.arg(&launch.fields.prim_uy);
    builder.arg(&launch.fields.prim_uz);
    builder.arg(launch.temperatures);
    builder.arg(&launch.gradients.du_dx);
    builder.arg(&launch.gradients.du_dy);
    builder.arg(&launch.gradients.du_dz);
    builder.arg(&launch.gradients.dv_dx);
    builder.arg(&launch.gradients.dv_dy);
    builder.arg(&launch.gradients.dv_dz);
    builder.arg(&launch.gradients.dw_dx);
    builder.arg(&launch.gradients.dw_dy);
    builder.arg(&launch.gradients.dw_dz);
    builder.arg(&launch.gradients.dt_dx);
    builder.arg(&launch.gradients.dt_dy);
    builder.arg(&launch.gradients.dt_dz);
    builder.arg(&mut launch.fields.res_mx);
    builder.arg(&mut launch.fields.res_my);
    builder.arg(&mut launch.fields.res_mz);
    builder.arg(&mut launch.fields.res_e);
    builder.arg(&launch.transport);
    // SAFETY: 边界面 owner atomic scatter；布局与 `viscous_boundary_f32` 一致。
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA 粘性边界面 kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}

pub(super) fn launch_viscous_face_transport(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    face_geom: &mut cudarc::driver::CudaSlice<super::viscous_face_geom::DeviceViscousFaceGeom>,
    num_faces: u32,
    temperatures: &cudarc::driver::CudaSlice<f32>,
    params: super::viscous_transport_params::DeviceViscousTransportParams,
) -> Result<()> {
    let num_blocks = num_faces.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(face_geom);
    builder.arg(&num_faces);
    builder.arg(temperatures);
    builder.arg(&params);
    // SAFETY: 每面独立写 `mu/lambda`；布局与 `viscous_face_transport_f32` 一致。
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA 粘性输运 kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}
