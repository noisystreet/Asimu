//! 无粘/粘性内面 bucket launch（从 `inviscid.rs` 拆分）。

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use super::super::buffers::CudaFieldBuffers;
use super::super::gradient_buffers::CudaGradientBuffers;
use super::super::viscous_mesh_cache::{CudaViscousBucketCache, CudaViscousFaceGeomBuffer};
use super::BLOCK_THREADS;
use crate::error::{AsimuError, Result};

pub(crate) fn launch_viscous_interior_color_buckets(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    buckets: &CudaViscousBucketCache,
    face_geom: &mut CudaViscousFaceGeomBuffer,
    fields: &mut CudaFieldBuffers,
    gradients_buf: &CudaGradientBuffers,
) -> Result<()> {
    for color in 0..buckets.num_colors() {
        let num_faces = buckets.bucket_len(color)?;
        if num_faces == 0 {
            continue;
        }
        let bucket = buckets.bucket_faces(color)?;
        super::super::viscous::launch_viscous_bucket(
            stream,
            function,
            bucket,
            num_faces,
            face_geom.face_geom(),
            fields,
            gradients_buf,
        )?;
    }
    Ok(())
}

pub(crate) struct InviscidBucketLaunchParams {
    pub gamma: f32,
    pub flux_scheme: u32,
    pub entropy_fix: u32,
}

pub(crate) fn launch_inviscid_bucket(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    bucket_faces: &cudarc::driver::CudaSlice<u32>,
    num_faces: u32,
    face_geom: &cudarc::driver::CudaSlice<super::super::buffers::DeviceFaceGeom>,
    fields: &mut CudaFieldBuffers,
    launch: InviscidBucketLaunchParams,
) -> Result<()> {
    let InviscidBucketLaunchParams {
        gamma,
        flux_scheme,
        entropy_fix,
    } = launch;
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
    builder.arg(&fields.prim_rho);
    builder.arg(&fields.prim_p);
    builder.arg(&fields.prim_ux);
    builder.arg(&fields.prim_uy);
    builder.arg(&fields.prim_uz);
    builder.arg(&mut fields.res_rho);
    builder.arg(&mut fields.res_mx);
    builder.arg(&mut fields.res_my);
    builder.arg(&mut fields.res_mz);
    builder.arg(&mut fields.res_e);
    builder.arg(&gamma);
    builder.arg(&flux_scheme);
    builder.arg(&entropy_fix);
    // SAFETY: 着色桶内面无共享单元；参数布局与 `inviscid_first_order_bucket_f32` 一致。
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}

pub(crate) fn launch_inviscid_boundary(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    faces: &cudarc::driver::CudaSlice<
        super::super::boundary_face_geom::ExecInviscidBoundaryFaceStatic,
    >,
    num_faces: u32,
    ghosts: &cudarc::driver::CudaSlice<
        super::super::spectral_radius_topology::DeviceSpectralGhostPrim,
    >,
    fields: &mut CudaFieldBuffers,
    launch: InviscidBucketLaunchParams,
) -> Result<()> {
    if num_faces == 0 {
        return Ok(());
    }
    let num_blocks = num_faces.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(faces);
    builder.arg(&num_faces);
    builder.arg(ghosts);
    builder.arg(&fields.prim_rho);
    builder.arg(&fields.prim_p);
    builder.arg(&fields.prim_ux);
    builder.arg(&fields.prim_uy);
    builder.arg(&fields.prim_uz);
    builder.arg(&mut fields.res_rho);
    builder.arg(&mut fields.res_mx);
    builder.arg(&mut fields.res_my);
    builder.arg(&mut fields.res_mz);
    builder.arg(&mut fields.res_e);
    builder.arg(&launch.gamma);
    builder.arg(&launch.flux_scheme);
    builder.arg(&launch.entropy_fix);
    // SAFETY: 边界面 owner atomic scatter；布局与 `inviscid_first_order_boundary_f32` 一致。
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA 无粘边界面 kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}
