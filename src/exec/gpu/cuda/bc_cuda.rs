//! 可压缩 BC device 拓扑缓存 + launch。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use tracing::info_span;

use super::bc_cuda_topology::{
    DeviceBcFaceStatic, DeviceBcPatchParams, ExecCompressibleBcTopology,
};
use super::transfer::clone_htod;
use crate::error::{AsimuError, Result};

const BLOCK_THREADS: u32 = 256;

pub struct CudaBcMeshCache {
    faces: CudaSlice<DeviceBcFaceStatic>,
    patches: CudaSlice<DeviceBcPatchParams>,
}

impl CudaBcMeshCache {
    pub fn try_upload(stream: &Arc<CudaStream>, topo: &ExecCompressibleBcTopology) -> Result<Self> {
        let num_faces = topo.num_faces();
        let faces = if num_faces == 0 {
            clone_htod(
                stream,
                "init_bc_faces_empty",
                &[DeviceBcFaceStatic::default()],
            )?
        } else {
            clone_htod(stream, "init_bc_faces", &topo.faces)?
        };
        let patches = clone_htod(stream, "init_bc_patches", &topo.patches)?;
        Ok(Self { faces, patches })
    }

    pub(crate) fn faces(&self) -> &CudaSlice<DeviceBcFaceStatic> {
        &self.faces
    }

    pub(crate) fn patches(&self) -> &CudaSlice<DeviceBcPatchParams> {
        &self.patches
    }
}

pub struct ApplyBcGhostsLaunchArgs<'a> {
    pub num_faces: u32,
    pub gamma: f32,
    pub gas_r: f32,
    pub min_pressure: f32,
    pub nondim_flag: f32,
    pub fs_mach: f32,
    pub fs_pressure: f32,
    pub fs_temperature: f32,
    pub fs_dir_x: f32,
    pub fs_dir_y: f32,
    pub fs_dir_z: f32,
    pub bc_mesh: &'a CudaBcMeshCache,
    pub cons_rho: &'a CudaSlice<f32>,
    pub cons_mx: &'a CudaSlice<f32>,
    pub cons_my: &'a CudaSlice<f32>,
    pub cons_mz: &'a CudaSlice<f32>,
    pub cons_e: &'a CudaSlice<f32>,
    pub ghost_out: &'a mut CudaSlice<super::boundary_face_geom::BoundaryConservedGhostHost>,
}

pub fn launch_apply_compressible_boundary_ghosts_f32(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    args: ApplyBcGhostsLaunchArgs<'_>,
) -> Result<()> {
    let _span = info_span!(
        "cuda_apply_compressible_boundary_ghosts",
        faces = args.num_faces
    )
    .entered();
    if args.num_faces == 0 {
        return Ok(());
    }
    let num_blocks = args.num_faces.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&args.num_faces);
    builder.arg(&args.gamma);
    builder.arg(&args.gas_r);
    builder.arg(&args.min_pressure);
    builder.arg(&args.nondim_flag);
    builder.arg(&args.fs_mach);
    builder.arg(&args.fs_pressure);
    builder.arg(&args.fs_temperature);
    builder.arg(&args.fs_dir_x);
    builder.arg(&args.fs_dir_y);
    builder.arg(&args.fs_dir_z);
    builder.arg(args.bc_mesh.faces());
    builder.arg(args.bc_mesh.patches());
    builder.arg(args.cons_rho);
    builder.arg(args.cons_mx);
    builder.arg(args.cons_my);
    builder.arg(args.cons_mz);
    builder.arg(args.cons_e);
    builder.arg(args.ghost_out);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!(
                "CUDA apply_boundary_ghosts kernel launch 失败: {e:?}"
            ))
        })?;
    }
    Ok(())
}
