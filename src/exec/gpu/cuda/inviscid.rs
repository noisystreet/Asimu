//! 一阶无粘内面 CUDA 装配（着色桶 flux + scatter）。

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use tracing::info_span;

use super::buffers::CudaFieldBuffers;
use super::face_geom::ExecInteriorFaceTopology;
use super::mesh_cache::CudaMeshDeviceCache;
use super::module::CudaInviscidModule;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedResidualT, PrimitiveFieldsT};

const BLOCK_THREADS: u32 = 256;

/// CUDA 一阶无粘通量格式（与 `inviscid_first_order_f32.cu` 一致）。
pub const CUDA_FLUX_SCHEME_ROE: u32 = 0;
pub const CUDA_FLUX_SCHEME_HVL: u32 = 1;

/// 一阶无粘内面 kernel launch 参数。
#[derive(Clone, Copy, Debug)]
pub struct CudaFirstOrderInviscidParams {
    pub gamma: f32,
    pub flux_scheme: u32,
    pub roe_entropy_fix: bool,
}

/// G1 CUDA 后端：模块、网格缓存、场缓冲。
pub struct CudaBackendState {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: CudaInviscidModule,
    mesh: Option<CudaMeshDeviceCache>,
    fields: Option<CudaFieldBuffers>,
    mesh_topo_key: Option<usize>,
}

impl CudaBackendState {
    pub fn try_new() -> Result<Self> {
        let context = CudaContext::new(0)
            .map_err(|e| AsimuError::Exec(format!("CUDA 设备初始化失败: {e:?}")))?;
        let stream = context.default_stream();
        let module = CudaInviscidModule::try_load(&context)?;
        Ok(Self {
            context,
            stream,
            module,
            mesh: None,
            fields: None,
            mesh_topo_key: None,
        })
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub fn ensure_mesh(&mut self, topo: &ExecInteriorFaceTopology, topo_key: usize) -> Result<()> {
        if self.mesh_topo_key == Some(topo_key) && self.mesh.is_some() {
            return Ok(());
        }
        let mesh = CudaMeshDeviceCache::try_upload(&self.stream, topo)?;
        self.mesh = Some(mesh);
        self.mesh_topo_key = Some(topo_key);
        Ok(())
    }

    pub fn ensure_fields(&mut self, num_cells: usize) -> Result<()> {
        let need_alloc = self
            .fields
            .as_ref()
            .is_none_or(|f| f.num_cells() != num_cells);
        if need_alloc {
            self.fields = Some(CudaFieldBuffers::try_new(&self.stream, num_cells)?);
        }
        Ok(())
    }

    pub fn assemble_first_order_inviscid_interior(
        &mut self,
        residual: &mut ConservedResidualT<f32>,
        primitives: &PrimitiveFieldsT<f32>,
        topo: &ExecInteriorFaceTopology,
        topo_key: usize,
        params: CudaFirstOrderInviscidParams,
    ) -> Result<()> {
        let entropy_fix = u32::from(params.roe_entropy_fix);
        self.ensure_mesh(topo, topo_key)?;
        self.ensure_fields(primitives.num_cells())?;
        let mesh = self.mesh.as_ref().expect("mesh cache after ensure");
        let fields = self.fields.as_mut().expect("field buffers after ensure");

        fields.upload_primitives(&self.stream, primitives)?;
        fields.zero_residual(&self.stream)?;

        let _span = info_span!(
            "cuda_inviscid_first_order_interior",
            faces = topo.num_interior_faces(),
            colors = topo.num_colors(),
            flux_scheme = params.flux_scheme,
        )
        .entered();

        for color in 0..mesh.num_colors() {
            let num_faces = mesh.bucket_len(color)?;
            if num_faces == 0 {
                continue;
            }
            let bucket = mesh.bucket_faces(color)?;
            launch_inviscid_bucket(
                &self.stream,
                &self.module.function,
                bucket,
                num_faces,
                mesh.face_geom(),
                fields,
                InviscidBucketLaunchParams {
                    gamma: params.gamma,
                    flux_scheme: params.flux_scheme,
                    entropy_fix,
                },
            )?;
        }

        fields.download_residual(&self.stream, residual)?;
        Ok(())
    }

    pub fn sync_to_host(&mut self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| AsimuError::Exec(format!("CUDA 同步失败: {e:?}")))
    }

    pub fn sync_to_device(&mut self) -> Result<()> {
        Ok(())
    }
}

struct InviscidBucketLaunchParams {
    gamma: f32,
    flux_scheme: u32,
    entropy_fix: u32,
}

fn launch_inviscid_bucket(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    bucket_faces: &cudarc::driver::CudaSlice<u32>,
    num_faces: u32,
    face_geom: &cudarc::driver::CudaSlice<super::buffers::DeviceFaceGeom>,
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
