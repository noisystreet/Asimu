//! 一阶无粘内面 CUDA 装配（着色桶 flux + scatter）。

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use tracing::info_span;

use super::buffers::CudaFieldBuffers;
use super::face_geom::ExecInteriorFaceTopology;
use super::gradient_buffers::CudaGradientBuffers;
use super::idwls::{launch_idwls_solve_gradient, launch_idwls_viscous_accumulate};
use super::idwls_mesh_cache::{
    CudaIdwlsMeshDeviceCache, CudaIdwlsRhsDeviceBuffers, IdwlsViscousRhsHostOut,
};
use super::idwls_topology::ExecIdwlsViscousTopology;
use super::mesh_cache::CudaMeshDeviceCache;
use super::module::{
    CudaIdwlsModule, CudaInviscidModule, CudaSpectralRadiusModule, CudaViscousModule,
};
use super::pipeline::CudaPipelineState;
use super::spectral_radius::{launch_finalize_cell_dts, launch_spectral_radius_accumulate};
use super::spectral_radius_mesh_cache::CudaSpectralMeshDeviceCache;
use super::spectral_radius_topology::ExecSpectralRadiusTopology;
use super::spmv::{
    CudaCsrSpmvCache, CusparseHandle, destroy_cusparse_handle, try_create_cusparse_handle,
};
use super::viscous_mesh_cache::{CudaViscousBucketCache, CudaViscousFaceGeomBuffer};
use crate::discretization::unstructured_face_cache_f32::LsqPrecomputedCellF32;
use crate::discretization::unstructured_idwls_exec_topo::IdwlsGhostSampleHost;
use crate::error::{AsimuError, Result};
use crate::exec::CsrSpmvView;
use crate::exec::spectral_radius_cuda::SpectralRadiusCudaInput;
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

/// G1+G2 CUDA 后端：模块、网格缓存、场缓冲。
pub struct CudaBackendState {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: CudaInviscidModule,
    pub(crate) viscous_module: CudaViscousModule,
    idwls_module: CudaIdwlsModule,
    spectral_module: CudaSpectralRadiusModule,
    mesh: Option<CudaMeshDeviceCache>,
    fields: Option<CudaFieldBuffers>,
    mesh_topo_key: Option<usize>,
    idwls_mesh: Option<CudaIdwlsMeshDeviceCache>,
    idwls_rhs: Option<CudaIdwlsRhsDeviceBuffers>,
    idwls_mesh_key: Option<usize>,
    spectral_mesh: Option<CudaSpectralMeshDeviceCache>,
    spectral_mesh_key: Option<usize>,
    viscous_buckets: Option<CudaViscousBucketCache>,
    viscous_face_geom: Option<CudaViscousFaceGeomBuffer>,
    viscous_bucket_key: Option<usize>,
    gradients: Option<CudaGradientBuffers>,
    cusparse_handle: CusparseHandle,
    spmv_cache: CudaCsrSpmvCache,
    /// host 侧 primitive 自上次 H2D 后是否已更新。
    primitives_dirty: bool,
    pipeline: CudaPipelineState,
    idwls_lsq_geometry: Option<cudarc::driver::CudaSlice<LsqPrecomputedCellF32>>,
    idwls_lsq_key: Option<usize>,
}

impl CudaBackendState {
    pub fn try_new() -> Result<Self> {
        let context = CudaContext::new(0)
            .map_err(|e| AsimuError::Exec(format!("CUDA 设备初始化失败: {e:?}")))?;
        let stream = context.default_stream();
        let module = CudaInviscidModule::try_load(&context)?;
        let viscous_module = CudaViscousModule::try_load(&context)?;
        let idwls_module = CudaIdwlsModule::try_load(&context)?;
        let spectral_module = CudaSpectralRadiusModule::try_load(&context)?;
        let cusparse_handle = try_create_cusparse_handle()?;
        tracing::info!("cuda_cusparse_handle_created");
        Ok(Self {
            context,
            stream,
            module,
            viscous_module,
            idwls_module,
            spectral_module,
            mesh: None,
            fields: None,
            mesh_topo_key: None,
            idwls_mesh: None,
            idwls_rhs: None,
            idwls_mesh_key: None,
            spectral_mesh: None,
            spectral_mesh_key: None,
            viscous_buckets: None,
            viscous_face_geom: None,
            viscous_bucket_key: None,
            gradients: None,
            cusparse_handle,
            spmv_cache: CudaCsrSpmvCache::new(),
            primitives_dirty: true,
            pipeline: CudaPipelineState::default(),
            idwls_lsq_geometry: None,
            idwls_lsq_key: None,
        })
    }

    /// BC / 守恒场刷新后调用：下一步 RHS 前将 primitive 上传 device。
    pub fn mark_host_primitives_updated(&mut self) {
        self.primitives_dirty = true;
    }

    pub(crate) fn reset_pipeline_step(&mut self) {
        self.pipeline.reset_step();
    }

    pub(crate) fn enable_rhs_device_pipeline(&mut self) {
        self.pipeline.rhs_pipeline_active = true;
    }

    #[must_use]
    pub(crate) fn rhs_pipeline_active(&self) -> bool {
        self.pipeline.rhs_pipeline_active
    }

    #[must_use]
    pub(crate) fn timestep_on_device(&self) -> bool {
        self.pipeline.timestep_on_device
    }

    pub(crate) fn upload_residual_from_host(
        &mut self,
        residual: &ConservedResidualT<f32>,
    ) -> Result<()> {
        self.ensure_fields(residual.num_cells())?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        fields.upload_full_residual(&self.stream, residual)?;
        self.pipeline.residual_on_device = true;
        Ok(())
    }

    pub(crate) fn flush_residual_to_host(
        &mut self,
        residual: &mut ConservedResidualT<f32>,
    ) -> Result<()> {
        if !self.pipeline.residual_on_device {
            return Ok(());
        }
        let fields = self.fields.as_ref().expect("field buffers");
        fields.download_residual(&self.stream, residual)?;
        self.pipeline.residual_on_device = false;
        Ok(())
    }

    pub(crate) fn flush_gradients_to_host(
        &mut self,
        gradients: &mut crate::discretization::gradient_typed::GradientFieldsT<f32>,
    ) -> Result<()> {
        if !self.pipeline.gradients_on_device {
            return Ok(());
        }
        let buf = self.gradients.as_ref().expect("gradient buffers");
        buf.download_to_host(&self.stream, gradients)?;
        self.pipeline.gradients_on_device = false;
        Ok(())
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
        defer_residual_d2h: bool,
    ) -> Result<()> {
        let entropy_fix = u32::from(params.roe_entropy_fix);
        self.ensure_mesh(topo, topo_key)?;
        self.ensure_fields(primitives.num_cells())?;
        let mesh = self.mesh.as_ref().expect("mesh cache after ensure");
        let fields = self.fields.as_mut().expect("field buffers after ensure");

        if self.primitives_dirty {
            fields.upload_primitives(&self.stream, primitives)?;
            self.primitives_dirty = false;
        }
        if !self.pipeline.residual_on_device {
            fields.zero_residual(&self.stream)?;
        }

        let _span = info_span!(
            "cuda_inviscid_first_order_interior",
            faces = topo.num_interior_faces(),
            colors = topo.num_colors(),
            flux_scheme = params.flux_scheme,
            defer_d2h = defer_residual_d2h,
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

        if defer_residual_d2h {
            self.pipeline.residual_on_device = true;
        } else {
            fields.download_residual(&self.stream, residual)?;
            self.pipeline.residual_on_device = false;
        }
        Ok(())
    }

    pub fn sync_to_host(&mut self) -> Result<()> {
        let _span = info_span!("cuda_sync_to_host").entered();
        self.stream
            .synchronize()
            .map_err(|e| AsimuError::Exec(format!("CUDA 同步失败: {e:?}")))
    }

    /// BC 更新后将 host primitive 写回 device（仅当 `primitives_dirty`）。
    pub fn sync_primitives_to_device(&mut self, primitives: &PrimitiveFieldsT<f32>) -> Result<()> {
        if !self.primitives_dirty {
            return Ok(());
        }
        self.ensure_fields(primitives.num_cells())?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        fields.upload_primitives(&self.stream, primitives)?;
        self.primitives_dirty = false;
        Ok(())
    }

    pub fn sync_to_device(&mut self, primitives: Option<&PrimitiveFieldsT<f32>>) -> Result<()> {
        if let Some(prim) = primitives {
            self.sync_primitives_to_device(prim)?;
        }
        Ok(())
    }

    pub fn assemble_viscous_interior(
        &mut self,
        residual: &mut ConservedResidualT<f32>,
        primitives: &PrimitiveFieldsT<f32>,
        gradients: &crate::discretization::gradient_typed::GradientFieldsT<f32>,
        topo: &super::viscous_face_geom::ExecViscousInteriorTopology,
        topo_key: usize,
        defer_residual_d2h: bool,
    ) -> Result<()> {
        self.ensure_viscous_resources(topo, topo_key)?;
        self.ensure_fields(primitives.num_cells())?;
        self.ensure_gradient_buffers(primitives.num_cells())?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        let gradients_buf = self
            .gradients
            .as_mut()
            .expect("gradient buffers after ensure");
        let buckets = self
            .viscous_buckets
            .as_ref()
            .expect("viscous buckets after ensure");
        let face_geom = self
            .viscous_face_geom
            .as_mut()
            .expect("viscous face geom after ensure");

        if self.primitives_dirty {
            fields.upload_primitives(&self.stream, primitives)?;
            self.primitives_dirty = false;
        }
        if !self.pipeline.gradients_on_device {
            gradients_buf.upload(&self.stream, gradients)?;
        }
        face_geom.refresh(&self.stream, &topo.faces)?;
        if !self.pipeline.residual_on_device {
            fields.upload_momentum_energy_residual(&self.stream, residual)?;
        }

        let _span = info_span!(
            "cuda_viscous_interior",
            faces = topo.num_interior_faces(),
            colors = topo.num_colors(),
            defer_d2h = defer_residual_d2h,
            gradients_on_device = self.pipeline.gradients_on_device,
        )
        .entered();

        launch_viscous_interior_color_buckets(
            &self.stream,
            &self.viscous_module.function,
            buckets,
            face_geom,
            fields,
            gradients_buf,
        )?;

        if defer_residual_d2h {
            self.pipeline.residual_on_device = true;
        } else {
            fields.download_momentum_energy_residual(&self.stream, residual)?;
            self.pipeline.residual_on_device = false;
        }
        Ok(())
    }

    fn ensure_idwls_lsq_geometry(
        &mut self,
        geometry: &[LsqPrecomputedCellF32],
        topo_key: usize,
    ) -> Result<()> {
        if self.idwls_lsq_key == Some(topo_key) && self.idwls_lsq_geometry.is_some() {
            return Ok(());
        }
        use super::transfer::clone_htod;
        self.idwls_lsq_geometry = Some(clone_htod(&self.stream, "idwls_lsq_geometry", geometry)?);
        self.idwls_lsq_key = Some(topo_key);
        Ok(())
    }

    fn ensure_idwls_mesh(
        &mut self,
        topo: &ExecIdwlsViscousTopology,
        topo_key: usize,
    ) -> Result<()> {
        if self.idwls_mesh_key == Some(topo_key)
            && self.idwls_mesh.is_some()
            && self.idwls_rhs.is_some()
        {
            return Ok(());
        }
        let (mesh, rhs) = CudaIdwlsMeshDeviceCache::try_upload(&self.stream, topo)?;
        self.idwls_mesh = Some(mesh);
        self.idwls_rhs = Some(rhs);
        self.idwls_mesh_key = Some(topo_key);
        Ok(())
    }

    /// P4+P1：device 累加 IDWLS RHS 并求解梯度（跳过 RHS D2H）。
    pub fn accumulate_and_solve_idwls_viscous_gradients(
        &mut self,
        primitives: &PrimitiveFieldsT<f32>,
        topo: &ExecIdwlsViscousTopology,
        topo_key: usize,
        lsq_geometry: &[LsqPrecomputedCellF32],
        temperatures: &[f32],
        boundary_ghosts: &[IdwlsGhostSampleHost],
    ) -> Result<()> {
        self.ensure_idwls_mesh(topo, topo_key)?;
        self.ensure_idwls_lsq_geometry(lsq_geometry, topo_key)?;
        self.ensure_fields(primitives.num_cells())?;
        self.ensure_gradient_buffers(primitives.num_cells())?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        if self.primitives_dirty {
            fields.upload_primitives(&self.stream, primitives)?;
            self.primitives_dirty = false;
        }
        let mesh = self.idwls_mesh.as_mut().expect("idwls mesh after ensure");
        mesh.upload_temperature(&self.stream, temperatures)?;
        mesh.upload_boundary_ghosts(&self.stream, boundary_ghosts)?;
        let rhs = self.idwls_rhs.as_mut().expect("idwls rhs after ensure");
        launch_idwls_viscous_accumulate(
            &self.stream,
            &self.idwls_module.accumulate,
            mesh,
            fields,
            rhs,
        )?;
        let lsq = self.idwls_lsq_geometry.as_ref().expect("lsq geometry");
        let gradients = self.gradients.as_mut().expect("gradient buffers");
        launch_idwls_solve_gradient(
            &self.stream,
            &self.idwls_module.solve_gradient,
            lsq,
            rhs,
            gradients,
        )?;
        self.pipeline.gradients_on_device = true;
        Ok(())
    }

    /// P4：device 上累加粘性 IDWLS RHS，D2H 写回 host `IdwlsRhsBuffer` f32 槽。
    pub fn accumulate_idwls_viscous_rhs(
        &mut self,
        primitives: &PrimitiveFieldsT<f32>,
        topo: &ExecIdwlsViscousTopology,
        topo_key: usize,
        temperatures: &[f32],
        boundary_ghosts: &[IdwlsGhostSampleHost],
        out: IdwlsViscousRhsHostOut<'_>,
    ) -> Result<()> {
        self.ensure_idwls_mesh(topo, topo_key)?;
        self.ensure_fields(primitives.num_cells())?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        if self.primitives_dirty {
            fields.upload_primitives(&self.stream, primitives)?;
            self.primitives_dirty = false;
        }
        let mesh = self.idwls_mesh.as_mut().expect("idwls mesh after ensure");
        mesh.upload_temperature(&self.stream, temperatures)?;
        mesh.upload_boundary_ghosts(&self.stream, boundary_ghosts)?;
        let rhs = self.idwls_rhs.as_mut().expect("idwls rhs after ensure");
        launch_idwls_viscous_accumulate(
            &self.stream,
            &self.idwls_module.accumulate,
            mesh,
            fields,
            rhs,
        )?;
        rhs.download_into(&self.stream, out)?;
        Ok(())
    }

    fn ensure_spectral_mesh(
        &mut self,
        topo: &ExecSpectralRadiusTopology,
        topo_key: usize,
    ) -> Result<()> {
        if self.spectral_mesh_key == Some(topo_key) && self.spectral_mesh.is_some() {
            return Ok(());
        }
        self.spectral_mesh = Some(CudaSpectralMeshDeviceCache::try_upload(&self.stream, topo)?);
        self.spectral_mesh_key = Some(topo_key);
        Ok(())
    }

    /// 非结构单元谱半径 + device 上 finalize `cell_dts`（P1 可延迟 D2H）。
    pub fn compute_spectral_radius_unstructured_f32(
        &mut self,
        input: &SpectralRadiusCudaInput<'_>,
        sigma_out: &mut [f32],
    ) -> Result<()> {
        self.ensure_spectral_mesh(input.topo, input.topo_key)?;
        self.ensure_fields(input.primitives.num_cells())?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        if self.primitives_dirty {
            fields.upload_primitives(&self.stream, input.primitives)?;
            self.primitives_dirty = false;
        }
        let mesh = self
            .spectral_mesh
            .as_mut()
            .expect("spectral mesh after ensure");
        mesh.upload_boundary_ghosts(&self.stream, input.boundary_ghosts)?;
        mesh.upload_diffusivity(&self.stream, input.diffusivity)?;
        launch_spectral_radius_accumulate(
            &self.stream,
            &self.spectral_module.accumulate,
            mesh,
            fields,
            input.gamma,
            input.diffusivity.is_some(),
        )?;
        launch_finalize_cell_dts(
            &self.stream,
            &self.spectral_module.finalize_dts,
            mesh,
            input.cfl,
            input.fixed_dt,
        )?;
        self.pipeline.timestep_on_device = true;
        if input.defer_timestep_d2h {
            return Ok(());
        }
        let n = mesh.num_cells();
        if sigma_out.len() != n {
            return Err(AsimuError::Field(format!("host sigma 长度须为 {n}")));
        }
        let mut cell_dts = vec![0.0f32; n];
        mesh.download_timestep(&self.stream, sigma_out, &mut cell_dts)?;
        self.pipeline.timestep_on_device = false;
        let _ = cell_dts;
        Ok(())
    }

    pub fn download_timestep_f32(
        &mut self,
        sigma_out: &mut [f32],
        cell_dts_out: &mut [f32],
        local_time_step: bool,
    ) -> Result<()> {
        if !self.pipeline.timestep_on_device {
            return Err(AsimuError::Exec(
                "CUDA timestep 未在 device 上；请先调用谱半径 CUDA 路径".to_string(),
            ));
        }
        let mesh = self.spectral_mesh.as_ref().expect("spectral mesh");
        mesh.download_timestep(&self.stream, sigma_out, cell_dts_out)?;
        if !local_time_step {
            let min_dt = cell_dts_out
                .iter()
                .copied()
                .filter(|d| d.is_finite() && *d > 0.0)
                .fold(f32::INFINITY, f32::min);
            if min_dt.is_finite() {
                cell_dts_out.fill(min_dt);
            }
        }
        self.pipeline.timestep_on_device = false;
        Ok(())
    }

    fn ensure_viscous_resources(
        &mut self,
        topo: &super::viscous_face_geom::ExecViscousInteriorTopology,
        topo_key: usize,
    ) -> Result<()> {
        let need_buckets = self
            .viscous_bucket_key
            .is_none_or(|k| k != topo_key || self.viscous_buckets.is_none());
        if need_buckets {
            self.viscous_buckets = Some(CudaViscousBucketCache::try_upload(&self.stream, topo)?);
            self.viscous_bucket_key = Some(topo_key);
        }
        let need_geom = self
            .viscous_face_geom
            .as_ref()
            .is_none_or(|g| g.face_geom().len() != topo.faces.len());
        if need_geom {
            self.viscous_face_geom = Some(CudaViscousFaceGeomBuffer::try_upload(
                &self.stream,
                &topo.faces,
            )?);
        }
        Ok(())
    }

    fn ensure_gradient_buffers(&mut self, num_cells: usize) -> Result<()> {
        let need_alloc = self
            .gradients
            .as_ref()
            .is_none_or(|g| g.num_cells() != num_cells);
        if need_alloc {
            self.gradients = Some(CudaGradientBuffers::try_new(&self.stream, num_cells)?);
        }
        Ok(())
    }

    /// G3：cuSPARSE CSR SpMV（f64；隐式路径预研入口）。
    pub fn csr_spmv(
        &mut self,
        matrix: &CsrSpmvView<'_>,
        x: &[crate::core::Real],
        y: &mut [crate::core::Real],
    ) -> Result<()> {
        super::spmv::csr_spmv_f64(
            &self.stream,
            self.cusparse_handle,
            &mut self.spmv_cache,
            matrix,
            x,
            y,
        )
    }
}

impl Drop for CudaBackendState {
    fn drop(&mut self) {
        let _ = destroy_cusparse_handle(self.cusparse_handle);
    }
}

fn launch_viscous_interior_color_buckets(
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
        super::viscous::launch_viscous_bucket(
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
