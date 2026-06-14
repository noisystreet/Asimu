//! 边界面 CUDA 装配（P2：无粘/粘性；残差与梯度保持 device）。

use tracing::info_span;

use super::super::boundary_face_geom::{
    CudaViscousBoundaryInput, ExecInviscidBoundaryTopology, ExecViscousBoundaryTopology,
};
use super::super::boundary_mesh_cache::{
    CudaInviscidBoundaryMeshCache, CudaViscousBoundaryMeshCache,
};
use super::super::spectral_radius_topology::DeviceSpectralGhostPrim;
use super::super::viscous::{ViscousBoundaryLaunch, launch_viscous_boundary};
use super::super::viscous_transport_params::build_device_viscous_transport_params;
use super::inviscid_launch::{InviscidBucketLaunchParams, launch_inviscid_boundary};
use super::{CudaBackendState, CudaFirstOrderInviscidParams};
use crate::core::Real;
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::gradient_unstructured_f32::cell_static_temperatures_f32;
use crate::discretization::unstructured_boundary_exec_topo::{
    prepare_idwls_boundary_ghost_samples_f32, prepare_inviscid_boundary_ghost_prims_f32,
    prepare_viscous_boundary_ghost_prims_f32,
};
use crate::discretization::unstructured_spectral_exec_topo::SpectralGhostPrimHost;
use crate::discretization::{BoundaryGhostBuffer, UnstructuredSolverMeshCache};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedResidualT, PrimitiveFieldsT};
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

/// prepare 后一次性上传 RHS 所需 boundary ghost / 单元温度至 device。
pub struct CudaPrepareRhsDeviceInput<'a> {
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFieldsT<f32>,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub min_pressure: Real,
}

impl CudaBackendState {
    pub fn prepare_rhs_device_state(&mut self, input: CudaPrepareRhsDeviceInput<'_>) -> Result<()> {
        let topo_key = std::ptr::from_ref(input.mesh_cache).addr();
        let face_topo = &input.mesh_cache.face_topology_f32;

        let idwls_topo = &input.mesh_cache.idwls_viscous_topo;
        self.ensure_idwls_mesh(idwls_topo, topo_key)?;
        let idwls_ghosts = prepare_idwls_boundary_ghost_samples_f32(
            face_topo,
            input.ghosts,
            input.eos,
            input.viscous,
            input.min_pressure,
        )?;
        let idwls_mesh = self.idwls_mesh.as_mut().expect("idwls mesh after ensure");
        idwls_mesh.upload_boundary_ghosts(&self.stream, &idwls_ghosts)?;
        let mut temps = Vec::new();
        cell_static_temperatures_f32(input.primitives, input.eos, input.viscous, &mut temps)?;
        idwls_mesh.upload_temperature(&self.stream, &temps)?;

        let inviscid_topo = &input.mesh_cache.cuda_inviscid_boundary_topo;
        if inviscid_topo.num_faces() > 0 {
            self.ensure_inviscid_boundary_mesh(inviscid_topo, topo_key)?;
            let inv_ghosts = prepare_inviscid_boundary_ghost_prims_f32(
                face_topo,
                input.ghosts,
                input.eos,
                input.min_pressure,
            )?;
            let ghosts_device: Vec<DeviceSpectralGhostPrim> = inv_ghosts
                .iter()
                .map(|g| DeviceSpectralGhostPrim {
                    rho: g.rho,
                    pressure: g.pressure,
                    u: g.u,
                    v: g.v,
                    w: g.w,
                })
                .collect();
            let inv_mesh = self
                .inviscid_boundary_mesh
                .as_mut()
                .expect("inviscid boundary mesh after ensure");
            inv_mesh.upload_ghosts(&self.stream, &ghosts_device)?;
        }

        let viscous_topo = &input.mesh_cache.cuda_viscous_boundary_topo;
        if viscous_topo.num_faces() > 0 {
            self.ensure_viscous_boundary_mesh(viscous_topo, topo_key)?;
            let visc_ghosts = prepare_viscous_boundary_ghost_prims_f32(
                face_topo,
                input.ghosts,
                input.eos,
                input.viscous,
                input.min_pressure,
            )?;
            let visc_mesh = self
                .viscous_boundary_mesh
                .as_mut()
                .expect("viscous boundary mesh after ensure");
            visc_mesh.upload_ghosts(&self.stream, &visc_ghosts)?;
        }

        self.pipeline.boundary_ghosts_on_device = true;
        self.pipeline.cell_temps_on_device = true;
        Ok(())
    }

    pub(crate) fn ensure_inviscid_boundary_mesh(
        &mut self,
        topo: &ExecInviscidBoundaryTopology,
        topo_key: usize,
    ) -> Result<()> {
        if self.inviscid_boundary_topo_key == Some(topo_key)
            && self.inviscid_boundary_mesh.is_some()
        {
            return Ok(());
        }
        let mesh = CudaInviscidBoundaryMeshCache::try_upload(&self.stream, topo)?;
        self.inviscid_boundary_mesh = Some(mesh);
        self.inviscid_boundary_topo_key = Some(topo_key);
        Ok(())
    }

    pub(crate) fn ensure_viscous_boundary_mesh(
        &mut self,
        topo: &ExecViscousBoundaryTopology,
        topo_key: usize,
    ) -> Result<()> {
        if self.viscous_boundary_topo_key == Some(topo_key) && self.viscous_boundary_mesh.is_some()
        {
            return Ok(());
        }
        let mesh = CudaViscousBoundaryMeshCache::try_upload(&self.stream, topo)?;
        self.viscous_boundary_mesh = Some(mesh);
        self.viscous_boundary_topo_key = Some(topo_key);
        Ok(())
    }

    pub(crate) fn viscous_transport_temps_slice<'a>(
        pipeline: &super::super::pipeline::CudaPipelineState,
        idwls_mesh: Option<&'a super::super::idwls_mesh_cache::CudaIdwlsMeshDeviceCache>,
        viscous_transport_temps: Option<&'a cudarc::driver::CudaSlice<f32>>,
    ) -> Result<&'a cudarc::driver::CudaSlice<f32>> {
        if pipeline.cell_temps_on_device {
            let mesh = idwls_mesh.ok_or_else(|| {
                AsimuError::Exec("cell_temps_on_device 但 IDWLS mesh 未初始化".to_string())
            })?;
            return Ok(mesh.temperature());
        }
        viscous_transport_temps
            .ok_or_else(|| AsimuError::Exec("粘性传输温度未在 device 上".to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn assemble_first_order_inviscid_boundary(
        &mut self,
        residual: &mut ConservedResidualT<f32>,
        primitives: &PrimitiveFieldsT<f32>,
        topo: &ExecInviscidBoundaryTopology,
        topo_key: usize,
        boundary_ghosts: &[SpectralGhostPrimHost],
        params: CudaFirstOrderInviscidParams,
        defer_residual_d2h: bool,
    ) -> Result<()> {
        if topo.num_faces() == 0 {
            return Ok(());
        }
        self.ensure_inviscid_boundary_mesh(topo, topo_key)?;
        self.ensure_fields(primitives.num_cells())?;
        if !self.pipeline.boundary_ghosts_on_device {
            let ghosts_device: Vec<DeviceSpectralGhostPrim> = boundary_ghosts
                .iter()
                .map(|g| DeviceSpectralGhostPrim {
                    rho: g.rho,
                    pressure: g.pressure,
                    u: g.u,
                    v: g.v,
                    w: g.w,
                })
                .collect();
            self.inviscid_boundary_mesh
                .as_mut()
                .expect("inviscid boundary mesh after ensure")
                .upload_ghosts(&self.stream, &ghosts_device)?;
        }
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        if self.primitives_dirty {
            fields.upload_primitives(&self.stream, primitives)?;
            self.primitives_dirty = false;
            self.pipeline.host_bc_primitives_synced = true;
        }
        let mesh = self
            .inviscid_boundary_mesh
            .as_ref()
            .expect("inviscid boundary mesh after ensure");
        let entropy_fix = u32::from(params.roe_entropy_fix);
        let _span = info_span!(
            "cuda_inviscid_first_order_boundary",
            faces = topo.num_faces(),
            flux_scheme = params.flux_scheme,
            defer_d2h = defer_residual_d2h,
        )
        .entered();
        launch_inviscid_boundary(
            &self.stream,
            &self.module.boundary_function,
            mesh.faces(),
            topo.num_faces() as u32,
            mesh.ghosts(),
            fields,
            InviscidBucketLaunchParams {
                gamma: params.gamma,
                flux_scheme: params.flux_scheme,
                entropy_fix,
            },
        )?;
        Self::finish_boundary_residual_transfer(
            &mut self.pipeline,
            &self.stream,
            fields,
            residual,
            defer_residual_d2h,
        )?;
        Ok(())
    }

    pub fn assemble_viscous_boundary(
        &mut self,
        residual: &mut ConservedResidualT<f32>,
        primitives: &PrimitiveFieldsT<f32>,
        gradients: &GradientFieldsT<f32>,
        input: CudaViscousBoundaryInput<'_>,
        defer_residual_d2h: bool,
    ) -> Result<()> {
        let topo = input.topo;
        if topo.num_faces() == 0 {
            return Ok(());
        }
        self.ensure_viscous_boundary_mesh(topo, input.topo_key)?;
        self.ensure_fields(primitives.num_cells())?;
        self.ensure_gradient_buffers(primitives.num_cells())?;
        self.upload_viscous_transport_temps(input.temperatures)?;
        self.prepare_viscous_boundary_device_inputs(primitives, gradients, input.boundary_ghosts)?;
        let transport = build_device_viscous_transport_params(input.viscous, input.eos)?;
        let _span = info_span!(
            "cuda_viscous_boundary",
            faces = topo.num_faces(),
            defer_d2h = defer_residual_d2h,
        )
        .entered();
        let temps = Self::viscous_transport_temps_slice(
            &self.pipeline,
            self.idwls_mesh.as_ref(),
            self.viscous_transport_temps.as_ref(),
        )?;
        let mesh = self
            .viscous_boundary_mesh
            .as_ref()
            .expect("viscous boundary mesh after ensure");
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        let gradients_buf = self
            .gradients
            .as_mut()
            .expect("gradient buffers after ensure");
        launch_viscous_boundary(
            &self.stream,
            &self.viscous_module.boundary_function,
            ViscousBoundaryLaunch {
                faces: mesh.faces(),
                num_faces: topo.num_faces() as u32,
                ghosts: mesh.ghosts(),
                temperatures: temps,
                fields,
                gradients: gradients_buf,
                transport,
            },
        )?;
        Self::finish_boundary_residual_transfer(
            &mut self.pipeline,
            &self.stream,
            fields,
            residual,
            defer_residual_d2h,
        )?;
        Ok(())
    }

    fn prepare_viscous_boundary_device_inputs(
        &mut self,
        primitives: &PrimitiveFieldsT<f32>,
        gradients: &GradientFieldsT<f32>,
        boundary_ghosts: &[super::super::boundary_face_geom::ViscousBoundaryGhostHost],
    ) -> Result<()> {
        if !self.pipeline.boundary_ghosts_on_device {
            self.viscous_boundary_mesh
                .as_mut()
                .expect("viscous boundary mesh after ensure")
                .upload_ghosts(&self.stream, boundary_ghosts)?;
        }
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        if self.primitives_dirty {
            fields.upload_primitives(&self.stream, primitives)?;
            self.primitives_dirty = false;
            self.pipeline.host_bc_primitives_synced = true;
        }
        let gradients_buf = self
            .gradients
            .as_mut()
            .expect("gradient buffers after ensure");
        if !self.pipeline.gradients_on_device {
            gradients_buf.upload(&self.stream, gradients)?;
            self.pipeline.gradients_on_device = true;
        }
        Ok(())
    }

    fn finish_boundary_residual_transfer(
        pipeline: &mut super::super::pipeline::CudaPipelineState,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        fields: &super::super::buffers::CudaFieldBuffers,
        residual: &mut ConservedResidualT<f32>,
        defer_d2h: bool,
    ) -> Result<()> {
        if defer_d2h {
            pipeline.residual_on_device = true;
        } else {
            fields.download_residual(stream, residual)?;
            pipeline.residual_on_device = false;
        }
        Ok(())
    }

    pub(crate) fn upload_viscous_transport_temps(&mut self, temperatures: &[f32]) -> Result<()> {
        if self.pipeline.cell_temps_on_device {
            return Ok(());
        }
        let num_cells = temperatures.len();
        let need_temps = self
            .viscous_transport_temps
            .as_ref()
            .is_none_or(|t| t.len() != num_cells);
        if need_temps {
            use super::super::transfer::clone_htod;
            self.viscous_transport_temps = Some(clone_htod(
                &self.stream,
                "viscous_transport_temps_alloc",
                temperatures,
            )?);
        } else {
            use super::super::transfer::memcpy_htod;
            let temps = self
                .viscous_transport_temps
                .as_mut()
                .expect("viscous transport temps");
            memcpy_htod(&self.stream, "viscous_transport_temps", temperatures, temps)?;
        }
        Ok(())
    }
}
