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
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::unstructured_spectral_exec_topo::SpectralGhostPrimHost;
use crate::error::Result;
use crate::field::{ConservedResidualT, PrimitiveFieldsT};

impl CudaBackendState {
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
        let mesh = self
            .inviscid_boundary_mesh
            .as_mut()
            .expect("inviscid boundary mesh after ensure");
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
        mesh.upload_ghosts(&self.stream, &ghosts_device)?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        if self.primitives_dirty {
            fields.upload_primitives(&self.stream, primitives)?;
            self.primitives_dirty = false;
        }
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
        if defer_residual_d2h {
            self.pipeline.residual_on_device = true;
        } else {
            fields.download_residual(&self.stream, residual)?;
            self.pipeline.residual_on_device = false;
        }
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
        let mesh = self
            .viscous_boundary_mesh
            .as_mut()
            .expect("viscous boundary mesh after ensure");
        mesh.upload_ghosts(&self.stream, input.boundary_ghosts)?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        let gradients_buf = self
            .gradients
            .as_mut()
            .expect("gradient buffers after ensure");
        if self.primitives_dirty {
            fields.upload_primitives(&self.stream, primitives)?;
            self.primitives_dirty = false;
        }
        if !self.pipeline.gradients_on_device {
            gradients_buf.upload(&self.stream, gradients)?;
            self.pipeline.gradients_on_device = true;
        }
        let transport = build_device_viscous_transport_params(input.viscous, input.eos)?;
        let _span = info_span!(
            "cuda_viscous_boundary",
            faces = topo.num_faces(),
            defer_d2h = defer_residual_d2h,
        )
        .entered();
        let temps = self
            .viscous_transport_temps
            .as_ref()
            .expect("viscous transport temps after upload");
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
        if defer_residual_d2h {
            self.pipeline.residual_on_device = true;
        } else {
            fields.download_residual(&self.stream, residual)?;
            self.pipeline.residual_on_device = false;
        }
        Ok(())
    }

    pub(crate) fn upload_viscous_transport_temps(&mut self, temperatures: &[f32]) -> Result<()> {
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
