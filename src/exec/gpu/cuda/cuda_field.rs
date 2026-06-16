//! 守恒场 device 驻留与 P5 prepare 路径（`inviscid` 子模块）。

use super::super::field::{
    CellStaticTemperatureLaunchArgs, FieldConservedSlices, FieldPrimitiveSlices,
    ViscousDiffusivityLaunchArgs, launch_cell_static_temperature_f32,
    launch_cell_viscous_diffusivity_max, launch_enforce_conserved_positivity_f32,
    launch_fill_primitives_from_conserved,
};
use super::super::viscous_transport_params::build_device_viscous_transport_params;
use super::CudaBackendState;
use crate::core::Real;
use crate::discretization::UnstructuredSolverMeshCache;
use crate::error::{AsimuError, Result};
use crate::field::ConservedFieldsT;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

impl CudaBackendState {
    #[must_use]
    pub(crate) fn conserved_on_device(&self) -> bool {
        self.pipeline.conserved_on_device
    }

    #[must_use]
    pub(crate) fn spectral_diffusivity_on_device(&self) -> bool {
        self.pipeline.spectral_diffusivity_on_device
    }

    #[must_use]
    pub(crate) fn boundary_ghosts_on_device(&self) -> bool {
        self.pipeline.boundary_ghosts_on_device
    }

    #[must_use]
    pub(crate) fn lusgs_diagonal_on_device(&self) -> bool {
        self.pipeline.lusgs_diagonal_on_device
    }

    pub fn upload_conserved_for_integration(
        &mut self,
        conserved: &ConservedFieldsT<f32>,
    ) -> Result<()> {
        if self.pipeline.conserved_on_device {
            return Ok(());
        }
        self.ensure_fields(conserved.num_cells())?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        fields.upload_conserved(&self.stream, conserved)?;
        self.pipeline.conserved_on_device = true;
        Ok(())
    }

    pub fn download_conserved_if_on_device(
        &mut self,
        fields: &mut ConservedFieldsT<f32>,
    ) -> Result<()> {
        if !self.pipeline.conserved_on_device {
            return Ok(());
        }
        let buffers = self.fields.as_ref().expect("field buffers");
        buffers.download_conserved(&self.stream, fields)?;
        self.pipeline.conserved_on_device = false;
        Ok(())
    }

    /// device 守恒场正性钳制（对齐 host `enforce_positivity` 语义；避免步末全表 D2H）。
    pub fn enforce_conserved_positivity_on_device(
        &mut self,
        eos: &IdealGasEoS,
        min_pressure: Real,
    ) -> Result<()> {
        if !self.pipeline.conserved_on_device {
            return Ok(());
        }
        let n = self.fields.as_ref().map(|f| f.num_cells()).unwrap_or(0);
        if n == 0 {
            return Ok(());
        }
        let field_bufs = self.fields.as_ref().expect("field buffers");
        launch_enforce_conserved_positivity_f32(
            &self.stream,
            &self.field_module.enforce_conserved_positivity,
            n as u32,
            eos.gamma as f32,
            min_pressure as f32,
            &FieldConservedSlices {
                rho: &field_bufs.cons_rho,
                mx: &field_bufs.cons_mx,
                my: &field_bufs.cons_my,
                mz: &field_bufs.cons_mz,
                e: &field_bufs.cons_e,
            },
        )?;
        Ok(())
    }

    /// BC 后：device 填原变量 + 单元温/扩散系数（P5；守恒场已驻留时跳过 H2D）。
    pub fn fill_primitives_and_diffusivity_on_device(
        &mut self,
        fields: &ConservedFieldsT<f32>,
        mesh_cache: &UnstructuredSolverMeshCache,
        eos: &IdealGasEoS,
        viscous: &ViscousPhysicsConfig,
        min_pressure: Real,
    ) -> Result<()> {
        let topo_key = std::ptr::from_ref(mesh_cache).addr();
        let spectral_topo = &mesh_cache.spectral_radius_topo;
        self.ensure_fields(fields.num_cells())?;
        self.ensure_spectral_mesh(spectral_topo, topo_key)?;
        self.ensure_idwls_mesh(&mesh_cache.idwls_viscous_topo, topo_key)?;
        let field_bufs = self.fields.as_mut().expect("field buffers after ensure");
        if !self.pipeline.conserved_on_device {
            field_bufs.upload_conserved(&self.stream, fields)?;
            self.pipeline.conserved_on_device = true;
        }
        launch_fill_primitives_from_conserved(
            &self.stream,
            &self.field_module.fill_primitives,
            fields.num_cells() as u32,
            eos.gamma as f32,
            min_pressure as f32,
            &FieldConservedSlices {
                rho: &field_bufs.cons_rho,
                mx: &field_bufs.cons_mx,
                my: &field_bufs.cons_my,
                mz: &field_bufs.cons_mz,
                e: &field_bufs.cons_e,
            },
            &FieldPrimitiveSlices {
                rho: &field_bufs.prim_rho,
                p: &field_bufs.prim_p,
                ux: &field_bufs.prim_ux,
                uy: &field_bufs.prim_uy,
                uz: &field_bufs.prim_uz,
            },
        )?;
        self.primitives_dirty = false;
        self.pipeline.host_bc_primitives_synced = true;
        let nondim_flag = if viscous.is_nondimensional() {
            1.0_f32
        } else {
            0.0_f32
        };
        let idwls_mesh = self.idwls_mesh.as_mut().expect("idwls mesh after ensure");
        launch_cell_static_temperature_f32(
            &self.stream,
            &self.field_module.cell_static_temperature,
            CellStaticTemperatureLaunchArgs {
                num_cells: fields.num_cells() as u32,
                gamma: eos.gamma as f32,
                gas_r: eos.gas_constant as f32,
                nondim_flag,
                prim_rho: &field_bufs.prim_rho,
                prim_p: &field_bufs.prim_p,
                temp_out: idwls_mesh.temperature_mut(),
            },
        )?;
        self.pipeline.cell_temps_on_device = true;
        if !self.pipeline.spectral_diffusivity_on_device {
            let transport = build_device_viscous_transport_params(viscous, eos)?;
            let mesh = self
                .spectral_mesh
                .as_mut()
                .expect("spectral mesh after ensure");
            launch_cell_viscous_diffusivity_max(
                &self.stream,
                &self.field_module.viscous_diffusivity_max,
                ViscousDiffusivityLaunchArgs {
                    num_cells: fields.num_cells() as u32,
                    gamma: eos.gamma as f32,
                    gas_r: eos.gas_constant as f32,
                    nondim_flag,
                    transport,
                    prim_rho: &field_bufs.prim_rho,
                    prim_p: &field_bufs.prim_p,
                    diffusivity_out: mesh.diffusivity_mut(),
                },
            )?;
            self.pipeline.spectral_diffusivity_on_device = true;
        }
        Ok(())
    }

    #[must_use]
    pub(crate) fn u_n_on_device(&self) -> bool {
        self.pipeline.u_n_on_device
    }

    /// 双时间步物理步初：device 守恒场 D2D 快照 \(U^n\)（P3b）。
    pub fn snapshot_u_n_on_device(&mut self, conserved: &ConservedFieldsT<f32>) -> Result<()> {
        self.ensure_fields(conserved.num_cells())?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        if !self.pipeline.conserved_on_device {
            fields.upload_conserved(&self.stream, conserved)?;
            self.pipeline.conserved_on_device = true;
        }
        fields.snapshot_u_n_on_device(&self.stream)?;
        self.pipeline.u_n_on_device = true;
        Ok(())
    }

    /// D2H 下载 device \(U^n\)（物理步边界与 host 存储项衔接）。
    pub fn download_u_n_on_device(&mut self, u_n_out: &mut ConservedFieldsT<f32>) -> Result<()> {
        if !self.pipeline.u_n_on_device {
            return Err(AsimuError::Exec(
                "CUDA U^n 未在 device 上；请先调用 snapshot_u_n_on_device".to_string(),
            ));
        }
        let fields = self.fields.as_ref().expect("field buffers");
        fields.download_u_n(&self.stream, u_n_out)
    }
}
