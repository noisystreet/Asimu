//! LU-SGS 对角更新与 device timestep 下载（`inviscid` 子模块，可访问私有字段）。

use super::super::lusgs_diagonal::launch_lusgs_diagonal_update;
use super::CudaBackendState;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT};

impl CudaBackendState {
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

    pub fn download_min_cell_dt_f32(&mut self) -> Result<f32> {
        if !self.pipeline.timestep_on_device {
            return Err(AsimuError::Exec(
                "CUDA timestep 未在 device 上；请先调用谱半径 CUDA 路径".to_string(),
            ));
        }
        let mesh = self.spectral_mesh.as_mut().expect("spectral mesh");
        mesh.download_min_cell_dt(&self.stream, &self.spectral_module)
    }

    pub fn lusgs_diagonal_update_f32(
        &mut self,
        base: &ConservedFieldsT<f32>,
        residual: &ConservedResidualT<f32>,
        stage: &mut ConservedFieldsT<f32>,
        omega: f32,
    ) -> Result<()> {
        if !self.pipeline.timestep_on_device {
            return Err(AsimuError::Exec(
                "CUDA LU-SGS 对角更新需要 device 上 σ/Δt_i".to_string(),
            ));
        }
        self.ensure_fields(base.num_cells())?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        fields.upload_conserved(&self.stream, base)?;
        if !self.pipeline.residual_on_device {
            fields.upload_full_residual(&self.stream, residual)?;
        }
        let mesh = self.spectral_mesh.as_ref().expect("spectral mesh");
        launch_lusgs_diagonal_update(
            &self.stream,
            &self.lusgs_module.diagonal_update,
            fields,
            mesh.sigma(),
            mesh.cell_dts(),
            omega,
        )?;
        fields.download_conserved(&self.stream, stage)?;
        self.pipeline.residual_on_device = true;
        self.pipeline.timestep_on_device = false;
        Ok(())
    }
}
