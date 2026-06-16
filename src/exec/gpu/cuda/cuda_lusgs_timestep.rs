//! LU-SGS 对角更新、扫掠与 device timestep 下载（`inviscid` 子模块，可访问私有字段）。

use super::super::lusgs_diagonal::{launch_lusgs_diagonal_update, launch_residual_density_sum_sq};
use super::super::lusgs_sweep::{
    LusgsSweepCudaHostInput, LusgsSweepCudaLaunchBuffers, LusgsSweepCudaScalars,
    launch_lusgs_sweep_unstructured_serial,
};
use super::super::lusgs_sweep_mesh_cache::upload_u0_snapshot;
use super::CudaBackendState;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT};
use crate::solver::compressible::lu_sgs_common::{
    LuSgsSweepScalarsF32, stabilize_sweep_update_f32,
};

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

    /// LU-SGS 对角更新：守恒场已在 device 时跳过 H2D/D2H（P4）。
    pub fn lusgs_diagonal_update_f32(
        &mut self,
        base: &ConservedFieldsT<f32>,
        residual: &ConservedResidualT<f32>,
        omega: f32,
        inv_dt_phys: f32,
    ) -> Result<()> {
        if !self.pipeline.timestep_on_device {
            return Err(AsimuError::Exec(
                "CUDA LU-SGS 对角更新需要 device 上 σ/Δt_i".to_string(),
            ));
        }
        self.ensure_fields(base.num_cells())?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        if !self.pipeline.conserved_on_device {
            fields.upload_conserved(&self.stream, base)?;
            self.pipeline.conserved_on_device = true;
        }
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
            inv_dt_phys,
        )?;
        self.pipeline.residual_on_device = true;
        self.pipeline.timestep_on_device = false;
        self.pipeline.conserved_on_device = true;
        self.pipeline.lusgs_diagonal_on_device = true;
        Ok(())
    }

    /// device 密度残差 RMS（单 float D2H；替代全量残差 D2H）。
    pub fn density_residual_rms_f32(&mut self) -> Result<f32> {
        if !self.pipeline.residual_on_device {
            return Err(AsimuError::Exec(
                "CUDA 密度残差 RMS 需要 residual 在 device 上".to_string(),
            ));
        }
        let fields = self.fields.as_ref().expect("field buffers");
        let n = fields.num_cells();
        if n == 0 {
            return Ok(0.0);
        }
        if self
            .residual_sum_sq_scratch
            .as_ref()
            .is_none_or(|s| s.len() != 1)
        {
            self.residual_sum_sq_scratch = Some(
                self.stream
                    .alloc_zeros::<f32>(1)
                    .map_err(|e| AsimuError::Exec(format!("CUDA sum_sq 分配失败: {e:?}")))?,
            );
        }
        let sum_buf = self
            .residual_sum_sq_scratch
            .as_mut()
            .expect("sum_sq scratch after ensure");
        launch_residual_density_sum_sq(
            &self.stream,
            &self.lusgs_module.residual_density_sum_sq,
            &fields.res_rho,
            n as u32,
            sum_buf,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| AsimuError::Exec(format!("CUDA 同步失败: {e:?}")))?;
        let sum_sq =
            super::super::transfer::clone_dtoh(&self.stream, "residual_density_sum_sq", sum_buf)?;
        let rms = (sum_sq[0] / n as f32).sqrt();
        Ok(rms)
    }

    /// LU-SGS 非结构双扫：device 前/后扫 + host 线搜索 stabilize。
    pub fn lusgs_sweep_update_f32(&mut self, input: LusgsSweepCudaHostInput<'_>) -> Result<()> {
        self.lusgs_sweep_validate_lengths(&input)?;
        self.lusgs_sweep_prepare_buffers(&input)?;
        let u0_bufs = self.lusgs_sweep_upload_u0(input.u0)?;
        self.lusgs_sweep_launch_device(u0_bufs, &input.scalars)?;
        self.lusgs_sweep_stabilize_host(input)
    }
}

struct LusgsSweepU0Device {
    rho: cudarc::driver::CudaSlice<f32>,
    mx: cudarc::driver::CudaSlice<f32>,
    my: cudarc::driver::CudaSlice<f32>,
    mz: cudarc::driver::CudaSlice<f32>,
    e: cudarc::driver::CudaSlice<f32>,
}

impl CudaBackendState {
    fn lusgs_sweep_validate_lengths(&self, input: &LusgsSweepCudaHostInput<'_>) -> Result<()> {
        if !self.pipeline.timestep_on_device {
            return Err(AsimuError::Exec(
                "CUDA LU-SGS 扫掠需要 device 上 σ/Δt_i".to_string(),
            ));
        }
        let n = input.fields.num_cells();
        if input.residual.num_cells() != n
            || input.host_sigma.len() != n
            || input.host_cell_dts.len() != n
            || input.host_volumes.len() != n
        {
            return Err(AsimuError::Exec(
                "CUDA LU-SGS 扫掠：场/残差/timestep 长度不一致".to_string(),
            ));
        }
        Ok(())
    }

    fn lusgs_sweep_prepare_buffers(&mut self, input: &LusgsSweepCudaHostInput<'_>) -> Result<()> {
        self.ensure_fields(input.fields.num_cells())?;
        self.ensure_lusgs_sweep_mesh(input.sweep_topo, input.topo_key)?;
        let field_bufs = self.fields.as_mut().expect("field buffers after ensure");
        if !self.pipeline.conserved_on_device {
            field_bufs.upload_conserved(&self.stream, input.fields)?;
            self.pipeline.conserved_on_device = true;
        }
        if !self.pipeline.residual_on_device {
            field_bufs.upload_full_residual(&self.stream, input.residual)?;
        }
        if self.primitives_dirty {
            field_bufs.upload_primitives(&self.stream, input.primitives)?;
            self.primitives_dirty = false;
        }
        Ok(())
    }

    fn lusgs_sweep_upload_u0(&mut self, u0: &ConservedFieldsT<f32>) -> Result<LusgsSweepU0Device> {
        let n = u0.num_cells();
        let mut rho = take_or_alloc_u0(&mut self.lusgs_sweep_u0_rho, &self.stream, n)?;
        let mut mx = take_or_alloc_u0(&mut self.lusgs_sweep_u0_mx, &self.stream, n)?;
        let mut my = take_or_alloc_u0(&mut self.lusgs_sweep_u0_my, &self.stream, n)?;
        let mut mz = take_or_alloc_u0(&mut self.lusgs_sweep_u0_mz, &self.stream, n)?;
        let mut e = take_or_alloc_u0(&mut self.lusgs_sweep_u0_e, &self.stream, n)?;
        upload_u0_snapshot(
            &self.stream,
            u0,
            &mut rho,
            &mut mx,
            &mut my,
            &mut mz,
            &mut e,
        )?;
        Ok(LusgsSweepU0Device { rho, mx, my, mz, e })
    }

    fn lusgs_sweep_launch_device(
        &mut self,
        u0: LusgsSweepU0Device,
        scalars: &LusgsSweepCudaScalars,
    ) -> Result<()> {
        let field_bufs = self.fields.as_ref().expect("field buffers");
        let sweep_mesh = self.lusgs_sweep_mesh.as_ref().expect("lusgs sweep mesh");
        let spectral_mesh = self.spectral_mesh.as_ref().expect("spectral mesh");
        launch_lusgs_sweep_unstructured_serial(
            &self.stream,
            &self.lusgs_module.sweep_unstructured_serial,
            &LusgsSweepCudaLaunchBuffers {
                fields: field_bufs,
                sweep_mesh,
                sigma: spectral_mesh.sigma(),
                cell_dts: spectral_mesh.cell_dts(),
                u0_rho: &u0.rho,
                u0_mx: &u0.mx,
                u0_my: &u0.my,
                u0_mz: &u0.mz,
                u0_e: &u0.e,
            },
            scalars,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| AsimuError::Exec(format!("CUDA LU-SGS 扫掠同步失败: {e:?}")))?;
        self.lusgs_sweep_u0_rho = Some(u0.rho);
        self.lusgs_sweep_u0_mx = Some(u0.mx);
        self.lusgs_sweep_u0_my = Some(u0.my);
        self.lusgs_sweep_u0_mz = Some(u0.mz);
        self.lusgs_sweep_u0_e = Some(u0.e);
        Ok(())
    }

    fn lusgs_sweep_stabilize_host(&mut self, input: LusgsSweepCudaHostInput<'_>) -> Result<()> {
        let field_bufs = self.fields.as_mut().expect("field buffers");
        let u0_host = input.u0.clone();
        field_bufs.download_conserved(&self.stream, input.fields)?;
        let u_sweep = input.fields.clone();
        if !self.pipeline.residual_on_device {
            return Err(AsimuError::Exec(
                "CUDA LU-SGS 扫掠后 residual 标志不一致".to_string(),
            ));
        }
        field_bufs.download_residual(&self.stream, input.residual)?;
        self.pipeline.residual_on_device = false;
        let scalars = LuSgsSweepScalarsF32 {
            dt: input.host_cell_dts,
            sigma: input.host_sigma,
            volumes: input.host_volumes,
            omega: input.scalars.omega,
            gamma: input.scalars.gamma,
            inv_dt_phys: input.scalars.inv_dt_phys,
        };
        stabilize_sweep_update_f32(
            input.fields,
            &u0_host,
            &u_sweep,
            input.residual,
            input.scalars.min_pressure,
            input.scalars.gamma,
            &scalars,
        )?;
        field_bufs.upload_conserved(&self.stream, input.fields)?;
        self.pipeline.conserved_on_device = true;
        self.pipeline.timestep_on_device = false;
        self.pipeline.lusgs_diagonal_on_device = false;
        self.pipeline.lusgs_sweep_on_device = true;
        self.primitives_dirty = true;
        Ok(())
    }
}

fn take_or_alloc_u0(
    slot: &mut Option<cudarc::driver::CudaSlice<f32>>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    n: usize,
) -> Result<cudarc::driver::CudaSlice<f32>> {
    match slot.take() {
        Some(buf) if buf.len() == n => Ok(buf),
        _ => stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| AsimuError::Exec(format!("CUDA LU-SGS 扫掠 u0 缓冲分配失败: {e:?}"))),
    }
}
