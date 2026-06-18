//! [`ExecutionContext`] CUDA 扩展方法（从 `context.rs` 拆分以满足复杂度门禁）。

use crate::error::Result;
use crate::exec::gpu::cuda;
use crate::exec::spectral_radius_cuda;

use super::ExecutionContext;

impl ExecutionContext {
    /// CUDA P1：步初清零整条 device 管线状态。
    #[cfg(feature = "cuda")]
    pub fn cuda_reset_full_pipeline_step(&mut self) -> Result<()> {
        self.backend_state.cuda_mut()?.reset_full_pipeline_step();
        Ok(())
    }

    /// CUDA：步间重置（保留守恒场 device 驻留）。
    #[cfg(feature = "cuda")]
    pub fn cuda_reset_between_timesteps(&mut self) -> Result<()> {
        self.backend_state.cuda_mut()?.reset_between_timesteps();
        Ok(())
    }

    /// CUDA P1：步初重置 device 管线状态。
    #[cfg(feature = "cuda")]
    pub fn cuda_reset_pipeline_step(&mut self) -> Result<()> {
        self.backend_state.cuda_mut()?.reset_pipeline_step();
        Ok(())
    }

    /// CUDA P1：启用 RHS device 管线（粘性链）。
    #[cfg(feature = "cuda")]
    pub fn cuda_enable_rhs_device_pipeline(&mut self) -> Result<()> {
        self.backend_state.cuda_mut()?.enable_rhs_device_pipeline();
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_rhs_pipeline_active(&self) -> bool {
        self.backend_state
            .cuda_rhs_pipeline_active()
            .unwrap_or(false)
    }

    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_timestep_on_device(&self) -> bool {
        self.backend_state
            .cuda_timestep_on_device()
            .unwrap_or(false)
    }

    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_residual_on_device(&self) -> bool {
        self.backend_state
            .cuda_residual_on_device()
            .unwrap_or(false)
    }

    /// CUDA P3：prepare 已刷新 BC/原变量并 H2D primitive。
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_host_bc_primitives_synced(&self) -> bool {
        self.backend_state
            .cuda_host_bc_primitives_synced()
            .unwrap_or(false)
    }

    /// CUDA P3：prepare 后一次性上传 RHS boundary ghost / 单元温度。
    #[cfg(feature = "cuda")]
    pub fn cuda_prepare_rhs_device_state(
        &mut self,
        input: cuda::CudaPrepareRhsDeviceInput<'_>,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .prepare_rhs_device_state(input)
    }

    /// CUDA P3：device 密度残差 RMS（单 float D2H）。
    #[cfg(feature = "cuda")]
    pub fn cuda_density_residual_rms_f32(&mut self) -> Result<f32> {
        self.backend_state.cuda_mut()?.density_residual_rms_f32()
    }

    /// CUDA P4：守恒场已在 device（LU-SGS 对角跳过 H2D/D2H）。
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_conserved_on_device(&self) -> bool {
        self.backend_state
            .cuda_conserved_on_device()
            .unwrap_or(false)
    }

    /// CUDA P3：边界面 ghost 原变量已在 device（prepare_rhs device BC 后）。
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_boundary_ghosts_on_device(&self) -> bool {
        self.backend_state
            .cuda_boundary_ghosts_on_device()
            .unwrap_or(false)
    }

    /// CUDA P5：谱半径粘性扩散系数已在 device。
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_spectral_diffusivity_on_device(&self) -> bool {
        self.backend_state
            .cuda_spectral_diffusivity_on_device()
            .unwrap_or(false)
    }

    /// CUDA P4：LU-SGS 对角已在 device 写回守恒场。
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_lusgs_diagonal_on_device(&self) -> bool {
        self.backend_state
            .cuda_lusgs_diagonal_on_device()
            .unwrap_or(false)
    }

    /// CUDA LU-SGS 双扫已在 device 写回守恒场。
    #[cfg(feature = "cuda")]
    pub fn cuda_lusgs_sweep_on_device(&self) -> bool {
        self.backend_state
            .cuda_lusgs_sweep_on_device()
            .unwrap_or(false)
    }

    /// CUDA 非结构 LU-SGS 双扫（device 前/后扫 + host stabilize）。
    #[cfg(feature = "cuda")]
    pub fn cuda_lusgs_sweep_update_f32(
        &mut self,
        input: crate::exec::gpu::cuda::lusgs_sweep::LusgsSweepCudaHostInput<'_>,
    ) -> Result<()> {
        self.backend_state.cuda_mut()?.lusgs_sweep_update_f32(input)
    }

    /// CUDA P3b：双时间步 \(U^n\) 快照在 device 上有效。
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_u_n_on_device(&self) -> bool {
        self.backend_state.cuda_u_n_on_device().unwrap_or(false)
    }

    /// CUDA P3b：物理步初 device D2D 快照 \(U^n\)。
    #[cfg(feature = "cuda")]
    pub fn cuda_snapshot_u_n_on_device(
        &mut self,
        fields: &crate::field::ConservedFieldsT<f32>,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .snapshot_u_n_on_device(fields)
    }

    /// CUDA P3b：D2H 下载 device \(U^n\) 至 host 缓冲（物理步边界同步）。
    #[cfg(feature = "cuda")]
    pub fn cuda_download_u_n_on_device(
        &mut self,
        u_n_out: &mut crate::field::ConservedFieldsT<f32>,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .download_u_n_on_device(u_n_out)
    }

    /// CUDA P3b：device 叠加 BDF1 物理存储项。
    #[cfg(feature = "cuda")]
    pub fn cuda_add_physical_storage_residual_f32(&mut self, dt_phys: f32) -> Result<()> {
        if !(dt_phys.is_finite() && dt_phys > 0.0) {
            return Err(crate::error::AsimuError::Field(
                "dual_time: dt_phys 须为正有限".to_string(),
            ));
        }
        self.backend_state
            .cuda_mut()?
            .add_physical_storage_residual_f32(1.0 / dt_phys)
    }

    /// CUDA P4：LU-SGS 步初上传守恒基态至 device。
    #[cfg(feature = "cuda")]
    pub fn cuda_upload_conserved_for_integration(
        &mut self,
        conserved: &crate::field::ConservedFieldsT<f32>,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .upload_conserved_for_integration(conserved)
    }

    /// CUDA P4：步末按需 D2H 守恒场（间隔输出 / 算例结束）。
    #[cfg(feature = "cuda")]
    pub fn cuda_download_conserved_if_on_device(
        &mut self,
        fields: &mut crate::field::ConservedFieldsT<f32>,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .download_conserved_if_on_device(fields)
    }

    /// 只读 D2H 拷贝 device 守恒场；不修改 `conserved_on_device`（内层诊断）。
    #[cfg(feature = "cuda")]
    pub fn cuda_copy_conserved_to_host(
        &mut self,
        fields: &mut crate::field::ConservedFieldsT<f32>,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .copy_conserved_to_host(fields)
    }

    /// CUDA P7：device 守恒场正性钳制（替代步末全表 D2H）。
    #[cfg(feature = "cuda")]
    pub fn cuda_enforce_conserved_positivity_on_device(
        &mut self,
        eos: &crate::physics::IdealGasEoS,
        min_pressure: crate::core::Real,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .enforce_conserved_positivity_on_device(eos, min_pressure)
    }

    /// CUDA P5：BC 后 device 填原变量与谱半径扩散系数。
    #[cfg(feature = "cuda")]
    pub fn cuda_fill_primitives_and_diffusivity_on_device(
        &mut self,
        fields: &crate::field::ConservedFieldsT<f32>,
        mesh_cache: &crate::discretization::UnstructuredSolverMeshCache,
        eos: &crate::physics::IdealGasEoS,
        viscous: &crate::physics::ViscousPhysicsConfig,
        min_pressure: crate::core::Real,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .fill_primitives_and_diffusivity_on_device(
                fields,
                mesh_cache,
                eos,
                viscous,
                min_pressure,
            )
    }

    /// CUDA P1：边界面 CPU scatter 后上传残差至 device。
    #[cfg(feature = "cuda")]
    pub fn cuda_upload_residual_for_rhs(
        &mut self,
        residual: &crate::field::ConservedResidualT<f32>,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .upload_residual_from_host(residual)
    }

    /// CUDA P1：批量 D2H \(\sigma_i\) + `cell_dts`。
    #[cfg(feature = "cuda")]
    pub fn cuda_download_timestep_f32(
        &mut self,
        sigma_out: &mut [f32],
        cell_dts_out: &mut [f32],
        local_time_step: bool,
    ) -> Result<()> {
        self.backend_state.cuda_mut()?.download_timestep_f32(
            sigma_out,
            cell_dts_out,
            local_time_step,
        )
    }

    /// CUDA：镜像 device σ/Δtᵢ 到 host，保留 `timestep_on_device`（LU-SGS 双扫 stabilize）。
    #[cfg(feature = "cuda")]
    pub fn cuda_mirror_timestep_f32_to_host(
        &mut self,
        sigma_out: &mut [f32],
        cell_dts_out: &mut [f32],
        local_time_step: bool,
    ) -> Result<()> {
        self.backend_state.cuda_mut()?.mirror_timestep_f32_to_host(
            sigma_out,
            cell_dts_out,
            local_time_step,
        )
    }

    /// CUDA P1：RHS 管线结束，仅残差 D2H（梯度可延后至边界面装配前）。
    #[cfg(feature = "cuda")]
    pub fn cuda_flush_rhs_residual(
        &mut self,
        residual: &mut crate::field::ConservedResidualT<f32>,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .flush_residual_to_host(residual)
    }

    /// CUDA P1：梯度 device → host（粘性边界面装配前按需调用）。
    #[cfg(feature = "cuda")]
    pub fn cuda_download_gradients_to_host(
        &mut self,
        gradients: &mut crate::discretization::gradient_typed::GradientFieldsT<f32>,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .flush_gradients_to_host(gradients)
    }

    /// CUDA P1：RHS 管线结束，残差/梯度 D2H。
    #[cfg(feature = "cuda")]
    pub fn cuda_flush_rhs_pipeline(
        &mut self,
        residual: &mut crate::field::ConservedResidualT<f32>,
        gradients: &mut crate::discretization::gradient_typed::GradientFieldsT<f32>,
    ) -> Result<()> {
        self.cuda_flush_rhs_residual(residual)?;
        self.cuda_download_gradients_to_host(gradients)?;
        Ok(())
    }

    /// CUDA P1：device 上 `cell_dts` 最小正有限值（单 float D2H）。
    #[cfg(feature = "cuda")]
    pub fn cuda_download_min_cell_dt_f32(&mut self) -> Result<f32> {
        self.backend_state.cuda_mut()?.download_min_cell_dt_f32()
    }

    /// CUDA P4：LU-SGS 对角更新（守恒/residual 在 device 时零拷贝）。
    #[cfg(feature = "cuda")]
    pub fn cuda_lusgs_diagonal_update_f32(
        &mut self,
        base: &crate::field::ConservedFieldsT<f32>,
        residual: &crate::field::ConservedResidualT<f32>,
        omega: f32,
        inv_dt_phys: f32,
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .lusgs_diagonal_update_f32(base, residual, omega, inv_dt_phys)
    }

    /// CUDA G1：一阶无粘内面着色桶 flux + scatter（Roe / HVL）。
    #[cfg(feature = "cuda")]
    pub fn cuda_assemble_first_order_inviscid_interior(
        &mut self,
        residual: &mut crate::field::ConservedResidualT<f32>,
        primitives: &crate::field::PrimitiveFieldsT<f32>,
        topo: &crate::exec::gpu::cuda::ExecInteriorFaceTopology,
        topo_key: usize,
        params: crate::exec::gpu::cuda::CudaFirstOrderInviscidParams,
    ) -> Result<()> {
        let defer = self
            .backend_state
            .cuda_rhs_pipeline_active()
            .unwrap_or(false);
        self.backend_state
            .cuda_mut()?
            .assemble_first_order_inviscid_interior(
                residual, primitives, topo, topo_key, params, defer,
            )
    }

    /// CUDA P2：一阶无粘边界面 flux + atomic scatter（ghost 每步 H2D）。
    #[cfg(feature = "cuda")]
    pub fn cuda_assemble_first_order_inviscid_boundary(
        &mut self,
        residual: &mut crate::field::ConservedResidualT<f32>,
        primitives: &crate::field::PrimitiveFieldsT<f32>,
        topo: &crate::exec::gpu::cuda::ExecInviscidBoundaryTopology,
        topo_key: usize,
        boundary_ghosts: &[crate::discretization::unstructured_spectral_exec_topo::SpectralGhostPrimHost],
        params: crate::exec::gpu::cuda::CudaFirstOrderInviscidParams,
    ) -> Result<()> {
        let defer = self
            .backend_state
            .cuda_rhs_pipeline_active()
            .unwrap_or(false);
        self.backend_state
            .cuda_mut()?
            .assemble_first_order_inviscid_boundary(
                residual,
                primitives,
                topo,
                topo_key,
                boundary_ghosts,
                params,
                defer,
            )
    }

    /// CUDA P2：粘性边界面 flux + atomic scatter（读 device 梯度）。
    #[cfg(feature = "cuda")]
    pub fn cuda_assemble_viscous_boundary_f32(
        &mut self,
        residual: &mut crate::field::ConservedResidualT<f32>,
        primitives: &crate::field::PrimitiveFieldsT<f32>,
        gradients: &crate::discretization::gradient_typed::GradientFieldsT<f32>,
        input: crate::exec::gpu::cuda::CudaViscousBoundaryInput<'_>,
    ) -> Result<()> {
        let defer = self
            .backend_state
            .cuda_rhs_pipeline_active()
            .unwrap_or(false);
        self.backend_state
            .cuda_mut()?
            .assemble_viscous_boundary(residual, primitives, gradients, input, defer)
    }

    /// CUDA G2：粘性内面着色桶 flux + scatter（仅动量/能量）。
    #[cfg(feature = "cuda")]
    pub fn cuda_assemble_viscous_interior(
        &mut self,
        residual: &mut crate::field::ConservedResidualT<f32>,
        primitives: &crate::field::PrimitiveFieldsT<f32>,
        gradients: &crate::discretization::gradient_typed::GradientFieldsT<f32>,
        topo: &crate::exec::gpu::cuda::ExecViscousInteriorTopology,
        topo_key: usize,
    ) -> Result<()> {
        let defer = self
            .backend_state
            .cuda_rhs_pipeline_active()
            .unwrap_or(false);
        self.backend_state
            .cuda_mut()?
            .assemble_viscous_interior(residual, primitives, gradients, topo, topo_key, defer)
    }

    /// CUDA：内面粘性输运系数 \(\mu,\lambda\) device kernel。
    #[cfg(feature = "cuda")]
    pub fn cuda_prepare_viscous_face_transport_f32(
        &mut self,
        topo: &crate::exec::gpu::cuda::ExecViscousInteriorTopology,
        topo_key: usize,
        temperatures: &[f32],
        viscous: &crate::physics::ViscousPhysicsConfig,
        eos: &crate::physics::IdealGasEoS,
    ) -> Result<()> {
        let params = crate::exec::gpu::cuda::build_device_viscous_transport_params(viscous, eos)?;
        self.backend_state
            .cuda_mut()?
            .prepare_viscous_face_transport_f32(topo, topo_key, temperatures, params)
    }

    /// CUDA P1：IDWLS RHS 累加 + device 3×3 求解梯度。
    #[cfg(feature = "cuda")]
    pub fn cuda_accumulate_and_solve_idwls_viscous_gradients(
        &mut self,
        primitives: &crate::field::PrimitiveFieldsT<f32>,
        topo: &crate::exec::gpu::cuda::ExecIdwlsViscousTopology,
        topo_key: usize,
        lsq_geometry: &[crate::discretization::unstructured_face_cache_f32::LsqPrecomputedCellF32],
        temperatures: &[f32],
        boundary_ghosts: &[crate::discretization::unstructured_idwls_exec_topo::IdwlsGhostSampleHost],
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .accumulate_and_solve_idwls_viscous_gradients(
                primitives,
                topo,
                topo_key,
                lsq_geometry,
                temperatures,
                boundary_ghosts,
            )
    }

    /// CUDA P4：粘性 IDWLS RHS 单元并行累加（CPU solve 回退路径）。
    #[cfg(feature = "cuda")]
    pub fn cuda_accumulate_idwls_viscous_rhs(
        &mut self,
        primitives: &crate::field::PrimitiveFieldsT<f32>,
        topo: &crate::exec::gpu::cuda::ExecIdwlsViscousTopology,
        topo_key: usize,
        temperatures: &[f32],
        boundary_ghosts: &[crate::discretization::unstructured_idwls_exec_topo::IdwlsGhostSampleHost],
    ) -> Result<()> {
        let (bu, bv, bw, bt) = self.scratch.idwls_mut().viscous_arrays_mut_f32();
        self.backend_state.cuda_mut()?.accumulate_idwls_viscous_rhs(
            primitives,
            topo,
            topo_key,
            temperatures,
            boundary_ghosts,
            cuda::IdwlsViscousRhsHostOut { bu, bv, bw, bt },
        )
    }

    /// CUDA：非结构 f32 单元谱半径（单元并行 kernel）。
    #[cfg(feature = "cuda")]
    pub fn cuda_compute_spectral_radius_unstructured_f32(
        &mut self,
        input: &spectral_radius_cuda::SpectralRadiusCudaInput<'_>,
        sigma_out: &mut [f32],
    ) -> Result<()> {
        self.backend_state
            .cuda_mut()?
            .compute_spectral_radius_unstructured_f32(input, sigma_out)
    }

    /// CUDA G3：cuSPARSE CSR SpMV（f64）。
    #[cfg(feature = "cuda")]
    pub(crate) fn dispatch_cuda_csr_spmv(
        &mut self,
        matrix: &crate::exec::CsrSpmvView<'_>,
        x: &[crate::core::Real],
        y: &mut [crate::core::Real],
    ) -> Result<()> {
        self.backend_state.try_csr_spmv_cuda(matrix, x, y)
    }
}
