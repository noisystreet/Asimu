//! 非结构 typed 时间步准备（BC/原变量刷新、谱半径、局部 \(\Delta t\)）。

use tracing::{info, info_span};

#[cfg(feature = "cuda")]
use crate::core::ExecDevice;
use crate::core::{ComputeFloat, Real, format_log_sci4};
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::exec::ExecutionContext;
use crate::field::{ConservedFieldsT, PrimitiveFillFromConserved};
use crate::solver::compressible::spectral_radius_unstructured::{
    SpectralRadiusUnstructuredTypedParams, UnstructuredSpectralRadiusTyped,
};
#[cfg(feature = "cuda")]
use crate::solver::time::TimeIntegrationScheme;
use crate::solver::{
    RefreshCompressibleStateTypedInput, finalize_cell_dts_from_sigma,
    finalize_cell_dts_from_sigma_f32, min_positive_dt, min_positive_dt_f32,
    refresh_compressible_ghosts_and_primitives_typed,
};

use super::{UnstructuredRunEnvTyped, UnstructuredStepWorkTyped};

/// 时间步准备阶段谱半径输出；`cell_dts` 为 `Some` 时表示已在 GPU/CPU 同路径完成 finalize。
pub(crate) struct TimestepPrepareSpectral<T> {
    pub sigma: Vec<T>,
    pub cell_dts: Option<Vec<T>>,
}

pub(crate) fn prepare_unstructured_timestep_typed<
    T: ComputeFloat
        + UnstructuredSpectralRadiusAtPrepare
        + UnstructuredTimestepFromSigma
        + UnstructuredCudaPrepareSync
        + PrimitiveFillFromConserved,
>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    cfl: Real,
    p_floor: Real,
) -> Result<Real> {
    let n = env.config.mesh.num_cells();
    let _prepare_span = info_span!(
        "unstructured_prepare_timestep_typed",
        cells = n,
        precision = T::PRECISION.label(),
        local_time_step = env.config.local_time_step,
    )
    .entered();
    fields.enforce_positivity(env.config.eos, p_floor);
    work.ghosts
        .ensure_face_capacity(env.config.mesh.num_faces());
    {
        let _span = info_span!("unstructured_refresh_state_typed", cells = n).entered();
        T::refresh_state_for_prepare(env, fields, work, p_floor)?;
        T::maybe_prepare_cuda_rhs_device_state(env, work, p_floor)?;
    }
    let fixed_dt = if env.config.dual_time.is_some() {
        None
    } else {
        env.config.fixed_dt.filter(|dt| *dt > 0.0 && dt.is_finite())
    };
    let prepared = compute_spectral_radius_at_prepare(env, work, p_floor, cfl, fixed_dt)?;
    {
        let _span = info_span!(
            "unstructured_finalize_cell_dts_typed",
            cells = n,
            fixed_dt = fixed_dt.is_some(),
        )
        .entered();
        T::store_sigma_and_cell_dts(
            work,
            prepared,
            cfl,
            fixed_dt,
            env.config.local_time_step,
            env.config.dual_time.is_some(),
        )
    }
}

fn compute_spectral_radius_at_prepare<
    T: ComputeFloat + UnstructuredSpectralRadiusAtPrepare + UnstructuredTimestepFromSigma,
>(
    env: &UnstructuredRunEnvTyped<'_>,
    work: &mut UnstructuredStepWorkTyped<T>,
    p_floor: Real,
    cfl: Real,
    fixed_dt: Option<Real>,
) -> Result<TimestepPrepareSpectral<T>> {
    let n = env.config.mesh.num_cells();
    let _span = info_span!("unstructured_spectral_radius_typed", cells = n).entered();
    T::compute_spectral_radius_at_prepare(
        env,
        work,
        p_floor,
        cfl,
        fixed_dt,
        env.config.local_time_step,
    )
}

pub(super) fn log_pseudo_timestep_stats_f32(
    inner: u32,
    dt_phys: Real,
    sigma: &[f32],
    cell_dts: &[f32],
) {
    let sigma_real: Vec<Real> = sigma.iter().map(|v| *v as Real).collect();
    let dt_real: Vec<Real> = cell_dts.iter().map(|v| *v as Real).collect();
    log_pseudo_timestep_stats(inner, dt_phys, &sigma_real, &dt_real);
}

fn log_pseudo_timestep_stats(inner: u32, dt_phys: Real, sigma: &[Real], cell_dts: &[Real]) {
    if sigma.len() != cell_dts.len() || sigma.is_empty() {
        return;
    }
    let mut sigma_min = Real::INFINITY;
    let mut sigma_max: Real = 0.0;
    let mut dt_min = Real::INFINITY;
    let mut dt_max: Real = 0.0;
    let mut dt_sigma_min = Real::INFINITY;
    let mut dt_sigma_max: Real = 0.0;
    let mut dt_inv_phys_min = Real::INFINITY;
    let mut dt_inv_phys_max: Real = 0.0;
    let inv_dt_phys = if dt_phys > 0.0 { 1.0 / dt_phys } else { 0.0 };
    for (&s, &dt) in sigma.iter().zip(cell_dts.iter()) {
        if !(s.is_finite() && dt.is_finite() && s > 0.0 && dt > 0.0) {
            continue;
        }
        sigma_min = sigma_min.min(s);
        sigma_max = sigma_max.max(s);
        dt_min = dt_min.min(dt);
        dt_max = dt_max.max(dt);
        let dt_sigma = dt * s;
        dt_sigma_min = dt_sigma_min.min(dt_sigma);
        dt_sigma_max = dt_sigma_max.max(dt_sigma);
        let dt_inv_phys = dt * inv_dt_phys;
        dt_inv_phys_min = dt_inv_phys_min.min(dt_inv_phys);
        dt_inv_phys_max = dt_inv_phys_max.max(dt_inv_phys);
    }
    if !sigma_min.is_finite() {
        return;
    }
    info!(
        parent: None,
        inner,
        sigma_min = %format_log_sci4(sigma_min),
        sigma_max = %format_log_sci4(sigma_max),
        dtau_min = %format_log_sci4(dt_min),
        dtau_max = %format_log_sci4(dt_max),
        dtau_sigma_min = %format_log_sci4(dt_sigma_min),
        dtau_sigma_max = %format_log_sci4(dt_sigma_max),
        dtau_inv_dt_phys_min = %format_log_sci4(dt_inv_phys_min),
        dtau_inv_dt_phys_max = %format_log_sci4(dt_inv_phys_max),
        "dual_time 伪时间步诊断",
    );
}

/// BC/原变量刷新后同步 device primitive（f32 CUDA 单步一次 H2D）。
pub(crate) trait UnstructuredCudaPrepareSync:
    UnstructuredSpectralRadiusTyped + PrimitiveFillFromConserved
{
    fn sync_primitives_after_refresh(work: &mut UnstructuredStepWorkTyped<Self>) -> Result<()>;

    fn refresh_state_for_prepare(
        env: &UnstructuredRunEnvTyped<'_>,
        fields: &mut ConservedFieldsT<Self>,
        work: &mut UnstructuredStepWorkTyped<Self>,
        p_floor: Real,
    ) -> Result<()> {
        refresh_compressible_ghosts_and_primitives_typed(RefreshCompressibleStateTypedInput {
            boundary_mesh: env.config.mesh,
            patches: env.config.patches,
            fields,
            ghosts: &mut work.ghosts,
            eos: env.config.eos,
            freestream: env.config.freestream,
            reference: env.config.reference,
            viscous: env.config.viscous,
            min_pressure: p_floor,
            primitives: &mut work.primitives,
        })?;
        Self::sync_primitives_after_refresh(work)
    }

    fn maybe_prepare_cuda_rhs_device_state(
        env: &UnstructuredRunEnvTyped<'_>,
        work: &mut UnstructuredStepWorkTyped<Self>,
        p_floor: Real,
    ) -> Result<()> {
        let _ = (env, work, p_floor);
        Ok(())
    }

    fn step_density_residual_rms(work: &mut UnstructuredStepWorkTyped<Self>) -> Result<Real>;

    fn dual_time_storage_inv_dt_coeff(
        work: &UnstructuredStepWorkTyped<Self>,
        dt_phys: Real,
    ) -> Real {
        work.dual_time_state.physical_storage_inv_dt_coeff(dt_phys)
    }

    fn log_dual_time_pseudo_timestep_stats(
        _work: &mut UnstructuredStepWorkTyped<Self>,
        _inner: u32,
        _dt_phys: Real,
        _local_time_step: bool,
    ) -> Result<()> {
        Ok(())
    }

    fn maybe_upload_lusgs_integration_base(
        work: &mut UnstructuredStepWorkTyped<Self>,
    ) -> Result<()>;

    fn lusgs_skip_copy_stage_after_diagonal(work: &UnstructuredStepWorkTyped<Self>) -> bool;

    /// 诊断用对角 trial 是否应跳过；CUDA device 管线下该 trial 会消费 timestep 驻留状态。
    fn skip_lusgs_diag_trial_probe(_work: &UnstructuredStepWorkTyped<Self>) -> bool {
        false
    }

    fn maybe_enforce_conserved_after_integration(
        work: &mut UnstructuredStepWorkTyped<Self>,
        eos: &crate::physics::IdealGasEoS,
        min_pressure: Real,
    ) -> Result<()>;

    fn maybe_download_conserved_for_output(
        work: &mut UnstructuredStepWorkTyped<Self>,
        fields: &mut ConservedFieldsT<Self>,
    ) -> Result<()>;

    /// 双时间步物理步初冻结 \(U^n\)。
    fn snapshot_dual_time_u_n(
        work: &mut UnstructuredStepWorkTyped<Self>,
        fields: &ConservedFieldsT<Self>,
    ) -> Result<()> {
        work.dual_time_state.snapshot_u_n(fields)
    }

    /// 叠加物理存储项至有效残差：首个物理步 BDF1，之后 BDF2。
    fn add_dual_time_storage_residual(
        work: &mut UnstructuredStepWorkTyped<Self>,
        fields: &ConservedFieldsT<Self>,
        dt_phys: Real,
    ) -> Result<()> {
        crate::solver::time::add_physical_storage_residual_from_state(
            &mut work.storage.k1,
            fields,
            &work.dual_time_state,
            dt_phys,
        )
    }

    /// 双时间步内层迭代初：同步 LU-SGS 积分基态（CUDA 保留 device 守恒场）。
    fn prepare_dual_time_inner_base(
        work: &mut UnstructuredStepWorkTyped<Self>,
        fields: &mut ConservedFieldsT<Self>,
    ) -> Result<()> {
        work.storage.u0.copy_from(fields)
    }

    /// 内层 LU-SGS 后诊断：相对 \(U^n\) 的密度偏移（debug 级）。
    fn debug_log_dual_time_inner_vs_u_n(
        fields: &ConservedFieldsT<Self>,
        work: &mut UnstructuredStepWorkTyped<Self>,
        inner: u32,
        dt_phys: Real,
    ) {
        let _ = (fields, work, inner, dt_phys);
    }

    /// 内层 LU-SGS 后复算 \(R_{\mathrm{eff}}\) 前：将积分后的守恒场同步到 host `fields`（CUDA 只读 D2H）。
    fn sync_fields_for_post_lusgs_rhs_probe(
        _work: &mut UnstructuredStepWorkTyped<Self>,
        _fields: &mut ConservedFieldsT<Self>,
    ) -> Result<()> {
        Ok(())
    }
}

impl UnstructuredCudaPrepareSync for f64 {
    fn sync_primitives_after_refresh(_work: &mut UnstructuredStepWorkTyped<f64>) -> Result<()> {
        Ok(())
    }

    fn step_density_residual_rms(work: &mut UnstructuredStepWorkTyped<f64>) -> Result<Real> {
        Ok(work.storage.k1.density_rms_norm())
    }

    fn log_dual_time_pseudo_timestep_stats(
        work: &mut UnstructuredStepWorkTyped<f64>,
        inner: u32,
        dt_phys: Real,
        _local_time_step: bool,
    ) -> Result<()> {
        log_pseudo_timestep_stats(
            inner,
            dt_phys,
            &work.timestep.sigma,
            &work.timestep.cell_dts,
        );
        Ok(())
    }

    fn maybe_upload_lusgs_integration_base(
        _work: &mut UnstructuredStepWorkTyped<f64>,
    ) -> Result<()> {
        Ok(())
    }

    fn lusgs_skip_copy_stage_after_diagonal(_work: &UnstructuredStepWorkTyped<f64>) -> bool {
        false
    }

    fn maybe_enforce_conserved_after_integration(
        _work: &mut UnstructuredStepWorkTyped<f64>,
        _eos: &crate::physics::IdealGasEoS,
        _min_pressure: Real,
    ) -> Result<()> {
        Ok(())
    }

    fn maybe_download_conserved_for_output(
        _work: &mut UnstructuredStepWorkTyped<f64>,
        _fields: &mut ConservedFieldsT<f64>,
    ) -> Result<()> {
        Ok(())
    }

    fn debug_log_dual_time_inner_vs_u_n(
        fields: &ConservedFieldsT<f64>,
        work: &mut UnstructuredStepWorkTyped<f64>,
        inner: u32,
        dt_phys: Real,
    ) {
        super::unstructured_dual_time_typed::log_inner_state_vs_u_n(
            fields,
            &work.dual_time_state.u_at_physical_level,
            inner,
            dt_phys,
        );
    }
}

/// 时间步准备阶段的谱半径（f32 可走 CUDA）。
pub(crate) trait UnstructuredSpectralRadiusAtPrepare:
    UnstructuredSpectralRadiusTyped
{
    fn compute_spectral_radius_at_prepare(
        env: &UnstructuredRunEnvTyped<'_>,
        work: &mut UnstructuredStepWorkTyped<Self>,
        p_floor: Real,
        cfl: Real,
        fixed_dt: Option<Real>,
        local_time_step: bool,
    ) -> Result<TimestepPrepareSpectral<Self>>;
}

impl UnstructuredSpectralRadiusAtPrepare for f32 {
    fn compute_spectral_radius_at_prepare(
        env: &UnstructuredRunEnvTyped<'_>,
        work: &mut UnstructuredStepWorkTyped<f32>,
        p_floor: Real,
        cfl: Real,
        fixed_dt: Option<Real>,
        local_time_step: bool,
    ) -> Result<TimestepPrepareSpectral<f32>> {
        let params = SpectralRadiusUnstructuredTypedParams {
            mesh: env.config.mesh,
            mesh_cache: &work.mesh_cache,
            boundaries: env.config.patches,
            ghosts: &work.ghosts,
            primitives: &work.primitives,
            eos: env.config.eos,
            min_pressure: p_floor,
            viscous: env.config.viscous,
        };
        #[cfg(feature = "cuda")]
        {
            let keep_timestep_on_device =
                f32_cuda_keep_timestep_on_device(env.config.time_scheme, &work.exec);
            let (sigma, cell_dts) =
                crate::solver::compressible::spectral_radius_unstructured_f32_cuda::compute_spectral_radius_f32_with_exec(
                    &params,
                    &mut work.exec,
                    cfl,
                    fixed_dt,
                    local_time_step,
                    keep_timestep_on_device,
                )?;
            Ok(TimestepPrepareSpectral { sigma, cell_dts })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (cfl, fixed_dt, local_time_step);
            let sigma = Self::cell_spectral_radius_unstructured_typed(&params)?;
            Ok(TimestepPrepareSpectral {
                sigma,
                cell_dts: None,
            })
        }
    }
}

impl UnstructuredSpectralRadiusAtPrepare for f64 {
    fn compute_spectral_radius_at_prepare(
        env: &UnstructuredRunEnvTyped<'_>,
        work: &mut UnstructuredStepWorkTyped<f64>,
        p_floor: Real,
        _cfl: Real,
        _fixed_dt: Option<Real>,
        _local_time_step: bool,
    ) -> Result<TimestepPrepareSpectral<Real>> {
        let sigma = Self::cell_spectral_radius_unstructured_typed(
            &SpectralRadiusUnstructuredTypedParams {
                mesh: env.config.mesh,
                mesh_cache: &work.mesh_cache,
                boundaries: env.config.patches,
                ghosts: &work.ghosts,
                primitives: &work.primitives,
                eos: env.config.eos,
                min_pressure: p_floor,
                viscous: env.config.viscous,
            },
        )?;
        Ok(TimestepPrepareSpectral {
            sigma,
            cell_dts: None,
        })
    }
}

/// 谱半径结果写入时间步缓冲（f32 原生 \(\sigma_i\)，无 prepare 边界转换）。
pub(crate) trait UnstructuredTimestepFromSigma: UnstructuredSpectralRadiusTyped {
    /// 写入 \(\sigma_i\) 与 `cell_dts` 并返回全局最小 \(\Delta t\)。
    ///
    /// `skip_cuda_min_dt_d2h`：`dual_time` 内层为 true 时跳过 `spectral_min_cell_dt` D2H（返回值未被消费）。
    fn store_sigma_and_cell_dts(
        work: &mut UnstructuredStepWorkTyped<Self>,
        prepared: TimestepPrepareSpectral<Self>,
        cfl: Real,
        fixed_dt: Option<Real>,
        local_time_step: bool,
        skip_cuda_min_dt_d2h: bool,
    ) -> Result<Real>;
}

impl UnstructuredTimestepFromSigma for f32 {
    fn store_sigma_and_cell_dts(
        work: &mut UnstructuredStepWorkTyped<f32>,
        prepared: TimestepPrepareSpectral<f32>,
        cfl: Real,
        fixed_dt: Option<Real>,
        local_time_step: bool,
        skip_cuda_min_dt_d2h: bool,
    ) -> Result<Real> {
        work.timestep.sigma_f32 = prepared.sigma;
        if let Some(cell_dts) = prepared.cell_dts {
            work.timestep.cell_dts_f32 = cell_dts;
            Ok(min_positive_dt_f32(&work.timestep.cell_dts_f32) as Real)
        } else {
            #[cfg(feature = "cuda")]
            if work.exec.cuda_timestep_on_device() {
                work.timestep.cell_dts_f32.clear();
                if let Some(dt) = fixed_dt {
                    return Ok(dt);
                }
                if skip_cuda_min_dt_d2h {
                    // dual_time 内层返回值未被消费；LU-SGS 直接读 device σ/Δt_i。
                    return Ok(0.0);
                }
                return Ok(work.exec.cuda_download_min_cell_dt_f32()? as Real);
            }
            #[cfg(not(feature = "cuda"))]
            let _ = skip_cuda_min_dt_d2h;
            work.timestep.cell_dts_f32 = finalize_cell_dts_from_sigma_f32(
                &work.volumes_f32,
                &work.timestep.sigma_f32,
                cfl as f32,
                fixed_dt.map(|d| d as f32),
                local_time_step,
            )?;
            Ok(min_positive_dt_f32(&work.timestep.cell_dts_f32) as Real)
        }
    }
}

impl UnstructuredTimestepFromSigma for f64 {
    fn store_sigma_and_cell_dts(
        work: &mut UnstructuredStepWorkTyped<f64>,
        prepared: TimestepPrepareSpectral<Real>,
        cfl: Real,
        fixed_dt: Option<Real>,
        local_time_step: bool,
        _skip_cuda_min_dt_d2h: bool,
    ) -> Result<Real> {
        work.timestep.sigma = prepared.sigma;
        work.timestep.cell_dts = if let Some(cell_dts) = prepared.cell_dts {
            cell_dts
        } else {
            finalize_cell_dts_from_sigma(
                &work.volumes,
                &work.timestep.sigma,
                cfl,
                fixed_dt,
                local_time_step,
            )?
        };
        Ok(min_positive_dt(&work.timestep.cell_dts))
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn f32_cuda_prepare_device_refresh(
    env: &UnstructuredRunEnvTyped<'_>,
    work: &UnstructuredStepWorkTyped<f32>,
) -> bool {
    env.config.viscous.is_some() && work.exec.device() == ExecDevice::GpuCuda
}

#[cfg(feature = "cuda")]
pub(crate) fn f32_cuda_viscous_rhs_pipeline(
    env: &UnstructuredRunEnvTyped<'_>,
    exec: &ExecutionContext,
) -> bool {
    env.config.viscous.is_some() && exec.device() == ExecDevice::GpuCuda
}

/// CUDA f32：谱半径 \(\sigma_i\)/`cell_dts` 驻留 device，供 LU-SGS 对角/双扫与 dual_time 内层跳过批量 D2H。
#[cfg(feature = "cuda")]
fn f32_cuda_keep_timestep_on_device(
    time_scheme: TimeIntegrationScheme,
    exec: &ExecutionContext,
) -> bool {
    if exec.device() != ExecDevice::GpuCuda {
        return false;
    }
    matches!(
        time_scheme,
        TimeIntegrationScheme::LuSgs | TimeIntegrationScheme::DualTime
    )
}

#[cfg(test)]
mod cuda_timestep_tests {
    #[cfg(feature = "cuda")]
    #[test]
    fn dual_time_cuda_keeps_timestep_on_device_with_lusgs_sweep() {
        use super::f32_cuda_keep_timestep_on_device;
        use crate::core::ExecDevice;
        use crate::exec::{ExecConfig, ExecutionContext, MeshExecMetrics};
        use crate::solver::time::TimeIntegrationScheme;

        let exec = ExecutionContext::new(
            ExecConfig {
                device: ExecDevice::GpuCuda,
                ..ExecConfig::default()
            },
            MeshExecMetrics::empty(),
        )
        .expect("cuda");
        assert!(f32_cuda_keep_timestep_on_device(
            TimeIntegrationScheme::DualTime,
            &exec,
        ));
        assert!(f32_cuda_keep_timestep_on_device(
            TimeIntegrationScheme::LuSgs,
            &exec,
        ));
        let cpu = ExecutionContext::for_unit_test();
        assert!(!f32_cuda_keep_timestep_on_device(
            TimeIntegrationScheme::DualTime,
            &cpu,
        ));
    }
}
