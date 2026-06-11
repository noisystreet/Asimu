//! 可压缩无粘 Euler 显式求解（RK4 / 一阶 Euler + FVM 残差）。
//!
//! 理论：[`docs/theory/time_integration.md`](../../docs/theory/time_integration.md)、
//! [`inviscid_flux.md`](../../docs/theory/inviscid_flux.md)

use std::time::Instant;

#[path = "compressible_context.rs"]
mod compressible_context;
#[path = "compressible_rhs.rs"]
mod compressible_rhs;
#[path = "gmres_block_preconditioner_3d.rs"]
mod gmres_block_preconditioner_3d;
#[path = "gmres_implicit_3d.rs"]
mod gmres_implicit_3d;
#[path = "lu_sgs_sweep_3d.rs"]
mod lu_sgs_sweep_3d;

use crate::solver::spectral_radius::{
    SpectralRadius3dParams, cell_local_dt_spectral, cell_spectral_radius_3d,
};
pub use compressible_context::{
    CompressibleAdvanceContext1d, CompressibleAdvanceContext3d, ResidualCorrection3d,
    ResidualCorrection3dHandle,
};
use compressible_rhs::EvaluateRhs3d;
pub use gmres_implicit_3d::{GmresImplicitConfig, GmresImplicitDelta, GmresPreconditionerKind};
use gmres_implicit_3d::{
    GmresStepLog, GmresStepTiming, apply_delta_with_line_search, log_gmres_step_diagnostics,
};
use lu_sgs_sweep_3d::{LuSgsSweep3dParams, lu_sgs_sweep_3d};

use crate::core::{Real, format_log_fixed4, format_log_sci4, log10_positive};
use crate::discretization::assemble_inviscid_residual_1d;
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual};
use crate::mesh::{StructuredMesh1d, StructuredMesh3d};
use crate::physics::{FreestreamParams, IdealGasEoS, ViscousPhysicsConfig};
use crate::solver::compressible_helpers::{
    RefreshCompressibleStateInput, finalize_cell_dts_from_sigma,
    refresh_compressible_ghosts_and_primitives,
};
use crate::solver::state::SolverState;
use crate::solver::time::{
    CflSchedule, LuSgsConfig, ResidualSmoothingConfig, Rk4Storage, RungeKutta4Config,
    RungeKutta4Integrator, TimeIntegrationScheme, TimeIntegrator, euler_step, euler_step_local,
    min_positive_dt, rk4_step, rk4_step_local, smooth_residual_3d_limited,
};
use crate::solver::wave_speed::max_wave_speed;
use tracing::{info, info_span, instrument};

/// 稳态伪时间 / 瞬态物理时间。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressibleTimeMode {
    Steady,
    Transient,
}

/// 显式可压缩 Euler 求解器配置。
#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleEulerConfig {
    pub time: RungeKutta4Config,
    pub inviscid: crate::discretization::InviscidFluxConfig,
    /// `Some` 时叠加层流粘性通量（Navier-Stokes）。
    pub viscous: Option<ViscousPhysicsConfig>,
    pub cfl_schedule: CflSchedule,
    pub time_mode: CompressibleTimeMode,
    pub local_time_step: bool,
    /// 时间积分格式（`rk4` 默认；`euler` 排错；`lu_sgs`/`gmres` 隐式伪时间）。
    pub time_scheme: TimeIntegrationScheme,
    /// `lu_sgs` 松弛因子等（显式格式下忽略）。
    pub lu_sgs: LuSgsConfig,
    pub gmres: GmresImplicitConfig,
    pub residual_smoothing: ResidualSmoothingConfig,
}

impl Default for CompressibleEulerConfig {
    fn default() -> Self {
        Self {
            time: RungeKutta4Config::default(),
            inviscid: crate::discretization::InviscidFluxConfig::default(),
            viscous: None,
            cfl_schedule: CflSchedule::constant(0.4),
            time_mode: CompressibleTimeMode::Transient,
            local_time_step: false,
            time_scheme: TimeIntegrationScheme::Rk4,
            lu_sgs: LuSgsConfig::default(),
            gmres: GmresImplicitConfig::default(),
            residual_smoothing: ResidualSmoothingConfig::default(),
        }
    }
}

/// 单步推进结果。
#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleStepInfo {
    pub dt: Real,
    pub physical_time: Real,
    pub step: u64,
    /// 全场 \(\mathrm{RMS}(\dot\rho)=\|\dot\rho\|_2/\sqrt{N}\)。
    pub residual_rms: Real,
    /// \(\log_{10}(\mathrm{RMS}(\dot\rho))\)，便于跨量级对比。
    pub residual_log10: Real,
    /// 本步使用的 CFL 数。
    pub cfl: Real,
    pub is_final: bool,
    /// 本步是否因 log₁₀(RMS(ρ̇)) ≤ `[time].tolerance` 触发早停（由算例编排层设置）。
    pub converged: bool,
}

/// 可压缩 Euler 显式 RK4 求解器。
#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleEulerSolver {
    pub config: CompressibleEulerConfig,
}

impl CompressibleEulerSolver {
    #[must_use]
    pub const fn new(config: CompressibleEulerConfig) -> Self {
        Self { config }
    }

    fn rhs_context_3d<'a>(
        &'a self,
        ctx: &'a mut CompressibleAdvanceContext3d<'_>,
        inviscid: &'a crate::discretization::InviscidFluxConfig,
        min_pressure: Real,
    ) -> EvaluateRhs3d<'a> {
        EvaluateRhs3d {
            mesh: ctx.mesh,
            structured: ctx.structured,
            patches: ctx.patches,
            ghosts: ctx.ghosts,
            eos: ctx.eos,
            freestream: ctx.freestream,
            reference: ctx.reference,
            inviscid,
            viscous: self.config.viscous.as_ref(),
            min_pressure,
            primitive_scratch: &mut ctx.primitive_scratch,
            gradient_scratch: &mut ctx.gradient_scratch,
            residual_correction: ctx.residual_correction.clone(),
        }
    }

    fn smooth_residual_if_enabled(
        &self,
        mesh: &StructuredMesh3d,
        base: &ConservedFields,
        residual: &mut ConservedResidual,
        update_scales: &[Real],
        eos: &IdealGasEoS,
        min_pressure: Real,
    ) -> Result<()> {
        if self.config.time_mode != CompressibleTimeMode::Steady {
            return Ok(());
        }
        let config = self.config.residual_smoothing;
        if !config.enabled {
            return Ok(());
        }
        let _span = info_span!(
            "residual_smoothing",
            epsilon = config.epsilon,
            sweeps = config.sweeps,
        )
        .entered();
        smooth_residual_3d_limited(
            residual,
            base,
            update_scales,
            mesh,
            eos,
            min_pressure,
            config,
        )
    }

    /// 1D 瞬态推进：每步刷新边界 ghost、装配残差、RK4 更新守恒量。
    pub fn advance_step_1d(
        &self,
        ctx: &CompressibleAdvanceContext1d<'_>,
        fields: &mut ConservedFields,
        storage: &mut Rk4Storage,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
    ) -> Result<CompressibleStepInfo> {
        let cfl = self.cfl_for_step(state);
        let dt = self.suggest_dt_1d(ctx.mesh, fields, ctx.eos, cfl)?;
        integrator.config.dt = dt;
        let p_floor = 1.0e-6;
        let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
            let boundaries = ctx.boundary.resolve(u)?;
            assemble_inviscid_residual_1d(
                ctx.mesh,
                u,
                r,
                ctx.eos,
                &self.config.inviscid,
                &boundaries,
                p_floor,
            )
        };
        self.advance_explicit_step(fields, storage, dt, None, evaluate, None)?;
        // 监控量 = ‖R(U^n)‖：显式推进 stage1/euler_rhs 已写入 k1，勿再步末重算。
        let last_residual = storage.k1.density_rms_norm();
        let time_info = integrator.advance(state)?;
        Ok(CompressibleStepInfo {
            dt: time_info.dt,
            physical_time: time_info.physical_time,
            step: time_info.step,
            residual_rms: last_residual,
            residual_log10: log10_positive(last_residual),
            cfl,
            is_final: time_info.is_final,
            converged: false,
        })
    }

    /// 3D 推进：每 RK 阶段重算边界 ghost 与残差；支持全局/逐单元时间步。
    #[instrument(
        skip(self, ctx, fields, storage, state, integrator),
        level = "debug",
        fields(step = state.time_step.saturating_add(1))
    )]
    pub fn advance_step_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &mut ConservedFields,
        storage: &mut Rk4Storage,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
    ) -> Result<CompressibleStepInfo> {
        let cfl = {
            let _span = info_span!("cfl_schedule").entered();
            self.cfl_for_step(state)
        };
        let p_floor = Self::positivity_pressure_floor(ctx.freestream);
        if self.config.time_scheme == TimeIntegrationScheme::Gmres {
            return self
                .advance_gmres_step_3d(ctx, fields, storage, state, integrator, cfl, p_floor);
        }
        if self.config.time_scheme == TimeIntegrationScheme::LuSgs {
            return self
                .advance_lusgs_step_3d(ctx, fields, storage, state, integrator, cfl, p_floor);
        }
        self.advance_explicit_step_3d(ctx, fields, storage, state, integrator, cfl, p_floor)
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_gmres_step_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &mut ConservedFields,
        storage: &mut Rk4Storage,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
        cfl: Real,
        p_floor: Real,
    ) -> Result<CompressibleStepInfo> {
        let step_start = Instant::now();
        if !self.config.local_time_step {
            return Err(crate::error::AsimuError::Config(
                "time.scheme = gmres 须配合 [time].local_time_step = true（稳态伪时间）"
                    .to_string(),
            ));
        }
        let compute_dt_start = Instant::now();
        let (dt, cell_dts, sigma) = {
            let _span = info_span!("compute_dt").entered();
            let (cell_dts, sigma) = self.prepare_lusgs_timestep_3d(ctx, fields, cfl, p_floor)?;
            (min_positive_dt(&cell_dts), cell_dts, sigma)
        };
        let compute_dt_ms = elapsed_ms(compute_dt_start);
        integrator.config.dt = dt;
        storage.ensure_capacity(fields.num_cells())?;
        storage.u0.copy_from(fields)?;
        let implicit_solve_start = Instant::now();
        let delta = {
            let _span = info_span!("gmres_implicit_solve").entered();
            self.solve_gmres_implicit_delta_3d(
                ctx,
                &storage.u0,
                &cell_dts,
                &sigma,
                p_floor,
                self.config.gmres,
            )?
        };
        let implicit_solve_ms = elapsed_ms(implicit_solve_start);
        let line_search_start = Instant::now();
        let update = {
            let _span = info_span!("gmres_line_search").entered();
            apply_delta_with_line_search(
                fields,
                &mut storage.stage,
                &storage.u0,
                &delta,
                ctx.eos,
                p_floor,
            )?
        };
        let line_search_ms = elapsed_ms(line_search_start);
        // 监控量 = ‖R(U^0)‖：隐式求解阶段已装配 base_residual，勿再步末重算。
        let step_residual = delta.base_residual_rms;
        let step_total_ms = elapsed_ms(step_start);
        log_gmres_step_diagnostics(GmresStepLog {
            step: state.time_step.saturating_add(1),
            dt,
            cfl,
            delta: &delta,
            update,
            residual_rms: step_residual,
            timing: GmresStepTiming {
                compute_dt_ms,
                implicit_solve_ms,
                line_search_ms,
                post_residual_ms: 0.0,
                step_total_ms,
            },
        });
        let time_info = integrator.advance(state)?;
        Ok(CompressibleStepInfo {
            dt: time_info.dt,
            physical_time: time_info.physical_time,
            step: time_info.step,
            residual_rms: step_residual,
            residual_log10: log10_positive(step_residual),
            cfl,
            is_final: time_info.is_final,
            converged: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_explicit_step_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &mut ConservedFields,
        storage: &mut Rk4Storage,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
        cfl: Real,
        p_floor: Real,
    ) -> Result<CompressibleStepInfo> {
        let inviscid = self.config.inviscid;
        let (dt, cell_dts) = {
            let _span = info_span!(
                "compute_dt",
                cells = ctx.structured.num_cells(),
                local_time_step = self.config.local_time_step,
            )
            .entered();
            {
                let _span = info_span!("enforce_positivity_pre").entered();
                fields.enforce_positivity(ctx.eos, p_floor);
            }
            let cell_dts = self.compute_cell_dts_3d(ctx, fields, cfl, p_floor)?;
            (min_positive_dt(&cell_dts), cell_dts)
        };
        integrator.config.dt = dt;
        let eos = *ctx.eos;
        let step_residual = {
            let _span = info_span!("rhs_monitor").entered();
            self.rhs_context_3d(ctx, &inviscid, p_floor)
                .run(fields, &mut storage.k1)?;
            storage.k1.density_rms_norm()
        };
        let cell_dts_arg = if self.config.local_time_step {
            Some(cell_dts.as_slice())
        } else {
            None
        };
        let global_dt_scales;
        let smoothing_scales = if let Some(local_dt) = cell_dts_arg {
            local_dt
        } else {
            global_dt_scales = vec![dt; fields.num_cells()];
            global_dt_scales.as_slice()
        };
        {
            let _span = info_span!(
                "time_integration",
                scheme = self.config.time_scheme.label(),
                local_time_step = self.config.local_time_step,
            )
            .entered();
            let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
                self.rhs_context_3d(ctx, &inviscid, p_floor).run(u, r)?;
                self.smooth_residual_if_enabled(
                    ctx.structured,
                    u,
                    r,
                    smoothing_scales,
                    &eos,
                    p_floor,
                )
            };
            self.advance_explicit_step(
                fields,
                storage,
                dt,
                cell_dts_arg,
                evaluate,
                Some((&eos, p_floor)),
            )?;
        }
        {
            let _span = info_span!("enforce_positivity_post").entered();
            fields.enforce_positivity(&eos, p_floor);
        }
        let time_info = {
            let _span = info_span!("advance_clock").entered();
            integrator.advance(state)?
        };
        Ok(CompressibleStepInfo {
            dt: time_info.dt,
            physical_time: time_info.physical_time,
            step: time_info.step,
            residual_rms: step_residual,
            residual_log10: log10_positive(step_residual),
            cfl,
            is_final: time_info.is_final,
            converged: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_lusgs_step_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &mut ConservedFields,
        storage: &mut Rk4Storage,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
        cfl: Real,
        p_floor: Real,
    ) -> Result<CompressibleStepInfo> {
        if !self.config.local_time_step {
            return Err(crate::error::AsimuError::Config(
                "time.scheme = lu_sgs 须配合 [time].local_time_step = true（稳态伪时间）"
                    .to_string(),
            ));
        }
        let inviscid = self.config.inviscid;
        let (dt, cell_dts, sigma) = {
            let _span = info_span!(
                "compute_dt",
                cells = ctx.structured.num_cells(),
                scheme = "lu_sgs",
            )
            .entered();
            let (cell_dts, sigma) = self.prepare_lusgs_timestep_3d(ctx, fields, cfl, p_floor)?;
            (min_positive_dt(&cell_dts), cell_dts, sigma)
        };
        integrator.config.dt = dt;
        let structured = ctx.structured;
        let eos = ctx.eos;
        let volumes = structured.cell_volumes();
        let lu_sgs = self.config.lu_sgs;
        let update_scales: Vec<Real> = cell_dts
            .iter()
            .zip(sigma.iter())
            .map(|(&dt_i, &sigma_i)| lu_sgs.omega * dt_i / (1.0 + dt_i * sigma_i))
            .collect();
        {
            let _span = info_span!(
                "time_integration",
                scheme = "lu_sgs",
                local_time_step = true,
            )
            .entered();
            fields.enforce_positivity(eos, p_floor);
            storage.u0.copy_from(fields)?;
            {
                let _span = info_span!("lu_sgs_rhs").entered();
                self.rhs_context_3d(ctx, &inviscid, p_floor)
                    .run(&storage.u0, &mut storage.k1)?;
                self.smooth_residual_if_enabled(
                    structured,
                    &storage.u0,
                    &mut storage.k1,
                    &update_scales,
                    eos,
                    p_floor,
                )?;
            }
            if lu_sgs.sweep {
                let mut sweep_params = LuSgsSweep3dParams {
                    mesh: structured,
                    eos,
                    primitives: &mut ctx.primitive_scratch,
                    min_pressure: p_floor,
                    backward_damping: lu_sgs.sweep_backward_damping,
                };
                let _span = info_span!("lu_sgs_sweep").entered();
                lu_sgs_sweep_3d(
                    fields,
                    &storage.k1,
                    &mut sweep_params,
                    &cell_dts,
                    &sigma,
                    &volumes,
                    lu_sgs.omega,
                    eos.gamma,
                )?;
            } else {
                storage.stage.assign_lusgs_diagonal_update(
                    &storage.u0,
                    &storage.k1,
                    &sigma,
                    &cell_dts,
                    lu_sgs.omega,
                    eos.gamma,
                    p_floor,
                )?;
                fields.copy_from(&storage.stage)?;
            }
            fields.enforce_positivity(eos, p_floor);
        }
        // 监控量 = ‖R(U^0)‖：k1 在 lu_sgs_rhs 已装配，扫掠/对角更新不覆盖 k1。
        let step_residual = {
            let _span = info_span!("rhs_monitor").entered();
            storage.k1.density_rms_norm()
        };
        fields.enforce_positivity(ctx.eos, p_floor);
        let time_info = integrator.advance(state)?;
        Ok(CompressibleStepInfo {
            dt: time_info.dt,
            physical_time: time_info.physical_time,
            step: time_info.step,
            residual_rms: step_residual,
            residual_log10: log10_positive(step_residual),
            cfl,
            is_final: time_info.is_final,
            converged: false,
        })
    }

    /// 1D 多步瞬态积分直至 `max_steps`。
    pub fn run_transient_1d(
        &self,
        ctx: &CompressibleAdvanceContext1d<'_>,
        fields: &mut ConservedFields,
    ) -> Result<Vec<CompressibleStepInfo>> {
        let mut storage = Rk4Storage::new(ctx.mesh.num_cells())?;
        let mut state = SolverState::default();
        let mut integrator = RungeKutta4Integrator::new(self.config.time);
        let mut history = Vec::new();
        loop {
            let info =
                self.advance_step_1d(ctx, fields, &mut storage, &mut state, &mut integrator)?;
            let is_final = info.is_final;
            info!(
                step = info.step,
                dt = %format_log_sci4(info.dt),
                t = %format_log_sci4(info.physical_time),
                log10_residual = %format_log_fixed4(info.residual_log10),
                cfl = info.cfl,
                is_final,
                "可压缩 Euler 1D 时间步"
            );
            history.push(info);
            if is_final {
                break;
            }
        }
        Ok(history)
    }

    /// 3D 多步瞬态积分直至 `max_steps`。
    pub fn run_transient_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &mut ConservedFields,
    ) -> Result<Vec<CompressibleStepInfo>> {
        let mut storage = Rk4Storage::new(ctx.structured.num_cells())?;
        let mut state = SolverState::default();
        let mut integrator = RungeKutta4Integrator::new(self.config.time);
        let mut history = Vec::new();
        loop {
            let info =
                self.advance_step_3d(ctx, fields, &mut storage, &mut state, &mut integrator)?;
            let is_final = info.is_final;
            info!(
                step = info.step,
                dt = %format_log_sci4(info.dt),
                t = %format_log_sci4(info.physical_time),
                log10_residual = %format_log_fixed4(info.residual_log10),
                cfl = info.cfl,
                is_final,
                "可压缩 Euler 3D 时间步"
            );
            history.push(info);
            if is_final {
                break;
            }
        }
        Ok(history)
    }

    fn cfl_for_step(&self, state: &SolverState) -> Real {
        let next_step = state.time_step.saturating_add(1);
        self.config
            .cfl_schedule
            .at_step(next_step, self.config.time.max_steps)
    }

    fn positivity_pressure_floor(freestream: &FreestreamParams) -> Real {
        crate::field::positivity_pressure_floor(freestream.pressure)
    }

    fn advance_explicit_step<F>(
        &self,
        fields: &mut ConservedFields,
        storage: &mut Rk4Storage,
        dt_global: Real,
        cell_dts: Option<&[Real]>,
        evaluate_rhs: F,
        positivity: Option<(&IdealGasEoS, Real)>,
    ) -> Result<()>
    where
        F: FnMut(&ConservedFields, &mut ConservedResidual) -> Result<()>,
    {
        let _span = info_span!(
            "explicit_step",
            scheme = self.config.time_scheme.label(),
            local = cell_dts.is_some(),
        )
        .entered();
        let (eos, min_pressure) = match positivity {
            Some((eos, p)) => (Some(eos), p),
            None => (None, 1.0e-6),
        };
        match (self.config.time_scheme, cell_dts) {
            (TimeIntegrationScheme::Rk4, Some(dt)) => {
                rk4_step_local(fields, storage, dt, evaluate_rhs, eos, min_pressure)
            }
            (TimeIntegrationScheme::Rk4, None) => {
                rk4_step(fields, storage, dt_global, evaluate_rhs)
            }
            (TimeIntegrationScheme::Euler, Some(dt)) => {
                euler_step_local(fields, storage, dt, evaluate_rhs, eos, min_pressure)
            }
            (TimeIntegrationScheme::Euler, None) => {
                euler_step(fields, storage, dt_global, evaluate_rhs, eos, min_pressure)
            }
            (TimeIntegrationScheme::LuSgs, _) => Err(crate::error::AsimuError::Solver(
                "advance_explicit_step 不支持 lu_sgs".to_string(),
            )),
            (TimeIntegrationScheme::Gmres, _) => Err(crate::error::AsimuError::Solver(
                "advance_explicit_step 不支持 gmres".to_string(),
            )),
            (TimeIntegrationScheme::Simplec, _) => Err(crate::error::AsimuError::Solver(
                "advance_explicit_step 不支持 simplec".to_string(),
            )),
            (TimeIntegrationScheme::Piso, _) => Err(crate::error::AsimuError::Solver(
                "advance_explicit_step 不支持 piso".to_string(),
            )),
        }
    }

    fn suggest_dt_1d(
        &self,
        mesh: &StructuredMesh1d,
        fields: &ConservedFields,
        eos: &IdealGasEoS,
        cfl: Real,
    ) -> Result<Real> {
        if let Some(dt) = positive_fixed_dt(self.config.time.dt) {
            return Ok(dt);
        }
        let max_speed = max_wave_speed(fields, eos, 1.0e-6)?;
        crate::solver::time::suggested_dt_cfl(mesh.dx(), max_speed, cfl)
    }

    /// Blazek 结构网格局部时间步：\(\Delta t_i=\mathrm{CFL}/\sigma_i\)；RK4 / LU-SGS 共用。
    fn compute_cell_dts_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &mut ConservedFields,
        cfl: Real,
        p_floor: Real,
    ) -> Result<Vec<Real>> {
        let n = fields.num_cells();
        if let Some(dt) = positive_fixed_dt(self.config.time.dt) {
            return Ok(vec![dt; n]);
        }
        let (cell_dts, _) = self.prepare_spectral_timestep_3d(ctx, fields, cfl, p_floor)?;
        if self.config.local_time_step {
            Ok(cell_dts)
        } else {
            let dt = min_positive_dt(&cell_dts);
            Ok(vec![dt; n])
        }
    }

    /// LU-SGS：与显式 RK 共用有限体积 face-sum 谱半径时间步。
    fn prepare_lusgs_timestep_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &mut ConservedFields,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)> {
        let (_, sigma) = self.prepare_spectral_timestep_3d(ctx, fields, cfl, p_floor)?;
        let cell_dts = finalize_cell_dts_from_sigma(
            &ctx.structured.cell_volumes(),
            &sigma,
            cfl,
            positive_fixed_dt(self.config.time.dt),
            true,
        )?;
        Ok((cell_dts, sigma))
    }

    /// 刷新 BC ghost、原始变量，并计算 Blazek face-sum 谱半径 \((\Delta t_i,\sigma_i)\)。
    fn prepare_spectral_timestep_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &mut ConservedFields,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)> {
        fields.enforce_positivity(ctx.eos, p_floor);
        refresh_compressible_ghosts_and_primitives(RefreshCompressibleStateInput {
            boundary_mesh: ctx.mesh,
            patches: ctx.patches,
            fields,
            ghosts: ctx.ghosts,
            eos: ctx.eos,
            freestream: ctx.freestream,
            reference: ctx.reference,
            viscous: ctx.viscous,
            min_pressure: p_floor,
            primitives: &mut ctx.primitive_scratch,
        })?;
        let params = self.spectral_radius_params(ctx, p_floor);
        let sigma = cell_spectral_radius_3d(&params)?;
        let volumes = params.mesh.cell_volumes();
        let cell_dts = cell_local_dt_spectral(&volumes, &sigma, cfl)?;
        Ok((cell_dts, sigma))
    }

    fn spectral_radius_params<'a>(
        &self,
        ctx: &'a CompressibleAdvanceContext3d<'_>,
        p_floor: Real,
    ) -> SpectralRadius3dParams<'a> {
        SpectralRadius3dParams {
            mesh: ctx.structured,
            boundary_mesh: ctx.mesh,
            boundaries: ctx.patches,
            ghosts: ctx.ghosts,
            primitives: &ctx.primitive_scratch,
            eos: ctx.eos,
            min_pressure: p_floor,
            viscous: ctx.viscous,
        }
    }
}

fn positive_fixed_dt(dt: Real) -> Option<Real> {
    if dt > 0.0 { Some(dt) } else { None }
}

fn elapsed_ms(start: Instant) -> Real {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
#[path = "compressible_tests.rs"]
mod tests;
