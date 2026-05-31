//! 可压缩无粘 Euler 显式求解（RK4 / 一阶 Euler + FVM 残差）。
//!
//! 理论：[`docs/theory/time_integration.md`](../../docs/theory/time_integration.md)、
//! [`inviscid_flux.md`](../../docs/theory/inviscid_flux.md)

#[path = "compressible_rhs.rs"]
mod compressible_rhs;

use compressible_rhs::EvaluateRhs3d;

use crate::boundary::BoundarySet;
use crate::core::{Real, format_log_fixed4, format_log_sci4, log10_positive};
use crate::discretization::{BoundaryGhostBuffer, assemble_inviscid_residual_1d};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual};
use crate::mesh::{BoundaryMesh3d, StructuredMesh1d, StructuredMesh3d};
use crate::physics::{FreestreamParams, IdealGasEoS, PrimitiveState};
use crate::solver::state::SolverState;
use crate::solver::time::{
    CflSchedule, Rk4Storage, RungeKutta4Config, RungeKutta4Integrator, TimeIntegrationScheme,
    TimeIntegrator, euler_step, euler_step_local, local_dt_cfl, min_positive_dt, rk4_step,
    rk4_step_local,
};
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
    pub cfl_schedule: CflSchedule,
    pub time_mode: CompressibleTimeMode,
    pub local_time_step: bool,
    /// 时间积分格式（`rk4` 默认；`euler` 用于排除 RK 多阶段 bug）。
    pub time_scheme: TimeIntegrationScheme,
}

impl Default for CompressibleEulerConfig {
    fn default() -> Self {
        Self {
            time: RungeKutta4Config::default(),
            inviscid: crate::discretization::InviscidFluxConfig::default(),
            cfl_schedule: CflSchedule::constant(0.4),
            time_mode: CompressibleTimeMode::Transient,
            local_time_step: false,
            time_scheme: TimeIntegrationScheme::Rk4,
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

/// 3D 单步推进上下文（减少参数个数）。
pub struct CompressibleAdvanceContext3d<'a> {
    pub mesh: &'a dyn BoundaryMesh3d,
    pub structured: &'a StructuredMesh3d,
    pub patches: &'a BoundarySet,
    pub ghosts: &'a mut BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
}

/// 1D 多步推进上下文。
pub struct CompressibleAdvanceContext1d<'a> {
    pub mesh: &'a StructuredMesh1d,
    pub boundary: crate::discretization::InviscidBoundary1d,
    pub eos: &'a IdealGasEoS,
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
        let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
            let boundaries = ctx.boundary.resolve(u)?;
            assemble_inviscid_residual_1d(
                ctx.mesh,
                u,
                r,
                ctx.eos,
                &self.config.inviscid,
                &boundaries,
            )
        };
        self.advance_explicit_step(fields, storage, dt, None, evaluate, None)?;
        let boundaries = ctx.boundary.resolve(fields)?;
        assemble_inviscid_residual_1d(
            ctx.mesh,
            fields,
            &mut storage.k1,
            ctx.eos,
            &self.config.inviscid,
            &boundaries,
        )?;
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
        level = "info",
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
            let cell_dts =
                self.suggest_cell_dts_3d(ctx.structured, fields, ctx.eos, cfl, p_floor)?;
            (min_positive_dt(&cell_dts), cell_dts)
        };
        integrator.config.dt = dt;
        let mesh = ctx.mesh;
        let structured = ctx.structured;
        let patches = ctx.patches;
        let eos = ctx.eos;
        let freestream = ctx.freestream;
        let mut rhs_ctx = EvaluateRhs3d {
            mesh,
            structured,
            patches,
            ghosts: ctx.ghosts,
            eos,
            freestream,
            inviscid: &inviscid,
        };
        let step_residual = {
            let _span = info_span!("rhs_monitor").entered();
            rhs_ctx.run(fields, &mut storage.k1)?;
            storage.k1.density_rms_norm()
        };
        let cell_dts_arg = if self.config.local_time_step {
            Some(cell_dts.as_slice())
        } else {
            None
        };
        {
            let _span = info_span!(
                "time_integration",
                scheme = self.config.time_scheme.label(),
                local_time_step = self.config.local_time_step,
            )
            .entered();
            let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
                EvaluateRhs3d {
                    mesh,
                    structured,
                    patches,
                    ghosts: ctx.ghosts,
                    eos,
                    freestream,
                    inviscid: &inviscid,
                }
                .run(u, r)
            };
            self.advance_explicit_step(
                fields,
                storage,
                dt,
                cell_dts_arg,
                evaluate,
                Some((eos, p_floor)),
            )?;
        }
        {
            let _span = info_span!("enforce_positivity_post").entered();
            fields.enforce_positivity(ctx.eos, p_floor);
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

    fn suggest_dt_3d(
        &self,
        mesh: &StructuredMesh3d,
        fields: &ConservedFields,
        eos: &IdealGasEoS,
        cfl: Real,
        min_pressure: Real,
    ) -> Result<Real> {
        if let Some(dt) = positive_fixed_dt(self.config.time.dt) {
            return Ok(dt);
        }
        let min_spacing = mesh.min_positive_spacing()?;
        let max_speed = max_wave_speed(fields, eos, min_pressure)?;
        crate::solver::time::suggested_dt_cfl(min_spacing, max_speed, cfl)
    }

    fn suggest_cell_dts_3d(
        &self,
        mesh: &StructuredMesh3d,
        fields: &ConservedFields,
        eos: &IdealGasEoS,
        cfl: Real,
        min_pressure: Real,
    ) -> Result<Vec<Real>> {
        let n = fields.num_cells();
        if let Some(dt) = positive_fixed_dt(self.config.time.dt) {
            return Ok(vec![dt; n]);
        }
        if self.config.local_time_step {
            let lengths = mesh.cell_cfl_lengths()?;
            let speeds = cell_wave_speeds(fields, eos, min_pressure)?;
            return local_dt_cfl(&lengths, &speeds, cfl);
        }
        Ok(vec![
            self.suggest_dt_3d(
                mesh,
                fields,
                eos,
                cfl,
                min_pressure
            )?;
            n
        ])
    }
}

fn positive_fixed_dt(dt: Real) -> Option<Real> {
    if dt > 0.0 { Some(dt) } else { None }
}

/// 全场最大波速 \(|u| + a\)（CFL 估计）。
pub fn max_wave_speed(
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<Real> {
    let mut max_speed = Real::EPSILON;
    for i in 0..fields.num_cells() {
        let prim = crate::field::primitive_from_conserved_relaxed(
            eos,
            &fields.cell_state(i)?,
            min_pressure,
        )?;
        max_speed = max_speed.max(wave_speed_primitive(&prim, eos)?);
    }
    Ok(max_speed)
}

fn cell_wave_speeds(
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<Vec<Real>> {
    let mut speeds = Vec::with_capacity(fields.num_cells());
    for i in 0..fields.num_cells() {
        let prim = crate::field::primitive_from_conserved_relaxed(
            eos,
            &fields.cell_state(i)?,
            min_pressure,
        )?;
        speeds.push(wave_speed_primitive(&prim, eos)?);
    }
    Ok(speeds)
}

fn wave_speed_primitive(prim: &PrimitiveState, eos: &IdealGasEoS) -> Result<Real> {
    let rho = prim.density.max(1.0e-12);
    let pressure = prim.pressure.max(1.0e-6);
    let speed = (prim.velocity[0] * prim.velocity[0]
        + prim.velocity[1] * prim.velocity[1]
        + prim.velocity[2] * prim.velocity[2])
        .sqrt();
    Ok(speed + (eos.gamma * pressure / rho).sqrt())
}

#[cfg(test)]
#[path = "compressible_tests.rs"]
mod tests;
