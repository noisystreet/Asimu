//! 时间推进抽象（ADR 0005）。
//!
//! 理论：[`docs/theory/time_integration.md`](../../../docs/theory/time_integration.md)

mod common;
pub mod euler;
pub mod lu_sgs;
pub mod rk4;
pub mod scheme;

use crate::core::Real;
use crate::error::Result;
use crate::solver::state::SolverState;

/// 稳态或瞬态模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeMode {
    Steady,
    Transient,
}

/// 单步时间推进摘要。
#[derive(Debug, Clone, PartialEq)]
pub struct TimeStepInfo {
    pub dt: Real,
    pub physical_time: Real,
    pub step: u64,
    pub is_final: bool,
}

/// 时间推进策略。
pub trait TimeIntegrator {
    fn mode(&self) -> TimeMode;
    fn advance(&mut self, state: &mut SolverState) -> Result<TimeStepInfo>;
}

pub use euler::{euler_step, euler_step_local};
pub use lu_sgs::{LuSgsConfig, lu_sgs_step, lu_sgs_step_local, lu_sgs_step_sweep_local};
pub use rk4::{Rk4Storage, RungeKutta4Config, RungeKutta4Integrator, rk4_step, rk4_step_local};
pub use scheme::TimeIntegrationScheme;

/// CFL 调度：在指定步数区间内从 `initial` 线性增至 `max`，之后保持 `max`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CflSchedule {
    pub initial: Real,
    pub max: Real,
    /// 线性爬升步数（第 1 步…`ramp_steps`）；`None` 表示在整段 `max_steps` 内爬升。
    pub ramp_steps: Option<u64>,
}

impl CflSchedule {
    #[must_use]
    pub const fn constant(cfl: Real) -> Self {
        Self {
            initial: cfl,
            max: cfl,
            ramp_steps: None,
        }
    }

    /// 第 `step` 步（1…`max_steps`）使用的 CFL。
    #[must_use]
    pub fn at_step(&self, step: u64, max_steps: u64) -> Real {
        if (self.max - self.initial).abs() <= Real::EPSILON {
            return self.initial;
        }
        let ramp_end = self
            .ramp_steps
            .unwrap_or(max_steps)
            .clamp(1, max_steps.max(1));
        let step = step.max(1);
        if step >= ramp_end {
            return self.max;
        }
        if ramp_end <= 1 {
            return self.initial;
        }
        let t = (step - 1) as Real / (ramp_end - 1) as Real;
        self.initial + t * (self.max - self.initial)
    }
}

/// CFL 建议时间步（可压缩流显式推进）。
pub fn suggested_dt_cfl(min_spacing: Real, max_wave_speed: Real, cfl: Real) -> Result<Real> {
    if min_spacing <= 0.0 || max_wave_speed <= 0.0 || cfl <= 0.0 {
        return Err(crate::error::AsimuError::Solver(
            "suggested_dt_cfl 参数须为正".to_string(),
        ));
    }
    Ok(cfl * min_spacing / max_wave_speed)
}

/// 逐单元 CFL 时间步：\(\Delta t_i = \mathrm{CFL}\, h_i / (|u|+a)_i\)。
pub fn local_dt_cfl(lengths: &[Real], wave_speeds: &[Real], cfl: Real) -> Result<Vec<Real>> {
    if lengths.len() != wave_speeds.len() {
        return Err(crate::error::AsimuError::Solver(
            "local_dt_cfl: lengths 与 wave_speeds 长度不一致".to_string(),
        ));
    }
    if cfl <= 0.0 {
        return Err(crate::error::AsimuError::Solver(
            "local_dt_cfl: CFL 须为正".to_string(),
        ));
    }
    let mut dt = Vec::with_capacity(lengths.len());
    for (&h, &speed) in lengths.iter().zip(wave_speeds.iter()) {
        if h <= 0.0 || speed <= 0.0 {
            return Err(crate::error::AsimuError::Solver(
                "local_dt_cfl: 间距与波速须为正".to_string(),
            ));
        }
        dt.push(cfl * h / speed);
    }
    Ok(dt)
}

/// 时间步数组的最小值（日志/伪时间累积用）。
#[must_use]
pub fn min_positive_dt(dt: &[Real]) -> Real {
    dt.iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

/// 稳态伪时间推进：递增 `pseudo_step` 直至达到上限。
#[derive(Debug, Clone)]
pub struct SteadyStateIntegrator {
    pub max_pseudo_steps: u32,
}

impl SteadyStateIntegrator {
    #[must_use]
    pub const fn new(max_pseudo_steps: u32) -> Self {
        Self { max_pseudo_steps }
    }
}

impl Default for SteadyStateIntegrator {
    fn default() -> Self {
        Self::new(1)
    }
}

impl TimeIntegrator for SteadyStateIntegrator {
    fn mode(&self) -> TimeMode {
        TimeMode::Steady
    }

    fn advance(&mut self, state: &mut SolverState) -> Result<TimeStepInfo> {
        state.pseudo_step = state.pseudo_step.saturating_add(1);
        let step = u64::from(state.pseudo_step);
        let is_final = state.pseudo_step >= self.max_pseudo_steps;
        Ok(TimeStepInfo {
            dt: Real::INFINITY,
            physical_time: 0.0,
            step,
            is_final,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::state::SolverState;

    #[test]
    fn cfl_schedule_linear_ramp() {
        let sched = CflSchedule {
            initial: 0.01,
            max: 0.05,
            ramp_steps: None,
        };
        assert!((sched.at_step(1, 10) - 0.01).abs() < 1.0e-12);
        assert!((sched.at_step(10, 10) - 0.05).abs() < 1.0e-12);
        assert!((sched.at_step(6, 10) - 0.032_222_222_222_222_22).abs() < 1.0e-12);
    }

    #[test]
    fn cfl_schedule_ramp_over_interval_then_holds_max() {
        let sched = CflSchedule {
            initial: 0.01,
            max: 0.5,
            ramp_steps: Some(5),
        };
        assert!((sched.at_step(1, 10) - 0.01).abs() < 1.0e-12);
        assert!((sched.at_step(5, 10) - 0.5).abs() < 1.0e-12);
        assert!((sched.at_step(6, 10) - 0.5).abs() < 1.0e-12);
        assert!((sched.at_step(10, 10) - 0.5).abs() < 1.0e-12);
        assert!((sched.at_step(3, 10) - 0.255).abs() < 1.0e-12);
    }

    #[test]
    fn cfl_schedule_constant_when_max_unset() {
        let sched = CflSchedule::constant(0.4);
        assert!((sched.at_step(1, 100) - 0.4).abs() < 1.0e-12);
        assert!((sched.at_step(100, 100) - 0.4).abs() < 1.0e-12);
    }

    #[test]
    fn local_dt_cfl_per_cell() {
        let lengths = vec![0.01, 0.02];
        let speeds = vec![340.0, 170.0];
        let dt = local_dt_cfl(&lengths, &speeds, 0.5).expect("dt");
        assert!((dt[0] - 0.5 * 0.01 / 340.0).abs() < 1.0e-12);
        assert!((dt[1] - 0.5 * 0.02 / 170.0).abs() < 1.0e-12);
        assert!((min_positive_dt(&dt) - dt[0]).abs() < 1.0e-12);
    }

    #[test]
    fn suggested_dt_positive() {
        let dt = suggested_dt_cfl(0.01, 340.0, 0.5).expect("dt");
        assert!(dt > 0.0);
    }

    #[test]
    fn steady_integrator_reaches_final_step() {
        let mut integrator = SteadyStateIntegrator::new(3);
        let mut state = SolverState::default();
        for _ in 0..2 {
            let info = integrator.advance(&mut state).expect("advance");
            assert!(!info.is_final);
        }
        let final_info = integrator.advance(&mut state).expect("advance");
        assert!(final_info.is_final);
        assert_eq!(final_info.step, 3);
    }
}
