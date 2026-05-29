//! 时间推进抽象（ADR 0005）。
//!
//! 理论：[`docs/theory/time_integration.md`](../../../docs/theory/time_integration.md)（规划）

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

/// 时间推进策略；v0.2 用具体 struct，v0.5+ 可扩展 enum dispatch。
pub trait TimeIntegrator {
    fn mode(&self) -> TimeMode;
    fn advance(&mut self, state: &mut SolverState) -> Result<TimeStepInfo>;
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
