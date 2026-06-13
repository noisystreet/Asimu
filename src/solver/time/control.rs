//! 时间步循环控制：统一 max_steps / 容差早停判定（ADR 0005 编排层）。

use crate::core::Real;
use crate::core::convergence::compressible_log10_tolerance_met;
use crate::solver::CompressibleStepInfo;

/// 可压缩显式/隐式伪时间外层循环控制参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientStepControl {
    pub log10_tolerance: Option<Real>,
}

impl TransientStepControl {
    #[must_use]
    pub const fn new(log10_tolerance: Option<Real>) -> Self {
        Self { log10_tolerance }
    }

    /// 本步是否应停止外层循环（步数上限或残差收敛）。
    #[must_use]
    pub fn should_stop(&self, step: &CompressibleStepInfo) -> bool {
        step.is_final || compressible_log10_tolerance_met(step.residual_log10, self.log10_tolerance)
    }

    /// 为步信息写入 `converged` 并返回是否早停。
    #[must_use]
    pub fn finalize_step(&self, step: &mut CompressibleStepInfo) -> bool {
        step.converged =
            compressible_log10_tolerance_met(step.residual_log10, self.log10_tolerance);
        self.should_stop(step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log10_positive;

    fn step(log10_res: Real, is_final: bool) -> CompressibleStepInfo {
        CompressibleStepInfo {
            dt: 1.0e-4,
            physical_time: 0.0,
            step: 1,
            residual_rms: 10_f64.powf(log10_res),
            residual_log10: log10_res,
            cfl: 0.4,
            is_final,
            converged: false,
        }
    }

    #[test]
    fn stops_on_final_or_tolerance() {
        let ctrl = TransientStepControl::new(Some(-6.0));
        assert!(ctrl.should_stop(&step(-7.0, false)));
        assert!(!ctrl.should_stop(&step(-5.0, false)));
        assert!(ctrl.should_stop(&step(-5.0, true)));
    }

    #[test]
    fn finalize_sets_converged_flag() {
        let ctrl = TransientStepControl::new(Some(-6.0));
        let mut info = step(-7.0, false);
        assert!(ctrl.finalize_step(&mut info));
        assert!(info.converged);
    }

    #[test]
    fn no_tolerance_never_converges_early() {
        let ctrl = TransientStepControl::new(None);
        let mut info = step(log10_positive(1.0e-20), false);
        assert!(!ctrl.finalize_step(&mut info));
        assert!(!info.converged);
    }
}
