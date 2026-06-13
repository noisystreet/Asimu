//! 收敛判据（可压 log₁₀ 残差、不可压稳态窗口等）。

use crate::core::Real;

/// \(\log_{10}(\mathrm{RMS})\) 是否达到 `[time].tolerance` 早停条件。
#[must_use]
pub fn log10_residual_converged(log10_residual: Real, log10_tolerance: Real) -> bool {
    super::real::residual_converged(log10_residual, log10_tolerance)
}

/// 可压步信息是否满足 log₁₀(RMS) 容差早停（`tolerance` 为 `None` 时不早停）。
#[must_use]
pub fn compressible_log10_tolerance_met(
    log10_residual: Real,
    log10_tolerance: Option<Real>,
) -> bool {
    log10_tolerance.is_some_and(|tol| log10_residual_converged(log10_residual, tol))
}

/// 不可压稳态 SIMPLEC/PISO 连续收敛窗口（与 `[time].min_steps` 联动）。
#[must_use]
pub fn incompressible_steady_convergence_window(min_steps: u64) -> usize {
    (min_steps as usize).clamp(1, 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressible_tolerance_uses_log10_residual() {
        assert!(compressible_log10_tolerance_met(-7.0, Some(-6.0)));
        assert!(!compressible_log10_tolerance_met(-5.0, Some(-6.0)));
        assert!(!compressible_log10_tolerance_met(-7.0, None));
    }

    #[test]
    fn incompressible_window_clamps_to_32() {
        assert_eq!(incompressible_steady_convergence_window(0), 1);
        assert_eq!(incompressible_steady_convergence_window(100), 32);
    }
}
