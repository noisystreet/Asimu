//! 守恒量更新的正性判定与增量限制。

use crate::core::Real;
use crate::physics::ConservedState;

/// \(\mathbf{U} \leftarrow \mathbf{U}_0 + \lambda\,\Delta\mathbf{U}\)。
#[must_use]
pub fn state_after_increment(
    base: &ConservedState,
    increment: [Real; 5],
    factor: Real,
) -> ConservedState {
    ConservedState {
        density: base.density + factor * increment[0],
        momentum: [
            base.momentum[0] + factor * increment[1],
            base.momentum[1] + factor * increment[2],
            base.momentum[2] + factor * increment[3],
        ],
        total_energy: base.total_energy + factor * increment[4],
    }
}

/// 密度正、内能正且有限（与 `primitive_from_conserved` 前置条件一致）。
#[must_use]
pub fn is_physical_conserved(state: &ConservedState, gamma: Real, min_pressure: Real) -> bool {
    let rho = state.density;
    if rho <= 0.0 || !rho.is_finite() || !state.total_energy.is_finite() {
        return false;
    }
    let ke = 0.5
        * (state.momentum[0] * state.momentum[0]
            + state.momentum[1] * state.momentum[1]
            + state.momentum[2] * state.momentum[2])
        / rho;
    let min_internal = min_pressure.max(0.0) / (gamma - 1.0);
    let internal = state.total_energy - ke;
    internal.is_finite() && internal > min_internal
}

/// 返回不超过 `scale` 的最大可行增量系数（不可行时返回 0）。
#[must_use]
pub fn max_physical_increment_scale(
    base: &ConservedState,
    increment: [Real; 5],
    scale: Real,
    gamma: Real,
    min_pressure: Real,
) -> Real {
    if scale <= 0.0 {
        return 0.0;
    }
    if is_physical_conserved(
        &state_after_increment(base, increment, scale),
        gamma,
        min_pressure,
    ) {
        return scale;
    }
    let mut alpha = 0.5;
    for _ in 0..12 {
        let trial = alpha * scale;
        if is_physical_conserved(
            &state_after_increment(base, increment, trial),
            gamma,
            min_pressure,
        ) {
            return trial;
        }
        alpha *= 0.5;
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::ConservedState;

    #[test]
    fn rejects_near_vacuum_internal_energy_with_pressure_floor() {
        let base = ConservedState {
            density: 1.0,
            momentum: [0.0, 0.0, 0.0],
            total_energy: 1.0e-8,
        };
        let increment = [0.0, 0.0, 0.0, 0.0, 0.0];
        let p_floor = 0.01;
        assert!(!is_physical_conserved(&base, 1.4, p_floor));
        assert_eq!(
            max_physical_increment_scale(&base, increment, 1.0, 1.4, p_floor),
            0.0
        );
    }

    #[test]
    fn limits_increment_that_would_break_internal_energy() {
        let base = ConservedState {
            density: 1.0,
            momentum: [1.0, 0.0, 0.0],
            total_energy: 1.0,
        };
        let increment = [0.0, 10.0, 0.0, 0.0, 0.0];
        let scale = max_physical_increment_scale(&base, increment, 1.0, 1.4, 0.0);
        assert!(scale > 0.0 && scale < 1.0);
        assert!(is_physical_conserved(
            &state_after_increment(&base, increment, scale),
            1.4,
            0.0
        ));
    }
}
