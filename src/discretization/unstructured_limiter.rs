//! 非结构梯度限制器（Barth–Jespersen / Venkatakrishnan）。
//!
//! 理论：[`docs/adr/0012`](../../docs/adr/0012-unstructured-gradient-limiters.md)

use crate::core::Real;

/// 非结构 IDWLS 梯度外推用限制器（与结构化 `SlopeLimiter` 独立）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnstructuredGradientLimiter {
    #[default]
    BarthJespersen,
    Venkatakrishnan,
}

impl UnstructuredGradientLimiter {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::BarthJespersen => "barth_jespersen",
            Self::Venkatakrishnan => "venkatakrishnan",
        }
    }

    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "barth_jespersen" | "barth-jespersen" | "bj" => Some(Self::BarthJespersen),
            "venkatakrishnan" | "venk" | "v" => Some(Self::Venkatakrishnan),
            _ => None,
        }
    }
}

/// Barth–Jespersen 单样本限制因子 \(\psi_{i,m}\in[0,1]\)（式 BJ）。
#[must_use]
pub fn barth_jespersen_sample_factor(
    phi_i: Real,
    phi_min: Real,
    phi_max: Real,
    grad_dot_dr: Real,
) -> Real {
    if grad_dot_dr > 0.0 {
        ((phi_max - phi_i) / grad_dot_dr).clamp(0.0, 1.0)
    } else if grad_dot_dr < 0.0 {
        ((phi_min - phi_i) / grad_dot_dr).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Venkatakrishnan 光滑函数 \(\varphi(\xi)\)（式 V2）。
#[must_use]
pub fn venkatakrishnan_phi(xi: Real) -> Real {
    (xi * xi + 2.0 * xi) / (xi * xi + xi + 2.0)
}

/// Venkatakrishnan 单样本限制因子。
#[must_use]
pub fn venkatakrishnan_sample_factor(phi_i: Real, phi_m: Real, grad_dot_dr: Real) -> Real {
    if grad_dot_dr.abs() <= Real::EPSILON {
        return 1.0;
    }
    let xi = (phi_m - phi_i) / (2.0 * grad_dot_dr);
    venkatakrishnan_phi(xi).clamp(0.0, 1.0)
}

/// 对单元所有 LSQ 样本取最小限制因子。
#[must_use]
pub fn limit_cell_gradient_factor(
    limiter: UnstructuredGradientLimiter,
    phi_i: Real,
    phi_min: Real,
    phi_max: Real,
    grad: [Real; 3],
    samples: &[([Real; 3], Real)],
) -> Real {
    let mut psi = 1.0_f64;
    for &(dr, phi_m) in samples {
        let grad_dot_dr = grad[0] * dr[0] + grad[1] * dr[1] + grad[2] * dr[2];
        let sample_psi = match limiter {
            UnstructuredGradientLimiter::BarthJespersen => {
                barth_jespersen_sample_factor(phi_i, phi_min, phi_max, grad_dot_dr)
            }
            UnstructuredGradientLimiter::Venkatakrishnan => {
                venkatakrishnan_sample_factor(phi_i, phi_m, grad_dot_dr)
            }
        };
        psi = psi.min(sample_psi);
    }
    psi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    #[test]
    fn barth_jespersen_limits_positive_slope_to_neighbor_max() {
        let psi = barth_jespersen_sample_factor(1.0, 0.5, 2.0, 4.0);
        assert!(approx_eq(psi, 0.25, 1.0e-12));
    }

    #[test]
    fn venkatakrishnan_is_smooth_near_zero_xi() {
        let v0 = venkatakrishnan_phi(0.0);
        let v1 = venkatakrishnan_phi(1.0e-6);
        assert!(approx_eq(v0, 0.0, 1.0e-12));
        assert!(v1 > 0.0);
        assert!(v1 < 1.0);
    }

    #[test]
    fn uniform_field_keeps_unit_limiter() {
        let samples = [([1.0, 0.0, 0.0], 1.0), ([0.0, 1.0, 0.0], 1.0)];
        let psi = limit_cell_gradient_factor(
            UnstructuredGradientLimiter::BarthJespersen,
            1.0,
            1.0,
            1.0,
            [0.0, 0.0, 0.0],
            &samples,
        );
        assert!(approx_eq(psi, 1.0, 1.0e-12));
    }
}
