//! 默认数值标量类型（v0.2 起统一场变量与矩阵系数）。

/// 场变量、矩阵系数、通量、残差（v0.5+ 可通过 feature 切换 f32）。
pub type Real = f64;

/// 浮点比较容差工具。
#[must_use]
pub fn approx_eq(a: Real, b: Real, tol: Real) -> bool {
    (a - b).abs() <= tol
}

/// \(\log_{10}(\mathrm{RMS})\) 是否达到 `[time].tolerance`（log₁₀ 容差）早停条件。
#[must_use]
pub fn residual_converged(log10_residual: Real, log10_tolerance: Real) -> bool {
    log10_tolerance.is_finite() && log10_residual.is_finite() && log10_residual <= log10_tolerance
}

/// 日志：`dt` / `t` 等时间量（科学计数法，小数点后 4 位）。
#[must_use]
pub fn format_log_sci4(v: Real) -> String {
    format!("{:.4e}", v)
}

/// 日志：log₁₀ 残差等（固定小数点，小数点后 4 位）。
#[must_use]
pub fn format_log_fixed4(v: Real) -> String {
    format!("{:.4}", v)
}

/// 日志：CFL 等（固定小数点，小数点后 5 位）。
#[must_use]
pub fn format_log_fixed5(v: Real) -> String {
    format!("{:.5}", v)
}

/// \(\log_{10}(x)\)；\(x \le 0\) 时返回 \(-\infty\)（仅用于残差日志）。
#[must_use]
pub fn log10_positive(x: Real) -> Real {
    if x <= 0.0 {
        Real::NEG_INFINITY
    } else {
        x.log10()
    }
}

/// 标量运算扩展（便于后续切换 `Real` 底层类型）。
pub trait RealOps {
    fn approx_eq(self, other: Self, tol: Self) -> bool;
}

impl RealOps for Real {
    fn approx_eq(self, other: Self, tol: Self) -> bool {
        approx_eq(self, other, tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_eq_works() {
        assert!(approx_eq(1.0, 1.0 + 1.0e-13, 1.0e-10));
        assert!(!approx_eq(1.0, 1.1, 1.0e-2));
    }

    #[test]
    fn residual_converged_compares_log10_to_tolerance() {
        assert!(super::residual_converged(-7.0, -6.0));
        assert!(!super::residual_converged(-5.0, -6.0));
        assert!(super::residual_converged(6.0, 6.5));
    }

    #[test]
    fn format_log_sci4_uses_four_decimals() {
        assert_eq!(format_log_sci4(2.126_356_776_522_852e-12), "2.1264e-12");
        assert_eq!(format_log_sci4(3.058_568_162_810_190_4e-8), "3.0586e-8");
    }

    #[test]
    fn format_log_fixed4_uses_four_decimals() {
        assert_eq!(format_log_fixed4(6.441_857_557_058_491), "6.4419");
    }

    #[test]
    fn format_log_fixed5_uses_five_decimals() {
        assert_eq!(format_log_fixed5(0.001_23), "0.00123");
    }

    #[test]
    fn log10_positive_works() {
        assert!(log10_positive(10.0) > 0.99);
    }
}
