//! 默认数值标量类型（v0.2 起统一场变量与矩阵系数）。

/// 场变量、矩阵系数、通量、残差（v0.5+ 可通过 feature 切换 f32）。
pub type Real = f64;

/// 浮点比较容差工具。
#[must_use]
pub fn approx_eq(a: Real, b: Real, tol: Real) -> bool {
    (a - b).abs() <= tol
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
}
