//! 核心计算精度（ADR 0016）。
//!
//! 仅覆盖求解热路径；网格几何与 I/O 仍使用 [`super::Real`]（默认 `f64`）。

use crate::error::{AsimuError, Result};

use super::Real;

/// 运行时可选的核心计算精度。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ComputePrecision {
    #[default]
    F64,
    F32,
}

impl ComputePrecision {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
        }
    }
}

/// 解析 `[numerics].compute_precision`。
pub fn parse_compute_precision(raw: &str) -> Result<ComputePrecision> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "f64" => Ok(ComputePrecision::F64),
        "f32" => Ok(ComputePrecision::F32),
        other => Err(AsimuError::Config(format!(
            "未知 [numerics].compute_precision \"{other}\"；可选 f64 | f32"
        ))),
    }
}

mod sealed {
    pub trait Sealed {}
}

/// 核心计算标量；仅允许 `f32` 与 `f64` 实现。
pub trait ComputeFloat: Copy + Send + Sync + PartialEq + 'static + sealed::Sealed {
    const PRECISION: ComputePrecision;

    fn from_real(value: Real) -> Self;
    fn to_real(self) -> Real;
    fn zero() -> Self;
    fn add(self, rhs: Self) -> Self;
    fn add_mul_real(self, rhs: Self, scale: Real) -> Self;
}

impl sealed::Sealed for f32 {}
impl sealed::Sealed for f64 {}

impl ComputeFloat for f64 {
    const PRECISION: ComputePrecision = ComputePrecision::F64;

    fn from_real(value: Real) -> Self {
        value
    }

    fn to_real(self) -> Real {
        self
    }

    fn zero() -> Self {
        0.0
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn add_mul_real(self, rhs: Self, scale: Real) -> Self {
        self + scale * rhs
    }
}

impl ComputeFloat for f32 {
    const PRECISION: ComputePrecision = ComputePrecision::F32;

    fn from_real(value: Real) -> Self {
        value as f32
    }

    fn to_real(self) -> Real {
        f64::from(self)
    }

    fn zero() -> Self {
        0.0
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn add_mul_real(self, rhs: Self, scale: Real) -> Self {
        self + (scale as f32) * rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_compute_precision_accepts_f64_and_f32() {
        assert_eq!(
            parse_compute_precision("f64").expect("f64"),
            ComputePrecision::F64
        );
        assert_eq!(
            parse_compute_precision(" F32 ").expect("f32"),
            ComputePrecision::F32
        );
    }

    #[test]
    fn parse_compute_precision_rejects_unknown() {
        let err = parse_compute_precision("mixed").expect_err("unknown");
        assert!(err.to_string().contains("compute_precision"));
    }

    #[test]
    fn compute_float_round_trip() {
        let value = 1.25_f64;
        let f32_value = f32::from_real(value);
        assert!((f32_value.to_real() - value).abs() < 1.0e-6);
        assert_eq!(f64::from_real(value), value);
    }

    #[test]
    fn compute_float_add_mul_real() {
        let a = f32::from_real(1.0);
        let b = f32::from_real(2.0);
        assert!((a.add_mul_real(b, 0.5).to_real() - 2.0).abs() < 1.0e-6);
    }
}
