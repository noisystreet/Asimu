//! 标量场 SoA（核心计算精度泛型，ADR 0016 P1）。

use crate::core::{ComputeFloat, Real};
use crate::error::{AsimuError, Result};

/// 标量场，长度与网格单元数一致。
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarFieldT<T: ComputeFloat> {
    values: Vec<T>,
}

impl<T: ComputeFloat> ScalarFieldT<T> {
    /// 构造常值场；`num_cells` 必须大于 0。
    pub fn uniform(num_cells: usize, value: T) -> Result<Self> {
        if num_cells == 0 {
            return Err(AsimuError::Field("num_cells 必须大于 0".to_string()));
        }
        Ok(Self {
            values: vec![value; num_cells],
        })
    }

    /// 从已有数据构造；拒绝空向量。
    pub fn from_values(values: Vec<T>) -> Result<Self> {
        if values.is_empty() {
            return Err(AsimuError::Field("values 不能为空".to_string()));
        }
        Ok(Self { values })
    }

    /// 从工程标量 `Real` 切片构造 typed 场。
    pub fn from_real_values(values: Vec<Real>) -> Result<Self> {
        Self::from_values(values.into_iter().map(T::from_real).collect())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// 转为 `Real` 向量（I/O 与归约用）。
    #[must_use]
    pub fn to_real_values(&self) -> Vec<Real> {
        self.values.iter().map(|v| v.to_real()).collect()
    }
}

/// 默认工程标量场（`f64`）。
pub type ScalarField = ScalarFieldT<Real>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ComputePrecision;

    #[test]
    fn rejects_empty_field() {
        assert!(matches!(
            ScalarFieldT::<f64>::from_values(vec![]).unwrap_err(),
            AsimuError::Field(_)
        ));
    }

    #[test]
    fn uniform_field_has_length() {
        let field = ScalarField::uniform(4, 1.5).expect("field");
        assert_eq!(field.len(), 4);
        assert!(field.values().iter().all(|&v| v == 1.5));
    }

    #[test]
    fn f32_field_round_trips_through_real() {
        let field = ScalarFieldT::<f32>::from_real_values(vec![1.0, 2.5, 3.25]).expect("field");
        assert_eq!(field.len(), 3);
        assert_eq!(f32::PRECISION, ComputePrecision::F32);
        assert!((field.to_real_values()[1] - 2.5).abs() < 1.0e-6);
    }
}
