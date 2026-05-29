//! 物理场存储（SoA，v0.2 骨架）。
//!
//! 理论：[`docs/theory/fvm_diffusion.md`](../../docs/theory/fvm_diffusion.md)

use crate::core::Real;
use crate::error::{AsimuError, Result};

/// 标量场，长度与网格单元数一致。
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField {
    values: Vec<Real>,
}

impl ScalarField {
    /// 构造常值场；`num_cells` 必须大于 0。
    pub fn uniform(num_cells: usize, value: Real) -> Result<Self> {
        if num_cells == 0 {
            return Err(AsimuError::Field("num_cells 必须大于 0".to_string()));
        }
        Ok(Self {
            values: vec![value; num_cells],
        })
    }

    /// 从已有数据构造；拒绝空向量。
    pub fn from_values(values: Vec<Real>) -> Result<Self> {
        if values.is_empty() {
            return Err(AsimuError::Field("values 不能为空".to_string()));
        }
        Ok(Self { values })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn values(&self) -> &[Real] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [Real] {
        &mut self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_field() {
        assert!(matches!(
            ScalarField::from_values(vec![]).unwrap_err(),
            AsimuError::Field(_)
        ));
    }

    #[test]
    fn uniform_field_has_length() {
        let field = ScalarField::uniform(4, 1.5).expect("field");
        assert_eq!(field.len(), 4);
        assert!(field.values().iter().all(|&v| v == 1.5));
    }
}
