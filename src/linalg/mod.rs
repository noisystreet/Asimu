//! 稀疏线性系统（v0.2 骨架；完整 SpMV / CG 后续 PR）。
//!
//! 理论：[`docs/theory/linear_cg.md`](../../docs/theory/linear_cg.md)（规划）

use crate::core::Real;
use crate::error::{AsimuError, Result};

/// 线性系统 Ax = b 的右端项占位；矩阵存储后续 PR 引入。
#[derive(Debug, Clone, PartialEq)]
pub struct LinearSystem {
    rhs: Vec<Real>,
}

impl LinearSystem {
    pub fn new(rhs: Vec<Real>) -> Result<Self> {
        if rhs.is_empty() {
            return Err(AsimuError::Linalg("rhs 不能为空".to_string()));
        }
        Ok(Self { rhs })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.rhs.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rhs.is_empty()
    }

    pub fn rhs(&self) -> &[Real] {
        &self.rhs
    }

    pub fn rhs_mut(&mut self) -> &mut [Real] {
        &mut self.rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_rhs() {
        assert!(matches!(
            LinearSystem::new(vec![]).unwrap_err(),
            AsimuError::Linalg(_)
        ));
    }
}
