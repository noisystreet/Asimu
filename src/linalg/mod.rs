//! 稀疏线性系统与 Krylov 迭代求解器。
//!
//! 理论：[`docs/theory/linear_gmres.md`](../../docs/theory/linear_gmres.md)。

mod gmres;
mod pcg;
mod preconditioner;
mod sparse;

use crate::core::Real;
use crate::error::{AsimuError, Result};

pub use gmres::{GmresConfig, GmresReport, GmresSolver};
pub use pcg::{PcgConfig, PcgReport, PcgSolver};
pub use preconditioner::{
    CellBlockDiagonalPreconditioner, CsrJacobiPreconditioner, IdentityPreconditioner,
    LusgsDiagonalPreconditioner,
};
pub use sparse::{CsrMatrix, CsrMatrixView, Ilu0Preconditioner};

/// 线性算子 \(y=A x\)。允许矩阵无关实现。
pub trait LinearOperator {
    fn dimension(&self) -> usize;
    fn apply(&mut self, x: &[Real], y: &mut [Real]) -> Result<()>;
}

/// 左预条件器 \(z=M^{-1}r\)。
pub trait Preconditioner {
    fn dimension(&self) -> usize;
    fn apply(&self, rhs: &[Real], out: &mut [Real]) -> Result<()>;
}

/// 三对角线性系统 \(A x = b\)（行 `i` 与 `i±1` 耦合）。
#[derive(Debug, Clone, PartialEq)]
pub struct LinearSystem {
    rhs: Vec<Real>,
    diag: Vec<Real>,
    lower: Vec<Real>,
    upper: Vec<Real>,
}

impl LinearSystem {
    /// 创建全零 `n×n` 三对角系统。
    pub fn zeros(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(AsimuError::Linalg("系统尺寸必须大于 0".to_string()));
        }
        Ok(Self {
            rhs: vec![0.0; n],
            diag: vec![0.0; n],
            lower: vec![0.0; n],
            upper: vec![0.0; n],
        })
    }

    /// 兼容入口：以 `rhs` 长度初始化，矩阵系数为零。
    pub fn new(rhs: Vec<Real>) -> Result<Self> {
        Self::zeros(rhs.len()).map(|mut sys| {
            sys.rhs = rhs;
            sys
        })
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

    pub fn diag(&self) -> &[Real] {
        &self.diag
    }

    pub fn diag_mut(&mut self) -> &mut [Real] {
        &mut self.diag
    }

    pub fn lower(&self) -> &[Real] {
        &self.lower
    }

    pub fn upper(&self) -> &[Real] {
        &self.upper
    }

    /// 向 `(row, col)` 累加系数（仅三对角邻接）。
    pub fn add_coupling(&mut self, row: usize, col: usize, coeff: Real) {
        debug_assert!(row < self.len());
        if row == col {
            self.diag[row] += coeff;
        } else if col + 1 == row {
            self.lower[row] += coeff;
        } else if row + 1 == col {
            self.upper[row] += coeff;
        } else {
            panic!("非三对角耦合 ({row}, {col})");
        }
    }

    pub fn add_diagonal(&mut self, row: usize, coeff: Real) {
        self.diag[row] += coeff;
    }

    pub fn add_rhs(&mut self, row: usize, value: Real) {
        self.rhs[row] += value;
    }

    /// 强 Dirichlet：清零行非对角耦合，置对角为 1。
    pub fn set_dirichlet_row(&mut self, row: usize, value: Real) {
        self.lower[row] = 0.0;
        self.diag[row] = 1.0;
        self.upper[row] = 0.0;
        self.rhs[row] = value;
    }

    /// Thomas 算法求解三对角系统。
    pub fn solve_tridiagonal(&self) -> Result<Vec<Real>> {
        let n = self.len();
        if n == 1 {
            if self.diag[0].abs() < Real::EPSILON {
                return Err(AsimuError::Linalg("奇异矩阵：对角元为零".to_string()));
            }
            return Ok(vec![self.rhs[0] / self.diag[0]]);
        }

        let mut c_prime = vec![0.0; n];
        let mut d_prime = vec![0.0; n];

        let denom0 = self.diag[0];
        if denom0.abs() < Real::EPSILON {
            return Err(AsimuError::Linalg(
                "Thomas 分解失败：对角元为零".to_string(),
            ));
        }
        c_prime[0] = self.upper[0] / denom0;
        d_prime[0] = self.rhs[0] / denom0;

        for i in 1..n {
            let denom = self.diag[i] - self.lower[i] * c_prime[i - 1];
            if denom.abs() < Real::EPSILON {
                return Err(AsimuError::Linalg(
                    "Thomas 分解失败：对角元为零".to_string(),
                ));
            }
            if i < n - 1 {
                c_prime[i] = self.upper[i] / denom;
            }
            d_prime[i] = (self.rhs[i] - self.lower[i] * d_prime[i - 1]) / denom;
        }

        let mut x = vec![0.0; n];
        x[n - 1] = d_prime[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }
        Ok(x)
    }
}

impl LinearOperator for LinearSystem {
    fn dimension(&self) -> usize {
        self.len()
    }

    fn apply(&mut self, x: &[Real], y: &mut [Real]) -> Result<()> {
        let n = self.len();
        ensure_vector_len(x, n, "tridiagonal input")?;
        ensure_vector_len(y, n, "tridiagonal output")?;
        for i in 0..n {
            let mut value = self.diag[i] * x[i];
            if i > 0 {
                value += self.lower[i] * x[i - 1];
            }
            if i + 1 < n {
                value += self.upper[i] * x[i + 1];
            }
            y[i] = value;
        }
        Ok(())
    }
}

pub(crate) fn ensure_vector_len(values: &[Real], expected: usize, label: &str) -> Result<()> {
    if values.len() != expected {
        return Err(AsimuError::Linalg(format!(
            "{label} 长度 {} 与期望 {expected} 不一致",
            values.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_system() {
        assert!(matches!(
            LinearSystem::zeros(0).unwrap_err(),
            AsimuError::Linalg(_)
        ));
    }

    #[test]
    fn solves_tridiagonal_system() {
        let mut sys = LinearSystem::zeros(3).expect("system");
        sys.add_coupling(0, 0, 2.0);
        sys.add_coupling(0, 1, -1.0);
        sys.add_coupling(1, 0, -1.0);
        sys.add_coupling(1, 1, 2.0);
        sys.add_coupling(1, 2, -1.0);
        sys.add_coupling(2, 1, -1.0);
        sys.add_coupling(2, 2, 2.0);
        sys.rhs_mut().copy_from_slice(&[1.0, 0.0, 1.0]);
        let x = sys.solve_tridiagonal().expect("solve");
        assert!((x[0] - 1.0).abs() < 1.0e-10);
        assert!((x[1] - 1.0).abs() < 1.0e-10);
        assert!((x[2] - 1.0).abs() < 1.0e-10);
    }
}
