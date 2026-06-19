use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::{LinearOperator, Preconditioner, ensure_vector_len};

/// PCG 参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgConfig {
    pub max_iters: usize,
    pub tolerance: Real,
}

impl Default for PcgConfig {
    fn default() -> Self {
        Self {
            max_iters: 500,
            tolerance: 1.0e-10,
        }
    }
}

/// PCG 收敛报告。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgReport {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: Real,
}

pub struct PcgSolver {
    config: PcgConfig,
}

impl PcgSolver {
    pub fn new(config: PcgConfig) -> Result<Self> {
        if config.max_iters == 0 || !config.tolerance.is_finite() || config.tolerance <= 0.0 {
            return Err(AsimuError::Linalg(
                "PCG max_iters/tolerance 参数无效".to_string(),
            ));
        }
        Ok(Self { config })
    }

    pub fn solve<A, M>(
        &self,
        op: &mut A,
        precond: &mut M,
        b: &[Real],
        x: &mut [Real],
    ) -> Result<PcgReport>
    where
        A: LinearOperator,
        M: Preconditioner,
    {
        let n = op.dimension();
        ensure_vector_len(b, n, "pcg rhs")?;
        ensure_vector_len(x, n, "pcg solution")?;
        if precond.dimension() != n {
            return Err(AsimuError::Linalg("PCG 预条件器尺寸不一致".to_string()));
        }

        let mut r = vec![0.0; n];
        let mut z = vec![0.0; n];
        let mut p = vec![0.0; n];
        let mut ap = vec![0.0; n];

        op.apply(x, &mut ap)?;
        for i in 0..n {
            r[i] = b[i] - ap[i];
        }
        let mut residual = norm(&r);
        if residual <= self.config.tolerance {
            return Ok(PcgReport {
                converged: true,
                iterations: 0,
                residual_norm: residual,
            });
        }

        precond.apply(&r, &mut z)?;
        p.copy_from_slice(&z);
        let mut rz_old = dot(&r, &z);

        for iter in 1..=self.config.max_iters {
            op.apply(&p, &mut ap)?;
            let denom = dot(&p, &ap);
            if denom.abs() <= Real::EPSILON {
                return Ok(PcgReport {
                    converged: false,
                    iterations: iter,
                    residual_norm: residual,
                });
            }
            let alpha = rz_old / denom;
            axpy(x, &p, alpha);
            axpy(&mut r, &ap, -alpha);
            residual = norm(&r);
            if residual <= self.config.tolerance {
                return Ok(PcgReport {
                    converged: true,
                    iterations: iter,
                    residual_norm: residual,
                });
            }
            precond.apply(&r, &mut z)?;
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz_old;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz_old = rz_new;
        }

        Ok(PcgReport {
            converged: residual <= self.config.tolerance,
            iterations: self.config.max_iters,
            residual_norm: residual,
        })
    }
}

fn dot(a: &[Real], b: &[Real]) -> Real {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(values: &[Real]) -> Real {
    values
        .iter()
        .map(|value| value * value)
        .sum::<Real>()
        .sqrt()
}

fn axpy(dst: &mut [Real], src: &[Real], scale: Real) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += scale * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{CsrJacobiPreconditioner, CsrMatrix};

    #[test]
    fn solves_spd_tridiagonal_system() {
        let matrix = CsrMatrix::from_rows(
            3,
            3,
            vec![
                vec![(0, 2.0), (1, -1.0)],
                vec![(0, -1.0), (1, 2.0), (2, -1.0)],
                vec![(1, -1.0), (2, 2.0)],
            ],
        )
        .expect("matrix");
        let mut precond = CsrJacobiPreconditioner::from_matrix(&matrix).expect("jacobi");
        let mut op = matrix;
        let b = [1.0, 0.0, 0.0];
        let mut x = [0.0; 3];
        let report = PcgSolver::new(PcgConfig {
            max_iters: 20,
            tolerance: 1.0e-12,
        })
        .expect("solver")
        .solve(&mut op, &mut precond, &b, &mut x)
        .expect("solve");
        assert!(report.converged);
        assert!(report.iterations <= 5);
    }
}
