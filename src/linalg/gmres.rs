use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::{LinearOperator, Preconditioner, ensure_vector_len};

/// GMRES 参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GmresConfig {
    pub restart: usize,
    pub max_iters: usize,
    pub tolerance: Real,
}

impl Default for GmresConfig {
    fn default() -> Self {
        Self {
            restart: 30,
            max_iters: 100,
            tolerance: 1.0e-8,
        }
    }
}

/// GMRES 收敛报告。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GmresReport {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: Real,
}

pub struct GmresSolver {
    config: GmresConfig,
}

impl GmresSolver {
    pub fn new(config: GmresConfig) -> Result<Self> {
        if config.restart == 0
            || config.max_iters == 0
            || !config.tolerance.is_finite()
            || config.tolerance <= 0.0
        {
            return Err(AsimuError::Linalg(
                "GMRES restart/max_iters/tolerance 参数无效".to_string(),
            ));
        }
        Ok(Self { config })
    }

    pub fn solve<A, M>(
        &self,
        op: &mut A,
        precond: &M,
        b: &[Real],
        x: &mut [Real],
    ) -> Result<GmresReport>
    where
        A: LinearOperator,
        M: Preconditioner,
    {
        let n = op.dimension();
        ensure_vector_len(b, n, "gmres rhs")?;
        ensure_vector_len(x, n, "gmres solution")?;
        if precond.dimension() != n {
            return Err(AsimuError::Linalg("GMRES 预条件器尺寸不一致".to_string()));
        }
        let mut work = GmresWork::new(n, self.config.restart);
        let mut total_iters = 0usize;
        let mut residual = compute_preconditioned_residual(op, precond, b, x, &mut work)?;
        if residual <= self.config.tolerance {
            return Ok(GmresReport {
                converged: true,
                iterations: 0,
                residual_norm: residual,
            });
        }
        while total_iters < self.config.max_iters {
            let cycle_iters = self.restart_cycle(
                op,
                precond,
                x,
                &mut work,
                residual,
                self.config.max_iters - total_iters,
            )?;
            total_iters += cycle_iters;
            residual = compute_preconditioned_residual(op, precond, b, x, &mut work)?;
            if residual <= self.config.tolerance || cycle_iters == 0 {
                return Ok(GmresReport {
                    converged: residual <= self.config.tolerance,
                    iterations: total_iters,
                    residual_norm: residual,
                });
            }
        }
        Ok(GmresReport {
            converged: false,
            iterations: total_iters,
            residual_norm: residual,
        })
    }

    fn restart_cycle<A, M>(
        &self,
        op: &mut A,
        precond: &M,
        x: &mut [Real],
        work: &mut GmresWork,
        beta: Real,
        remaining_iters: usize,
    ) -> Result<usize>
    where
        A: LinearOperator,
        M: Preconditioner,
    {
        let m = self.config.restart.min(remaining_iters);
        if m == 0 || beta <= Real::EPSILON {
            return Ok(0);
        }
        for (vi, ri) in work.v[0].iter_mut().zip(work.z.iter()) {
            *vi = *ri / beta;
        }
        work.g.fill(0.0);
        work.g[0] = beta;
        let mut used = 0usize;
        for j in 0..m {
            apply_preconditioned_operator(op, precond, &work.v[j], &mut work.av, &mut work.w)?;
            for i in 0..=j {
                work.h[i][j] = dot(&work.w, &work.v[i]);
                axpy(&mut work.w, &work.v[i], -work.h[i][j]);
            }
            work.h[j + 1][j] = norm(&work.w);
            if work.h[j + 1][j] > Real::EPSILON {
                for (dst, src) in work.v[j + 1].iter_mut().zip(work.w.iter()) {
                    *dst = *src / work.h[j + 1][j];
                }
            }
            apply_existing_givens(work, j);
            let (cs, sn) = givens(work.h[j][j], work.h[j + 1][j]);
            work.cs[j] = cs;
            work.sn[j] = sn;
            apply_givens_to_column(work, j, cs, sn);
            apply_givens_to_rhs(work, j, cs, sn);
            used = j + 1;
            if work.g[j + 1].abs() <= self.config.tolerance {
                break;
            }
        }
        update_solution(x, work, used);
        Ok(used)
    }
}

struct GmresWork {
    v: Vec<Vec<Real>>,
    h: Vec<Vec<Real>>,
    cs: Vec<Real>,
    sn: Vec<Real>,
    g: Vec<Real>,
    r: Vec<Real>,
    z: Vec<Real>,
    w: Vec<Real>,
    av: Vec<Real>,
}

impl GmresWork {
    fn new(n: usize, restart: usize) -> Self {
        Self {
            v: vec![vec![0.0; n]; restart + 1],
            h: vec![vec![0.0; restart]; restart + 1],
            cs: vec![0.0; restart],
            sn: vec![0.0; restart],
            g: vec![0.0; restart + 1],
            r: vec![0.0; n],
            z: vec![0.0; n],
            w: vec![0.0; n],
            av: vec![0.0; n],
        }
    }
}

fn dot(a: &[Real], b: &[Real]) -> Real {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[Real]) -> Real {
    dot(a, a).sqrt()
}

fn axpy(dst: &mut [Real], src: &[Real], scale: Real) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += scale * s;
    }
}

fn compute_preconditioned_residual<A, M>(
    op: &mut A,
    precond: &M,
    b: &[Real],
    x: &[Real],
    work: &mut GmresWork,
) -> Result<Real>
where
    A: LinearOperator,
    M: Preconditioner,
{
    op.apply(x, &mut work.av)?;
    for ((ri, bi), axi) in work.r.iter_mut().zip(b.iter()).zip(work.av.iter()) {
        *ri = *bi - *axi;
    }
    precond.apply(&work.r, &mut work.z)?;
    Ok(norm(&work.z))
}

fn apply_preconditioned_operator<A, M>(
    op: &mut A,
    precond: &M,
    x: &[Real],
    ax: &mut [Real],
    out: &mut [Real],
) -> Result<()>
where
    A: LinearOperator,
    M: Preconditioner,
{
    op.apply(x, ax)?;
    precond.apply(ax, out)
}

fn givens(a: Real, b: Real) -> (Real, Real) {
    if b.abs() <= Real::EPSILON {
        (1.0, 0.0)
    } else {
        let r = (a * a + b * b).sqrt();
        (a / r, b / r)
    }
}

fn apply_existing_givens(work: &mut GmresWork, col: usize) {
    for i in 0..col {
        let temp = work.cs[i] * work.h[i][col] + work.sn[i] * work.h[i + 1][col];
        work.h[i + 1][col] = -work.sn[i] * work.h[i][col] + work.cs[i] * work.h[i + 1][col];
        work.h[i][col] = temp;
    }
}

fn apply_givens_to_column(work: &mut GmresWork, col: usize, cs: Real, sn: Real) {
    let temp = cs * work.h[col][col] + sn * work.h[col + 1][col];
    work.h[col + 1][col] = 0.0;
    work.h[col][col] = temp;
}

fn apply_givens_to_rhs(work: &mut GmresWork, row: usize, cs: Real, sn: Real) {
    let temp = cs * work.g[row] + sn * work.g[row + 1];
    work.g[row + 1] = -sn * work.g[row] + cs * work.g[row + 1];
    work.g[row] = temp;
}

fn update_solution(x: &mut [Real], work: &GmresWork, used: usize) {
    if used == 0 {
        return;
    }
    let mut y = vec![0.0; used];
    for i in (0..used).rev() {
        let mut rhs = work.g[i];
        for (j, yj) in y.iter().enumerate().take(used).skip(i + 1) {
            rhs -= work.h[i][j] * yj;
        }
        let diag = work.h[i][i];
        if diag.abs() <= Real::EPSILON {
            y[i] = 0.0;
            continue;
        }
        y[i] = rhs / diag;
    }
    for (basis, coeff) in work.v.iter().take(used).zip(y.iter()) {
        axpy(x, basis, *coeff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{CsrMatrix, IdentityPreconditioner, Ilu0Preconditioner};

    #[test]
    fn gmres_solves_small_csr_system() {
        let mut matrix = CsrMatrix::from_rows(
            2,
            2,
            vec![vec![(0, 4.0), (1, 1.0)], vec![(0, 2.0), (1, 3.0)]],
        )
        .expect("csr");
        let b = [1.0, 1.0];
        let mut x = [0.0, 0.0];
        let solver = GmresSolver::new(GmresConfig {
            restart: 2,
            max_iters: 4,
            tolerance: 1.0e-12,
        })
        .expect("gmres");
        let report = solver
            .solve(&mut matrix, &IdentityPreconditioner::new(2), &b, &mut x)
            .expect("solve");
        assert!(report.converged, "{report:?}");
        assert!((x[0] - 0.2).abs() < 1.0e-10);
        assert!((x[1] - 0.2).abs() < 1.0e-10);
    }

    #[test]
    fn gmres_uses_ilu0_preconditioner() {
        let mut matrix = CsrMatrix::from_rows(
            3,
            3,
            vec![
                vec![(0, 2.0), (1, -1.0)],
                vec![(0, -1.0), (1, 2.0), (2, -1.0)],
                vec![(1, -1.0), (2, 2.0)],
            ],
        )
        .expect("csr");
        let ilu = Ilu0Preconditioner::factor(&matrix).expect("ilu");
        let mut x = [0.0; 3];
        let report = GmresSolver::new(GmresConfig {
            restart: 3,
            max_iters: 3,
            tolerance: 1.0e-12,
        })
        .expect("gmres")
        .solve(&mut matrix, &ilu, &[1.0, 0.0, 1.0], &mut x)
        .expect("solve");
        assert!(report.converged, "{report:?}");
        assert!((x[0] - 1.0).abs() < 1.0e-10);
        assert!((x[1] - 1.0).abs() < 1.0e-10);
        assert!((x[2] - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn gmres_accepts_compatible_hessenberg_breakdown() {
        let mut matrix = CsrMatrix::from_rows(2, 2, vec![vec![(0, 1.0)], vec![(0, 0.0), (1, 0.0)]])
            .expect("csr");
        let mut x = [0.0; 2];
        let report = GmresSolver::new(GmresConfig {
            restart: 2,
            max_iters: 2,
            tolerance: 1.0e-12,
        })
        .expect("gmres")
        .solve(
            &mut matrix,
            &IdentityPreconditioner::new(2),
            &[1.0, 0.0],
            &mut x,
        )
        .expect("solve");

        assert!(report.converged, "{report:?}");
        assert!((x[0] - 1.0).abs() < 1.0e-10);
        assert!(x[1].abs() < 1.0e-10);
    }
}
