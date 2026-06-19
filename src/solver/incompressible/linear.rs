use crate::core::Real;
use crate::discretization::{
    IncompressibleMomentumPredictorSystem, IncompressiblePressureCorrectionSystem,
};
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::linalg::{
    CsrJacobiPreconditioner, CsrMatrix, CsrMatrixView, GmresConfig, GmresSolver,
    IdentityPreconditioner, PcgConfig, PcgSolver,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompressiblePressureLinearSolverKind {
    Gmres,
    Pcg,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressiblePressureLinearSolverConfig {
    pub kind: IncompressiblePressureLinearSolverKind,
    pub max_iters: usize,
    pub tolerance: Real,
    pub gmres_restart: usize,
}

impl IncompressiblePressureLinearSolverConfig {
    #[must_use]
    pub fn gmres_config(self) -> GmresConfig {
        GmresConfig {
            restart: self.gmres_restart,
            max_iters: self.max_iters,
            tolerance: self.tolerance,
        }
    }

    #[must_use]
    pub fn pcg_config(self) -> PcgConfig {
        PcgConfig {
            max_iters: self.max_iters,
            tolerance: self.tolerance,
        }
    }
}

impl Default for IncompressiblePressureLinearSolverConfig {
    fn default() -> Self {
        Self {
            kind: IncompressiblePressureLinearSolverKind::Pcg,
            max_iters: 500,
            tolerance: 1.0e-10,
            gmres_restart: 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IncompressibleLinearSolverConfig {
    pub momentum: GmresConfig,
    pub pressure: IncompressiblePressureLinearSolverConfig,
}

pub(crate) struct PressureCorrectionSolveDiagnostic {
    pub(crate) converged: bool,
    pub(crate) iterations: usize,
    pub(crate) residual_norm: Real,
    pub(crate) max_abs_correction: Real,
    pub(crate) correction: Vec<Real>,
}

pub(crate) struct MomentumPredictorSolveDiagnostic {
    pub(crate) converged: bool,
    pub(crate) iterations: usize,
    pub(crate) residual_norm: Real,
    pub(crate) max_abs_equation_residual: Real,
    pub(crate) max_abs_velocity_delta: Real,
    pub(crate) predicted_fields: IncompressibleFields,
}

pub(crate) fn solve_pressure_correction(
    system: &IncompressiblePressureCorrectionSystem,
    config: IncompressiblePressureLinearSolverConfig,
) -> Result<PressureCorrectionSolveDiagnostic> {
    let n = system.matrix.nrows();
    let mut matrix = CsrMatrixView::new(&system.matrix);
    let mut pressure_correction = vec![0.0; n];
    let (converged, iterations, residual_norm) = match config.kind {
        IncompressiblePressureLinearSolverKind::Gmres => {
            let mut preconditioner = IdentityPreconditioner::new(n);
            let solver = GmresSolver::new(config.gmres_config())?;
            let report = solver.solve(
                &mut matrix,
                &mut preconditioner,
                &system.rhs,
                &mut pressure_correction,
            )?;
            (report.converged, report.iterations, report.residual_norm)
        }
        IncompressiblePressureLinearSolverKind::Pcg => {
            let mut preconditioner = CsrJacobiPreconditioner::from_matrix(&system.matrix)?;
            let solver = PcgSolver::new(config.pcg_config())?;
            let report = solver.solve(
                &mut matrix,
                &mut preconditioner,
                &system.rhs,
                &mut pressure_correction,
            )?;
            (report.converged, report.iterations, report.residual_norm)
        }
    };
    let max_abs_correction = pressure_correction
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    Ok(PressureCorrectionSolveDiagnostic {
        converged,
        iterations,
        residual_norm,
        max_abs_correction,
        correction: pressure_correction,
    })
}

pub(crate) fn solve_momentum_predictor(
    system: &IncompressibleMomentumPredictorSystem,
    fields: &IncompressibleFields,
    gmres_config: GmresConfig,
) -> Result<MomentumPredictorSolveDiagnostic> {
    let u = solve_momentum_component(system, &system.rhs_x, gmres_config)?;
    let v = solve_momentum_component(system, &system.rhs_y, gmres_config)?;
    let w = solve_momentum_component(system, &system.rhs_z, gmres_config)?;
    let max_abs_equation_residual =
        max_linear_system_residual(&system.matrix, &u.solution, &system.rhs_x)?
            .max(max_linear_system_residual(
                &system.matrix,
                &v.solution,
                &system.rhs_y,
            )?)
            .max(max_linear_system_residual(
                &system.matrix,
                &w.solution,
                &system.rhs_z,
            )?);
    let max_abs_velocity_delta = max_velocity_delta(fields, &u.solution, &v.solution, &w.solution);
    let predicted_fields = IncompressibleFields {
        pressure: fields.pressure.clone(),
        velocity_x: ScalarField::from_values(u.solution)?,
        velocity_y: ScalarField::from_values(v.solution)?,
        velocity_z: ScalarField::from_values(w.solution)?,
    };
    Ok(MomentumPredictorSolveDiagnostic {
        converged: u.converged && v.converged && w.converged,
        iterations: u.iterations.max(v.iterations).max(w.iterations),
        residual_norm: u.residual_norm.max(v.residual_norm).max(w.residual_norm),
        max_abs_equation_residual,
        max_abs_velocity_delta,
        predicted_fields,
    })
}

struct MomentumComponentSolve {
    solution: Vec<Real>,
    converged: bool,
    iterations: usize,
    residual_norm: Real,
}

fn solve_momentum_component(
    system: &IncompressibleMomentumPredictorSystem,
    rhs: &[Real],
    gmres_config: GmresConfig,
) -> Result<MomentumComponentSolve> {
    let n = system.matrix.nrows();
    let mut matrix = CsrMatrixView::new(&system.matrix);
    let mut preconditioner = IdentityPreconditioner::new(n);
    let solver = GmresSolver::new(gmres_config)?;
    let mut solution = vec![0.0; n];
    let report = solver.solve(&mut matrix, &mut preconditioner, rhs, &mut solution)?;
    Ok(MomentumComponentSolve {
        solution,
        converged: report.converged,
        iterations: report.iterations,
        residual_norm: report.residual_norm,
    })
}

fn max_linear_system_residual(matrix: &CsrMatrix, solution: &[Real], rhs: &[Real]) -> Result<Real> {
    if solution.len() != matrix.ncols() || rhs.len() != matrix.nrows() {
        return Err(AsimuError::Linalg(
            "线性系统残差向量长度与矩阵尺寸不一致".to_string(),
        ));
    }
    let mut max_residual: Real = 0.0;
    for (row, rhs_value) in rhs.iter().enumerate().take(matrix.nrows()) {
        let ax = matrix
            .row_entries(row)
            .map(|(col, value)| value * solution[col])
            .sum::<Real>();
        max_residual = max_residual.max((ax - rhs_value).abs());
    }
    Ok(max_residual)
}

fn max_velocity_delta(fields: &IncompressibleFields, u: &[Real], v: &[Real], w: &[Real]) -> Real {
    let mut max_delta: Real = 0.0;
    for idx in 0..fields.velocity_x.len() {
        max_delta = max_delta.max((u[idx] - fields.velocity_x.values()[idx]).abs());
        max_delta = max_delta.max((v[idx] - fields.velocity_y.values()[idx]).abs());
        max_delta = max_delta.max((w[idx] - fields.velocity_z.values()[idx]).abs());
    }
    max_delta
}
