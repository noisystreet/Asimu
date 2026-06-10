//! 不可压缩 SIMPLEC 求解编排。

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::{
    IncompressibleMomentumPredictorConfig, IncompressiblePressureCorrectionConfig,
    assemble_incompressible_momentum_predictor_with_boundary_3d,
    assemble_incompressible_pressure_correction_3d, compute_incompressible_divergence_3d,
    compute_incompressible_rhie_chow_divergence_3d,
};
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::linalg::{CsrMatrix, GmresConfig, GmresSolver, IdentityPreconditioner};
use crate::mesh::StructuredMesh3d;

const SIMPLEC_DIVERGENCE_LIMIT: Real = 1.0e50;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressibleLinearSolverConfig {
    pub momentum: GmresConfig,
    pub pressure: GmresConfig,
}

impl Default for IncompressibleLinearSolverConfig {
    fn default() -> Self {
        Self {
            momentum: GmresConfig::default(),
            pressure: GmresConfig {
                restart: 64,
                max_iters: 500,
                tolerance: 1.0e-10,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IncompressibleSimplecConfig<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub density: Real,
    pub kinematic_viscosity: Real,
    pub body_force: [Real; 3],
    pub velocity_under_relaxation: Real,
    pub pressure_under_relaxation: Real,
    pub pseudo_time_step: Real,
    pub boundary: &'a BoundarySet,
    pub max_iterations: usize,
    pub tolerance: Option<Real>,
    pub linear_solvers: IncompressibleLinearSolverConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleSimplecDiagnostic {
    pub max_abs_divergence: Real,
    pub max_abs_predicted_divergence: Real,
    pub max_abs_corrected_divergence: Real,
    pub pressure_system_rows: usize,
    pub pressure_system_nnz: usize,
    pub pressure_solve_converged: bool,
    pub pressure_solve_iterations: usize,
    pub pressure_solve_residual: Real,
    pub max_abs_pressure_correction: Real,
    pub momentum_system_rows: usize,
    pub momentum_system_nnz: usize,
    pub max_momentum_d_coefficient: Real,
    pub momentum_solve_converged: bool,
    pub momentum_solve_iterations: usize,
    pub momentum_solve_residual: Real,
    pub max_abs_momentum_equation_residual: Real,
    pub max_abs_predicted_velocity_delta: Real,
    pub max_abs_corrected_velocity_delta: Real,
    pub simplec_iterations: usize,
    pub simplec_converged: bool,
    pub simplec_final_residual: Real,
    pub simplec_final_momentum_residual: Real,
    pub simplec_residual_history: Vec<Real>,
    pub simplec_momentum_residual_history: Vec<Real>,
    pub corrected_fields: IncompressibleFields,
}

pub fn run_incompressible_simplec(
    initial_fields: &IncompressibleFields,
    config: IncompressibleSimplecConfig<'_>,
) -> Result<IncompressibleSimplecDiagnostic> {
    let mut current_fields = initial_fields.clone();
    let max_iterations = config.max_iterations.max(1);
    let mut history = Vec::with_capacity(max_iterations);
    let mut momentum_history = Vec::with_capacity(max_iterations);
    let mut last = None;
    for _ in 0..max_iterations {
        let mut diagnostic = assemble_simplec_step(&current_fields, &config)?;
        let residual = diagnostic.max_abs_corrected_divergence;
        let momentum_residual = diagnostic.max_abs_momentum_equation_residual;
        let velocity_delta = diagnostic.max_abs_corrected_velocity_delta;
        validate_simplec_step(residual, momentum_residual, velocity_delta)?;
        history.push(residual);
        momentum_history.push(momentum_residual);
        current_fields = diagnostic.corrected_fields.clone();
        let converged = config
            .tolerance
            .map(|tolerance| {
                residual <= tolerance
                    && momentum_residual <= tolerance
                    && velocity_delta <= tolerance
            })
            .unwrap_or(false);
        diagnostic.simplec_iterations = history.len();
        diagnostic.simplec_converged = converged;
        diagnostic.simplec_final_residual = residual;
        diagnostic.simplec_final_momentum_residual = momentum_residual;
        diagnostic.simplec_residual_history = history.clone();
        diagnostic.simplec_momentum_residual_history = momentum_history.clone();
        if converged {
            return Ok(diagnostic);
        }
        last = Some(diagnostic);
    }
    let mut diagnostic =
        last.ok_or_else(|| AsimuError::Solver("SIMPLEC 至少需要一次外层迭代".to_string()))?;
    diagnostic.simplec_converged = config
        .tolerance
        .map(|tolerance| {
            diagnostic.simplec_final_residual <= tolerance
                && diagnostic.simplec_final_momentum_residual <= tolerance
                && diagnostic.max_abs_corrected_velocity_delta <= tolerance
        })
        .unwrap_or(false);
    Ok(diagnostic)
}

fn validate_simplec_step(
    residual: Real,
    momentum_residual: Real,
    velocity_delta: Real,
) -> Result<()> {
    if !residual.is_finite() || !momentum_residual.is_finite() || !velocity_delta.is_finite() {
        return Err(AsimuError::Solver("SIMPLEC 残差出现非有限值".to_string()));
    }
    if residual > SIMPLEC_DIVERGENCE_LIMIT
        || momentum_residual > SIMPLEC_DIVERGENCE_LIMIT
        || velocity_delta > SIMPLEC_DIVERGENCE_LIMIT
    {
        return Err(AsimuError::Solver(format!(
            "SIMPLEC 发散：continuity={residual:.4e}, momentum={momentum_residual:.4e}, velocity_delta={velocity_delta:.4e}"
        )));
    }
    Ok(())
}

fn assemble_simplec_step(
    fields: &IncompressibleFields,
    config: &IncompressibleSimplecConfig<'_>,
) -> Result<IncompressibleSimplecDiagnostic> {
    let mesh = config.mesh;
    let divergence = compute_incompressible_divergence_3d(mesh, fields)?;
    let max_abs_divergence = divergence
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    let momentum_system = assemble_incompressible_momentum_predictor_with_boundary_3d(
        mesh,
        fields,
        config.boundary,
        IncompressibleMomentumPredictorConfig::new(
            config.kinematic_viscosity,
            config.pseudo_time_step,
        )?
        .with_body_force(config.body_force)?
        .with_velocity_under_relaxation(config.velocity_under_relaxation)?,
    )?;
    let max_momentum_d_coefficient = momentum_system
        .d_coefficient
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    let momentum_solution =
        solve_momentum_predictor(&momentum_system, fields, config.linear_solvers.momentum)?;
    let predicted_divergence = compute_incompressible_rhie_chow_divergence_3d(
        mesh,
        &momentum_solution.predicted_fields,
        &momentum_system.d_coefficient,
        config.boundary,
    )?;
    let max_abs_predicted_divergence = predicted_divergence
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    let system = assemble_incompressible_pressure_correction_3d(
        mesh,
        &predicted_divergence,
        &momentum_system.d_coefficient,
        config.boundary,
        IncompressiblePressureCorrectionConfig::new(config.density, 0, 0.0)?,
    )?;
    let pressure_solution = solve_pressure_correction(&system, config.linear_solvers.pressure)?;
    let max_abs_corrected_divergence = max_pressure_correction_continuity_residual(
        &system.matrix,
        &system.rhs,
        &pressure_solution.correction,
        config.density,
    )?;
    let corrected_fields = corrected_incompressible_fields(
        mesh,
        fields,
        &momentum_solution.predicted_fields,
        &pressure_solution.correction,
        momentum_system.d_coefficient.values(),
        config.pressure_under_relaxation,
        config.boundary.has_periodic_pair("i_min", "i_max"),
    )?;
    let max_abs_corrected_velocity_delta = max_velocity_delta(
        fields,
        corrected_fields.velocity_x.values(),
        corrected_fields.velocity_y.values(),
        corrected_fields.velocity_z.values(),
    );
    Ok(IncompressibleSimplecDiagnostic {
        max_abs_divergence,
        max_abs_predicted_divergence,
        max_abs_corrected_divergence,
        pressure_system_rows: system.matrix.nrows(),
        pressure_system_nnz: system.matrix.values().len(),
        pressure_solve_converged: pressure_solution.converged,
        pressure_solve_iterations: pressure_solution.iterations,
        pressure_solve_residual: pressure_solution.residual_norm,
        max_abs_pressure_correction: pressure_solution.max_abs_correction,
        momentum_system_rows: momentum_system.matrix.nrows(),
        momentum_system_nnz: momentum_system.matrix.values().len(),
        max_momentum_d_coefficient,
        momentum_solve_converged: momentum_solution.converged,
        momentum_solve_iterations: momentum_solution.iterations,
        momentum_solve_residual: momentum_solution.residual_norm,
        max_abs_momentum_equation_residual: momentum_solution.max_abs_equation_residual,
        max_abs_predicted_velocity_delta: momentum_solution.max_abs_velocity_delta,
        max_abs_corrected_velocity_delta,
        simplec_iterations: 0,
        simplec_converged: false,
        simplec_final_residual: max_abs_corrected_divergence,
        simplec_final_momentum_residual: momentum_solution.max_abs_equation_residual,
        simplec_residual_history: Vec::new(),
        simplec_momentum_residual_history: Vec::new(),
        corrected_fields,
    })
}

struct PressureCorrectionSolveDiagnostic {
    converged: bool,
    iterations: usize,
    residual_norm: Real,
    max_abs_correction: Real,
    correction: Vec<Real>,
}

struct MomentumPredictorSolveDiagnostic {
    converged: bool,
    iterations: usize,
    residual_norm: Real,
    max_abs_equation_residual: Real,
    max_abs_velocity_delta: Real,
    predicted_fields: IncompressibleFields,
}

fn solve_pressure_correction(
    system: &crate::discretization::IncompressiblePressureCorrectionSystem,
    gmres_config: GmresConfig,
) -> Result<PressureCorrectionSolveDiagnostic> {
    let n = system.matrix.nrows();
    let mut matrix = system.matrix.clone();
    let preconditioner = IdentityPreconditioner::new(n);
    let solver = GmresSolver::new(gmres_config)?;
    let mut pressure_correction = vec![0.0; n];
    let report = solver.solve(
        &mut matrix,
        &preconditioner,
        &system.rhs,
        &mut pressure_correction,
    )?;
    let max_abs_correction = pressure_correction
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    Ok(PressureCorrectionSolveDiagnostic {
        converged: report.converged,
        iterations: report.iterations,
        residual_norm: report.residual_norm,
        max_abs_correction,
        correction: pressure_correction,
    })
}

fn solve_momentum_predictor(
    system: &crate::discretization::IncompressibleMomentumPredictorSystem,
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
    system: &crate::discretization::IncompressibleMomentumPredictorSystem,
    rhs: &[Real],
    gmres_config: GmresConfig,
) -> Result<MomentumComponentSolve> {
    let n = system.matrix.nrows();
    let mut matrix = system.matrix.clone();
    let preconditioner = IdentityPreconditioner::new(n);
    let solver = GmresSolver::new(gmres_config)?;
    let mut solution = vec![0.0; n];
    let report = solver.solve(&mut matrix, &preconditioner, rhs, &mut solution)?;
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

fn max_pressure_correction_continuity_residual(
    matrix: &CsrMatrix,
    rhs: &[Real],
    correction: &[Real],
    density: Real,
) -> Result<Real> {
    if density <= 0.0 {
        return Err(AsimuError::Linalg(
            "压力校正连续性残差要求正密度".to_string(),
        ));
    }
    if correction.len() != matrix.ncols() || rhs.len() != matrix.nrows() {
        return Err(AsimuError::Linalg(
            "压力校正连续性残差向量长度与矩阵尺寸不一致".to_string(),
        ));
    }
    let mut max_residual: Real = 0.0;
    for (row, rhs_value) in rhs.iter().enumerate().take(matrix.nrows()) {
        if is_identity_constraint_row(matrix, row) {
            continue;
        }
        let ax = matrix
            .row_entries(row)
            .map(|(col, value)| value * correction[col])
            .sum::<Real>();
        max_residual = max_residual.max(((rhs_value - ax) / density).abs());
    }
    Ok(max_residual)
}

fn is_identity_constraint_row(matrix: &CsrMatrix, row: usize) -> bool {
    let mut entries = matrix.row_entries(row);
    let Some((col, value)) = entries.next() else {
        return false;
    };
    entries.next().is_none() && col == row && (value - 1.0).abs() <= Real::EPSILON
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

fn corrected_incompressible_fields(
    mesh: &StructuredMesh3d,
    current: &IncompressibleFields,
    predicted: &IncompressibleFields,
    pressure_correction: &[Real],
    d_coefficient: &[Real],
    pressure_under_relaxation: Real,
    periodic_x: bool,
) -> Result<IncompressibleFields> {
    let n = mesh.num_cells();
    if pressure_correction.len() != n || d_coefficient.len() != n {
        return Err(AsimuError::Field(
            "不可压缩修正场长度与网格单元数不一致".to_string(),
        ));
    }
    let spacing = CartesianSpacing::from_mesh(mesh)?;
    let mut pressure = Vec::with_capacity(n);
    let mut velocity_x = Vec::with_capacity(n);
    let mut velocity_y = Vec::with_capacity(n);
    let mut velocity_z = Vec::with_capacity(n);
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let cell = mesh.cell_index(i, j, k);
                let grad = pressure_correction_gradient(
                    mesh,
                    pressure_correction,
                    i,
                    j,
                    k,
                    spacing,
                    periodic_x,
                );
                let d = d_coefficient[cell];
                pressure.push(
                    current.pressure.values()[cell]
                        + pressure_under_relaxation * pressure_correction[cell],
                );
                velocity_x.push(predicted.velocity_x.values()[cell] - d * grad[0]);
                velocity_y.push(predicted.velocity_y.values()[cell] - d * grad[1]);
                velocity_z.push(predicted.velocity_z.values()[cell] - d * grad[2]);
            }
        }
    }
    Ok(IncompressibleFields {
        pressure: ScalarField::from_values(pressure)?,
        velocity_x: ScalarField::from_values(velocity_x)?,
        velocity_y: ScalarField::from_values(velocity_y)?,
        velocity_z: ScalarField::from_values(velocity_z)?,
    })
}

#[derive(Debug, Clone, Copy)]
struct CartesianSpacing {
    dx: Real,
    dy: Real,
    dz: Real,
}

impl CartesianSpacing {
    fn from_mesh(mesh: &StructuredMesh3d) -> Result<Self> {
        let dx = mesh.node_x(1, 0, 0) - mesh.node_x(0, 0, 0);
        let dy = mesh.node_y(0, 1, 0) - mesh.node_y(0, 0, 0);
        let dz = mesh.node_z(0, 0, 1) - mesh.node_z(0, 0, 0);
        if dx.abs() <= Real::EPSILON || dy.abs() <= Real::EPSILON || dz.abs() <= Real::EPSILON {
            return Err(AsimuError::Mesh(
                "不可压缩修正场要求正的 Cartesian 网格间距".to_string(),
            ));
        }
        Ok(Self {
            dx: dx.abs(),
            dy: dy.abs(),
            dz: dz.abs(),
        })
    }
}

fn pressure_correction_gradient(
    mesh: &StructuredMesh3d,
    pressure_correction: &[Real],
    i: usize,
    j: usize,
    k: usize,
    spacing: CartesianSpacing,
    periodic_x: bool,
) -> [Real; 3] {
    [
        (cell_value(
            mesh,
            pressure_correction,
            east_with_periodic(i, mesh.nx, periodic_x),
            j,
            k,
        ) - cell_value(
            mesh,
            pressure_correction,
            west_with_periodic(i, mesh.nx, periodic_x),
            j,
            k,
        )) / (2.0 * spacing.dx),
        (cell_value(mesh, pressure_correction, i, north(j, mesh.ny), k)
            - cell_value(mesh, pressure_correction, i, south(j), k))
            / (2.0 * spacing.dy),
        (cell_value(mesh, pressure_correction, i, j, top(k, mesh.nz))
            - cell_value(mesh, pressure_correction, i, j, bottom(k)))
            / (2.0 * spacing.dz),
    ]
}

fn cell_value(mesh: &StructuredMesh3d, values: &[Real], i: usize, j: usize, k: usize) -> Real {
    values[mesh.cell_index(i, j, k)]
}

fn west(i: usize) -> usize {
    i.saturating_sub(1)
}

fn east(i: usize, nx: usize) -> usize {
    (i + 1).min(nx - 1)
}

fn west_with_periodic(i: usize, nx: usize, periodic_x: bool) -> usize {
    if periodic_x && i == 0 {
        nx - 1
    } else {
        west(i)
    }
}

fn east_with_periodic(i: usize, nx: usize, periodic_x: bool) -> usize {
    if periodic_x && i + 1 == nx {
        0
    } else {
        east(i, nx)
    }
}

fn south(j: usize) -> usize {
    j.saturating_sub(1)
}

fn north(j: usize, ny: usize) -> usize {
    (j + 1).min(ny - 1)
}

fn bottom(k: usize) -> usize {
    k.saturating_sub(1)
}

fn top(k: usize, nz: usize) -> usize {
    (k + 1).min(nz - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplec_step_validation_rejects_divergence() {
        let err = validate_simplec_step(1.0e60, 1.0, 1.0).expect_err("divergence");
        assert!(err.to_string().contains("SIMPLEC 发散"));
    }

    #[test]
    fn simplec_step_validation_rejects_non_finite_values() {
        let err = validate_simplec_step(1.0, Real::INFINITY, 1.0).expect_err("non-finite");
        assert!(err.to_string().contains("非有限值"));
    }
}
