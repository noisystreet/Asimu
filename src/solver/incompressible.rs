//! 不可压缩 SIMPLEC 求解编排。

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::{
    IncompressibleMomentumPredictorConfig, IncompressiblePressureCorrectionConfig,
    apply_incompressible_boundary_conditions_3d,
    assemble_incompressible_momentum_predictor_with_boundary_3d,
    assemble_incompressible_pressure_correction_3d, compute_incompressible_face_flux_divergence_3d,
    compute_incompressible_rhie_chow_divergence_3d,
};
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::linalg::{
    CsrJacobiPreconditioner, CsrMatrix, CsrMatrixView, GmresConfig, GmresSolver,
    IdentityPreconditioner, PcgSolver,
};
use crate::mesh::StructuredMesh3d;
pub use crate::solver::incompressible_diagnostics::IncompressiblePressureVelocityAlgorithm;
use crate::solver::incompressible_diagnostics::{
    max_velocity_delta_by_region, pressure_velocity_algorithm, simplec_converged,
    validate_simplec_step,
};
use tracing::debug;

pub use crate::solver::incompressible_linear::{
    IncompressibleLinearSolverConfig, IncompressiblePressureLinearSolverConfig,
    IncompressiblePressureLinearSolverKind,
};

#[derive(Debug, Clone, Copy)]
pub struct IncompressiblePressureVelocityConfig<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub density: Real,
    pub kinematic_viscosity: Real,
    pub body_force: [Real; 3],
    pub velocity_under_relaxation: Real,
    pub pressure_under_relaxation: Real,
    pub pseudo_time_step: Real,
    pub convection_scheme: crate::discretization::IncompressibleConvectionScheme,
    pub pressure_correctors: usize,
    pub boundary: &'a BoundarySet,
    pub max_iterations: usize,
    pub min_iterations: usize,
    pub tolerance: Option<Real>,
    pub linear_solvers: IncompressibleLinearSolverConfig,
}

pub type IncompressibleSimplecConfig<'a> = IncompressiblePressureVelocityConfig<'a>;

#[derive(Debug, Clone, PartialEq)]
pub struct IncompressiblePressureVelocityDiagnostic {
    pub algorithm: IncompressiblePressureVelocityAlgorithm,
    pub pressure_correctors: usize,
    pub max_abs_divergence: Real,
    pub max_abs_predicted_divergence: Real,
    pub max_abs_corrected_divergence: Real,
    pub max_abs_underrelaxed_corrected_divergence: Real,
    pub max_abs_corrected_field_divergence_before_boundary: Real,
    pub max_abs_corrected_field_divergence_after_boundary: Real,
    pub pressure_correction_rhs_active_sum: Real,
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
    pub max_abs_corrected_velocity_delta_interior: Real,
    pub max_abs_corrected_velocity_delta_boundary: Real,
    pub simplec_iterations: usize,
    pub simplec_converged: bool,
    pub simplec_final_residual: Real,
    pub simplec_final_momentum_residual: Real,
    pub simplec_residual_history: Vec<Real>,
    pub simplec_momentum_residual_history: Vec<Real>,
    pub pressure_corrector_residual_history: Vec<Real>,
    pub pressure_corrector_max_correction_history: Vec<Real>,
    pub corrected_fields: IncompressibleFields,
}

pub type IncompressibleSimplecDiagnostic = IncompressiblePressureVelocityDiagnostic;

pub fn run_incompressible_pressure_velocity(
    initial_fields: &IncompressibleFields,
    config: IncompressiblePressureVelocityConfig<'_>,
) -> Result<IncompressiblePressureVelocityDiagnostic> {
    let mut current_fields = initial_fields.clone();
    let max_iterations = config.max_iterations.max(1);
    let mut history = Vec::with_capacity(max_iterations);
    let mut momentum_history = Vec::with_capacity(max_iterations);
    let mut corrector_residual_history = Vec::new();
    let mut corrector_max_correction_history = Vec::new();
    let mut last = None;
    for _ in 0..max_iterations {
        let mut diagnostic = assemble_simplec_step(&current_fields, &config)?;
        let residual = diagnostic.max_abs_underrelaxed_corrected_divergence;
        let momentum_residual = diagnostic.max_abs_momentum_equation_residual;
        let velocity_delta = diagnostic.max_abs_corrected_velocity_delta_interior;
        validate_simplec_step(residual, momentum_residual, velocity_delta)?;
        history.push(residual);
        momentum_history.push(momentum_residual);
        corrector_residual_history
            .extend_from_slice(&diagnostic.pressure_corrector_residual_history);
        corrector_max_correction_history
            .extend_from_slice(&diagnostic.pressure_corrector_max_correction_history);
        current_fields = diagnostic.corrected_fields.clone();
        let converged = simplec_converged(
            config.tolerance,
            config.min_iterations,
            history.len(),
            residual,
            momentum_residual,
            velocity_delta,
        );
        diagnostic.simplec_iterations = history.len();
        diagnostic.simplec_converged = converged;
        diagnostic.simplec_final_residual = residual;
        diagnostic.simplec_final_momentum_residual = momentum_residual;
        diagnostic.simplec_residual_history = history.clone();
        diagnostic.simplec_momentum_residual_history = momentum_history.clone();
        diagnostic.pressure_corrector_residual_history = corrector_residual_history.clone();
        diagnostic.pressure_corrector_max_correction_history =
            corrector_max_correction_history.clone();
        if converged {
            return Ok(diagnostic);
        }
        last = Some(diagnostic);
    }
    let mut diagnostic =
        last.ok_or_else(|| AsimuError::Solver("SIMPLEC 至少需要一次外层迭代".to_string()))?;
    diagnostic.simplec_converged = simplec_converged(
        config.tolerance,
        config.min_iterations,
        diagnostic.simplec_iterations,
        diagnostic.simplec_final_residual,
        diagnostic.simplec_final_momentum_residual,
        diagnostic.max_abs_corrected_velocity_delta_interior,
    );
    diagnostic.pressure_corrector_residual_history = corrector_residual_history;
    diagnostic.pressure_corrector_max_correction_history = corrector_max_correction_history;
    Ok(diagnostic)
}

pub fn run_incompressible_simplec(
    initial_fields: &IncompressibleFields,
    config: IncompressibleSimplecConfig<'_>,
) -> Result<IncompressibleSimplecDiagnostic> {
    run_incompressible_pressure_velocity(initial_fields, config)
}

fn assemble_simplec_step(
    fields: &IncompressibleFields,
    config: &IncompressibleSimplecConfig<'_>,
) -> Result<IncompressibleSimplecDiagnostic> {
    let mesh = config.mesh;
    let divergence = compute_incompressible_face_flux_divergence_3d(mesh, fields, config.boundary)?;
    let max_abs_divergence = max_abs_scalar_field(&divergence);
    let momentum_system = assemble_incompressible_momentum_predictor_with_boundary_3d(
        mesh,
        fields,
        config.boundary,
        IncompressibleMomentumPredictorConfig::new(
            config.kinematic_viscosity,
            config.pseudo_time_step,
        )?
        .with_body_force(config.body_force)?
        .with_velocity_under_relaxation(config.velocity_under_relaxation)?
        .with_convection_scheme(config.convection_scheme),
    )?;
    let max_momentum_d_coefficient = momentum_system
        .d_coefficient
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    let momentum_solution =
        solve_momentum_predictor(&momentum_system, fields, config.linear_solvers.momentum)?;
    let predicted_fields = predicted_fields_with_boundary(&momentum_solution, config)?;
    let predicted_divergence = compute_incompressible_rhie_chow_divergence_3d(
        mesh,
        &predicted_fields,
        &momentum_system.d_coefficient,
        config.boundary,
    )?;
    let max_abs_predicted_divergence = max_abs_scalar_field(&predicted_divergence);
    let mut pressure_step = solve_pressure_correction_step(
        &predicted_divergence,
        &momentum_system.d_coefficient,
        config,
    )?;
    let mut corrector_residuals = vec![pressure_step.max_abs_underrelaxed_corrected_divergence];
    let mut corrector_max_corrections = vec![pressure_step.solution.max_abs_correction];
    let mut corrected = build_corrected_fields_with_diagnostics(
        fields,
        &predicted_fields,
        &pressure_step.solution.correction,
        momentum_system.d_coefficient.values(),
        config,
    )?;
    (pressure_step, corrected) = apply_additional_pressure_correctors(
        pressure_step,
        corrected,
        &momentum_system,
        config,
        &mut corrector_residuals,
        &mut corrector_max_corrections,
    )?;
    let velocity_delta = max_velocity_delta_by_region(
        mesh,
        config.boundary,
        fields,
        corrected.fields.velocity_x.values(),
        corrected.fields.velocity_y.values(),
        corrected.fields.velocity_z.values(),
    )?;
    debug!(
        pressure_equation_residual = pressure_step.max_abs_corrected_divergence,
        underrelaxed_pressure_equation_residual =
            pressure_step.max_abs_underrelaxed_corrected_divergence,
        corrected_field_divergence_before_boundary = corrected.max_abs_divergence_before_boundary,
        corrected_field_divergence_after_boundary = corrected.max_abs_divergence_after_boundary,
        pressure_rhs_active_sum = pressure_step.rhs_active_sum,
        velocity_delta = velocity_delta.all,
        velocity_delta_interior = velocity_delta.interior,
        velocity_delta_boundary = velocity_delta.boundary,
        "SIMPLEC pressure-velocity diagnostic"
    );
    Ok(IncompressibleSimplecDiagnostic {
        algorithm: pressure_velocity_algorithm(config.pressure_correctors),
        pressure_correctors: config.pressure_correctors.max(1),
        max_abs_divergence,
        max_abs_predicted_divergence,
        max_abs_corrected_divergence: pressure_step.max_abs_corrected_divergence,
        max_abs_underrelaxed_corrected_divergence: pressure_step
            .max_abs_underrelaxed_corrected_divergence,
        max_abs_corrected_field_divergence_before_boundary: corrected
            .max_abs_divergence_before_boundary,
        max_abs_corrected_field_divergence_after_boundary: corrected
            .max_abs_divergence_after_boundary,
        pressure_correction_rhs_active_sum: pressure_step.rhs_active_sum,
        pressure_system_rows: pressure_step.system.matrix.nrows(),
        pressure_system_nnz: pressure_step.system.matrix.values().len(),
        pressure_solve_converged: pressure_step.solution.converged,
        pressure_solve_iterations: pressure_step.solution.iterations,
        pressure_solve_residual: pressure_step.solution.residual_norm,
        max_abs_pressure_correction: pressure_step.solution.max_abs_correction,
        momentum_system_rows: momentum_system.matrix.nrows(),
        momentum_system_nnz: momentum_system.matrix.values().len(),
        max_momentum_d_coefficient,
        momentum_solve_converged: momentum_solution.converged,
        momentum_solve_iterations: momentum_solution.iterations,
        momentum_solve_residual: momentum_solution.residual_norm,
        max_abs_momentum_equation_residual: momentum_solution.max_abs_equation_residual,
        max_abs_predicted_velocity_delta: momentum_solution.max_abs_velocity_delta,
        max_abs_corrected_velocity_delta: velocity_delta.all,
        max_abs_corrected_velocity_delta_interior: velocity_delta.interior,
        max_abs_corrected_velocity_delta_boundary: velocity_delta.boundary,
        simplec_iterations: 0,
        simplec_converged: false,
        simplec_final_residual: pressure_step.max_abs_underrelaxed_corrected_divergence,
        simplec_final_momentum_residual: momentum_solution.max_abs_equation_residual,
        simplec_residual_history: Vec::new(),
        simplec_momentum_residual_history: Vec::new(),
        pressure_corrector_residual_history: corrector_residuals,
        pressure_corrector_max_correction_history: corrector_max_corrections,
        corrected_fields: corrected.fields,
    })
}

fn apply_additional_pressure_correctors(
    mut pressure_step: PressureCorrectionStepDiagnostic,
    mut corrected: CorrectedFieldsDiagnostic,
    momentum_system: &crate::discretization::IncompressibleMomentumPredictorSystem,
    config: &IncompressibleSimplecConfig<'_>,
    residual_history: &mut Vec<Real>,
    max_correction_history: &mut Vec<Real>,
) -> Result<(PressureCorrectionStepDiagnostic, CorrectedFieldsDiagnostic)> {
    for _ in 1..config.pressure_correctors.max(1) {
        let divergence = compute_incompressible_rhie_chow_divergence_3d(
            config.mesh,
            &corrected.fields,
            &momentum_system.d_coefficient,
            config.boundary,
        )?;
        pressure_step =
            solve_pressure_correction_step(&divergence, &momentum_system.d_coefficient, config)?;
        residual_history.push(pressure_step.max_abs_underrelaxed_corrected_divergence);
        max_correction_history.push(pressure_step.solution.max_abs_correction);
        corrected = build_corrected_fields_with_diagnostics(
            &corrected.fields,
            &corrected.fields,
            &pressure_step.solution.correction,
            momentum_system.d_coefficient.values(),
            config,
        )?;
    }
    Ok((pressure_step, corrected))
}

struct CorrectedFieldsDiagnostic {
    fields: IncompressibleFields,
    max_abs_divergence_before_boundary: Real,
    max_abs_divergence_after_boundary: Real,
}

struct PressureCorrectionStepDiagnostic {
    system: crate::discretization::IncompressiblePressureCorrectionSystem,
    solution: PressureCorrectionSolveDiagnostic,
    max_abs_corrected_divergence: Real,
    max_abs_underrelaxed_corrected_divergence: Real,
    rhs_active_sum: Real,
}

fn solve_pressure_correction_step(
    predicted_divergence: &ScalarField,
    d_coefficient: &ScalarField,
    config: &IncompressibleSimplecConfig<'_>,
) -> Result<PressureCorrectionStepDiagnostic> {
    let system = assemble_incompressible_pressure_correction_3d(
        config.mesh,
        predicted_divergence,
        d_coefficient,
        config.boundary,
        IncompressiblePressureCorrectionConfig::new(config.density, 0, 0.0)?,
    )?;
    let solution = solve_pressure_correction(&system, config.linear_solvers.pressure)?;
    let max_abs_corrected_divergence = max_pressure_correction_continuity_residual(
        &system.matrix,
        &system.rhs,
        &solution.correction,
        config.density,
    )?;
    let max_abs_underrelaxed_corrected_divergence =
        max_scaled_pressure_correction_continuity_residual(
            &system.matrix,
            &system.rhs,
            &solution.correction,
            config.density,
            config.pressure_under_relaxation,
        )?;
    let rhs_active_sum = pressure_correction_active_rhs_sum(&system.matrix, &system.rhs)?;
    Ok(PressureCorrectionStepDiagnostic {
        system,
        solution,
        max_abs_corrected_divergence,
        max_abs_underrelaxed_corrected_divergence,
        rhs_active_sum,
    })
}

fn predicted_fields_with_boundary(
    momentum: &MomentumPredictorSolveDiagnostic,
    config: &IncompressibleSimplecConfig<'_>,
) -> Result<IncompressibleFields> {
    let mut fields = momentum.predicted_fields.clone();
    apply_incompressible_boundary_conditions_3d(config.mesh, &mut fields, config.boundary)?;
    Ok(fields)
}

fn build_corrected_fields_with_diagnostics(
    current: &IncompressibleFields,
    predicted: &IncompressibleFields,
    pressure_correction: &[Real],
    d_coefficient: &[Real],
    config: &IncompressibleSimplecConfig<'_>,
) -> Result<CorrectedFieldsDiagnostic> {
    let mesh = config.mesh;
    let mut fields = corrected_incompressible_fields(
        mesh,
        current,
        predicted,
        pressure_correction,
        d_coefficient,
        config.pressure_under_relaxation,
        config.boundary.has_periodic_pair("i_min", "i_max"),
    )?;
    let max_abs_divergence_before_boundary =
        max_abs_field_divergence(mesh, &fields, config.boundary)?;
    apply_incompressible_boundary_conditions_3d(mesh, &mut fields, config.boundary)?;
    let max_abs_divergence_after_boundary =
        max_abs_field_divergence(mesh, &fields, config.boundary)?;
    Ok(CorrectedFieldsDiagnostic {
        fields,
        max_abs_divergence_before_boundary,
        max_abs_divergence_after_boundary,
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
    config: IncompressiblePressureLinearSolverConfig,
) -> Result<PressureCorrectionSolveDiagnostic> {
    let n = system.matrix.nrows();
    let mut matrix = CsrMatrixView::new(&system.matrix);
    let mut pressure_correction = vec![0.0; n];
    let (converged, iterations, residual_norm) = match config.kind {
        IncompressiblePressureLinearSolverKind::Gmres => {
            let preconditioner = IdentityPreconditioner::new(n);
            let solver = GmresSolver::new(config.gmres_config())?;
            let report = solver.solve(
                &mut matrix,
                &preconditioner,
                &system.rhs,
                &mut pressure_correction,
            )?;
            (report.converged, report.iterations, report.residual_norm)
        }
        IncompressiblePressureLinearSolverKind::Pcg => {
            let preconditioner = CsrJacobiPreconditioner::from_matrix(&system.matrix)?;
            let solver = PcgSolver::new(config.pcg_config())?;
            let report = solver.solve(
                &mut matrix,
                &preconditioner,
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
    let mut matrix = CsrMatrixView::new(&system.matrix);
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
    max_scaled_pressure_correction_continuity_residual(matrix, rhs, correction, density, 1.0)
}

fn max_scaled_pressure_correction_continuity_residual(
    matrix: &CsrMatrix,
    rhs: &[Real],
    correction: &[Real],
    density: Real,
    correction_scale: Real,
) -> Result<Real> {
    if density <= 0.0 {
        return Err(AsimuError::Linalg(
            "压力校正连续性残差要求正密度".to_string(),
        ));
    }
    if !correction_scale.is_finite() {
        return Err(AsimuError::Linalg(
            "压力校正连续性残差缩放必须为有限值".to_string(),
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
        max_residual = max_residual.max(((rhs_value - correction_scale * ax) / density).abs());
    }
    Ok(max_residual)
}

fn pressure_correction_active_rhs_sum(matrix: &CsrMatrix, rhs: &[Real]) -> Result<Real> {
    if rhs.len() != matrix.nrows() {
        return Err(AsimuError::Linalg(
            "压力校正 RHS 长度与矩阵行数不一致".to_string(),
        ));
    }
    let mut sum = 0.0;
    for (row, value) in rhs.iter().enumerate().take(matrix.nrows()) {
        if !is_identity_constraint_row(matrix, row) {
            sum += *value;
        }
    }
    Ok(sum)
}

fn is_identity_constraint_row(matrix: &CsrMatrix, row: usize) -> bool {
    let mut entries = matrix.row_entries(row);
    let Some((col, value)) = entries.next() else {
        return false;
    };
    entries.next().is_none() && col == row && (value - 1.0).abs() <= Real::EPSILON
}

fn max_abs_field_divergence(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    boundary: &BoundarySet,
) -> Result<Real> {
    let divergence = compute_incompressible_face_flux_divergence_3d(mesh, fields, boundary)?;
    Ok(max_abs_scalar_field(&divergence))
}

fn max_abs_scalar_field(field: &ScalarField) -> Real {
    field
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()))
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
                velocity_x.push(
                    predicted.velocity_x.values()[cell] - pressure_under_relaxation * d * grad[0],
                );
                velocity_y.push(
                    predicted.velocity_y.values()[cell] - pressure_under_relaxation * d * grad[1],
                );
                velocity_z.push(
                    predicted.velocity_z.values()[cell] - pressure_under_relaxation * d * grad[2],
                );
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
#[path = "incompressible_tests.rs"]
mod tests;
