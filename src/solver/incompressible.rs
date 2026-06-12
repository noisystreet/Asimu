//! 不可压缩 SIMPLEC 求解编排。

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::{
    IncompressibleFaceFluxField, IncompressibleMomentumPredictorConfig,
    IncompressiblePressureCorrectionConfig, RhieChowVelocityCorrectionConfig,
    apply_incompressible_boundary_conditions_3d,
    assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d,
    assemble_incompressible_pressure_correction_3d, compute_incompressible_face_flux_divergence_3d,
    corrected_incompressible_fields_rhie_chow_3d,
};
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::linalg::CsrMatrix;
use crate::mesh::StructuredMesh3d;
pub use crate::solver::incompressible_diagnostics::IncompressiblePressureVelocityAlgorithm;
use crate::solver::incompressible_diagnostics::{
    PressureCouplingLog, SimplecStepLog, SimplecStepTiming, elapsed_ms, log_simplec_step,
    max_velocity_delta_by_region, pressure_velocity_algorithm, simplec_converged,
    validate_simplec_step,
};
use crate::solver::incompressible_linear::{
    MomentumPredictorSolveDiagnostic, PressureCorrectionSolveDiagnostic, solve_momentum_predictor,
    solve_pressure_correction,
};
use std::time::Instant;
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
    pub require_velocity_convergence: bool,
    pub snapshot_interval: Option<usize>,
    pub linear_solvers: IncompressibleLinearSolverConfig,
}

pub type IncompressibleSimplecConfig<'a> = IncompressiblePressureVelocityConfig<'a>;

#[derive(Debug, Clone, PartialEq)]
pub struct IncompressiblePressureVelocityStepInfo {
    pub step: u64,
    pub nondimensional_time: Real,
    pub continuity: Real,
    pub momentum_residual: Real,
    pub velocity_delta_interior: Real,
    pub face_flux_divergence: Real,
    pub pressure_equation_residual: Real,
    pub pressure_solve_converged: bool,
    pub pressure_solve_iterations: usize,
    pub pressure_solve_residual: Real,
    pub momentum_solve_converged: bool,
    pub momentum_solve_iterations: usize,
    pub momentum_solve_residual: Real,
    pub converged: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IncompressiblePressureVelocitySnapshot {
    pub step: u64,
    pub nondimensional_time: Real,
    pub fields: IncompressibleFields,
}

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
    pub step_history: Vec<IncompressiblePressureVelocityStepInfo>,
    pub snapshots: Vec<IncompressiblePressureVelocitySnapshot>,
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
    let mut step_history = Vec::with_capacity(max_iterations);
    let mut snapshots = Vec::new();
    let mut last = None;
    let mut current_face_flux: Option<IncompressibleFaceFluxField> = None;
    for step in 0..max_iterations {
        let step_no = step + 1;
        let (mut diagnostic, timing, next_face_flux) =
            assemble_simplec_step(&current_fields, current_face_flux.as_ref(), &config)?;
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
        current_face_flux = Some(next_face_flux);
        let converged = simplec_converged(
            config.tolerance,
            config.min_iterations,
            history.len(),
            residual,
            momentum_residual,
            velocity_convergence_metric(velocity_delta, config.require_velocity_convergence),
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
        step_history.push(incompressible_step_info(
            step_no as u64,
            &diagnostic,
            config.pseudo_time_step,
            converged,
        ));
        if snapshot_due(step_no, config.snapshot_interval) {
            snapshots.push(IncompressiblePressureVelocitySnapshot {
                step: step_no as u64,
                nondimensional_time: step_no as Real * config.pseudo_time_step,
                fields: diagnostic.corrected_fields.clone(),
            });
        }
        diagnostic.step_history = step_history.clone();
        diagnostic.snapshots = snapshots.clone();
        let is_final = converged || step_no == max_iterations;
        log_simplec_step(SimplecStepLog {
            step: step_no,
            algorithm: diagnostic.algorithm,
            continuity: residual,
            momentum: momentum_residual,
            velocity_delta,
            pressure_iters: diagnostic.pressure_solve_iterations,
            momentum_iters: diagnostic.momentum_solve_iterations,
            pressure_converged: diagnostic.pressure_solve_converged,
            momentum_converged: diagnostic.momentum_solve_converged,
            coupling: PressureCouplingLog {
                predicted_divergence: diagnostic.max_abs_predicted_divergence,
                pressure_equation_residual: diagnostic.max_abs_corrected_divergence,
                face_flux_divergence: diagnostic.max_abs_corrected_field_divergence_after_boundary,
                rhs_active_sum: diagnostic.pressure_correction_rhs_active_sum,
            },
            timing,
            converged,
            is_final,
        });
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
        velocity_convergence_metric(
            diagnostic.max_abs_corrected_velocity_delta_interior,
            config.require_velocity_convergence,
        ),
    );
    diagnostic.pressure_corrector_residual_history = corrector_residual_history;
    diagnostic.pressure_corrector_max_correction_history = corrector_max_correction_history;
    diagnostic.step_history = step_history;
    diagnostic.snapshots = snapshots;
    Ok(diagnostic)
}

fn incompressible_step_info(
    step: u64,
    diagnostic: &IncompressiblePressureVelocityDiagnostic,
    pseudo_time_step: Real,
    converged: bool,
) -> IncompressiblePressureVelocityStepInfo {
    IncompressiblePressureVelocityStepInfo {
        step,
        nondimensional_time: step as Real * pseudo_time_step,
        continuity: diagnostic.simplec_final_residual,
        momentum_residual: diagnostic.simplec_final_momentum_residual,
        velocity_delta_interior: diagnostic.max_abs_corrected_velocity_delta_interior,
        face_flux_divergence: diagnostic.max_abs_corrected_field_divergence_after_boundary,
        pressure_equation_residual: diagnostic.max_abs_corrected_divergence,
        pressure_solve_converged: diagnostic.pressure_solve_converged,
        pressure_solve_iterations: diagnostic.pressure_solve_iterations,
        pressure_solve_residual: diagnostic.pressure_solve_residual,
        momentum_solve_converged: diagnostic.momentum_solve_converged,
        momentum_solve_iterations: diagnostic.momentum_solve_iterations,
        momentum_solve_residual: diagnostic.momentum_solve_residual,
        converged,
    }
}

fn snapshot_due(step: usize, interval: Option<usize>) -> bool {
    interval.is_some_and(|value| value > 0 && step % value == 0)
}

fn velocity_convergence_metric(velocity_delta: Real, required: bool) -> Real {
    if required { velocity_delta } else { 0.0 }
}

pub fn run_incompressible_simplec(
    initial_fields: &IncompressibleFields,
    config: IncompressibleSimplecConfig<'_>,
) -> Result<IncompressibleSimplecDiagnostic> {
    run_incompressible_pressure_velocity(initial_fields, config)
}

fn assemble_simplec_step(
    fields: &IncompressibleFields,
    previous_face_flux: Option<&IncompressibleFaceFluxField>,
    config: &IncompressibleSimplecConfig<'_>,
) -> Result<(
    IncompressibleSimplecDiagnostic,
    SimplecStepTiming,
    IncompressibleFaceFluxField,
)> {
    let step_start = Instant::now();
    let mut timing = SimplecStepTiming::default();
    let mesh = config.mesh;
    let divergence_start = Instant::now();
    let divergence = compute_incompressible_face_flux_divergence_3d(mesh, fields, config.boundary)?;
    timing.divergence_ms = elapsed_ms(divergence_start);
    let max_abs_divergence = max_abs_scalar_field(&divergence);
    let momentum_assemble_start = Instant::now();
    let momentum_system = assemble_momentum_predictor(fields, previous_face_flux, config)?;
    timing.momentum_assemble_ms = elapsed_ms(momentum_assemble_start);
    let max_momentum_d_coefficient = momentum_system
        .d_coefficient
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    let momentum_solve_start = Instant::now();
    let momentum_solution =
        solve_momentum_predictor(&momentum_system, fields, config.linear_solvers.momentum)?;
    timing.momentum_solve_ms = elapsed_ms(momentum_solve_start);
    let rhie_chow_start = Instant::now();
    let predicted_fields = predicted_fields_with_boundary(&momentum_solution, config)?;
    let mut face_flux = IncompressibleFaceFluxField::from_rhie_chow(
        mesh,
        &predicted_fields,
        &momentum_system.d_coefficient,
        config.boundary,
    )?;
    let predicted_divergence = face_flux.divergence(mesh)?;
    timing.rhie_chow_ms = elapsed_ms(rhie_chow_start);
    let max_abs_predicted_divergence = max_abs_scalar_field(&predicted_divergence);
    let pressure_start = Instant::now();
    let mut pressure_step = solve_pressure_correction_step(
        &predicted_divergence,
        &momentum_system.d_coefficient,
        config,
    )?;
    timing.pressure_ms = elapsed_ms(pressure_start);
    let correct_start = Instant::now();
    let mut corrector_residuals = vec![pressure_step.max_abs_underrelaxed_corrected_divergence];
    let mut corrector_max_corrections = vec![pressure_step.solution.max_abs_correction];
    let mut accumulated_pressure_correction = pressure_step.solution.correction.clone();
    face_flux.apply_pressure_correction(
        mesh,
        momentum_system.d_coefficient.values(),
        &pressure_step.solution.correction,
        1.0,
    )?;
    let mut corrected = build_corrected_fields_with_diagnostics(
        fields,
        &predicted_fields,
        &accumulated_pressure_correction,
        momentum_system.d_coefficient.values(),
        config,
        &face_flux,
    )?;
    (pressure_step, corrected) = apply_additional_pressure_correctors(
        pressure_step,
        corrected,
        &momentum_system,
        config,
        PressureCorrectorLoopState {
            current_fields: fields,
            predicted_fields: &predicted_fields,
            accumulated_pressure_correction: &mut accumulated_pressure_correction,
            face_flux: &mut face_flux,
            residual_history: &mut corrector_residuals,
            max_correction_history: &mut corrector_max_corrections,
        },
    )?;
    let velocity_delta = max_velocity_delta_by_region(
        mesh,
        config.boundary,
        fields,
        corrected.fields.velocity_x.values(),
        corrected.fields.velocity_y.values(),
        corrected.fields.velocity_z.values(),
    )?;
    timing.correct_ms = elapsed_ms(correct_start);
    timing.step_total_ms = elapsed_ms(step_start);
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
    let diagnostic = IncompressibleSimplecDiagnostic {
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
        step_history: Vec::new(),
        snapshots: Vec::new(),
        corrected_fields: corrected.fields,
    };
    Ok((diagnostic, timing, face_flux))
}

fn assemble_momentum_predictor(
    fields: &IncompressibleFields,
    previous_face_flux: Option<&IncompressibleFaceFluxField>,
    config: &IncompressibleSimplecConfig<'_>,
) -> Result<crate::discretization::IncompressibleMomentumPredictorSystem> {
    let predictor_config = IncompressibleMomentumPredictorConfig::new(
        config.kinematic_viscosity,
        config.pseudo_time_step,
    )?
    .with_body_force(config.body_force)?
    .with_velocity_under_relaxation(config.velocity_under_relaxation)?
    .with_convection_scheme(config.convection_scheme);
    assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d(
        config.mesh,
        fields,
        config.boundary,
        predictor_config,
        previous_face_flux,
    )
}

struct PressureCorrectorLoopState<'a> {
    current_fields: &'a IncompressibleFields,
    predicted_fields: &'a IncompressibleFields,
    accumulated_pressure_correction: &'a mut [Real],
    face_flux: &'a mut IncompressibleFaceFluxField,
    residual_history: &'a mut Vec<Real>,
    max_correction_history: &'a mut Vec<Real>,
}

fn apply_additional_pressure_correctors(
    mut pressure_step: PressureCorrectionStepDiagnostic,
    mut corrected: CorrectedFieldsDiagnostic,
    momentum_system: &crate::discretization::IncompressibleMomentumPredictorSystem,
    config: &IncompressibleSimplecConfig<'_>,
    loop_state: PressureCorrectorLoopState<'_>,
) -> Result<(PressureCorrectionStepDiagnostic, CorrectedFieldsDiagnostic)> {
    let PressureCorrectorLoopState {
        current_fields,
        predicted_fields,
        accumulated_pressure_correction,
        face_flux,
        residual_history,
        max_correction_history,
    } = loop_state;
    for _ in 1..config.pressure_correctors.max(1) {
        let divergence = face_flux.divergence(config.mesh)?;
        pressure_step =
            solve_pressure_correction_step(&divergence, &momentum_system.d_coefficient, config)?;
        max_correction_history.push(pressure_step.solution.max_abs_correction);
        if accumulated_pressure_correction.len() != pressure_step.solution.correction.len() {
            return Err(AsimuError::Field(
                "PISO 压力校正累积长度与求解结果不一致".to_string(),
            ));
        }
        for (total, increment) in accumulated_pressure_correction
            .iter_mut()
            .zip(pressure_step.solution.correction.iter())
        {
            *total += *increment;
        }
        face_flux.apply_pressure_correction(
            config.mesh,
            momentum_system.d_coefficient.values(),
            &pressure_step.solution.correction,
            1.0,
        )?;
        residual_history.push(pressure_step.max_abs_underrelaxed_corrected_divergence);
        corrected = build_corrected_fields_with_diagnostics(
            current_fields,
            predicted_fields,
            accumulated_pressure_correction,
            momentum_system.d_coefficient.values(),
            config,
            face_flux,
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
    let solution = if pressure_correction_rhs_satisfies_coupling_tolerance(
        &system.rhs,
        config.density,
        config.tolerance,
    ) {
        zero_pressure_correction_solution(&system)
    } else {
        solve_pressure_correction(&system, config.linear_solvers.pressure)?
    };
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

fn pressure_correction_rhs_satisfies_coupling_tolerance(
    rhs: &[Real],
    density: Real,
    tolerance: Option<Real>,
) -> bool {
    tolerance.is_some_and(|tol| max_abs_slice(rhs) / density <= tol)
}

fn zero_pressure_correction_solution(
    system: &crate::discretization::IncompressiblePressureCorrectionSystem,
) -> PressureCorrectionSolveDiagnostic {
    PressureCorrectionSolveDiagnostic {
        converged: true,
        iterations: 0,
        residual_norm: l2_norm(&system.rhs),
        max_abs_correction: 0.0,
        correction: vec![0.0; system.matrix.nrows()],
    }
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
    face_flux: &IncompressibleFaceFluxField,
) -> Result<CorrectedFieldsDiagnostic> {
    let mesh = config.mesh;
    let mut fields =
        corrected_incompressible_fields_rhie_chow_3d(RhieChowVelocityCorrectionConfig {
            mesh,
            current,
            predicted,
            pressure_correction,
            d_coefficient,
            pressure_under_relaxation: config.pressure_under_relaxation,
            boundary: config.boundary,
            periodic_x: config.boundary.has_periodic_pair("i_min", "i_max"),
        })?;
    let max_abs_divergence_before_boundary =
        max_abs_field_divergence(mesh, &fields, config.boundary)?;
    apply_incompressible_boundary_conditions_3d(mesh, &mut fields, config.boundary)?;
    let max_abs_divergence_after_boundary = max_abs_scalar_field(&face_flux.divergence(mesh)?);
    Ok(CorrectedFieldsDiagnostic {
        fields,
        max_abs_divergence_before_boundary,
        max_abs_divergence_after_boundary,
    })
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
    max_abs_slice(field.values())
}

fn max_abs_slice(values: &[Real]) -> Real {
    values
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()))
}

fn l2_norm(values: &[Real]) -> Real {
    values
        .iter()
        .map(|value| value * value)
        .sum::<Real>()
        .sqrt()
}

#[cfg(test)]
#[path = "incompressible_tests.rs"]
mod tests;
