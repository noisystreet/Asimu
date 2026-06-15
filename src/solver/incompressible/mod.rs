mod coupling;
mod diagnostics;
mod linear;
mod pressure_reference;
mod projection;
use crate::boundary::BoundarySet;
use crate::core::{Real, elapsed_ms};
use crate::discretization::{
    IncompressibleFaceFluxField, IncompressibleMomentumPredictorConfig,
    IncompressiblePressureCorrectionConfig, RhieChowVelocityCorrectionConfig,
    apply_incompressible_boundary_conditions_3d,
    assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d,
    assemble_incompressible_pressure_correction_3d, compute_incompressible_face_flux_divergence_3d,
    corrected_incompressible_fields_rhie_chow_3d, incompressible_pressure_correction_dirichlet,
};
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::linalg::CsrMatrix;
use crate::mesh::StructuredMesh3d;
pub use diagnostics::IncompressiblePressureVelocityAlgorithm;
use diagnostics::{
    PressureCouplingLog, SimplecConvergenceCheck, SimplecStepLog, SimplecStepTiming,
    log_simplec_step, max_abs_active_face_flux_divergence, max_abs_field_divergence,
    max_abs_scalar_field, max_velocity_delta_by_region, simplec_converged, validate_simplec_step,
};
use linear::{
    PressureCorrectionSolveDiagnostic, solve_momentum_predictor, solve_pressure_correction,
};
use pressure_reference::volume_weighted_pressure_mean;
use std::time::Instant;

pub use linear::{
    IncompressibleLinearSolverConfig, IncompressiblePressureLinearSolverConfig,
    IncompressiblePressureLinearSolverKind,
};
pub use projection::{
    IncompressibleProjectionConfig, IncompressibleProjectionMode, IncompressibleProjectionStats,
    project_incompressible_fields_divergence_free_3d,
    project_incompressible_fields_divergence_free_with_d_3d,
    reconcile_rhie_chow_pressure_with_fixed_velocity_3d,
};

#[derive(Debug, Clone)]
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
    pub convergence_window: usize,
    pub snapshot_interval: Option<usize>,
    pub linear_solvers: IncompressibleLinearSolverConfig,
    /// 物理时间步进：每个外层步推进 \(t+\Delta t\)，不因 coupling 收敛提前退出。
    pub transient_mode: bool,
    /// 首步（或 IC 后）用于动量对流的 Rhie-Chow 面通量；缺省则回退 cell 插值。
    pub initial_face_flux: Option<IncompressibleFaceFluxField>,
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

pub struct IncompressiblePressureVelocityStepView<'a> {
    pub info: &'a IncompressiblePressureVelocityStepInfo,
    pub history: &'a [IncompressiblePressureVelocityStepInfo],
    pub fields: &'a IncompressibleFields,
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
    run_incompressible_pressure_velocity_with_observer(initial_fields, config, |_| Ok(()))
}
pub fn run_incompressible_pressure_velocity_with_observer(
    initial_fields: &IncompressibleFields,
    config: IncompressiblePressureVelocityConfig<'_>,
    mut observe_step: impl FnMut(IncompressiblePressureVelocityStepView<'_>) -> Result<()>,
) -> Result<IncompressiblePressureVelocityDiagnostic> {
    let mut current_fields = initial_fields.clone();
    let max_iterations = config.max_iterations.max(1);
    let mut history = Vec::with_capacity(max_iterations);
    let mut momentum_history = Vec::with_capacity(max_iterations);
    let mut velocity_history = Vec::with_capacity(max_iterations);
    let mut corrector_residual_history = Vec::new();
    let mut corrector_max_correction_history = Vec::new();
    let mut step_history = Vec::with_capacity(max_iterations);
    let mut snapshots = Vec::new();
    let mut last = None;
    let mut current_face_flux = config.initial_face_flux.clone();
    for step in 0..max_iterations {
        let step_no = step + 1;
        let (mut diagnostic, timing, next_face_flux) =
            assemble_simplec_step(&current_fields, current_face_flux.as_ref(), &config)?;
        let residual = diagnostic.max_abs_corrected_field_divergence_after_boundary;
        let momentum_residual = diagnostic.max_abs_momentum_equation_residual;
        let velocity_delta = diagnostic.max_abs_corrected_velocity_delta_interior;
        validate_simplec_step(residual, momentum_residual, velocity_delta)?;
        history.push(residual);
        momentum_history.push(momentum_residual);
        velocity_history.push(if config.require_velocity_convergence {
            velocity_delta
        } else {
            0.0
        });
        corrector_residual_history
            .extend_from_slice(&diagnostic.pressure_corrector_residual_history);
        corrector_max_correction_history
            .extend_from_slice(&diagnostic.pressure_corrector_max_correction_history);
        current_fields = diagnostic.corrected_fields.clone();
        current_face_flux = Some(next_face_flux);
        let converged = simplec_converged(SimplecConvergenceCheck {
            tolerance: config.tolerance,
            min_iterations: config.min_iterations,
            iterations: history.len(),
            residual_history: &history,
            momentum_history: &momentum_history,
            velocity_history: &velocity_history,
            convergence_window: config.convergence_window,
            linear_solvers_converged: diagnostic.pressure_solve_converged
                && diagnostic.momentum_solve_converged,
        });
        diagnostic.simplec_iterations = history.len();
        diagnostic.simplec_converged = converged;
        diagnostic.simplec_final_residual = residual;
        diagnostic.simplec_final_momentum_residual = momentum_residual;
        diagnostic.simplec_residual_history = history.clone();
        diagnostic.simplec_momentum_residual_history = momentum_history.clone();
        diagnostic.pressure_corrector_residual_history = corrector_residual_history.clone();
        diagnostic.pressure_corrector_max_correction_history =
            corrector_max_correction_history.clone();
        let step_info = incompressible_step_info(
            step_no as u64,
            &diagnostic,
            config.pseudo_time_step,
            converged,
        );
        step_history.push(step_info);
        if config
            .snapshot_interval
            .is_some_and(|value| value > 0 && step_no % value == 0)
        {
            snapshots.push(IncompressiblePressureVelocitySnapshot {
                step: step_no as u64,
                nondimensional_time: step_no as Real * config.pseudo_time_step,
                fields: diagnostic.corrected_fields.clone(),
            });
        }
        diagnostic.step_history = step_history.clone();
        diagnostic.snapshots = snapshots.clone();
        observe_step(IncompressiblePressureVelocityStepView {
            info: step_history.last().expect("step history just pushed"),
            history: &step_history,
            fields: &current_fields,
        })?;
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
        if converged && !config.transient_mode {
            return Ok(diagnostic);
        }
        last = Some(diagnostic);
    }
    let mut diagnostic =
        last.ok_or_else(|| AsimuError::Solver("SIMPLEC 至少需要一次外层迭代".to_string()))?;
    diagnostic.simplec_converged = simplec_converged(SimplecConvergenceCheck {
        tolerance: config.tolerance,
        min_iterations: config.min_iterations,
        iterations: diagnostic.simplec_iterations,
        residual_history: &history,
        momentum_history: &momentum_history,
        velocity_history: &velocity_history,
        convergence_window: config.convergence_window,
        linear_solvers_converged: diagnostic.pressure_solve_converged
            && diagnostic.momentum_solve_converged,
    });
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
    let predicted = coupling::build_predicted_rhie_chow_flux(
        &momentum_solution,
        &momentum_system.d_coefficient,
        config,
    )?;
    let coupling::PredictedRhieChowFlux {
        predicted_fields,
        mut face_flux,
        predicted_divergence,
        max_abs_predicted_divergence,
        pressure_reconciliation,
    } = predicted;
    timing.rhie_chow_ms = elapsed_ms(rhie_chow_start);
    let pressure_start = Instant::now();
    let mut pressure_step = solve_pressure_correction_step(
        &predicted_divergence,
        &momentum_system.d_coefficient,
        config,
    )?;
    timing.pressure_ms = elapsed_ms(pressure_start);
    let correct_start = Instant::now();
    let mut corrector_max_corrections = vec![pressure_step.solution.max_abs_correction];
    let mut accumulated_pressure_correction = pressure_reconciliation;
    for (total, increment) in accumulated_pressure_correction
        .iter_mut()
        .zip(pressure_step.solution.correction.iter())
    {
        *total += *increment;
    }
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
        &mut face_flux,
    )?;
    let mut corrector_residuals = vec![corrected.max_abs_divergence_after_boundary];
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
    coupling::log_simplec_step_diagnostics(&pressure_step, &corrected, &velocity_delta);
    let diagnostic =
        coupling::build_incompressible_step_diagnostic(coupling::SimplecStepDiagnosticInput {
            config,
            max_abs_divergence,
            max_abs_predicted_divergence,
            max_momentum_d_coefficient,
            momentum_system: &momentum_system,
            momentum_solution: &momentum_solution,
            pressure_step: &pressure_step,
            corrected: &corrected,
            velocity_delta: &velocity_delta,
            corrector_residuals,
            corrector_max_corrections,
        });
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
        corrected = build_corrected_fields_with_diagnostics(
            current_fields,
            predicted_fields,
            accumulated_pressure_correction,
            momentum_system.d_coefficient.values(),
            config,
            face_flux,
        )?;
        residual_history.push(corrected.max_abs_divergence_after_boundary);
    }
    Ok((pressure_step, corrected))
}

pub(crate) struct CorrectedFieldsDiagnostic {
    fields: IncompressibleFields,
    max_abs_divergence_before_boundary: Real,
    max_abs_divergence_after_boundary: Real,
}

pub(crate) struct PressureCorrectionStepDiagnostic {
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
    let mut solution = if pressure_correction_rhs_satisfies_coupling_tolerance(
        predicted_divergence,
        config.mesh,
        &system.rhs,
        config.density,
        config.tolerance,
    ) {
        zero_pressure_correction_solution(&system)
    } else {
        solve_pressure_correction(&system, config.linear_solvers.pressure)?
    };
    normalize_closed_pressure_reference(&mut solution.correction, config);
    let max_abs_corrected_divergence = max_pressure_correction_continuity_residual(
        &system.matrix,
        &system.rhs,
        &solution.correction,
        config.mesh,
        config.density,
    )?;
    let max_abs_underrelaxed_corrected_divergence =
        max_scaled_pressure_correction_continuity_residual(
            &system.matrix,
            &system.rhs,
            &solution.correction,
            config.mesh,
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
    predicted_divergence: &ScalarField,
    mesh: &StructuredMesh3d,
    rhs: &[Real],
    density: Real,
    tolerance: Option<Real>,
) -> bool {
    tolerance.is_some_and(|tol| {
        max_abs_scalar_field(predicted_divergence) <= tol
            && max_abs_pressure_rhs_divergence(mesh, rhs, density)
                .is_ok_and(|residual| residual <= tol)
    })
}

fn zero_pressure_correction_solution(
    system: &crate::discretization::IncompressiblePressureCorrectionSystem,
) -> PressureCorrectionSolveDiagnostic {
    PressureCorrectionSolveDiagnostic {
        converged: true,
        iterations: 0,
        residual_norm: system
            .rhs
            .iter()
            .map(|value| value * value)
            .sum::<Real>()
            .sqrt(),
        max_abs_correction: 0.0,
        correction: vec![0.0; system.matrix.nrows()],
    }
}

fn normalize_closed_pressure_reference(
    pressure_correction: &mut [Real],
    config: &IncompressibleSimplecConfig<'_>,
) {
    if has_pressure_correction_dirichlet(config.boundary) || pressure_correction.is_empty() {
        return;
    }
    let reference = volume_weighted_pressure_mean(pressure_correction, config.mesh);
    for value in pressure_correction {
        *value -= reference;
    }
}

fn has_pressure_correction_dirichlet(boundary: &BoundarySet) -> bool {
    boundary
        .patches()
        .iter()
        .any(|patch| incompressible_pressure_correction_dirichlet(&patch.kind))
}

fn build_corrected_fields_with_diagnostics(
    current: &IncompressibleFields,
    predicted: &IncompressibleFields,
    pressure_correction: &[Real],
    d_coefficient: &[Real],
    config: &IncompressibleSimplecConfig<'_>,
    face_flux: &mut IncompressibleFaceFluxField,
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
    face_flux.refresh_boundary_net(mesh, &fields, config.boundary)?;
    let max_abs_divergence_after_boundary =
        max_abs_active_face_flux_divergence(mesh, face_flux, config.boundary)?;
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
    mesh: &StructuredMesh3d,
    density: Real,
) -> Result<Real> {
    max_scaled_pressure_correction_continuity_residual(matrix, rhs, correction, mesh, density, 1.0)
}

fn max_scaled_pressure_correction_continuity_residual(
    matrix: &CsrMatrix,
    rhs: &[Real],
    correction: &[Real],
    mesh: &StructuredMesh3d,
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
    if rhs.len() != mesh.num_cells() {
        return Err(AsimuError::Linalg(
            "压力校正连续性残差长度与网格单元数不一致".to_string(),
        ));
    }
    let mut max_residual: Real = 0.0;
    for (row, rhs_value) in rhs.iter().enumerate().take(matrix.nrows()) {
        if is_identity_constraint_row(matrix, row) {
            continue;
        }
        let (i, j, k) = cell_ijk(mesh, row);
        let volume = mesh.cell_metric(i, j, k).volume;
        let ax = matrix
            .row_entries(row)
            .map(|(col, value)| value * correction[col])
            .sum::<Real>();
        max_residual =
            max_residual.max(((rhs_value - correction_scale * ax) / (density * volume)).abs());
    }
    Ok(max_residual)
}

fn max_abs_pressure_rhs_divergence(
    mesh: &StructuredMesh3d,
    rhs: &[Real],
    density: Real,
) -> Result<Real> {
    if density <= 0.0 {
        return Err(AsimuError::Linalg(
            "压力校正 RHS 散度要求正密度".to_string(),
        ));
    }
    if rhs.len() != mesh.num_cells() {
        return Err(AsimuError::Linalg(
            "压力校正 RHS 长度与网格单元数不一致".to_string(),
        ));
    }
    let mut max_residual: Real = 0.0;
    for (cell, rhs_value) in rhs.iter().enumerate() {
        let (i, j, k) = cell_ijk(mesh, cell);
        let volume = mesh.cell_metric(i, j, k).volume;
        max_residual = max_residual.max((rhs_value / (density * volume)).abs());
    }
    Ok(max_residual)
}

fn cell_ijk(mesh: &StructuredMesh3d, cell: usize) -> (usize, usize, usize) {
    let cells_per_layer = mesh.nx * mesh.ny;
    let k = cell / cells_per_layer;
    let rem = cell % cells_per_layer;
    let j = rem / mesh.nx;
    let i = rem % mesh.nx;
    (i, j, k)
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

#[cfg(test)]
#[path = "tests_main.rs"]
mod tests;
