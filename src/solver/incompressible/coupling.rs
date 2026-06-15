//! 动量预测后与 Rhie-Chow 面通量相关的 coupling helper。

use crate::core::Real;
use crate::discretization::{
    IncompressibleFaceFluxField, apply_incompressible_boundary_conditions_3d,
};
use crate::error::Result;
use crate::field::{IncompressibleFields, ScalarField};
use tracing::debug;

use super::diagnostics::{self, max_abs_scalar_field, pressure_velocity_algorithm};
use super::linear::MomentumPredictorSolveDiagnostic;
use super::projection::{
    IncompressibleProjectionConfig, reconcile_rhie_chow_pressure_with_fixed_velocity_3d,
};
use super::{
    CorrectedFieldsDiagnostic, IncompressibleSimplecConfig, IncompressibleSimplecDiagnostic,
    PressureCorrectionStepDiagnostic,
};

pub(super) struct PredictedRhieChowFlux {
    pub predicted_fields: IncompressibleFields,
    pub face_flux: IncompressibleFaceFluxField,
    pub predicted_divergence: ScalarField,
    pub max_abs_predicted_divergence: Real,
    pub pressure_reconciliation: Vec<Real>,
}

pub(super) struct SimplecStepDiagnosticInput<'a> {
    pub config: &'a IncompressibleSimplecConfig<'a>,
    pub max_abs_divergence: Real,
    pub max_abs_predicted_divergence: Real,
    pub max_momentum_d_coefficient: Real,
    pub momentum_system: &'a crate::discretization::IncompressibleMomentumPredictorSystem,
    pub momentum_solution: &'a MomentumPredictorSolveDiagnostic,
    pub pressure_step: &'a PressureCorrectionStepDiagnostic,
    pub corrected: &'a CorrectedFieldsDiagnostic,
    pub velocity_delta: &'a diagnostics::VelocityDeltaByRegion,
    pub corrector_residuals: Vec<Real>,
    pub corrector_max_corrections: Vec<Real>,
}

pub(super) fn build_predicted_rhie_chow_flux(
    momentum_solution: &MomentumPredictorSolveDiagnostic,
    d_coefficient: &ScalarField,
    config: &IncompressibleSimplecConfig<'_>,
) -> Result<PredictedRhieChowFlux> {
    let mut predicted_fields = momentum_solution.predicted_fields.clone();
    apply_incompressible_boundary_conditions_3d(
        config.mesh,
        &mut predicted_fields,
        config.boundary,
    )?;
    let pressure_reconciliation = if config.transient_mode {
        reconcile_rhie_chow_pressure_with_fixed_velocity_3d(
            &mut predicted_fields,
            d_coefficient,
            IncompressibleProjectionConfig::rhie_chow_pressure_only(
                config.mesh,
                config.boundary,
                config.density,
                config.linear_solvers.pressure,
                12,
                1.0e-8,
            ),
        )?
    } else {
        vec![0.0; config.mesh.num_cells()]
    };
    let face_flux = IncompressibleFaceFluxField::from_rhie_chow(
        config.mesh,
        &predicted_fields,
        d_coefficient,
        config.boundary,
    )?;
    let predicted_divergence = face_flux.divergence(config.mesh)?;
    Ok(PredictedRhieChowFlux {
        predicted_fields,
        face_flux,
        max_abs_predicted_divergence: max_abs_scalar_field(&predicted_divergence),
        predicted_divergence,
        pressure_reconciliation,
    })
}

pub(super) fn log_simplec_step_diagnostics(
    pressure_step: &PressureCorrectionStepDiagnostic,
    corrected: &CorrectedFieldsDiagnostic,
    velocity_delta: &diagnostics::VelocityDeltaByRegion,
) {
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
}

pub(super) fn build_incompressible_step_diagnostic(
    input: SimplecStepDiagnosticInput<'_>,
) -> IncompressibleSimplecDiagnostic {
    let SimplecStepDiagnosticInput {
        config,
        max_abs_divergence,
        max_abs_predicted_divergence,
        max_momentum_d_coefficient,
        momentum_system,
        momentum_solution,
        pressure_step,
        corrected,
        velocity_delta,
        corrector_residuals,
        corrector_max_corrections,
    } = input;
    IncompressibleSimplecDiagnostic {
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
        simplec_final_residual: corrected.max_abs_divergence_after_boundary,
        simplec_final_momentum_residual: momentum_solution.max_abs_equation_residual,
        simplec_residual_history: Vec::new(),
        simplec_momentum_residual_history: Vec::new(),
        pressure_corrector_residual_history: corrector_residuals,
        pressure_corrector_max_correction_history: corrector_max_corrections,
        step_history: Vec::new(),
        snapshots: Vec::new(),
        corrected_fields: corrected.fields.clone(),
    }
}
