use tracing::info;

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::{Real, format_log_fixed4, format_log_sci4, log10_positive};
use crate::discretization::{
    compute_incompressible_face_flux_divergence_3d, incompressible_pressure_correction_dirichlet,
};
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::mesh::{BoundaryMesh, StructuredMesh3d};

const SIMPLEC_DIVERGENCE_LIMIT: Real = 1.0e50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompressiblePressureVelocityAlgorithm {
    Simplec,
    Piso,
}

impl IncompressiblePressureVelocityAlgorithm {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Simplec => "simplec",
            Self::Piso => "piso",
        }
    }
}

#[must_use]
pub(crate) fn pressure_velocity_algorithm(
    pressure_correctors: usize,
) -> IncompressiblePressureVelocityAlgorithm {
    if pressure_correctors.max(1) > 1 {
        IncompressiblePressureVelocityAlgorithm::Piso
    } else {
        IncompressiblePressureVelocityAlgorithm::Simplec
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SimplecConvergenceCheck<'a> {
    pub(crate) tolerance: Option<Real>,
    pub(crate) min_iterations: usize,
    pub(crate) iterations: usize,
    pub(crate) residual_history: &'a [Real],
    pub(crate) momentum_history: &'a [Real],
    pub(crate) velocity_history: &'a [Real],
    pub(crate) convergence_window: usize,
    pub(crate) linear_solvers_converged: bool,
}

#[must_use]
pub(crate) fn simplec_converged(check: SimplecConvergenceCheck<'_>) -> bool {
    if check.iterations < check.min_iterations || !check.linear_solvers_converged {
        return false;
    }
    let Some(tol) = check.tolerance else {
        return false;
    };
    let required_window = check.convergence_window.max(1);
    if check.residual_history.len() < required_window
        || check.momentum_history.len() < required_window
        || check.velocity_history.len() < required_window
    {
        return false;
    }
    check.residual_history[check.residual_history.len() - required_window..]
        .iter()
        .chain(&check.momentum_history[check.momentum_history.len() - required_window..])
        .chain(&check.velocity_history[check.velocity_history.len() - required_window..])
        .all(|value| *value <= tol)
}

pub(crate) fn validate_simplec_step(
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

/// SIMPLEC/PISO 外层步各阶段耗时（毫秒）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct SimplecStepTiming {
    pub(crate) divergence_ms: Real,
    pub(crate) momentum_assemble_ms: Real,
    pub(crate) momentum_solve_ms: Real,
    pub(crate) rhie_chow_ms: Real,
    pub(crate) pressure_ms: Real,
    pub(crate) correct_ms: Real,
    pub(crate) step_total_ms: Real,
}

/// 压力方程与 face-flux 散度耦合诊断（单步）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct PressureCouplingLog {
    pub predicted_divergence: Real,
    pub pressure_equation_residual: Real,
    pub face_flux_divergence: Real,
    pub rhs_active_sum: Real,
}

/// SIMPLEC/PISO 外层步 info 日志字段。
#[derive(Debug, Clone, Copy)]
pub(crate) struct SimplecStepLog {
    pub step: usize,
    pub algorithm: IncompressiblePressureVelocityAlgorithm,
    pub continuity: Real,
    pub momentum: Real,
    pub velocity_delta: Real,
    pub pressure_iters: usize,
    pub momentum_iters: usize,
    pub pressure_converged: bool,
    pub momentum_converged: bool,
    pub coupling: PressureCouplingLog,
    pub timing: SimplecStepTiming,
    pub converged: bool,
    pub is_final: bool,
}

pub(crate) fn log_simplec_step(log: SimplecStepLog) {
    info!(
        step = log.step,
        algorithm = log.algorithm.label(),
        continuity = %format_log_sci4(log.continuity),
        momentum = %format_log_sci4(log.momentum),
        velocity_delta = %format_log_sci4(log.velocity_delta),
        log10_continuity = %format_log_fixed4(log10_positive(log.continuity)),
        pressure_iters = log.pressure_iters,
        momentum_iters = log.momentum_iters,
        pressure_converged = log.pressure_converged,
        momentum_converged = log.momentum_converged,
        predicted_divergence = %format_log_sci4(log.coupling.predicted_divergence),
        pressure_equation_residual = %format_log_sci4(log.coupling.pressure_equation_residual),
        face_flux_divergence = %format_log_sci4(log.coupling.face_flux_divergence),
        pressure_rhs_active_sum = %format_log_sci4(log.coupling.rhs_active_sum),
        profile_divergence_ms = %format_log_fixed4(log.timing.divergence_ms),
        profile_momentum_assemble_ms = %format_log_fixed4(log.timing.momentum_assemble_ms),
        profile_momentum_solve_ms = %format_log_fixed4(log.timing.momentum_solve_ms),
        profile_rhie_chow_ms = %format_log_fixed4(log.timing.rhie_chow_ms),
        profile_pressure_ms = %format_log_fixed4(log.timing.pressure_ms),
        profile_correct_ms = %format_log_fixed4(log.timing.correct_ms),
        profile_step_total_ms = %format_log_fixed4(log.timing.step_total_ms),
        converged = log.converged,
        is_final = log.is_final,
        "SIMPLEC 外层步"
    );
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VelocityDeltaByRegion {
    pub(crate) all: Real,
    pub(crate) interior: Real,
    pub(crate) boundary: Real,
}

pub(crate) fn max_velocity_delta_by_region(
    mesh: &StructuredMesh3d,
    boundary: &BoundarySet,
    fields: &IncompressibleFields,
    u: &[Real],
    v: &[Real],
    w: &[Real],
) -> Result<VelocityDeltaByRegion> {
    let mut constrained_owner = vec![false; mesh.num_cells()];
    for patch in boundary.patches() {
        if !is_velocity_constrained_kind(&patch.kind) {
            continue;
        }
        for face_id in &patch.face_ids {
            let owner = mesh.face_owner(*face_id)?;
            constrained_owner[owner.index() as usize] = true;
        }
    }

    let mut delta = VelocityDeltaByRegion {
        all: 0.0,
        interior: 0.0,
        boundary: 0.0,
    };
    for idx in 0..fields.velocity_x.len() {
        let cell_delta = (u[idx] - fields.velocity_x.values()[idx])
            .abs()
            .max((v[idx] - fields.velocity_y.values()[idx]).abs())
            .max((w[idx] - fields.velocity_z.values()[idx]).abs());
        delta.all = delta.all.max(cell_delta);
        if constrained_owner[idx] {
            delta.boundary = delta.boundary.max(cell_delta);
        } else {
            delta.interior = delta.interior.max(cell_delta);
        }
    }
    Ok(delta)
}

fn is_velocity_constrained_kind(kind: &BoundaryKind) -> bool {
    matches!(
        kind,
        BoundaryKind::Wall { .. }
            | BoundaryKind::MovingWall { .. }
            | BoundaryKind::IncompressibleVelocityInlet { .. }
            | BoundaryKind::Inlet { .. }
    )
}

pub(crate) fn max_abs_scalar_field(field: &ScalarField) -> Real {
    field
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()))
}

pub(crate) fn max_abs_field_divergence(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    boundary: &BoundarySet,
) -> Result<Real> {
    let divergence = compute_incompressible_face_flux_divergence_3d(mesh, fields, boundary)?;
    Ok(max_abs_scalar_field(&divergence))
}

/// 面通量散度在非 \(p'=0\) owner 单元上的最大值（开域出口不参与 SIMPLEC 连续性判据）。
pub(crate) fn max_abs_active_face_flux_divergence(
    mesh: &StructuredMesh3d,
    face_flux: &crate::discretization::IncompressibleFaceFluxField,
    boundary: &BoundarySet,
) -> Result<Real> {
    let divergence = face_flux.divergence(mesh)?;
    let mut dirichlet_cells = vec![false; mesh.num_cells()];
    for patch in boundary.patches() {
        if !incompressible_pressure_correction_dirichlet(&patch.kind) {
            continue;
        }
        for &face in &patch.face_ids {
            let owner = mesh.face_owner(face)?.index() as usize;
            dirichlet_cells[owner] = true;
        }
    }
    let mut max_abs: Real = 0.0;
    for (cell, value) in divergence.values().iter().enumerate() {
        if dirichlet_cells[cell] {
            continue;
        }
        max_abs = max_abs.max(value.abs());
    }
    Ok(max_abs)
}

#[cfg(test)]
mod tests {
    use super::{SimplecConvergenceCheck, simplec_converged};

    #[test]
    fn simplec_convergence_requires_full_recent_window_below_tolerance() {
        let residual = [2.0e-5, 9.0e-6, 8.0e-6];
        let momentum = [1.0e-9, 1.0e-9, 1.0e-9];
        let velocity = [9.0e-6, 8.0e-6, 7.0e-6];

        assert!(!simplec_converged(SimplecConvergenceCheck {
            tolerance: Some(1.0e-5),
            min_iterations: 1,
            iterations: residual.len(),
            residual_history: &residual,
            momentum_history: &momentum,
            velocity_history: &velocity,
            convergence_window: 3,
            linear_solvers_converged: true,
        }));
        assert!(simplec_converged(SimplecConvergenceCheck {
            tolerance: Some(1.0e-5),
            min_iterations: 1,
            iterations: residual.len(),
            residual_history: &residual,
            momentum_history: &momentum,
            velocity_history: &velocity,
            convergence_window: 2,
            linear_solvers_converged: true,
        }));
    }

    #[test]
    fn simplec_convergence_requires_linear_solvers_to_converge() {
        let residual = [8.0e-6, 7.0e-6];
        let momentum = [1.0e-9, 1.0e-9];
        let velocity = [6.0e-6, 5.0e-6];

        assert!(!simplec_converged(SimplecConvergenceCheck {
            tolerance: Some(1.0e-5),
            min_iterations: 1,
            iterations: residual.len(),
            residual_history: &residual,
            momentum_history: &momentum,
            velocity_history: &velocity,
            convergence_window: 2,
            linear_solvers_converged: false,
        }));
    }
}
