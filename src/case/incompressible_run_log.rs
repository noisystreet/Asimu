//! 不可压缩 3D 算例完成日志与摘要字符串。

use tracing::info;

use crate::core::{Real, format_log_sci4};
use crate::discretization::IncompressibleBoundaryApplyStats;
use crate::solver::IncompressibleSimplecDiagnostic;

pub(super) fn log_incompressible_completion(
    steps: u64,
    physical_time: Real,
    diagnostic: &IncompressibleSimplecDiagnostic,
    boundary_stats: &IncompressibleBoundaryApplyStats,
) {
    info!(
        steps,
        t = %format_log_sci4(physical_time),
        max_abs_divergence = %format_log_sci4(diagnostic.max_abs_divergence),
        max_abs_predicted_divergence = %format_log_sci4(diagnostic.max_abs_predicted_divergence),
        max_abs_corrected_divergence = %format_log_sci4(diagnostic.max_abs_corrected_divergence),
        max_abs_underrelaxed_corrected_divergence =
            %format_log_sci4(diagnostic.max_abs_underrelaxed_corrected_divergence),
        max_abs_corrected_field_divergence_before_boundary =
            %format_log_sci4(diagnostic.max_abs_corrected_field_divergence_before_boundary),
        max_abs_corrected_field_divergence_after_boundary =
            %format_log_sci4(diagnostic.max_abs_corrected_field_divergence_after_boundary),
        pressure_rhs_active_sum = %format_log_sci4(diagnostic.pressure_correction_rhs_active_sum),
        pressure_rows = diagnostic.pressure_system_rows,
        pressure_nnz = diagnostic.pressure_system_nnz,
        pressure_converged = diagnostic.pressure_solve_converged,
        pressure_iters = diagnostic.pressure_solve_iterations,
        pressure_residual = %format_log_sci4(diagnostic.pressure_solve_residual),
        max_abs_pressure_correction = %format_log_sci4(diagnostic.max_abs_pressure_correction),
        momentum_rows = diagnostic.momentum_system_rows,
        momentum_nnz = diagnostic.momentum_system_nnz,
        max_momentum_d = %format_log_sci4(diagnostic.max_momentum_d_coefficient),
        momentum_converged = diagnostic.momentum_solve_converged,
        momentum_iters = diagnostic.momentum_solve_iterations,
        momentum_residual = %format_log_sci4(diagnostic.momentum_solve_residual),
        max_abs_momentum_equation_residual = %format_log_sci4(diagnostic.max_abs_momentum_equation_residual),
        max_abs_predicted_velocity_delta = %format_log_sci4(diagnostic.max_abs_predicted_velocity_delta),
        max_abs_corrected_velocity_delta = %format_log_sci4(diagnostic.max_abs_corrected_velocity_delta),
        max_abs_corrected_velocity_delta_interior =
            %format_log_sci4(diagnostic.max_abs_corrected_velocity_delta_interior),
        max_abs_corrected_velocity_delta_boundary =
            %format_log_sci4(diagnostic.max_abs_corrected_velocity_delta_boundary),
        algorithm = diagnostic.algorithm.label(),
        pressure_correctors = diagnostic.pressure_correctors,
        simplec_iterations = diagnostic.simplec_iterations,
        simplec_converged = diagnostic.simplec_converged,
        simplec_final_residual = %format_log_sci4(diagnostic.simplec_final_residual),
        simplec_final_momentum_residual = %format_log_sci4(diagnostic.simplec_final_momentum_residual),
        boundary_velocity_cells = boundary_stats.velocity_cells,
        boundary_pressure_cells = boundary_stats.pressure_cells,
        boundary_ignored_faces = boundary_stats.ignored_faces,
        "不可压缩 3D I1 skeleton 完成"
    );
}

pub(super) fn incompressible_summary(
    steps: u64,
    diagnostic: &IncompressibleSimplecDiagnostic,
    boundary_stats: &IncompressibleBoundaryApplyStats,
) -> String {
    format!(
        "incompressible_3d_i1 algorithm={} pressure_correctors={} steps={steps} pressure_velocity_iters={} pressure_velocity_converged={} pressure_velocity_residual={} pressure_velocity_momentum_residual={} max|div(u)|={} max|div(u*)|={} max|div(u_corr_eq)|={} max|div(u_corr_underrelaxed_eq)|={} max|div(u_corr_pre_bc)|={} max|div(u_corr_post_bc)|={} pressure_rhs_active_sum={} pressure_rows={} pressure_nnz={} pressure_converged={} pressure_iters={} pressure_residual={} momentum_rows={} momentum_nnz={} momentum_converged={} momentum_iters={} momentum_residual={} bc_velocity_cells={} bc_pressure_cells={}",
        diagnostic.algorithm.label(),
        diagnostic.pressure_correctors,
        diagnostic.simplec_iterations,
        diagnostic.simplec_converged,
        format_log_sci4(diagnostic.simplec_final_residual),
        format_log_sci4(diagnostic.simplec_final_momentum_residual),
        format_log_sci4(diagnostic.max_abs_divergence),
        format_log_sci4(diagnostic.max_abs_predicted_divergence),
        format_log_sci4(diagnostic.max_abs_corrected_divergence),
        format_log_sci4(diagnostic.max_abs_underrelaxed_corrected_divergence),
        format_log_sci4(diagnostic.max_abs_corrected_field_divergence_before_boundary),
        format_log_sci4(diagnostic.max_abs_corrected_field_divergence_after_boundary),
        format_log_sci4(diagnostic.pressure_correction_rhs_active_sum),
        diagnostic.pressure_system_rows,
        diagnostic.pressure_system_nnz,
        diagnostic.pressure_solve_converged,
        diagnostic.pressure_solve_iterations,
        format_log_sci4(diagnostic.pressure_solve_residual),
        diagnostic.momentum_system_rows,
        diagnostic.momentum_system_nnz,
        diagnostic.momentum_solve_converged,
        diagnostic.momentum_solve_iterations,
        format_log_sci4(diagnostic.momentum_solve_residual),
        boundary_stats.velocity_cells,
        boundary_stats.pressure_cells
    )
}
