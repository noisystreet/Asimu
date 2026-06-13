//! 不可压缩 3D I0 占位求解器：初始化字段并写出流场。

use std::path::PathBuf;

#[cfg(not(feature = "io-cgns"))]
use tracing::warn;
use tracing::{info, info_span};

use crate::core::{Real, format_log_sci4};
use crate::discretization::{
    IncompressibleBoundaryApplyStats, apply_incompressible_boundary_conditions_3d,
};
use crate::error::{AsimuError, Result};
use crate::field::IncompressibleFields;
#[cfg(feature = "io-cgns")]
use crate::field::ScalarField;
use crate::io::write_incompressible_residual_csv;
use crate::io::{CaseSpec, CaseTimeConfig, CaseTimeMode, resolve_case_output_path};
#[cfg(feature = "io-cgns")]
use crate::io::{
    StructuredVertexSolution, VertexScalarFieldView, write_structured_vertex_solution_cgns,
};
use crate::mesh::StructuredMesh3d;
use crate::solver::{
    IncompressiblePressureVelocityStepInfo, IncompressibleSimplecConfig,
    IncompressibleSimplecDiagnostic, TimeIntegrationScheme,
    run_incompressible_pressure_velocity_with_observer,
};

use super::incompressible_profiles::{
    incompressible_centerline_profiles, lid_cavity_profile_error, poiseuille_profile_error,
};
use super::{CaseRunKind, CaseRunResult};

#[derive(Debug, Clone, PartialEq)]
pub struct Incompressible3dRunMetrics {
    pub algorithm: String,
    pub pressure_correctors: usize,
    pub steps: u64,
    pub physical_time: f64,
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
    pub max_abs_pressure: Real,
    pub max_abs_velocity: Real,
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
    pub boundary_velocity_cells: usize,
    pub boundary_pressure_cells: usize,
    pub boundary_ignored_faces: usize,
    pub centerline_profiles: Option<IncompressibleCenterlineProfiles>,
    pub poiseuille_profile_error: Option<IncompressibleProfileError>,
    pub lid_cavity_profile_error: Option<IncompressibleCenterlineProfileError>,
    pub written: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleLineSample {
    pub coordinate: Real,
    pub velocity_x: Real,
    pub velocity_y: Real,
    pub velocity_z: Real,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleCenterlineProfiles {
    pub vertical_u: Vec<IncompressibleLineSample>,
    pub horizontal_v: Vec<IncompressibleLineSample>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleProfileError {
    pub max_abs: Real,
    pub l2: Real,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleCenterlineProfileError {
    pub vertical_u: IncompressibleProfileError,
    pub horizontal_v: IncompressibleProfileError,
}

pub fn run(case: &CaseSpec) -> Result<CaseRunResult> {
    let mesh = case.mesh.as_3d()?;
    let config = case
        .incompressible
        .as_ref()
        .ok_or_else(|| AsimuError::Config("不可压缩算例须包含 [incompressible] 段".to_string()))?;
    let steps = case.resolved_max_steps();
    let dt = case.time.dt.unwrap_or(0.0);
    let mut fields =
        IncompressibleFields::uniform(mesh.num_cells(), config.pressure, config.velocity)?;
    fields.validate_len(mesh.num_cells())?;
    let boundary_stats =
        apply_incompressible_boundary_conditions_3d(mesh, &mut fields, &case.boundary)?;
    let pseudo_time_step = case.time.dt.filter(|value| *value > 0.0).unwrap_or(1.0);
    let mut written = Vec::new();
    let diagnostic = run_incompressible_pressure_velocity_with_observer(
        &fields,
        IncompressibleSimplecConfig {
            mesh,
            density: config.density,
            kinematic_viscosity: config.kinematic_viscosity,
            body_force: config.body_force,
            velocity_under_relaxation: config.velocity_under_relaxation,
            pressure_under_relaxation: config.pressure_under_relaxation,
            pseudo_time_step,
            convection_scheme: config.convection_scheme,
            pressure_correctors: incompressible_pressure_correctors(case, config.piso_correctors),
            boundary: &case.boundary,
            max_iterations: steps as usize,
            min_iterations: case.time.min_steps.unwrap_or(0) as usize,
            tolerance: case.time.tolerance,
            require_velocity_convergence: case.time.mode == CaseTimeMode::Steady,
            convergence_window: incompressible_convergence_window(&case.time),
            snapshot_interval: None,
            linear_solvers: config.linear_solvers,
        },
        |step| {
            written.extend(super::output_interval::maybe_write_incompressible_interval(
                case, mesh, step,
            )?);
            Ok(())
        },
    )?;

    let completed_steps = diagnostic.simplec_iterations as u64;
    let nondimensional_time = dt * completed_steps as f64;
    let physical_time = physical_time_from_nondimensional(case, nondimensional_time);
    written.extend(write_outputs(case, mesh, &diagnostic, nondimensional_time)?);
    let centerline_profiles = incompressible_centerline_profiles(
        case,
        mesh,
        config.kinematic_viscosity,
        config.body_force,
        &diagnostic.corrected_fields,
    );
    let poiseuille_profile_error = poiseuille_profile_error(
        case,
        mesh,
        config.kinematic_viscosity,
        config.body_force,
        &diagnostic.corrected_fields,
    );
    let lid_cavity_profile_error = lid_cavity_profile_error(case, centerline_profiles.as_ref());
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
        max_abs_corrected_velocity_delta_interior = %format_log_sci4(diagnostic.max_abs_corrected_velocity_delta_interior),
        max_abs_corrected_velocity_delta_boundary = %format_log_sci4(diagnostic.max_abs_corrected_velocity_delta_boundary),
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
    Ok(CaseRunResult {
        name: case.name.clone(),
        benchmark_id: case.benchmark_id.clone(),
        kind: CaseRunKind::Incompressible3dSteady,
        summary: incompressible_summary(steps, &diagnostic, &boundary_stats),
        diffusion: None,
        sod: None,
        compressible_3d: None,
        incompressible_3d: Some(build_run_metrics(
            completed_steps,
            physical_time,
            diagnostic,
            boundary_stats,
            BenchmarkDiagnostics {
                centerline_profiles,
                poiseuille_profile_error,
                lid_cavity_profile_error,
                written,
            },
        )),
    })
}

fn incompressible_convergence_window(time: &CaseTimeConfig) -> usize {
    if time.mode == CaseTimeMode::Steady {
        crate::core::incompressible_steady_convergence_window(time.min_steps.unwrap_or(0))
    } else {
        1
    }
}

fn incompressible_pressure_correctors(case: &CaseSpec, configured: usize) -> usize {
    match case.time.scheme {
        Some(TimeIntegrationScheme::Simplec) => 1,
        Some(TimeIntegrationScheme::Piso) => configured.max(1),
        _ => configured.max(1),
    }
}

fn physical_time_from_nondimensional(case: &CaseSpec, nondimensional_time: Real) -> Real {
    case.incompressible_reference
        .as_ref()
        .map(|reference| nondimensional_time * reference.time_scale())
        .unwrap_or(nondimensional_time)
}

fn incompressible_physical_time_scale(case: &CaseSpec) -> Real {
    case.incompressible_reference
        .as_ref()
        .map(|reference| reference.time_scale())
        .unwrap_or(1.0)
}

fn incompressible_summary(
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

struct BenchmarkDiagnostics {
    centerline_profiles: Option<IncompressibleCenterlineProfiles>,
    poiseuille_profile_error: Option<IncompressibleProfileError>,
    lid_cavity_profile_error: Option<IncompressibleCenterlineProfileError>,
    written: Vec<PathBuf>,
}

fn build_run_metrics(
    steps: u64,
    physical_time: Real,
    diagnostic: IncompressibleSimplecDiagnostic,
    boundary_stats: IncompressibleBoundaryApplyStats,
    benchmark: BenchmarkDiagnostics,
) -> Incompressible3dRunMetrics {
    let max_abs_pressure = max_abs_slice(diagnostic.corrected_fields.pressure.values());
    let max_abs_velocity = max_abs_incompressible_velocity(&diagnostic.corrected_fields);
    Incompressible3dRunMetrics {
        algorithm: diagnostic.algorithm.label().to_string(),
        pressure_correctors: diagnostic.pressure_correctors,
        steps,
        physical_time,
        max_abs_divergence: diagnostic.max_abs_divergence,
        max_abs_predicted_divergence: diagnostic.max_abs_predicted_divergence,
        max_abs_corrected_divergence: diagnostic.max_abs_corrected_divergence,
        max_abs_underrelaxed_corrected_divergence: diagnostic
            .max_abs_underrelaxed_corrected_divergence,
        max_abs_corrected_field_divergence_before_boundary: diagnostic
            .max_abs_corrected_field_divergence_before_boundary,
        max_abs_corrected_field_divergence_after_boundary: diagnostic
            .max_abs_corrected_field_divergence_after_boundary,
        pressure_correction_rhs_active_sum: diagnostic.pressure_correction_rhs_active_sum,
        pressure_system_rows: diagnostic.pressure_system_rows,
        pressure_system_nnz: diagnostic.pressure_system_nnz,
        pressure_solve_converged: diagnostic.pressure_solve_converged,
        pressure_solve_iterations: diagnostic.pressure_solve_iterations,
        pressure_solve_residual: diagnostic.pressure_solve_residual,
        max_abs_pressure_correction: diagnostic.max_abs_pressure_correction,
        max_abs_pressure,
        max_abs_velocity,
        momentum_system_rows: diagnostic.momentum_system_rows,
        momentum_system_nnz: diagnostic.momentum_system_nnz,
        max_momentum_d_coefficient: diagnostic.max_momentum_d_coefficient,
        momentum_solve_converged: diagnostic.momentum_solve_converged,
        momentum_solve_iterations: diagnostic.momentum_solve_iterations,
        momentum_solve_residual: diagnostic.momentum_solve_residual,
        max_abs_momentum_equation_residual: diagnostic.max_abs_momentum_equation_residual,
        max_abs_predicted_velocity_delta: diagnostic.max_abs_predicted_velocity_delta,
        max_abs_corrected_velocity_delta: diagnostic.max_abs_corrected_velocity_delta,
        max_abs_corrected_velocity_delta_interior: diagnostic
            .max_abs_corrected_velocity_delta_interior,
        max_abs_corrected_velocity_delta_boundary: diagnostic
            .max_abs_corrected_velocity_delta_boundary,
        simplec_iterations: diagnostic.simplec_iterations,
        simplec_converged: diagnostic.simplec_converged,
        simplec_final_residual: diagnostic.simplec_final_residual,
        simplec_final_momentum_residual: diagnostic.simplec_final_momentum_residual,
        simplec_residual_history: diagnostic.simplec_residual_history,
        simplec_momentum_residual_history: diagnostic.simplec_momentum_residual_history,
        pressure_corrector_residual_history: diagnostic.pressure_corrector_residual_history,
        pressure_corrector_max_correction_history: diagnostic
            .pressure_corrector_max_correction_history,
        boundary_velocity_cells: boundary_stats.velocity_cells,
        boundary_pressure_cells: boundary_stats.pressure_cells,
        boundary_ignored_faces: boundary_stats.ignored_faces,
        centerline_profiles: benchmark.centerline_profiles,
        poiseuille_profile_error: benchmark.poiseuille_profile_error,
        lid_cavity_profile_error: benchmark.lid_cavity_profile_error,
        written: benchmark.written,
    }
}

fn max_abs_incompressible_velocity(fields: &IncompressibleFields) -> Real {
    max_abs_slice(fields.velocity_x.values())
        .max(max_abs_slice(fields.velocity_y.values()))
        .max(max_abs_slice(fields.velocity_z.values()))
}

fn max_abs_slice(values: &[Real]) -> Real {
    values
        .iter()
        .fold(0.0, |max_value: Real, value| max_value.max(value.abs()))
}

fn write_outputs(
    case: &CaseSpec,
    mesh: &StructuredMesh3d,
    diagnostic: &IncompressibleSimplecDiagnostic,
    nondimensional_time: f64,
) -> Result<Vec<PathBuf>> {
    let Some(output) = &case.output else {
        return Ok(Vec::new());
    };
    let mut written = Vec::new();
    written.extend(write_incompressible_residual_outputs(
        case,
        &diagnostic.step_history,
    )?);
    if let Some(name) = &output.solution_cgns {
        let path = resolve_case_output_path(case.case_dir.as_deref(), &output.dir, name)?;
        let (mesh_out, fields_out, time_out) = prepare_dimensional_incompressible_output(
            case,
            mesh,
            &diagnostic.corrected_fields,
            nondimensional_time,
        )?;
        write_incompressible_cgns(&path, &mesh_out, &fields_out, time_out)?;
        info!(path = %path.display(), "已写出不可压缩流场 CGNS");
        written.push(path);
    }
    Ok(written)
}

pub(crate) fn write_incompressible_residual_outputs(
    case: &CaseSpec,
    history: &[IncompressiblePressureVelocityStepInfo],
) -> Result<Vec<PathBuf>> {
    let Some(output) = &case.output else {
        return Ok(Vec::new());
    };
    let mut written = Vec::new();
    if let Some(name) = &output.residual_csv {
        let path = resolve_case_output_path(case.case_dir.as_deref(), &output.dir, name)?;
        write_incompressible_residual_csv(
            &path,
            history,
            incompressible_physical_time_scale(case),
        )?;
        info!(path = %path.display(), "已写出不可压缩残差 CSV");
        written.push(path.clone());
        if let Some(plot_name) = &output.residual_plot {
            let plot_path =
                resolve_case_output_path(case.case_dir.as_deref(), &output.dir, plot_name)?;
            if let Err(err) = super::output_3d::plot_residual_csv(&path, &plot_path) {
                tracing::warn!(error = %err, "不可压缩残差曲线图未生成（需 python3 + matplotlib）");
            } else {
                info!(path = %plot_path.display(), "已写出不可压缩残差曲线图");
                written.push(plot_path);
            }
        }
    }
    Ok(written)
}

pub(crate) fn write_incompressible_interval_flow_cgns(
    case: &CaseSpec,
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    nondimensional_time: f64,
    path: &std::path::Path,
) -> Result<()> {
    let (mesh_out, fields_out, time_out) =
        prepare_dimensional_incompressible_output(case, mesh, fields, nondimensional_time)?;
    write_incompressible_cgns(path, &mesh_out, &fields_out, time_out)?;
    info!(
        path = %path.display(),
        "已写出不可压缩间隔流场 CGNS"
    );
    Ok(())
}

fn prepare_dimensional_incompressible_output(
    case: &CaseSpec,
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    nondimensional_time: f64,
) -> Result<(StructuredMesh3d, IncompressibleFields, f64)> {
    let Some(reference) = case.incompressible_reference.as_ref() else {
        return Ok((mesh.clone(), fields.clone(), nondimensional_time));
    };
    let mut mesh_out = mesh.clone();
    mesh_out.scale_coordinates(reference.length);
    let fields_out = fields.to_dimensional(reference)?;
    let time_out = nondimensional_time * reference.time_scale();
    Ok((mesh_out, fields_out, time_out))
}

#[cfg(feature = "io-cgns")]
fn write_incompressible_cgns(
    path: &std::path::Path,
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    physical_time: f64,
) -> Result<()> {
    let _span = info_span!("write_incompressible_cgns", path = %path.display()).entered();
    let vertex = gather_incompressible_vertex_fields(mesh, fields)?;
    let views = [
        VertexScalarFieldView {
            name: "Pressure",
            values: &vertex.pressure,
        },
        VertexScalarFieldView {
            name: "VelocityX",
            values: &vertex.velocity_x,
        },
        VertexScalarFieldView {
            name: "VelocityY",
            values: &vertex.velocity_y,
        },
        VertexScalarFieldView {
            name: "VelocityZ",
            values: &vertex.velocity_z,
        },
    ];
    write_structured_vertex_solution_cgns(
        path,
        mesh,
        StructuredVertexSolution {
            physical_time,
            fields: &views,
        },
    )
}

#[cfg(not(feature = "io-cgns"))]
fn write_incompressible_cgns(
    path: &std::path::Path,
    _mesh: &StructuredMesh3d,
    _fields: &IncompressibleFields,
    _physical_time: f64,
) -> Result<()> {
    let _span = info_span!("write_incompressible_cgns", path = %path.display()).entered();
    warn!("solution_cgns 须启用 feature io-cgns");
    Ok(())
}

#[cfg(feature = "io-cgns")]
struct IncompressibleVertexFields {
    pressure: Vec<f64>,
    velocity_x: Vec<f64>,
    velocity_y: Vec<f64>,
    velocity_z: Vec<f64>,
}

#[cfg(feature = "io-cgns")]
fn gather_incompressible_vertex_fields(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
) -> Result<IncompressibleVertexFields> {
    fields.validate_len(mesh.num_cells())?;
    Ok(IncompressibleVertexFields {
        pressure: gather_vertex_scalar(mesh, &fields.pressure)?,
        velocity_x: gather_vertex_scalar(mesh, &fields.velocity_x)?,
        velocity_y: gather_vertex_scalar(mesh, &fields.velocity_y)?,
        velocity_z: gather_vertex_scalar(mesh, &fields.velocity_z)?,
    })
}

#[cfg(feature = "io-cgns")]
fn gather_vertex_scalar(mesh: &StructuredMesh3d, field: &ScalarField) -> Result<Vec<f64>> {
    if field.len() != mesh.num_cells() {
        return Err(AsimuError::Field(format!(
            "不可压缩输出字段长度 {} 与单元数 {} 不一致",
            field.len(),
            mesh.num_cells()
        )));
    }
    let mut out = Vec::with_capacity(mesh.num_nodes());
    for k in 0..=mesh.nz {
        for j in 0..=mesh.ny {
            for i in 0..=mesh.nx {
                let mut sum = 0.0;
                let mut count = 0usize;
                let k0 = k.saturating_sub(1);
                let j0 = j.saturating_sub(1);
                let i0 = i.saturating_sub(1);
                let k1 = k.min(mesh.nz - 1);
                let j1 = j.min(mesh.ny - 1);
                let i1 = i.min(mesh.nx - 1);
                for ck in k0..=k1 {
                    for cj in j0..=j1 {
                        for ci in i0..=i1 {
                            sum += field.values()[mesh.cell_index(ci, cj, ck)];
                            count += 1;
                        }
                    }
                }
                out.push(sum / count.max(1) as f64);
            }
        }
    }
    Ok(out)
}
