//! 不可压缩 3D I0 占位求解器：初始化字段并写出流场。

use std::path::PathBuf;

#[cfg(not(feature = "io-cgns"))]
use tracing::warn;
use tracing::{info, info_span};

use crate::core::{Real, format_log_sci4};
use crate::discretization::apply_incompressible_boundary_conditions_3d;
use crate::error::{AsimuError, Result};
use crate::field::IncompressibleFields;
#[cfg(feature = "io-cgns")]
use crate::field::ScalarField;
use crate::io::{CaseSpec, resolve_case_output_path};
#[cfg(feature = "io-cgns")]
use crate::io::{
    StructuredVertexSolution, VertexScalarFieldView, write_structured_vertex_solution_cgns,
};
use crate::mesh::StructuredMesh3d;
use crate::solver::{IncompressibleSimplecConfig, run_incompressible_simplec};

use super::{CaseRunKind, CaseRunResult};

#[derive(Debug, Clone, PartialEq)]
pub struct Incompressible3dRunMetrics {
    pub steps: u64,
    pub physical_time: f64,
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
    pub boundary_velocity_cells: usize,
    pub boundary_pressure_cells: usize,
    pub boundary_ignored_faces: usize,
    pub centerline_profiles: Option<IncompressibleCenterlineProfiles>,
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

pub fn run(case: &CaseSpec) -> Result<CaseRunResult> {
    let mesh = case.mesh.as_3d()?;
    let config = case
        .incompressible
        .as_ref()
        .ok_or_else(|| AsimuError::Config("不可压缩算例须包含 [incompressible] 段".to_string()))?;
    let steps = case.time.max_steps.unwrap_or(1);
    let dt = case.time.dt.unwrap_or(0.0);
    let nondimensional_time = dt * steps as f64;
    let physical_time = case
        .incompressible_reference
        .as_ref()
        .map(|reference| nondimensional_time * reference.time_scale())
        .unwrap_or(nondimensional_time);
    let mut fields =
        IncompressibleFields::uniform(mesh.num_cells(), config.pressure, config.velocity)?;
    fields.validate_len(mesh.num_cells())?;
    let boundary_stats =
        apply_incompressible_boundary_conditions_3d(mesh, &mut fields, &case.boundary)?;
    let pseudo_time_step = case.time.dt.filter(|value| *value > 0.0).unwrap_or(1.0);
    let diagnostic = run_incompressible_simplec(
        &fields,
        IncompressibleSimplecConfig {
            mesh,
            density: config.density,
            kinematic_viscosity: config.kinematic_viscosity,
            body_force: config.body_force,
            velocity_under_relaxation: config.velocity_under_relaxation,
            pressure_under_relaxation: config.pressure_under_relaxation,
            pseudo_time_step,
            boundary: &case.boundary,
            max_iterations: steps as usize,
            tolerance: case.time.tolerance,
            linear_solvers: config.linear_solvers,
        },
    )?;

    let written = write_outputs(
        case,
        mesh,
        &diagnostic.corrected_fields,
        nondimensional_time,
    )?;
    let centerline_profiles =
        lid_cavity_centerline_profiles(case, mesh, &diagnostic.corrected_fields);
    info!(
        steps,
        t = %format_log_sci4(physical_time),
        max_abs_divergence = %format_log_sci4(diagnostic.max_abs_divergence),
        max_abs_predicted_divergence = %format_log_sci4(diagnostic.max_abs_predicted_divergence),
        max_abs_corrected_divergence = %format_log_sci4(diagnostic.max_abs_corrected_divergence),
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
        summary: format!(
            "incompressible_3d_i1 steps={steps} simplec_iters={} simplec_converged={} simplec_residual={} simplec_momentum_residual={} max|div(u)|={} max|div(u*)|={} max|div(u_corr)|={} pressure_rows={} pressure_nnz={} pressure_converged={} pressure_iters={} pressure_residual={} momentum_rows={} momentum_nnz={} momentum_converged={} momentum_iters={} momentum_residual={} bc_velocity_cells={} bc_pressure_cells={}",
            diagnostic.simplec_iterations,
            diagnostic.simplec_converged,
            format_log_sci4(diagnostic.simplec_final_residual),
            format_log_sci4(diagnostic.simplec_final_momentum_residual),
            format_log_sci4(diagnostic.max_abs_divergence),
            format_log_sci4(diagnostic.max_abs_predicted_divergence),
            format_log_sci4(diagnostic.max_abs_corrected_divergence),
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
        ),
        diffusion: None,
        sod: None,
        compressible_3d: None,
        incompressible_3d: Some(Incompressible3dRunMetrics {
            steps,
            physical_time,
            max_abs_divergence: diagnostic.max_abs_divergence,
            max_abs_predicted_divergence: diagnostic.max_abs_predicted_divergence,
            max_abs_corrected_divergence: diagnostic.max_abs_corrected_divergence,
            pressure_system_rows: diagnostic.pressure_system_rows,
            pressure_system_nnz: diagnostic.pressure_system_nnz,
            pressure_solve_converged: diagnostic.pressure_solve_converged,
            pressure_solve_iterations: diagnostic.pressure_solve_iterations,
            pressure_solve_residual: diagnostic.pressure_solve_residual,
            max_abs_pressure_correction: diagnostic.max_abs_pressure_correction,
            momentum_system_rows: diagnostic.momentum_system_rows,
            momentum_system_nnz: diagnostic.momentum_system_nnz,
            max_momentum_d_coefficient: diagnostic.max_momentum_d_coefficient,
            momentum_solve_converged: diagnostic.momentum_solve_converged,
            momentum_solve_iterations: diagnostic.momentum_solve_iterations,
            momentum_solve_residual: diagnostic.momentum_solve_residual,
            max_abs_momentum_equation_residual: diagnostic.max_abs_momentum_equation_residual,
            max_abs_predicted_velocity_delta: diagnostic.max_abs_predicted_velocity_delta,
            max_abs_corrected_velocity_delta: diagnostic.max_abs_corrected_velocity_delta,
            simplec_iterations: diagnostic.simplec_iterations,
            simplec_converged: diagnostic.simplec_converged,
            simplec_final_residual: diagnostic.simplec_final_residual,
            simplec_final_momentum_residual: diagnostic.simplec_final_momentum_residual,
            simplec_residual_history: diagnostic.simplec_residual_history,
            simplec_momentum_residual_history: diagnostic.simplec_momentum_residual_history,
            boundary_velocity_cells: boundary_stats.velocity_cells,
            boundary_pressure_cells: boundary_stats.pressure_cells,
            boundary_ignored_faces: boundary_stats.ignored_faces,
            centerline_profiles,
            written,
        }),
    })
}

fn lid_cavity_centerline_profiles(
    case: &CaseSpec,
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
) -> Option<IncompressibleCenterlineProfiles> {
    if case.benchmark_id.as_deref() != Some("lid_driven_cavity_re100") {
        return None;
    }
    let i_mid = mesh.nx / 2;
    let j_mid = mesh.ny / 2;
    let k_mid = mesh.nz / 2;
    let mut vertical_u = Vec::with_capacity(mesh.ny);
    for j in 0..mesh.ny {
        let cell = mesh.cell_index(i_mid, j, k_mid);
        vertical_u.push(IncompressibleLineSample {
            coordinate: cell_center_y(mesh, i_mid, j, k_mid),
            velocity_x: fields.velocity_x.values()[cell],
            velocity_y: fields.velocity_y.values()[cell],
            velocity_z: fields.velocity_z.values()[cell],
        });
    }
    let mut horizontal_v = Vec::with_capacity(mesh.nx);
    for i in 0..mesh.nx {
        let cell = mesh.cell_index(i, j_mid, k_mid);
        horizontal_v.push(IncompressibleLineSample {
            coordinate: cell_center_x(mesh, i, j_mid, k_mid),
            velocity_x: fields.velocity_x.values()[cell],
            velocity_y: fields.velocity_y.values()[cell],
            velocity_z: fields.velocity_z.values()[cell],
        });
    }
    Some(IncompressibleCenterlineProfiles {
        vertical_u,
        horizontal_v,
    })
}

fn cell_center_x(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> Real {
    0.5 * (mesh.node_x(i, j, k) + mesh.node_x(i + 1, j, k))
}

fn cell_center_y(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> Real {
    0.5 * (mesh.node_y(i, j, k) + mesh.node_y(i, j + 1, k))
}

fn write_outputs(
    case: &CaseSpec,
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    nondimensional_time: f64,
) -> Result<Vec<PathBuf>> {
    let Some(output) = &case.output else {
        return Ok(Vec::new());
    };
    let mut written = Vec::new();
    if let Some(name) = &output.solution_cgns {
        let path = resolve_case_output_path(case.case_dir.as_deref(), &output.dir, name)?;
        let (mesh_out, fields_out, time_out) =
            prepare_dimensional_incompressible_output(case, mesh, fields, nondimensional_time)?;
        write_incompressible_cgns(&path, &mesh_out, &fields_out, time_out)?;
        info!(path = %path.display(), "已写出不可压缩流场 CGNS");
        written.push(path);
    }
    Ok(written)
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
