//! 不可压缩 3D I0 占位求解器：初始化字段并写出流场。

use std::path::PathBuf;

#[cfg(not(feature = "io-cgns"))]
use tracing::warn;
use tracing::{info, info_span};

use crate::core::{Real, format_log_sci4};
use crate::discretization::{
    IncompressibleMomentumPredictorConfig, IncompressiblePressureCorrectionConfig,
    assemble_incompressible_momentum_predictor_3d, assemble_incompressible_pressure_poisson_3d,
    compute_incompressible_divergence_3d,
};
use crate::error::{AsimuError, Result};
use crate::field::IncompressibleFields;
#[cfg(feature = "io-cgns")]
use crate::field::ScalarField;
use crate::io::{CaseSpec, resolve_case_output_path};
#[cfg(feature = "io-cgns")]
use crate::io::{
    StructuredVertexSolution, VertexScalarFieldView, write_structured_vertex_solution_cgns,
};
use crate::linalg::{GmresConfig, GmresSolver, IdentityPreconditioner};
use crate::mesh::StructuredMesh3d;

use super::{CaseRunKind, CaseRunResult};

#[derive(Debug, Clone, PartialEq)]
pub struct Incompressible3dRunMetrics {
    pub steps: u64,
    pub physical_time: f64,
    pub max_abs_divergence: Real,
    pub pressure_system_rows: usize,
    pub pressure_system_nnz: usize,
    pub pressure_solve_converged: bool,
    pub pressure_solve_iterations: usize,
    pub pressure_solve_residual: Real,
    pub max_abs_pressure_correction: Real,
    pub momentum_system_rows: usize,
    pub momentum_system_nnz: usize,
    pub max_momentum_d_coefficient: Real,
    pub written: Vec<PathBuf>,
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
    let fields = IncompressibleFields::uniform(mesh.num_cells(), config.pressure, config.velocity)?;
    fields.validate_len(mesh.num_cells())?;
    let pseudo_time_step = case.time.dt.filter(|value| *value > 0.0).unwrap_or(1.0);
    let diagnostic = assemble_i1_diagnostic(
        mesh,
        &fields,
        config.density,
        config.kinematic_viscosity,
        pseudo_time_step,
    )?;

    let written = write_outputs(case, mesh, &fields, nondimensional_time)?;
    info!(
        steps,
        t = %format_log_sci4(physical_time),
        max_abs_divergence = %format_log_sci4(diagnostic.max_abs_divergence),
        pressure_rows = diagnostic.pressure_system_rows,
        pressure_nnz = diagnostic.pressure_system_nnz,
        pressure_converged = diagnostic.pressure_solve_converged,
        pressure_iters = diagnostic.pressure_solve_iterations,
        pressure_residual = %format_log_sci4(diagnostic.pressure_solve_residual),
        max_abs_pressure_correction = %format_log_sci4(diagnostic.max_abs_pressure_correction),
        momentum_rows = diagnostic.momentum_system_rows,
        momentum_nnz = diagnostic.momentum_system_nnz,
        max_momentum_d = %format_log_sci4(diagnostic.max_momentum_d_coefficient),
        "不可压缩 3D I1 skeleton 完成"
    );
    Ok(CaseRunResult {
        name: case.name.clone(),
        benchmark_id: case.benchmark_id.clone(),
        kind: CaseRunKind::Incompressible3dSteady,
        summary: format!(
            "incompressible_3d_i1 steps={steps} max|div(u)|={} pressure_rows={} pressure_nnz={} pressure_converged={} pressure_iters={} pressure_residual={} momentum_rows={} momentum_nnz={}",
            format_log_sci4(diagnostic.max_abs_divergence),
            diagnostic.pressure_system_rows,
            diagnostic.pressure_system_nnz,
            diagnostic.pressure_solve_converged,
            diagnostic.pressure_solve_iterations,
            format_log_sci4(diagnostic.pressure_solve_residual),
            diagnostic.momentum_system_rows,
            diagnostic.momentum_system_nnz
        ),
        diffusion: None,
        sod: None,
        compressible_3d: None,
        incompressible_3d: Some(Incompressible3dRunMetrics {
            steps,
            physical_time,
            max_abs_divergence: diagnostic.max_abs_divergence,
            pressure_system_rows: diagnostic.pressure_system_rows,
            pressure_system_nnz: diagnostic.pressure_system_nnz,
            pressure_solve_converged: diagnostic.pressure_solve_converged,
            pressure_solve_iterations: diagnostic.pressure_solve_iterations,
            pressure_solve_residual: diagnostic.pressure_solve_residual,
            max_abs_pressure_correction: diagnostic.max_abs_pressure_correction,
            momentum_system_rows: diagnostic.momentum_system_rows,
            momentum_system_nnz: diagnostic.momentum_system_nnz,
            max_momentum_d_coefficient: diagnostic.max_momentum_d_coefficient,
            written,
        }),
    })
}

struct IncompressibleI1Diagnostic {
    max_abs_divergence: Real,
    pressure_system_rows: usize,
    pressure_system_nnz: usize,
    pressure_solve_converged: bool,
    pressure_solve_iterations: usize,
    pressure_solve_residual: Real,
    max_abs_pressure_correction: Real,
    momentum_system_rows: usize,
    momentum_system_nnz: usize,
    max_momentum_d_coefficient: Real,
}

fn assemble_i1_diagnostic(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    density: Real,
    kinematic_viscosity: Real,
    pseudo_time_step: Real,
) -> Result<IncompressibleI1Diagnostic> {
    let divergence = compute_incompressible_divergence_3d(mesh, fields)?;
    let max_abs_divergence = divergence
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    let system = assemble_incompressible_pressure_poisson_3d(
        mesh,
        &divergence,
        IncompressiblePressureCorrectionConfig::new(density, 0, 0.0)?,
    )?;
    let pressure_solution = solve_pressure_correction(&system)?;
    let momentum_system = assemble_incompressible_momentum_predictor_3d(
        mesh,
        fields,
        IncompressibleMomentumPredictorConfig::new(kinematic_viscosity, pseudo_time_step)?,
    )?;
    let max_momentum_d_coefficient = momentum_system
        .d_coefficient
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    Ok(IncompressibleI1Diagnostic {
        max_abs_divergence,
        pressure_system_rows: system.matrix.nrows(),
        pressure_system_nnz: system.matrix.values().len(),
        pressure_solve_converged: pressure_solution.converged,
        pressure_solve_iterations: pressure_solution.iterations,
        pressure_solve_residual: pressure_solution.residual_norm,
        max_abs_pressure_correction: pressure_solution.max_abs_correction,
        momentum_system_rows: momentum_system.matrix.nrows(),
        momentum_system_nnz: momentum_system.matrix.values().len(),
        max_momentum_d_coefficient,
    })
}

struct PressureCorrectionSolveDiagnostic {
    converged: bool,
    iterations: usize,
    residual_norm: Real,
    max_abs_correction: Real,
}

fn solve_pressure_correction(
    system: &crate::discretization::IncompressiblePressureCorrectionSystem,
) -> Result<PressureCorrectionSolveDiagnostic> {
    let n = system.matrix.nrows();
    let mut matrix = system.matrix.clone();
    let preconditioner = IdentityPreconditioner::new(n);
    let solver = GmresSolver::new(GmresConfig::default())?;
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
    })
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
