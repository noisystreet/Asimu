//! 不可压缩 3D I0 占位求解器：初始化字段并写出流场。

use std::path::PathBuf;

#[cfg(not(feature = "io-cgns"))]
use tracing::warn;
use tracing::{info, info_span};

use crate::core::{Real, format_log_sci4};
use crate::discretization::{
    IncompressibleMomentumPredictorConfig, IncompressiblePressureCorrectionConfig,
    apply_incompressible_boundary_conditions_3d,
    assemble_incompressible_momentum_predictor_with_boundary_3d,
    assemble_incompressible_pressure_correction_3d, compute_incompressible_divergence_3d,
    compute_incompressible_rhie_chow_divergence_3d,
};
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
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
    pub max_abs_predicted_velocity_delta: Real,
    pub max_abs_corrected_velocity_delta: Real,
    pub simplec_iterations: usize,
    pub simplec_converged: bool,
    pub simplec_final_residual: Real,
    pub simplec_residual_history: Vec<Real>,
    pub boundary_velocity_cells: usize,
    pub boundary_pressure_cells: usize,
    pub boundary_ignored_faces: usize,
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
    let mut fields =
        IncompressibleFields::uniform(mesh.num_cells(), config.pressure, config.velocity)?;
    fields.validate_len(mesh.num_cells())?;
    let boundary_stats =
        apply_incompressible_boundary_conditions_3d(mesh, &mut fields, &case.boundary)?;
    let pseudo_time_step = case.time.dt.filter(|value| *value > 0.0).unwrap_or(1.0);
    let diagnostic = run_simplec_iterations(
        &fields,
        SimplecIterationParams {
            mesh,
            density: config.density,
            kinematic_viscosity: config.kinematic_viscosity,
            velocity_under_relaxation: config.velocity_under_relaxation,
            pseudo_time_step,
            boundary: &case.boundary,
            max_iterations: steps as usize,
            tolerance: case.time.tolerance,
        },
    )?;

    let written = write_outputs(
        case,
        mesh,
        &diagnostic.corrected_fields,
        nondimensional_time,
    )?;
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
        max_abs_predicted_velocity_delta = %format_log_sci4(diagnostic.max_abs_predicted_velocity_delta),
        max_abs_corrected_velocity_delta = %format_log_sci4(diagnostic.max_abs_corrected_velocity_delta),
        simplec_iterations = diagnostic.simplec_iterations,
        simplec_converged = diagnostic.simplec_converged,
        simplec_final_residual = %format_log_sci4(diagnostic.simplec_final_residual),
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
            "incompressible_3d_i1 steps={steps} simplec_iters={} simplec_converged={} simplec_residual={} max|div(u)|={} max|div(u*)|={} max|div(u_corr)|={} pressure_rows={} pressure_nnz={} pressure_converged={} pressure_iters={} pressure_residual={} momentum_rows={} momentum_nnz={} momentum_converged={} momentum_iters={} momentum_residual={} bc_velocity_cells={} bc_pressure_cells={}",
            diagnostic.simplec_iterations,
            diagnostic.simplec_converged,
            format_log_sci4(diagnostic.simplec_final_residual),
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
            max_abs_predicted_velocity_delta: diagnostic.max_abs_predicted_velocity_delta,
            max_abs_corrected_velocity_delta: diagnostic.max_abs_corrected_velocity_delta,
            simplec_iterations: diagnostic.simplec_iterations,
            simplec_converged: diagnostic.simplec_converged,
            simplec_final_residual: diagnostic.simplec_final_residual,
            simplec_residual_history: diagnostic.simplec_residual_history,
            boundary_velocity_cells: boundary_stats.velocity_cells,
            boundary_pressure_cells: boundary_stats.pressure_cells,
            boundary_ignored_faces: boundary_stats.ignored_faces,
            written,
        }),
    })
}

struct IncompressibleI1Diagnostic {
    max_abs_divergence: Real,
    max_abs_predicted_divergence: Real,
    max_abs_corrected_divergence: Real,
    pressure_system_rows: usize,
    pressure_system_nnz: usize,
    pressure_solve_converged: bool,
    pressure_solve_iterations: usize,
    pressure_solve_residual: Real,
    max_abs_pressure_correction: Real,
    momentum_system_rows: usize,
    momentum_system_nnz: usize,
    max_momentum_d_coefficient: Real,
    momentum_solve_converged: bool,
    momentum_solve_iterations: usize,
    momentum_solve_residual: Real,
    max_abs_predicted_velocity_delta: Real,
    max_abs_corrected_velocity_delta: Real,
    simplec_iterations: usize,
    simplec_converged: bool,
    simplec_final_residual: Real,
    simplec_residual_history: Vec<Real>,
    corrected_fields: IncompressibleFields,
}

struct SimplecIterationParams<'a> {
    mesh: &'a StructuredMesh3d,
    density: Real,
    kinematic_viscosity: Real,
    velocity_under_relaxation: Real,
    pseudo_time_step: Real,
    boundary: &'a crate::boundary::BoundarySet,
    max_iterations: usize,
    tolerance: Option<Real>,
}

fn run_simplec_iterations(
    initial_fields: &IncompressibleFields,
    params: SimplecIterationParams<'_>,
) -> Result<IncompressibleI1Diagnostic> {
    let mut current_fields = initial_fields.clone();
    let max_iterations = params.max_iterations.max(1);
    let mut history = Vec::with_capacity(max_iterations);
    let mut last = None;
    for _ in 0..max_iterations {
        let mut diagnostic = assemble_i1_diagnostic(
            params.mesh,
            &current_fields,
            params.density,
            params.kinematic_viscosity,
            params.velocity_under_relaxation,
            params.pseudo_time_step,
            params.boundary,
        )?;
        let residual = diagnostic.max_abs_corrected_divergence;
        if !residual.is_finite() {
            return Err(AsimuError::Solver(
                "SIMPLEC 连续性残差出现非有限值".to_string(),
            ));
        }
        history.push(residual);
        current_fields = diagnostic.corrected_fields.clone();
        let converged = params
            .tolerance
            .map(|tolerance| residual <= tolerance)
            .unwrap_or(false);
        diagnostic.simplec_iterations = history.len();
        diagnostic.simplec_converged = converged || params.tolerance.is_none();
        diagnostic.simplec_final_residual = residual;
        diagnostic.simplec_residual_history = history.clone();
        if converged {
            return Ok(diagnostic);
        }
        last = Some(diagnostic);
    }
    let mut diagnostic =
        last.ok_or_else(|| AsimuError::Solver("SIMPLEC 至少需要一次外层迭代".to_string()))?;
    diagnostic.simplec_converged = params
        .tolerance
        .map(|tolerance| diagnostic.simplec_final_residual <= tolerance)
        .unwrap_or(true);
    Ok(diagnostic)
}

fn assemble_i1_diagnostic(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    density: Real,
    kinematic_viscosity: Real,
    velocity_under_relaxation: Real,
    pseudo_time_step: Real,
    boundary: &crate::boundary::BoundarySet,
) -> Result<IncompressibleI1Diagnostic> {
    let divergence = compute_incompressible_divergence_3d(mesh, fields)?;
    let max_abs_divergence = divergence
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    let momentum_system = assemble_incompressible_momentum_predictor_with_boundary_3d(
        mesh,
        fields,
        boundary,
        IncompressibleMomentumPredictorConfig::new(kinematic_viscosity, pseudo_time_step)?
            .with_velocity_under_relaxation(velocity_under_relaxation)?,
    )?;
    let max_momentum_d_coefficient = momentum_system
        .d_coefficient
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    let momentum_solution = solve_momentum_predictor(&momentum_system, fields)?;
    let predicted_divergence = compute_incompressible_rhie_chow_divergence_3d(
        mesh,
        &momentum_solution.predicted_fields,
        &momentum_system.d_coefficient,
        boundary,
    )?;
    let max_abs_predicted_divergence = predicted_divergence
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    let system = assemble_incompressible_pressure_correction_3d(
        mesh,
        &predicted_divergence,
        &momentum_system.d_coefficient,
        boundary,
        IncompressiblePressureCorrectionConfig::new(density, 0, 0.0)?,
    )?;
    let pressure_solution = solve_pressure_correction(&system)?;
    let corrected_fields = corrected_incompressible_fields(
        mesh,
        fields,
        &momentum_solution.predicted_fields,
        &pressure_solution.correction,
        momentum_system.d_coefficient.values(),
    )?;
    let corrected_divergence = compute_incompressible_divergence_3d(mesh, &corrected_fields)?;
    let max_abs_corrected_divergence = corrected_divergence
        .values()
        .iter()
        .fold(0.0, |acc: Real, value| acc.max(value.abs()));
    let max_abs_corrected_velocity_delta = max_velocity_delta(
        fields,
        corrected_fields.velocity_x.values(),
        corrected_fields.velocity_y.values(),
        corrected_fields.velocity_z.values(),
    );
    Ok(IncompressibleI1Diagnostic {
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
        max_abs_predicted_velocity_delta: momentum_solution.max_abs_velocity_delta,
        max_abs_corrected_velocity_delta,
        simplec_iterations: 0,
        simplec_converged: false,
        simplec_final_residual: max_abs_corrected_divergence,
        simplec_residual_history: Vec::new(),
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
    max_abs_velocity_delta: Real,
    predicted_fields: IncompressibleFields,
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
        correction: pressure_correction,
    })
}

fn solve_momentum_predictor(
    system: &crate::discretization::IncompressibleMomentumPredictorSystem,
    fields: &IncompressibleFields,
) -> Result<MomentumPredictorSolveDiagnostic> {
    let u = solve_momentum_component(system, &system.rhs_x)?;
    let v = solve_momentum_component(system, &system.rhs_y)?;
    let w = solve_momentum_component(system, &system.rhs_z)?;
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
) -> Result<MomentumComponentSolve> {
    let n = system.matrix.nrows();
    let mut matrix = system.matrix.clone();
    let preconditioner = IdentityPreconditioner::new(n);
    let solver = GmresSolver::new(GmresConfig::default())?;
    let mut solution = vec![0.0; n];
    let report = solver.solve(&mut matrix, &preconditioner, rhs, &mut solution)?;
    Ok(MomentumComponentSolve {
        solution,
        converged: report.converged,
        iterations: report.iterations,
        residual_norm: report.residual_norm,
    })
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
) -> Result<IncompressibleFields> {
    let n = mesh.num_cells();
    if pressure_correction.len() != n || d_coefficient.len() != n {
        return Err(AsimuError::Field(
            "不可压缩修正场长度与网格单元数不一致".to_string(),
        ));
    }
    let spacing = CaseCartesianSpacing::from_mesh(mesh)?;
    let mut pressure = Vec::with_capacity(n);
    let mut velocity_x = Vec::with_capacity(n);
    let mut velocity_y = Vec::with_capacity(n);
    let mut velocity_z = Vec::with_capacity(n);
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let cell = mesh.cell_index(i, j, k);
                let grad =
                    pressure_correction_gradient(mesh, pressure_correction, i, j, k, spacing);
                let d = d_coefficient[cell];
                pressure.push(current.pressure.values()[cell] + pressure_correction[cell]);
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
struct CaseCartesianSpacing {
    dx: Real,
    dy: Real,
    dz: Real,
}

impl CaseCartesianSpacing {
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
    spacing: CaseCartesianSpacing,
) -> [Real; 3] {
    [
        (cell_value(mesh, pressure_correction, east(i, mesh.nx), j, k)
            - cell_value(mesh, pressure_correction, west(i), j, k))
            / (2.0 * spacing.dx),
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
