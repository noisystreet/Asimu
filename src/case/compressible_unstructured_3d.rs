//! 3D 非结构可压缩算例编排（单域混合单元面循环）。

use std::path::PathBuf;

use tracing::{debug_span, info};

use crate::case::{CaseRunKind, CaseRunResult, validate};
use crate::core::{ComputePrecision, Real, format_log_fixed4, format_log_sci4, log10_positive};
use crate::error::{AsimuError, Result};
use crate::exec::ExecConfig;
use crate::field::{ConservedFields, ConservedFieldsT};
use crate::io::{CaseSpec, resolve_case_output_path};
use crate::mesh::UnstructuredMesh3d;
use crate::solver::UnstructuredComputeBackend;
use crate::solver::{
    CompressibleEulerConfig, CompressibleEulerSolver, CompressibleStepInfo, CompressibleTimeMode,
    RungeKutta4Config, UnstructuredDriverConfig, run_unstructured_typed_with_observer,
};

use super::Compressible3dRunMetrics;

pub(super) fn run(case: &CaseSpec) -> Result<CaseRunResult> {
    let mesh = case.mesh.as_unstructured_3d()?;
    match case.numerics.compute_precision {
        ComputePrecision::F64 => run_compressible_unstructured_3d_typed::<f64>(case, mesh),
        ComputePrecision::F32 => run_compressible_unstructured_3d_typed::<f32>(case, mesh),
    }
}

struct UnstructuredPreparedRun {
    inviscid: crate::discretization::InviscidFluxConfig,
    solver: CompressibleEulerSolver,
    eos: crate::physics::IdealGasEoS,
    freestream: crate::physics::FreestreamParams,
    fields: ConservedFields,
    driver_time: UnstructuredDriverTimeConfig,
}

struct UnstructuredDriverTimeConfig {
    fixed_dt: Option<Real>,
    local_time_step: bool,
    time_scheme: crate::solver::TimeIntegrationScheme,
    lu_sgs: crate::solver::LuSgsConfig,
    cfl_schedule: crate::solver::CflSchedule,
    max_steps: u64,
    residual_tolerance: Option<Real>,
}

fn prepare_unstructured_run(
    case: &CaseSpec,
    mesh: &UnstructuredMesh3d,
) -> Result<UnstructuredPreparedRun> {
    let disc = case.compressible_discretization()?;
    let inviscid = disc.inviscid();
    validate::unstructured_compressible(case)?;
    validate::unstructured_boundary_coverage(mesh, &case.boundary)?;
    let eos = case.physics.eos()?;
    let freestream = case
        .freestream
        .or(case.fluid_initial.freestream)
        .ok_or_else(|| AsimuError::Field("3D 可压缩算例须指定 [freestream]".to_string()))?;
    let solver = build_compressible_solver(case, &inviscid)?;
    let fields = case.build_conserved_fields()?;
    Ok(UnstructuredPreparedRun {
        inviscid,
        solver,
        eos,
        freestream,
        fields,
        driver_time: UnstructuredDriverTimeConfig {
            fixed_dt: case.time.dt,
            local_time_step: case.time.uses_local_time_step(),
            time_scheme: case.time.resolved_time_scheme(),
            lu_sgs: case.time.resolved_lusgs_config()?,
            cfl_schedule: case.cfl_schedule()?,
            max_steps: case.resolved_max_steps(),
            residual_tolerance: validate::residual_tolerance(case),
        },
    })
}

fn run_compressible_unstructured_3d_typed<T: UnstructuredComputeBackend>(
    case: &CaseSpec,
    mesh: &UnstructuredMesh3d,
) -> Result<CaseRunResult> {
    let prepared = {
        let _span = debug_span!("prepare_unstructured_solver").entered();
        prepare_unstructured_run(case, mesh)?
    };
    let UnstructuredPreparedRun {
        inviscid,
        solver,
        eos,
        freestream,
        fields,
        driver_time,
    } = prepared;
    let equation_label = unstructured_equation_label(case);
    log_unstructured_start(
        mesh,
        &inviscid,
        &driver_time,
        equation_label,
        Some(T::PRECISION.label()),
    );
    let driver = UnstructuredDriverConfig {
        solver: &solver,
        mesh,
        eos: &eos,
        freestream: &freestream,
        inviscid: &inviscid,
        patches: &case.boundary,
        reference: case.reference.as_ref(),
        viscous: case.physics.viscous.as_ref(),
        fixed_dt: driver_time.fixed_dt,
        local_time_step: driver_time.local_time_step,
        time_scheme: driver_time.time_scheme,
        lu_sgs: driver_time.lu_sgs,
        cfl_schedule: driver_time.cfl_schedule,
        max_steps: driver_time.max_steps,
        residual_tolerance: driver_time.residual_tolerance,
        exec_config: ExecConfig::from_numerics(&case.numerics),
        observer_field_sync_interval: case.output.as_ref().and_then(|o| o.solution_every),
    };
    let mut fields_t = ConservedFieldsT::<T>::from_real_fields(&fields)?;
    let mut interval_paths = Vec::new();
    let (history, fields) =
        run_unstructured_typed_with_observer::<T>(&driver, &mut fields_t, |step| {
            interval_paths.extend(
                super::output_interval::maybe_write_compressible_unstructured_interval(
                    case, mesh, step,
                )?,
            );
            Ok(())
        })?;
    let last = history
        .last()
        .ok_or_else(|| AsimuError::Solver("非结构 typed 推进未产生任何时间步".to_string()))?;
    let metrics = Compressible3dRunMetrics {
        steps: last.step,
        final_time: last.physical_time,
        residual_rms: last.residual_rms,
        residual_log10: log10_positive(last.residual_rms),
        scheme: inviscid.short_label().to_string(),
        limiter: inviscid.limiter_label().to_string(),
        converged: last.converged,
    };
    log_unstructured_complete(
        &metrics,
        inviscid.short_label(),
        inviscid.limiter_label(),
        mesh,
        equation_label,
    );
    let output_paths = write_unstructured_outputs(case, mesh, &fields, &history)?;
    for path in &interval_paths {
        info!(path = %path.display(), "算例间隔流场输出");
    }
    for path in output_paths {
        info!(path = %path.display(), "非结构算例输出");
    }
    Ok(build_unstructured_case_result(
        case,
        mesh,
        metrics,
        inviscid.short_label(),
        equation_label,
    ))
}

fn build_unstructured_case_result(
    case: &CaseSpec,
    mesh: &UnstructuredMesh3d,
    metrics: Compressible3dRunMetrics,
    scheme: &str,
    equation_label: &str,
) -> CaseRunResult {
    CaseRunResult {
        name: case.name.clone(),
        benchmark_id: case.benchmark_id.clone(),
        kind: CaseRunKind::Compressible3dTransient,
        summary: format!(
            "3D unstructured {} {} t={} log10={} steps={} converged={} cells={}",
            equation_label,
            scheme,
            format_log_sci4(metrics.final_time),
            format_log_fixed4(metrics.residual_log10),
            metrics.steps,
            metrics.converged,
            mesh.num_cells()
        ),
        diffusion: None,
        sod: None,
        compressible_3d: Some(metrics),
        incompressible_3d: None,
    }
}

pub(crate) fn build_compressible_solver(
    case: &CaseSpec,
    inviscid: &crate::discretization::InviscidFluxConfig,
) -> Result<CompressibleEulerSolver> {
    Ok(CompressibleEulerSolver::new(CompressibleEulerConfig {
        time: RungeKutta4Config {
            dt: case.time.dt.unwrap_or(0.0),
            max_steps: case.resolved_max_steps(),
        },
        inviscid: *inviscid,
        viscous: case.physics.viscous.clone(),
        cfl_schedule: case.cfl_schedule()?,
        time_mode: match case.time.mode {
            crate::io::CaseTimeMode::Steady => CompressibleTimeMode::Steady,
            crate::io::CaseTimeMode::Transient => CompressibleTimeMode::Transient,
        },
        local_time_step: case.time.uses_local_time_step(),
        time_scheme: case.time.resolved_time_scheme(),
        lu_sgs: case.time.resolved_lusgs_config()?,
        gmres: case.time.resolved_gmres_config(),
        residual_smoothing: case.time.residual_smoothing_config(),
    }))
}

fn log_unstructured_start(
    mesh: &UnstructuredMesh3d,
    inviscid: &crate::discretization::InviscidFluxConfig,
    driver_time: &UnstructuredDriverTimeConfig,
    equation_label: &str,
    precision: Option<&str>,
) {
    match precision {
        Some(precision) => info!(
            cells = mesh.num_cells(),
            faces = mesh.num_faces(),
            max_steps = driver_time.max_steps,
            scheme = inviscid.short_label(),
            limiter = inviscid.limiter_label(),
            time_scheme = driver_time.time_scheme.label(),
            equation = equation_label,
            precision,
            "开始非结构 3D 可压缩求解"
        ),
        None => info!(
            cells = mesh.num_cells(),
            faces = mesh.num_faces(),
            max_steps = driver_time.max_steps,
            scheme = inviscid.short_label(),
            limiter = inviscid.limiter_label(),
            time_scheme = driver_time.time_scheme.label(),
            equation = equation_label,
            "开始非结构 3D 可压缩求解"
        ),
    }
}

fn log_unstructured_complete(
    metrics: &Compressible3dRunMetrics,
    scheme: &str,
    limiter: &str,
    mesh: &UnstructuredMesh3d,
    equation_label: &str,
) {
    info!(
        steps = metrics.steps,
        t = %format_log_sci4(metrics.final_time),
        log10_residual = %format_log_fixed4(metrics.residual_log10),
        converged = metrics.converged,
        scheme,
        limiter,
        equation = equation_label,
        cells = mesh.num_cells(),
        faces = mesh.num_faces(),
        "非结构 3D 可压缩求解完成",
    );
}

fn unstructured_equation_label(case: &CaseSpec) -> &'static str {
    if case.physics.viscous.is_some() {
        "Navier-Stokes"
    } else {
        "Euler"
    }
}

pub(crate) fn write_unstructured_interval_flow(
    case: &CaseSpec,
    mesh: &UnstructuredMesh3d,
    fields: &ConservedFields,
    physical_time: Real,
    path: PathBuf,
) -> Result<()> {
    write_unstructured_flow_cgns(case, mesh, fields, physical_time, path)
}

fn write_unstructured_outputs(
    case: &CaseSpec,
    mesh: &UnstructuredMesh3d,
    fields: &ConservedFields,
    history: &[CompressibleStepInfo],
) -> Result<Vec<PathBuf>> {
    let mut written = super::output_3d::write_residual_outputs(case, history)?;
    let Some(output) = &case.output else {
        return Ok(written);
    };
    let Some(name) = &output.solution_cgns else {
        return Ok(written);
    };
    let cgns_path = resolve_case_output_path(case.case_dir.as_deref(), &output.dir, name)?;
    let physical_time = history.last().map(|s| s.physical_time).unwrap_or(0.0);
    write_unstructured_flow_cgns(case, mesh, fields, physical_time, cgns_path.clone())?;
    info!(
        path = %cgns_path.display(),
        cells = mesh.num_cells(),
        t = %format_log_sci4(physical_time),
        "已写出非结构流场 CGNS"
    );
    written.push(cgns_path.clone());
    #[cfg(feature = "io-vtk")]
    if output.solution_vtk {
        let vtu_path = cgns_path.with_extension("vtu");
        write_unstructured_flow_vtu(case, mesh, fields, physical_time, vtu_path.clone())?;
        written.push(vtu_path);
    }
    Ok(written)
}

fn write_unstructured_flow_cgns(
    case: &CaseSpec,
    mesh: &UnstructuredMesh3d,
    fields: &ConservedFields,
    physical_time: Real,
    path: PathBuf,
) -> Result<()> {
    let (fields_out, eos_out, _time_out, p_floor) =
        super::output_3d::prepare_dimensional_flow_output(case, fields, physical_time)?;
    #[cfg(feature = "io-cgns")]
    {
        crate::io::write_flow_cgns_unstructured(
            &path,
            mesh,
            &fields_out,
            &eos_out,
            physical_time,
            p_floor,
        )
    }
    #[cfg(not(feature = "io-cgns"))]
    {
        let _ = (mesh, fields_out, eos_out, p_floor, path);
        Err(AsimuError::Config(
            "非结构流场 CGNS 写出须启用 feature io-cgns".to_string(),
        ))
    }
}

fn write_unstructured_flow_vtu(
    case: &CaseSpec,
    mesh: &UnstructuredMesh3d,
    fields: &ConservedFields,
    physical_time: Real,
    path: PathBuf,
) -> Result<()> {
    let (fields_out, eos_out, _time_out, p_floor) =
        super::output_3d::prepare_dimensional_flow_output(case, fields, physical_time)?;
    #[cfg(feature = "io-vtk")]
    {
        crate::io::write_flow_vtu_unstructured(&path, mesh, &fields_out, &eos_out, p_floor)
    }
    #[cfg(not(feature = "io-vtk"))]
    {
        let _ = (mesh, fields_out, eos_out, p_floor, path);
        Err(AsimuError::Config(
            "非结构流场 VTU 写出须启用 feature io-vtk".to_string(),
        ))
    }
}

#[cfg(test)]
#[path = "compressible_unstructured_3d_lusgs_test.rs"]
mod lusgs_monitoring_tests;
