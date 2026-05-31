//! 3D 可压缩 Euler / Navier-Stokes 算例编排（`[euler]` / `[navier_stokes]` + CGNS/结构化 3D 网格）。

use tracing::info;

use crate::case::{CaseRunKind, CaseRunResult};
use crate::core::{Real, format_log_fixed4, format_log_sci4, log10_positive, residual_converged};
use crate::discretization::{BoundaryGhostBuffer, GradientFields};
use crate::error::{AsimuError, Result};
use crate::field::PrimitiveFields;
use crate::io::{CaseSpec, CaseTimeMode};
use crate::solver::{
    CompressibleAdvanceContext3d, CompressibleEulerConfig, CompressibleEulerSolver,
    CompressibleStepInfo, CompressibleTimeMode, Rk4Storage, RungeKutta4Config,
    RungeKutta4Integrator, SolverState,
};

/// 3D 可压缩 Euler 运行指标。
#[derive(Debug, Clone, PartialEq)]
pub struct Compressible3dRunMetrics {
    pub steps: u64,
    pub final_time: Real,
    pub residual_rms: Real,
    pub residual_log10: Real,
    pub scheme: String,
    pub limiter: String,
    pub converged: bool,
}

pub fn run(case: &CaseSpec) -> Result<CaseRunResult> {
    let mesh = case.mesh.as_3d()?;
    let disc = case.compressible_discretization()?;
    let eos = case.physics.eos()?;
    let freestream = case
        .freestream
        .or(case.fluid_initial.freestream)
        .ok_or_else(|| AsimuError::Field("3D 可压缩算例须指定 [freestream]".to_string()))?;
    let inviscid = disc.inviscid();
    let solver = build_compressible_solver(case, &inviscid)?;
    let mut fields = case.build_conserved_fields()?;
    let mut ghosts = BoundaryGhostBuffer::new();
    let mut ctx = CompressibleAdvanceContext3d {
        mesh,
        structured: mesh,
        patches: &case.boundary,
        ghosts: &mut ghosts,
        eos: &eos,
        freestream: &freestream,
        primitive_scratch: PrimitiveFields::zeros(mesh.num_cells())?,
        gradient_scratch: GradientFields::zeros(mesh.num_cells())?,
        viscous: solver.config.viscous.as_ref(),
    };
    let scheme = inviscid.short_label().to_string();
    let limiter = inviscid.limiter_label().to_string();
    let time_mode = solver_time_mode(case.time.mode);
    let local_time_step = case.time.uses_local_time_step();
    let interval_flow = case
        .output
        .as_ref()
        .is_some_and(|o| o.wants_interval_flow());
    let mut snapshot_paths: Vec<std::path::PathBuf> = Vec::new();
    let history = run_transient_3d_with_snapshots(
        &solver,
        time_mode,
        local_time_step,
        &mut ctx,
        &mut fields,
        case.resolved_tolerance(),
        interval_flow.then_some(SnapshotWriter {
            case,
            mesh,
            eos: &eos,
            paths: &mut snapshot_paths,
        }),
    )?;
    let last = history
        .last()
        .ok_or_else(|| AsimuError::Solver("3D 可压缩推进未产生任何时间步".to_string()))?;
    let metrics = build_run_metrics(last, &scheme, &limiter);
    log_run_complete(
        &metrics,
        &scheme,
        &limiter,
        time_mode,
        local_time_step,
        mesh.num_cells(),
    );
    let output_paths =
        super::output_3d::write_compressible_3d_outputs(case, mesh, &fields, &eos, &history)?;
    log_written_paths(&snapshot_paths, &output_paths);
    Ok(build_case_run_result(
        case,
        mesh,
        &metrics,
        &scheme,
        &limiter,
        time_mode,
        local_time_step,
    ))
}

fn log_written_paths(snapshot_paths: &[std::path::PathBuf], output_paths: &[std::path::PathBuf]) {
    for path in snapshot_paths {
        info!(path = %path.display(), "算例间隔流场输出");
    }
    for path in output_paths {
        info!(path = %path.display(), "算例输出");
    }
}

fn build_case_run_result(
    case: &CaseSpec,
    mesh: &crate::mesh::StructuredMesh3d,
    metrics: &Compressible3dRunMetrics,
    scheme: &str,
    limiter: &str,
    time_mode: CompressibleTimeMode,
    local_time_step: bool,
) -> CaseRunResult {
    CaseRunResult {
        name: case.name.clone(),
        benchmark_id: case.benchmark_id.clone(),
        kind: CaseRunKind::Compressible3dTransient,
        summary: format!(
            "3D Euler {}/{} {} t={} log10={} steps={} converged={} lts={} cells={}",
            limiter,
            scheme,
            time_mode_label(time_mode),
            format_log_sci4(metrics.final_time),
            format_log_fixed4(metrics.residual_log10),
            metrics.steps,
            metrics.converged,
            local_time_step,
            mesh.num_cells()
        ),
        diffusion: None,
        sod: None,
        compressible_3d: Some(metrics.clone()),
    }
}

fn build_compressible_solver(
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
        time_mode: solver_time_mode(case.time.mode),
        local_time_step: case.time.uses_local_time_step(),
        time_scheme: case.time.resolved_time_scheme(),
        lu_sgs: case.time.resolved_lusgs_config()?,
    }))
}

fn solver_time_mode(mode: CaseTimeMode) -> CompressibleTimeMode {
    match mode {
        CaseTimeMode::Steady => CompressibleTimeMode::Steady,
        CaseTimeMode::Transient => CompressibleTimeMode::Transient,
    }
}

fn build_run_metrics(
    last: &CompressibleStepInfo,
    scheme: &str,
    limiter: &str,
) -> Compressible3dRunMetrics {
    Compressible3dRunMetrics {
        steps: last.step,
        final_time: last.physical_time,
        residual_rms: last.residual_rms,
        residual_log10: log10_positive(last.residual_rms),
        scheme: scheme.to_string(),
        limiter: limiter.to_string(),
        converged: last.converged,
    }
}

fn log_run_complete(
    metrics: &Compressible3dRunMetrics,
    scheme: &str,
    limiter: &str,
    time_mode: CompressibleTimeMode,
    local_time_step: bool,
    cells: usize,
) {
    info!(
        steps = metrics.steps,
        t = %format_log_sci4(metrics.final_time),
        log10_residual = %format_log_fixed4(metrics.residual_log10),
        converged = metrics.converged,
        scheme = %scheme,
        limiter = %limiter,
        local_time_step,
        cells,
        "3D 可压缩 Euler {}求解完成",
        time_mode_label(time_mode),
    );
}

struct SnapshotWriter<'a> {
    case: &'a CaseSpec,
    mesh: &'a crate::mesh::StructuredMesh3d,
    eos: &'a crate::physics::IdealGasEoS,
    paths: &'a mut Vec<std::path::PathBuf>,
}

fn run_transient_3d_with_snapshots(
    solver: &CompressibleEulerSolver,
    _time_mode: CompressibleTimeMode,
    _local_time_step: bool,
    ctx: &mut CompressibleAdvanceContext3d<'_>,
    fields: &mut crate::field::ConservedFields,
    tolerance: Option<Real>,
    mut snapshot: Option<SnapshotWriter<'_>>,
) -> Result<Vec<CompressibleStepInfo>> {
    let mut storage = Rk4Storage::new(ctx.structured.num_cells())?;
    let mut state = SolverState::default();
    let mut integrator = RungeKutta4Integrator::new(solver.config.time);
    let mut history = Vec::new();
    loop {
        let mut step_info =
            solver.advance_step_3d(ctx, fields, &mut storage, &mut state, &mut integrator)?;
        let converged =
            tolerance.is_some_and(|tol| residual_converged(step_info.residual_log10, tol));
        step_info.converged = converged;
        let stop = step_info.is_final || converged;
        info!(
            step = step_info.step,
            dt = %format_log_sci4(step_info.dt),
            t = %format_log_sci4(step_info.physical_time),
            log10_residual = %format_log_fixed4(step_info.residual_log10),
            cfl = step_info.cfl,
            is_final = stop,
            converged,
            "时间步"
        );
        if converged {
            info!(
                step = step_info.step,
                tolerance = ?tolerance,
                log10_residual = %format_log_fixed4(step_info.residual_log10),
                "log₁₀ 残差已达 [time].tolerance，提前停止"
            );
        }
        if let Some(ref mut writer) = snapshot {
            if let Some(path) = super::output_3d::maybe_write_flow_snapshot(
                writer.case,
                writer.mesh,
                fields,
                writer.eos,
                &step_info,
            )? {
                writer.paths.push(path);
            }
        }
        history.push(step_info);
        if stop {
            break;
        }
    }
    Ok(history)
}

fn time_mode_label(mode: CompressibleTimeMode) -> &'static str {
    match mode {
        CompressibleTimeMode::Steady => "稳态",
        CompressibleTimeMode::Transient => "瞬态",
    }
}
