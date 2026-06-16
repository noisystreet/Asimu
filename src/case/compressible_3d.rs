//! 3D 可压缩 Euler / Navier-Stokes 算例编排（`[euler]` / `[navier_stokes]` + CGNS/结构化 3D 网格）。

use tracing::{debug_span, info, warn};

use crate::case::{CaseRunKind, CaseRunResult};
use crate::core::{ComputePrecision, Real, format_log_fixed4, format_log_sci4, log10_positive};
use crate::error::{AsimuError, Result};
use crate::io::{CaseSpec, CaseTimeMode};
use crate::mesh::MultiBlockStructuredMesh3d;
use crate::solver::{
    CompressibleEulerConfig, CompressibleEulerSolver, CompressibleStepInfo, CompressibleTimeMode,
    MultiblockStructuredDriverInput, RungeKutta4Config, StructuredComputeBackend,
    run_multiblock_structured_typed_with_observer, run_multiblock_structured_with_observer,
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
    /// 末物理步双时间步内层迭代次数（非 dual_time 为 0）。
    pub inner_iterations: u32,
}

pub fn run(case: &CaseSpec) -> Result<CaseRunResult> {
    let mesh = case.mesh.as_multiblock_3d()?;
    match case.numerics.compute_precision {
        ComputePrecision::F64 => run_compressible_3d(case, mesh),
        ComputePrecision::F32 => run_compressible_3d_typed::<f32>(case, mesh),
    }
}

fn run_compressible_3d(
    case: &CaseSpec,
    mesh: &MultiBlockStructuredMesh3d,
) -> Result<CaseRunResult> {
    let (eos, freestream, solver, scheme, limiter, time_mode, local_time_step) = {
        let _span = debug_span!("prepare_compressible_solver").entered();
        case.validate_multiblock_compressible()?;
        let disc = case.compressible_discretization()?;
        let eos = case.physics.eos()?;
        let freestream = case
            .freestream
            .or(case.fluid_initial.freestream)
            .ok_or_else(|| AsimuError::Field("3D 可压缩算例须指定 [freestream]".to_string()))?;
        let inviscid = disc.inviscid();
        let solver = build_compressible_solver(case, &inviscid)?;
        let scheme = inviscid.short_label().to_string();
        let limiter = inviscid.limiter_label().to_string();
        let time_mode = solver_time_mode(case.time.mode);
        let local_time_step = case.time.uses_local_time_step();
        (
            eos,
            freestream,
            solver,
            scheme,
            limiter,
            time_mode,
            local_time_step,
        )
    };
    if !mesh.interfaces().is_empty() {
        warn!(
            blocks = mesh.num_blocks(),
            interfaces = mesh.interfaces().len(),
            "多块 3D 求解按 block 同步推进，1-to-1 接口使用共享无粘通量守恒装配"
        );
    }
    let initial_fields = {
        let _span = debug_span!(
            "load_block_initial_fields",
            blocks = mesh.num_blocks(),
            restart = case.restart.is_some()
        )
        .entered();
        case.build_multiblock_conserved_fields(mesh.blocks())?
    };
    log_run_start(
        mesh,
        &scheme,
        &limiter,
        time_mode,
        local_time_step,
        case.resolved_max_steps(),
        None,
    );
    let mut snapshot_paths = Vec::new();
    let (history, fields) = run_multiblock_structured_with_observer(
        MultiblockStructuredDriverInput {
            solver: &solver,
            eos: &eos,
            freestream: &freestream,
            mesh,
            global_boundary: &case.boundary,
            reference: case.reference.as_ref(),
            residual_tolerance: super::validate::residual_tolerance(case),
            initial_fields,
        },
        |step| {
            snapshot_paths.extend(
                super::output_interval::maybe_write_compressible_structured_interval(
                    case, mesh, step,
                )?,
            );
            Ok(())
        },
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
    let output_paths = {
        let _span = debug_span!("write_compressible_outputs").entered();
        super::output_3d::write_compressible_3d_outputs(case, mesh, &fields, &history)?
    };
    log_written_paths(&snapshot_paths, &output_paths);
    Ok(build_case_run_result(
        case,
        mesh.num_cells(),
        &metrics,
        &scheme,
        &limiter,
        time_mode,
        local_time_step,
    ))
}

fn run_compressible_3d_typed<T: StructuredComputeBackend>(
    case: &CaseSpec,
    mesh: &MultiBlockStructuredMesh3d,
) -> Result<CaseRunResult> {
    let (eos, freestream, solver, scheme, limiter, time_mode, local_time_step) = {
        let _span = debug_span!("prepare_compressible_solver").entered();
        case.validate_multiblock_compressible()?;
        let disc = case.compressible_discretization()?;
        let eos = case.physics.eos()?;
        let freestream = case
            .freestream
            .or(case.fluid_initial.freestream)
            .ok_or_else(|| AsimuError::Field("3D 可压缩算例须指定 [freestream]".to_string()))?;
        let inviscid = disc.inviscid();
        let solver = build_compressible_solver(case, &inviscid)?;
        let scheme = inviscid.short_label().to_string();
        let limiter = inviscid.limiter_label().to_string();
        let time_mode = solver_time_mode(case.time.mode);
        let local_time_step = case.time.uses_local_time_step();
        (
            eos,
            freestream,
            solver,
            scheme,
            limiter,
            time_mode,
            local_time_step,
        )
    };
    let initial_fields = {
        let _span = debug_span!(
            "load_block_initial_fields",
            blocks = mesh.num_blocks(),
            restart = case.restart.is_some()
        )
        .entered();
        case.build_multiblock_conserved_fields(mesh.blocks())?
    };
    log_run_start(
        mesh,
        &scheme,
        &limiter,
        time_mode,
        local_time_step,
        case.resolved_max_steps(),
        Some(T::PRECISION.label()),
    );
    let mut snapshot_paths = Vec::new();
    let (history, fields) = run_multiblock_structured_typed_with_observer::<T>(
        MultiblockStructuredDriverInput {
            solver: &solver,
            eos: &eos,
            freestream: &freestream,
            mesh,
            global_boundary: &case.boundary,
            reference: case.reference.as_ref(),
            residual_tolerance: super::validate::residual_tolerance(case),
            initial_fields,
        },
        |step| {
            snapshot_paths.extend(
                super::output_interval::maybe_write_compressible_structured_interval(
                    case, mesh, step,
                )?,
            );
            Ok(())
        },
    )?;
    let last = history
        .last()
        .ok_or_else(|| AsimuError::Solver("3D 可压缩 typed 推进未产生任何时间步".to_string()))?;
    let metrics = build_run_metrics(last, &scheme, &limiter);
    log_run_complete(
        &metrics,
        &scheme,
        &limiter,
        time_mode,
        local_time_step,
        mesh.num_cells(),
    );
    let output_paths = {
        let _span = debug_span!("write_compressible_outputs").entered();
        super::output_3d::write_compressible_3d_outputs(case, mesh, &fields, &history)?
    };
    log_written_paths(&snapshot_paths, &output_paths);
    Ok(build_case_run_result(
        case,
        mesh.num_cells(),
        &metrics,
        &scheme,
        &limiter,
        time_mode,
        local_time_step,
    ))
}

fn log_run_start(
    mesh: &MultiBlockStructuredMesh3d,
    scheme: &str,
    limiter: &str,
    time_mode: CompressibleTimeMode,
    local_time_step: bool,
    max_steps: u64,
    precision: Option<&str>,
) {
    match precision {
        Some(precision) => info!(
            blocks = mesh.num_blocks(),
            interfaces = mesh.interfaces().len(),
            cells = mesh.num_cells(),
            max_steps,
            scheme,
            limiter,
            local_time_step,
            precision,
            "开始 3D 可压缩 {}求解",
            time_mode_label(time_mode),
        ),
        None => info!(
            blocks = mesh.num_blocks(),
            interfaces = mesh.interfaces().len(),
            cells = mesh.num_cells(),
            max_steps,
            scheme,
            limiter,
            local_time_step,
            "开始 3D 可压缩 {}求解",
            time_mode_label(time_mode),
        ),
    }
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
    num_cells: usize,
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
            num_cells
        ),
        diffusion: None,
        sod: None,
        compressible_3d: Some(metrics.clone()),
        incompressible_3d: None,
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
        gmres: case.time.resolved_gmres_config(),
        residual_smoothing: case.time.residual_smoothing_config(),
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
        inner_iterations: last.inner_iterations,
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

fn time_mode_label(mode: CompressibleTimeMode) -> &'static str {
    match mode {
        CompressibleTimeMode::Steady => "稳态",
        CompressibleTimeMode::Transient => "瞬态",
    }
}
