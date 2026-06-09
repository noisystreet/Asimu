//! 3D 可压缩 Euler / Navier-Stokes 算例编排（`[euler]` / `[navier_stokes]` + CGNS/结构化 3D 网格）。

#[path = "compressible_3d_interface_flux.rs"]
mod compressible_3d_interface_flux;
#[path = "compressible_3d_multiblock.rs"]
mod compressible_3d_multiblock;

use std::{cell::RefCell, rc::Rc};

use compressible_3d_interface_flux::{
    InterfaceResidualContribution, SharedInterfaceResidualParams, apply_interface_residuals,
    compute_shared_interface_residuals,
};
use compressible_3d_multiblock::{
    BlockInterfaceLink, SharedInterfaceFace, build_multiblock_interface_metadata,
};

use tracing::{info, info_span, warn};

use crate::boundary::{BoundaryPatch, BoundarySet};
use crate::case::{CaseRunKind, CaseRunResult};
use crate::core::{Real, format_log_fixed4, format_log_sci4, log10_positive, residual_converged};
use crate::discretization::{BoundaryGhostBuffer, GhostCellState, GradientFields};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, PrimitiveFields};
use crate::io::{CaseSpec, CaseTimeMode};
use crate::mesh::{MultiBlockStructuredMesh3d, StructuredBlock3d};
use crate::physics::{FreestreamParams, IdealGasEoS};
use crate::solver::compressible::ResidualCorrection3d;
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
    let mesh = case.mesh.as_multiblock_3d()?;
    run_compressible_3d(case, mesh)
}

fn run_compressible_3d(
    case: &CaseSpec,
    mesh: &MultiBlockStructuredMesh3d,
) -> Result<CaseRunResult> {
    let _span = info_span!(
        "run_compressible_3d",
        blocks = mesh.num_blocks(),
        interfaces = mesh.interfaces().len(),
        cells = mesh.num_cells()
    )
    .entered();
    let (eos, freestream, solver, scheme, limiter, time_mode, local_time_step) = {
        let _span = info_span!("prepare_compressible_solver").entered();
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
    let interfaces = {
        let _span = info_span!("build_block_interface_metadata").entered();
        build_multiblock_interface_metadata(mesh)?
    };
    let mut states = {
        let _span = info_span!("build_block_run_states", blocks = mesh.num_blocks()).entered();
        build_block_run_states(case, mesh.blocks(), &interfaces.patches, solver.config.time)?
    };
    let mut snapshot_paths = Vec::new();
    let advance = BlockAdvanceEnv {
        case,
        solver: &solver,
        eos: &eos,
        freestream: &freestream,
        mesh,
        links: &interfaces.links,
        shared_faces: &interfaces.shared_faces,
    };
    let history = {
        let _span = info_span!("advance_block_history").entered();
        advance_block_history(&advance, &mut states, &mut snapshot_paths)?
    };

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
    let fields: Vec<ConservedFields> = {
        let _span = info_span!("collect_output_fields", blocks = states.len()).entered();
        states.iter().map(|state| state.fields.clone()).collect()
    };
    let output_paths = {
        let _span = info_span!("write_compressible_outputs").entered();
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

fn boundary_for_block(boundary: &BoundarySet, block_name: &str) -> BoundarySet {
    let prefix = format!("{block_name}/");
    let patches = boundary
        .patches()
        .iter()
        .filter_map(|patch| {
            let local_name = patch.name.strip_prefix(&prefix)?;
            Some(BoundaryPatch::new(
                local_name.to_string(),
                patch.face_ids.clone(),
                patch.kind.clone(),
            ))
        })
        .collect();
    BoundarySet::new(patches)
}

fn resolve_block_boundary(
    boundary: &BoundarySet,
    block: &StructuredBlock3d,
    num_blocks: usize,
) -> BoundarySet {
    let block_boundary = boundary_for_block(boundary, &block.name);
    if num_blocks == 1 && block_boundary.patches().is_empty() {
        return boundary.clone();
    }
    block_boundary
}

struct BlockRunState {
    fields: ConservedFields,
    ghosts: BoundaryGhostBuffer,
    boundary: BoundarySet,
    storage: Rk4Storage,
    solver_state: SolverState,
    integrator: RungeKutta4Integrator,
}

struct BlockAdvanceEnv<'a> {
    case: &'a CaseSpec,
    solver: &'a CompressibleEulerSolver,
    eos: &'a IdealGasEoS,
    freestream: &'a FreestreamParams,
    mesh: &'a MultiBlockStructuredMesh3d,
    links: &'a [Vec<BlockInterfaceLink>],
    shared_faces: &'a [SharedInterfaceFace],
}

struct InterfaceResidualCorrection {
    contributions: Vec<InterfaceResidualContribution>,
}

impl ResidualCorrection3d for InterfaceResidualCorrection {
    fn apply(&mut self, residual: &mut crate::field::ConservedResidual) -> Result<()> {
        apply_interface_residuals(residual, &self.contributions)
    }
}

fn build_block_run_states(
    case: &CaseSpec,
    blocks: &[StructuredBlock3d],
    interface_patches: &[Vec<BoundaryPatch>],
    time_config: RungeKutta4Config,
) -> Result<Vec<BlockRunState>> {
    let block_fields = {
        let _span = info_span!(
            "load_block_initial_fields",
            blocks = blocks.len(),
            restart = case.restart.is_some()
        )
        .entered();
        case.build_multiblock_conserved_fields(blocks)?
    };
    let mut states = Vec::with_capacity(blocks.len());
    for (index, (block, fields)) in blocks.iter().zip(block_fields).enumerate() {
        let _span = info_span!(
            "build_block_state",
            block = %block.name,
            cells = block.mesh.num_cells(),
            interfaces = interface_patches[index].len()
        )
        .entered();
        let mut patches = resolve_block_boundary(&case.boundary, block, blocks.len())
            .patches()
            .to_vec();
        patches.extend(interface_patches[index].iter().cloned());
        let boundary = BoundarySet::new(patches);
        info!(
            block = %block.name,
            cells = block.mesh.num_cells(),
            patches = boundary.patches().len(),
            interfaces = interface_patches[index].len(),
            "初始化 3D 可压缩 block"
        );
        states.push(BlockRunState {
            fields,
            ghosts: BoundaryGhostBuffer::new(),
            boundary,
            storage: Rk4Storage::new(block.mesh.num_cells())?,
            solver_state: SolverState::default(),
            integrator: RungeKutta4Integrator::new(time_config),
        });
    }
    Ok(states)
}

fn advance_block_history(
    env: &BlockAdvanceEnv<'_>,
    states: &mut [BlockRunState],
    snapshot_paths: &mut Vec<std::path::PathBuf>,
) -> Result<Vec<CompressibleStepInfo>> {
    let mut history = Vec::new();
    loop {
        let aggregate = advance_block_step(env, states)?;
        let stop = aggregate.is_final || aggregate.converged;
        log_block_step(env.mesh.num_blocks(), &aggregate, stop, aggregate.converged);
        history.push(aggregate);
        maybe_write_interval_snapshot(
            env.case,
            env.mesh,
            states,
            history.last().expect("history"),
            &history,
            snapshot_paths,
        )?;
        if stop {
            break;
        }
    }
    Ok(history)
}

fn maybe_write_interval_snapshot(
    case: &CaseSpec,
    mesh: &MultiBlockStructuredMesh3d,
    states: &[BlockRunState],
    step: &CompressibleStepInfo,
    history: &[CompressibleStepInfo],
    paths: &mut Vec<std::path::PathBuf>,
) -> Result<()> {
    let _span = info_span!("maybe_write_interval_snapshot", step = step.step).entered();
    if !super::output_3d::interval_output_due(case, step.step) {
        return Ok(());
    }
    let fields: Vec<ConservedFields> = {
        let _span = info_span!("collect_snapshot_fields", blocks = states.len()).entered();
        states.iter().map(|state| state.fields.clone()).collect()
    };
    if let Some(path) =
        super::output_3d::maybe_write_interval_flow_snapshot(case, mesh, &fields, step)?
    {
        paths.push(path);
    }
    let _ = super::output_3d::maybe_write_residual_outputs(case, history, step)?;
    Ok(())
}

fn advance_block_step(
    env: &BlockAdvanceEnv<'_>,
    states: &mut [BlockRunState],
) -> Result<CompressibleStepInfo> {
    let _span = info_span!("advance_block_step").entered();
    let snapshots: Vec<ConservedFields> = {
        let _span = info_span!("clone_block_snapshots", blocks = states.len()).entered();
        states.iter().map(|state| state.fields.clone()).collect()
    };
    let interface_residuals = if env.shared_faces.is_empty() {
        vec![Vec::new(); states.len()]
    } else {
        let _span = info_span!(
            "compute_shared_interface_residuals",
            blocks = env.mesh.num_blocks()
        )
        .entered();
        compute_shared_interface_residuals(&SharedInterfaceResidualParams {
            blocks: env.mesh.blocks(),
            shared_faces: env.shared_faces,
            snapshots: &snapshots,
            eos: env.eos,
            freestream: env.freestream,
            inviscid: &env.solver.config.inviscid,
        })?
    };
    let mut step_infos = Vec::with_capacity(states.len());
    for (block_index, block) in env.mesh.blocks().iter().enumerate() {
        if !env.links[block_index].is_empty() {
            let _span = info_span!(
                "fill_interface_ghosts",
                block = %block.name,
                links = env.links[block_index].len()
            )
            .entered();
            fill_interface_ghosts(
                &mut states[block_index].ghosts,
                &env.links[block_index],
                &snapshots,
            )?;
        }
        step_infos.push(advance_single_block_step(
            env.case,
            env.solver,
            env.eos,
            env.freestream,
            block,
            &mut states[block_index],
            &interface_residuals[block_index],
        )?);
    }
    let mut aggregate = {
        let _span = info_span!("aggregate_block_step", blocks = step_infos.len()).entered();
        aggregate_block_step(&step_infos)?
    };
    aggregate.converged = env
        .case
        .resolved_tolerance()
        .is_some_and(|tol| residual_converged(aggregate.residual_log10, tol));
    Ok(aggregate)
}

fn advance_single_block_step(
    case: &CaseSpec,
    solver: &CompressibleEulerSolver,
    eos: &IdealGasEoS,
    freestream: &FreestreamParams,
    block: &StructuredBlock3d,
    state: &mut BlockRunState,
    interface_residual: &[InterfaceResidualContribution],
) -> Result<CompressibleStepInfo> {
    let _span = info_span!(
        "advance_single_block_step",
        block = %block.name,
        cells = block.mesh.num_cells(),
        interface_contributions = interface_residual.len()
    )
    .entered();
    let residual_correction: Option<Rc<RefCell<dyn ResidualCorrection3d>>> =
        if interface_residual.is_empty() {
            None
        } else {
            Some(Rc::new(RefCell::new(InterfaceResidualCorrection {
                contributions: interface_residual.to_vec(),
            })))
        };
    let mut ctx = CompressibleAdvanceContext3d {
        mesh: &block.mesh,
        structured: &block.mesh,
        patches: &state.boundary,
        ghosts: &mut state.ghosts,
        eos,
        freestream,
        reference: case.reference.as_ref(),
        primitive_scratch: PrimitiveFields::zeros(block.mesh.num_cells())?,
        gradient_scratch: GradientFields::zeros(block.mesh.num_cells())?,
        viscous: solver.config.viscous.as_ref(),
        residual_correction,
    };
    let mut step_info = solver.advance_step_3d(
        &mut ctx,
        &mut state.fields,
        &mut state.storage,
        &mut state.solver_state,
        &mut state.integrator,
    )?;
    step_info.converged = case
        .resolved_tolerance()
        .is_some_and(|tol| residual_converged(step_info.residual_log10, tol));
    Ok(step_info)
}

fn fill_interface_ghosts(
    ghosts: &mut BoundaryGhostBuffer,
    links: &[BlockInterfaceLink],
    snapshots: &[ConservedFields],
) -> Result<()> {
    for link in links {
        let conserved = snapshots[link.donor_block_index].cell_state(link.donor_cell)?;
        ghosts.insert_face(link.face, GhostCellState { conserved });
    }
    Ok(())
}

fn log_block_step(blocks: usize, step_info: &CompressibleStepInfo, stop: bool, converged: bool) {
    let label = if blocks > 1 {
        "多块聚合时间步"
    } else {
        "时间步"
    };
    info!(
        step = step_info.step,
        dt = %format_log_sci4(step_info.dt),
        t = %format_log_sci4(step_info.physical_time),
        log10_residual = %format_log_fixed4(step_info.residual_log10),
        cfl = step_info.cfl,
        is_final = stop,
        converged,
        "{label}"
    );
}

fn aggregate_block_step(steps: &[CompressibleStepInfo]) -> Result<CompressibleStepInfo> {
    let first = steps
        .first()
        .ok_or_else(|| AsimuError::Solver("3D 求解没有 block 时间步".to_string()))?;
    if steps.len() == 1 {
        return Ok(first.clone());
    }
    let residual_rms = steps
        .iter()
        .map(|step| step.residual_rms)
        .fold(0.0, Real::max);
    Ok(CompressibleStepInfo {
        dt: steps.iter().map(|step| step.dt).fold(first.dt, Real::min),
        physical_time: steps
            .iter()
            .map(|step| step.physical_time)
            .fold(first.physical_time, Real::max),
        step: steps
            .iter()
            .map(|step| step.step)
            .max()
            .unwrap_or(first.step),
        residual_rms,
        residual_log10: log10_positive(residual_rms),
        cfl: first.cfl,
        is_final: steps.iter().all(|step| step.is_final),
        converged: steps.iter().all(|step| step.converged),
    })
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
