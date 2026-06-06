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
use crate::io::{CaseMesh, CaseSpec, CaseTimeMode};
use crate::mesh::{MultiBlockStructuredMesh3d, StructuredBlock3d, StructuredMesh3d};
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
    match &case.mesh {
        CaseMesh::Structured3d(mesh) => run_structured(case, mesh),
        CaseMesh::MultiBlockStructured3d(mesh) => run_multiblock(case, mesh),
        _ => Err(AsimuError::Mesh("3D 可压缩算例须使用 3D 网格".to_string())),
    }
}

fn run_structured(case: &CaseSpec, mesh: &StructuredMesh3d) -> Result<CaseRunResult> {
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
        reference: case.reference.as_ref(),
        primitive_scratch: PrimitiveFields::zeros(mesh.num_cells())?,
        gradient_scratch: GradientFields::zeros(mesh.num_cells())?,
        viscous: solver.config.viscous.as_ref(),
        residual_correction: None,
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
        super::output_3d::write_compressible_3d_outputs(case, mesh, &fields, &history)?;
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

fn run_multiblock(case: &CaseSpec, mesh: &MultiBlockStructuredMesh3d) -> Result<CaseRunResult> {
    let _span = info_span!(
        "run_multiblock_compressible_3d",
        blocks = mesh.num_blocks(),
        interfaces = mesh.interfaces().len(),
        cells = mesh.num_cells()
    )
    .entered();
    let (eos, freestream, solver, scheme, limiter, time_mode, local_time_step) = {
        let _span = info_span!("prepare_multiblock_solver").entered();
        validate_multiblock_case(case)?;
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
    warn_multiblock_limitations(mesh.num_blocks(), mesh.interfaces().len());
    let interfaces = {
        let _span = info_span!("build_multiblock_interface_metadata").entered();
        build_multiblock_interface_metadata(mesh)?
    };
    let mut states = {
        let _span = info_span!("build_multiblock_run_states", blocks = mesh.num_blocks()).entered();
        build_multiblock_run_states(case, mesh.blocks(), &interfaces.patches, solver.config.time)?
    };
    let mut snapshot_paths = Vec::new();
    let advance = MultiblockAdvanceEnv {
        case,
        solver: &solver,
        eos: &eos,
        freestream: &freestream,
        mesh,
        links: &interfaces.links,
        shared_faces: &interfaces.shared_faces,
    };
    let history = {
        let _span = info_span!("advance_multiblock_history").entered();
        advance_multiblock_history(&advance, &mut states, &mut snapshot_paths)?
    };

    let last = history
        .last()
        .ok_or_else(|| AsimuError::Solver("多块 3D 推进未产生任何时间步".to_string()))?;
    let metrics = build_run_metrics(last, &scheme, &limiter);
    log_run_complete(
        &metrics,
        &scheme,
        &limiter,
        time_mode,
        local_time_step,
        case.mesh.num_cells(),
    );
    let fields: Vec<ConservedFields> = {
        let _span = info_span!("collect_multiblock_output_fields", blocks = states.len()).entered();
        states.iter().map(|state| state.fields.clone()).collect()
    };
    let output_paths = {
        let _span = info_span!("write_multiblock_outputs").entered();
        super::output_3d::write_multiblock_compressible_3d_outputs(case, mesh, &fields, &history)?
    };
    log_written_paths(&snapshot_paths, &output_paths);
    Ok(build_case_run_result(
        case,
        case.mesh.num_cells(),
        &metrics,
        &scheme,
        &limiter,
        time_mode,
        local_time_step,
    ))
}

fn validate_multiblock_case(case: &CaseSpec) -> Result<()> {
    if case.time.resolved_time_scheme() != crate::solver::TimeIntegrationScheme::LuSgs {
        return Err(AsimuError::Config(
            "严格守恒多块 3D 求解当前要求 time.scheme = \"lu_sgs\"".to_string(),
        ));
    }
    if case.time.resolved_lusgs_config()?.sweep {
        return Err(AsimuError::Config(
            "严格守恒多块 3D 求解暂不支持 lusgs_sweep = true".to_string(),
        ));
    }
    Ok(())
}

fn warn_multiblock_limitations(blocks: usize, interfaces: usize) {
    warn!(
        blocks,
        interfaces, "多块 3D 求解按 block 同步推进，1-to-1 接口使用共享无粘通量守恒装配"
    );
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

struct BlockRunState {
    fields: ConservedFields,
    ghosts: BoundaryGhostBuffer,
    boundary: BoundarySet,
    storage: Rk4Storage,
    solver_state: SolverState,
    integrator: RungeKutta4Integrator,
}

struct MultiblockAdvanceEnv<'a> {
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

fn build_multiblock_run_states(
    case: &CaseSpec,
    blocks: &[StructuredBlock3d],
    interface_patches: &[Vec<BoundaryPatch>],
    time_config: RungeKutta4Config,
) -> Result<Vec<BlockRunState>> {
    let block_fields = {
        let _span = info_span!(
            "load_multiblock_initial_fields",
            blocks = blocks.len(),
            restart = case.restart.is_some()
        )
        .entered();
        case.build_multiblock_conserved_fields(blocks)?
    };
    let mut states = Vec::with_capacity(blocks.len());
    for (index, (block, fields)) in blocks.iter().zip(block_fields).enumerate() {
        let _span = info_span!(
            "build_multiblock_block_state",
            block = %block.name,
            cells = block.mesh.num_cells(),
            interfaces = interface_patches[index].len()
        )
        .entered();
        let mut patches = boundary_for_block(&case.boundary, &block.name)
            .patches()
            .to_vec();
        patches.extend(interface_patches[index].iter().cloned());
        let boundary = BoundarySet::new(patches);
        info!(
            block = %block.name,
            cells = block.mesh.num_cells(),
            patches = boundary.patches().len(),
            interfaces = interface_patches[index].len(),
            "初始化多块 3D 可压缩 block"
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

fn advance_multiblock_history(
    env: &MultiblockAdvanceEnv<'_>,
    states: &mut [BlockRunState],
    snapshot_paths: &mut Vec<std::path::PathBuf>,
) -> Result<Vec<CompressibleStepInfo>> {
    let mut history = Vec::new();
    loop {
        let aggregate = advance_multiblock_step(env, states)?;
        let stop = aggregate.is_final || aggregate.converged;
        log_multiblock_step(&aggregate, stop, aggregate.converged);
        history.push(aggregate);
        maybe_write_multiblock_snapshot(
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

fn maybe_write_multiblock_snapshot(
    case: &CaseSpec,
    mesh: &MultiBlockStructuredMesh3d,
    states: &[BlockRunState],
    step: &CompressibleStepInfo,
    history: &[CompressibleStepInfo],
    paths: &mut Vec<std::path::PathBuf>,
) -> Result<()> {
    let _span = info_span!("maybe_write_multiblock_snapshot", step = step.step).entered();
    if !super::output_3d::interval_output_due(case, step.step) {
        return Ok(());
    }
    let fields: Vec<ConservedFields> = {
        let _span =
            info_span!("collect_multiblock_snapshot_fields", blocks = states.len()).entered();
        states.iter().map(|state| state.fields.clone()).collect()
    };
    if let Some(path) =
        super::output_3d::maybe_write_multiblock_flow_snapshot(case, mesh, &fields, step)?
    {
        paths.push(path);
    }
    let _ = super::output_3d::maybe_write_residual_outputs(case, history, step)?;
    Ok(())
}

fn advance_multiblock_step(
    env: &MultiblockAdvanceEnv<'_>,
    states: &mut [BlockRunState],
) -> Result<CompressibleStepInfo> {
    let _span = info_span!("advance_multiblock_step").entered();
    let snapshots: Vec<ConservedFields> = {
        let _span = info_span!("clone_multiblock_snapshots", blocks = states.len()).entered();
        states.iter().map(|state| state.fields.clone()).collect()
    };
    let interface_residuals = {
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
        {
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
        step_infos.push(advance_block_step(
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
        let _span = info_span!("aggregate_multiblock_step", blocks = step_infos.len()).entered();
        aggregate_multiblock_step(&step_infos)?
    };
    aggregate.converged = env
        .case
        .resolved_tolerance()
        .is_some_and(|tol| residual_converged(aggregate.residual_log10, tol));
    Ok(aggregate)
}

fn advance_block_step(
    case: &CaseSpec,
    solver: &CompressibleEulerSolver,
    eos: &IdealGasEoS,
    freestream: &FreestreamParams,
    block: &StructuredBlock3d,
    state: &mut BlockRunState,
    interface_residual: &[InterfaceResidualContribution],
) -> Result<CompressibleStepInfo> {
    let _span = info_span!(
        "advance_multiblock_block",
        block = %block.name,
        cells = block.mesh.num_cells(),
        interface_contributions = interface_residual.len()
    )
    .entered();
    let correction = InterfaceResidualCorrection {
        contributions: interface_residual.to_vec(),
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
        residual_correction: Some(Rc::new(RefCell::new(correction))),
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

fn log_multiblock_step(step_info: &CompressibleStepInfo, stop: bool, converged: bool) {
    info!(
        step = step_info.step,
        dt = %format_log_sci4(step_info.dt),
        t = %format_log_sci4(step_info.physical_time),
        log10_residual = %format_log_fixed4(step_info.residual_log10),
        cfl = step_info.cfl,
        is_final = stop,
        converged,
        "多块聚合时间步"
    );
}

fn aggregate_multiblock_step(steps: &[CompressibleStepInfo]) -> Result<CompressibleStepInfo> {
    let first = steps
        .first()
        .ok_or_else(|| AsimuError::Solver("多块 3D 求解没有 block 时间步".to_string()))?;
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

struct SnapshotWriter<'a> {
    case: &'a CaseSpec,
    mesh: &'a crate::mesh::StructuredMesh3d,
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
        history.push(step_info);
        if let Some(ref mut writer) = snapshot {
            let step_info = history.last().expect("history");
            if super::output_3d::interval_output_due(writer.case, step_info.step) {
                if let Some(path) = super::output_3d::maybe_write_flow_snapshot(
                    writer.case,
                    writer.mesh,
                    fields,
                    step_info,
                )? {
                    writer.paths.push(path);
                }
                let _ = super::output_3d::maybe_write_residual_outputs(
                    writer.case,
                    &history,
                    step_info,
                )?;
            }
        }
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
