//! 多块 structured 3D 可压缩时间推进驱动（不含 case 输出编排）。

use std::{cell::RefCell, rc::Rc};

use tracing::{info, info_span};

use crate::boundary::{BoundaryPatch, BoundarySet};
use crate::core::{Real, format_log_fixed4, format_log_sci4, log10_positive};
use crate::discretization::{BoundaryGhostBuffer, GhostCellState, GradientFields};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, PrimitiveFields};
use crate::mesh::{MultiBlockStructuredMesh3d, StructuredBlock3d};
use crate::physics::{FreestreamParams, IdealGasEoS, ReferenceScales};
use crate::solver::compressible::ResidualCorrection3d;
use crate::solver::compressible_multiblock::{
    BlockInterfaceLink, SharedInterfaceFace, build_multiblock_interface_metadata,
};
use crate::solver::compressible_multiblock_interface::{
    InterfaceResidualContribution, SharedInterfaceResidualParams, apply_interface_residuals,
    compute_shared_interface_residuals,
};
use crate::solver::time::TransientStepControl;
use crate::solver::{
    CompressibleAdvanceContext3d, CompressibleEulerSolver, CompressibleStepInfo, Rk4Storage,
    RungeKutta4Config, RungeKutta4Integrator, SolverState,
};

/// 多块 structured 外层步只读视图（observer 回调参数）。
#[derive(Debug, Clone, Copy)]
pub struct CompressibleMultiblockStepView<'a> {
    pub info: &'a CompressibleStepInfo,
    pub history: &'a [CompressibleStepInfo],
    pub fields: &'a [ConservedFields],
}

/// 多块 structured 推进输入（由 case 层从 `CaseSpec` 组装）。
pub struct MultiblockStructuredDriverInput<'a> {
    pub solver: &'a CompressibleEulerSolver,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
    pub mesh: &'a MultiBlockStructuredMesh3d,
    pub global_boundary: &'a BoundarySet,
    pub reference: Option<&'a ReferenceScales>,
    pub residual_tolerance: Option<Real>,
    pub initial_fields: Vec<ConservedFields>,
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
    solver: &'a CompressibleEulerSolver,
    eos: &'a IdealGasEoS,
    freestream: &'a FreestreamParams,
    mesh: &'a MultiBlockStructuredMesh3d,
    reference: Option<&'a ReferenceScales>,
    residual_tolerance: Option<Real>,
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

/// 多块 structured 同步推进；每步调用 `observe_step`。
pub fn run_multiblock_structured_with_observer(
    input: MultiblockStructuredDriverInput<'_>,
    mut observe_step: impl FnMut(CompressibleMultiblockStepView<'_>) -> Result<()>,
) -> Result<(Vec<CompressibleStepInfo>, Vec<ConservedFields>)> {
    let interfaces = build_multiblock_interface_metadata(input.mesh)?;
    let mut states = build_block_run_states(
        input.mesh.blocks(),
        &interfaces.patches,
        input.solver.config.time,
        input.global_boundary,
        input.initial_fields,
    )?;
    let env = BlockAdvanceEnv {
        solver: input.solver,
        eos: input.eos,
        freestream: input.freestream,
        mesh: input.mesh,
        reference: input.reference,
        residual_tolerance: input.residual_tolerance,
        links: &interfaces.links,
        shared_faces: &interfaces.shared_faces,
    };
    let history = advance_block_history(&env, &mut states, &mut observe_step)?;
    let fields = states.into_iter().map(|state| state.fields).collect();
    Ok((history, fields))
}

fn build_block_run_states(
    blocks: &[StructuredBlock3d],
    interface_patches: &[Vec<BoundaryPatch>],
    time_config: RungeKutta4Config,
    global_boundary: &BoundarySet,
    block_fields: Vec<ConservedFields>,
) -> Result<Vec<BlockRunState>> {
    let mut states = Vec::with_capacity(blocks.len());
    for (index, (block, fields)) in blocks.iter().zip(block_fields).enumerate() {
        let _span = info_span!(
            "build_block_state",
            block = %block.name,
            cells = block.mesh.num_cells(),
            interfaces = interface_patches[index].len()
        )
        .entered();
        let mut patches = resolve_block_boundary(global_boundary, block, blocks.len())
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
    observe_step: &mut impl FnMut(CompressibleMultiblockStepView<'_>) -> Result<()>,
) -> Result<Vec<CompressibleStepInfo>> {
    let mut history = Vec::new();
    loop {
        let aggregate = advance_block_step(env, states)?;
        let stop = aggregate.is_final || aggregate.converged;
        log_block_step(env.mesh.num_blocks(), &aggregate, stop, aggregate.converged);
        history.push(aggregate);
        let fields: Vec<ConservedFields> = {
            let _span = info_span!("collect_observer_fields", blocks = states.len()).entered();
            states.iter().map(|state| state.fields.clone()).collect()
        };
        observe_step(CompressibleMultiblockStepView {
            info: history.last().expect("history"),
            history: &history,
            fields: &fields,
        })?;
        if stop {
            break;
        }
    }
    Ok(history)
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
            SingleBlockStepParams {
                solver: env.solver,
                eos: env.eos,
                freestream: env.freestream,
                reference: env.reference,
                residual_tolerance: env.residual_tolerance,
                block,
                interface_residual: &interface_residuals[block_index],
            },
            &mut states[block_index],
        )?);
    }
    let mut aggregate = {
        let _span = info_span!("aggregate_block_step", blocks = step_infos.len()).entered();
        aggregate_block_step(&step_infos)?
    };
    let control = TransientStepControl::new(env.residual_tolerance);
    aggregate.converged = crate::core::compressible_log10_tolerance_met(
        aggregate.residual_log10,
        env.residual_tolerance,
    );
    let _ = control.finalize_step(&mut aggregate);
    Ok(aggregate)
}

struct SingleBlockStepParams<'a> {
    solver: &'a CompressibleEulerSolver,
    eos: &'a IdealGasEoS,
    freestream: &'a FreestreamParams,
    reference: Option<&'a ReferenceScales>,
    residual_tolerance: Option<Real>,
    block: &'a StructuredBlock3d,
    interface_residual: &'a [InterfaceResidualContribution],
}

fn advance_single_block_step(
    params: SingleBlockStepParams<'_>,
    state: &mut BlockRunState,
) -> Result<CompressibleStepInfo> {
    let _span = info_span!(
        "advance_single_block_step",
        block = %params.block.name,
        cells = params.block.mesh.num_cells(),
        interface_contributions = params.interface_residual.len()
    )
    .entered();
    let residual_correction: Option<Rc<RefCell<dyn ResidualCorrection3d>>> =
        if params.interface_residual.is_empty() {
            None
        } else {
            Some(Rc::new(RefCell::new(InterfaceResidualCorrection {
                contributions: params.interface_residual.to_vec(),
            })))
        };
    let mut ctx = CompressibleAdvanceContext3d {
        mesh: &params.block.mesh,
        structured: &params.block.mesh,
        patches: &state.boundary,
        ghosts: &mut state.ghosts,
        eos: params.eos,
        freestream: params.freestream,
        reference: params.reference,
        primitive_scratch: PrimitiveFields::zeros(params.block.mesh.num_cells())?,
        gradient_scratch: GradientFields::zeros(params.block.mesh.num_cells())?,
        viscous: params.solver.config.viscous.as_ref(),
        residual_correction,
    };
    let mut step_info = params.solver.advance_step_3d(
        &mut ctx,
        &mut state.fields,
        &mut state.storage,
        &mut state.solver_state,
        &mut state.integrator,
    )?;
    step_info.converged = crate::core::compressible_log10_tolerance_met(
        step_info.residual_log10,
        params.residual_tolerance,
    );
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
