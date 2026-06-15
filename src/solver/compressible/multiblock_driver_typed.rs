//! 多块 structured 3D 可压缩 typed 时间推进驱动（含 1-to-1 接口通量）。

use tracing::info_span;

use crate::boundary::{BoundaryPatch, BoundarySet};
use crate::core::{ComputeFloat, Real, format_log_fixed4, format_log_sci4, log10_positive};
use crate::discretization::{BoundaryGhostBuffer, GradientFields};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedFieldsT, PrimitiveFields, PrimitiveFieldsT};
use crate::mesh::{MultiBlockStructuredMesh3d, StructuredBlock3d};
use crate::physics::{FreestreamParams, IdealGasEoS, ReferenceScales};
use crate::solver::compressible::CompressibleAdvanceContext3dTyped;
use crate::solver::compressible::multiblock::{
    BlockInterfaceLink, SharedInterfaceFace, build_multiblock_interface_metadata,
    fill_interface_ghosts,
};
use crate::solver::compressible::multiblock_interface::{
    SharedInterfaceResidualParams, compute_shared_interface_residuals,
};
use crate::solver::compressible::structured_compute_backend::StructuredComputeBackend;
use crate::solver::time::TransientStepControl;
use crate::solver::{
    CompressibleEulerSolver, CompressibleMultiblockStepView, CompressibleStepInfo,
    MultiblockStructuredDriverInput, Rk4StorageT, RungeKutta4Config, RungeKutta4Integrator,
    SolverState,
};

struct BlockRunStateTyped<T: ComputeFloat> {
    fields: ConservedFieldsT<T>,
    ghosts: BoundaryGhostBuffer,
    boundary: BoundarySet,
    primitive_scratch: PrimitiveFieldsT<T>,
    spectral_primitives: PrimitiveFields,
    gradient_scratch: GradientFields,
    storage: Rk4StorageT<T>,
    solver_state: SolverState,
    integrator: RungeKutta4Integrator,
}

struct BlockAdvanceEnvTyped<'a> {
    solver: &'a CompressibleEulerSolver,
    eos: &'a IdealGasEoS,
    freestream: &'a FreestreamParams,
    mesh: &'a MultiBlockStructuredMesh3d,
    reference: Option<&'a ReferenceScales>,
    residual_tolerance: Option<Real>,
    links: &'a [Vec<BlockInterfaceLink>],
    shared_faces: &'a [SharedInterfaceFace],
}

/// typed 多块 structured 同步推进（含 1-to-1 共享无粘接口通量）。
#[allow(private_bounds)]
pub fn run_multiblock_structured_typed_with_observer<T: StructuredComputeBackend>(
    input: MultiblockStructuredDriverInput<'_>,
    mut observe_step: impl FnMut(CompressibleMultiblockStepView<'_>) -> Result<()>,
) -> Result<(Vec<CompressibleStepInfo>, Vec<ConservedFields>)> {
    let interfaces = build_multiblock_interface_metadata(input.mesh)?;
    let mut states = build_block_run_states_typed::<T>(
        input.mesh.blocks(),
        &interfaces.patches,
        input.solver.config.time,
        input.global_boundary,
        input.initial_fields,
    )?;
    let env = BlockAdvanceEnvTyped {
        solver: input.solver,
        eos: input.eos,
        freestream: input.freestream,
        mesh: input.mesh,
        reference: input.reference,
        residual_tolerance: input.residual_tolerance,
        links: &interfaces.links,
        shared_faces: &interfaces.shared_faces,
    };
    let history = advance_block_history_typed(&env, &mut states, &mut observe_step)?;
    let fields = states
        .into_iter()
        .map(|state| state.fields.cast_real())
        .collect::<Result<Vec<_>>>()?;
    Ok((history, fields))
}

fn build_block_run_states_typed<T: StructuredComputeBackend>(
    blocks: &[StructuredBlock3d],
    interface_patches: &[Vec<BoundaryPatch>],
    time_config: RungeKutta4Config,
    global_boundary: &BoundarySet,
    block_fields: Vec<ConservedFields>,
) -> Result<Vec<BlockRunStateTyped<T>>> {
    let mut states = Vec::with_capacity(blocks.len());
    for (index, (block, fields)) in blocks.iter().zip(block_fields).enumerate() {
        let _span = info_span!(
            "build_block_state_typed",
            block = %block.name,
            cells = block.mesh.num_cells(),
            precision = T::PRECISION.label(),
        )
        .entered();
        let mut patches = resolve_block_boundary(global_boundary, block, blocks.len())
            .patches()
            .to_vec();
        patches.extend(interface_patches[index].iter().cloned());
        let boundary = BoundarySet::new(patches);
        let n = block.mesh.num_cells();
        states.push(BlockRunStateTyped {
            fields: ConservedFieldsT::from_real_fields(&fields)?,
            ghosts: BoundaryGhostBuffer::new(),
            boundary,
            primitive_scratch: PrimitiveFieldsT::zeros(n)?,
            spectral_primitives: PrimitiveFields::zeros(n)?,
            gradient_scratch: GradientFields::zeros(n)?,
            storage: Rk4StorageT::new(n)?,
            solver_state: SolverState::default(),
            integrator: RungeKutta4Integrator::new(time_config),
        });
    }
    Ok(states)
}

fn advance_block_history_typed<T: StructuredComputeBackend>(
    env: &BlockAdvanceEnvTyped<'_>,
    states: &mut [BlockRunStateTyped<T>],
    observe_step: &mut impl FnMut(CompressibleMultiblockStepView<'_>) -> Result<()>,
) -> Result<Vec<CompressibleStepInfo>> {
    let mut history = Vec::new();
    loop {
        let aggregate = advance_block_step_typed(env, states)?;
        let stop = aggregate.is_final || aggregate.converged;
        log_block_step(env.mesh.num_blocks(), &aggregate, stop, aggregate.converged);
        history.push(aggregate);
        let fields: Vec<ConservedFields> = {
            let _span = info_span!("collect_observer_fields", blocks = states.len()).entered();
            states
                .iter()
                .map(|state| state.fields.cast_real())
                .collect::<Result<Vec<_>>>()?
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

fn advance_block_step_typed<T: StructuredComputeBackend>(
    env: &BlockAdvanceEnvTyped<'_>,
    states: &mut [BlockRunStateTyped<T>],
) -> Result<CompressibleStepInfo> {
    let _span = info_span!("advance_block_step_typed", precision = T::PRECISION.label(),).entered();
    let snapshots: Vec<ConservedFields> = {
        let _span = info_span!("clone_block_snapshots", blocks = states.len()).entered();
        states
            .iter()
            .map(|state| state.fields.cast_real())
            .collect::<Result<Vec<_>>>()?
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
        let state = &mut states[block_index];
        let interface_slice = interface_residuals[block_index].as_slice();
        let mut ctx = CompressibleAdvanceContext3dTyped {
            mesh: &block.mesh,
            structured: &block.mesh,
            patches: &state.boundary,
            ghosts: &mut state.ghosts,
            eos: env.eos,
            freestream: env.freestream,
            reference: env.reference,
            primitive_scratch: std::mem::replace(
                &mut state.primitive_scratch,
                PrimitiveFieldsT::zeros(block.mesh.num_cells())?,
            ),
            spectral_primitives: std::mem::replace(
                &mut state.spectral_primitives,
                PrimitiveFields::zeros(block.mesh.num_cells())?,
            ),
            gradient_scratch: std::mem::replace(
                &mut state.gradient_scratch,
                GradientFields::zeros(block.mesh.num_cells())?,
            ),
            viscous: env.solver.config.viscous.as_ref(),
            interface_residual: if interface_slice.is_empty() {
                None
            } else {
                Some(interface_slice)
            },
        };
        let step_info = env.solver.advance_step_3d_typed(
            &mut ctx,
            &mut state.fields,
            &mut state.storage,
            &mut state.solver_state,
            &mut state.integrator,
        )?;
        state.primitive_scratch = ctx.primitive_scratch;
        state.spectral_primitives = ctx.spectral_primitives;
        state.gradient_scratch = ctx.gradient_scratch;
        step_infos.push(step_info);
    }
    let mut aggregate = aggregate_block_step(&step_infos)?;
    aggregate.converged = crate::core::compressible_log10_tolerance_met(
        aggregate.residual_log10,
        env.residual_tolerance,
    );
    let control = TransientStepControl::new(env.residual_tolerance);
    let _ = control.finalize_step(&mut aggregate);
    Ok(aggregate)
}

fn resolve_block_boundary(
    boundary: &BoundarySet,
    block: &StructuredBlock3d,
    num_blocks: usize,
) -> BoundarySet {
    let prefix = format!("{}/", block.name);
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
    let block_boundary = BoundarySet::new(patches);
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
    tracing::info!(
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
        .ok_or_else(|| AsimuError::Solver("3D typed 求解没有 block 时间步".to_string()))?;
    if steps.len() == 1 {
        return Ok(first.clone());
    }
    let residual_rms = steps
        .iter()
        .map(|step| step.residual_rms)
        .fold(0.0, Real::max);
    Ok(CompressibleStepInfo {
        dt: steps.iter().map(|step| step.dt).fold(first.dt, Real::min),
        physical_time: first.physical_time,
        step: first.step,
        residual_rms,
        residual_log10: log10_positive(residual_rms),
        cfl: steps.iter().map(|step| step.cfl).fold(first.cfl, Real::min),
        is_final: steps.iter().all(|step| step.is_final),
        converged: false,
    })
}
