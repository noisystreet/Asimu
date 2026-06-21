//! 非结构 3D 可压缩 typed 时间推进驱动（ADR 0016 P3）。

#[path = "gmres_block_preconditioner_unstructured.rs"]
mod gmres_block_preconditioner_unstructured;
#[path = "gmres_block_preconditioner_unstructured_math.rs"]
mod gmres_block_preconditioner_unstructured_math;
#[path = "gmres_block_preconditioner_unstructured_state.rs"]
mod gmres_block_preconditioner_unstructured_state;
#[path = "gmres_block_preconditioner_unstructured_viscous.rs"]
mod gmres_block_preconditioner_unstructured_viscous;
#[path = "gmres_implicit_unstructured_typed.rs"]
mod gmres_implicit_unstructured_typed;
#[path = "gmres_lusgs_sweep_preconditioner_unstructured.rs"]
mod gmres_lusgs_sweep_preconditioner_unstructured;
#[path = "unstructured_block_lusgs_typed.rs"]
mod unstructured_block_lusgs_typed;
#[path = "unstructured_cuda_prepare_f32.rs"]
mod unstructured_cuda_prepare_f32;
#[path = "unstructured_dual_time_typed.rs"]
mod unstructured_dual_time_typed;
#[path = "unstructured_explicit_typed.rs"]
mod unstructured_explicit_typed;
#[path = "unstructured_gmres_typed.rs"]
mod unstructured_gmres_typed;
#[path = "unstructured_lusgs_typed.rs"]
mod unstructured_lusgs_typed;
#[path = "unstructured_prepare_timestep_typed.rs"]
mod unstructured_prepare_timestep_typed;

use unstructured_block_lusgs_typed::advance_unstructured_block_lusgs_typed;
use unstructured_dual_time_typed::advance_unstructured_dual_time_typed;
use unstructured_explicit_typed::{
    UnstructuredExplicitTimeAdvance, advance_unstructured_explicit_typed,
};
use unstructured_gmres_typed::advance_unstructured_gmres_typed;
use unstructured_lusgs_typed::{
    UnstructuredLusgsDiagonalUpdate, UnstructuredLusgsSweep, UnstructuredLusgsSweepContext,
};
pub(crate) use unstructured_prepare_timestep_typed::{
    UnstructuredCudaPrepareSync, UnstructuredSpectralRadiusAtPrepare,
    UnstructuredTimestepFromSigma, prepare_unstructured_timestep_typed,
};

use std::time::Instant;

use tracing::{debug, info, info_span};

#[cfg(feature = "cuda")]
use crate::core::ExecDevice;
use crate::core::{
    ComputeFloat, Real, elapsed_ms, format_log_fixed4, format_log_fixed5, format_log_sci4,
    log10_positive,
};
use crate::discretization::InviscidFaceFluxTyped;
use crate::discretization::compressible::residual::InviscidAssemblyUnstructuredTypedParams;
use crate::discretization::compressible::residual::{
    InviscidTypedScatterBackend, ViscousTypedScatterBackend,
};
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::gradient_unstructured_f32::UnstructuredGradientLsqInputF32;
use crate::discretization::{
    BoundaryGhostBuffer, ReconstructionKind, UnstructuredGradientLsqInput,
    UnstructuredGradientScratchF32, UnstructuredSolverMeshCache,
    ViscousAssemblyUnstructuredF32Input, ViscousAssemblyUnstructuredScratch,
    ViscousAssemblyUnstructuredTypedInput, assemble_inviscid_residual_unstructured_typed,
    compute_gradients_and_assemble_viscous_unstructured_f32,
    compute_gradients_and_assemble_viscous_unstructured_typed,
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq,
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq_f32,
};
use crate::error::{AsimuError, Result};
use crate::exec::{ExecutionContext, MeshExecMetrics};
use crate::field::{
    ConservedFields, ConservedFieldsT, ConservedResidualT, LusgsDiagonalUpdateBackend,
    PrimitiveFieldsT, PrimitiveFillFromConserved,
};
use crate::solver::compressible::unstructured_driver::CompressibleUnstructuredStepView;
use crate::solver::time::{
    Rk4StorageT, RungeKutta4Config, RungeKutta4Integrator, TimeIntegrationScheme, TimeIntegrator,
    TransientStepControl,
};
use crate::solver::{
    CompressibleStepInfo, LuSgsUnstructuredCouplings, LuSgsUnstructuredSweepTyped,
    RefreshCompressibleStateTypedInput, SolverState, UnstructuredDriverConfig,
    refresh_compressible_ghosts_and_primitives_typed,
};

/// 非结构时间步缓冲（f64 与 f32 热路径分离）。
pub(crate) struct UnstructuredTimestepBuffers {
    pub sigma: Vec<Real>,
    pub cell_dts: Vec<Real>,
    pub sigma_f32: Vec<f32>,
    pub cell_dts_f32: Vec<f32>,
}

pub(crate) struct UnstructuredTypedRhsWork<'a, T: ComputeFloat> {
    ghosts: &'a mut BoundaryGhostBuffer,
    primitives: &'a mut PrimitiveFieldsT<T>,
    gradients: &'a mut GradientFieldsT<T>,
    viscous_scratch: &'a mut ViscousAssemblyUnstructuredScratch,
    viscous_grad_scratch_f32: &'a mut UnstructuredGradientScratchF32,
    mesh_cache: &'a UnstructuredSolverMeshCache,
    exec: &'a mut ExecutionContext,
}

pub(crate) struct UnstructuredStepWorkTyped<T: ComputeFloat> {
    storage: Rk4StorageT<T>,
    state: SolverState,
    integrator: RungeKutta4Integrator,
    ghosts: BoundaryGhostBuffer,
    primitives: PrimitiveFieldsT<T>,
    gradients: GradientFieldsT<T>,
    viscous_scratch: ViscousAssemblyUnstructuredScratch,
    viscous_grad_scratch_f32: UnstructuredGradientScratchF32,
    mesh_cache: UnstructuredSolverMeshCache,
    exec: ExecutionContext,
    volumes: Vec<Real>,
    volumes_f32: Vec<f32>,
    timestep: UnstructuredTimestepBuffers,
    lusgs_couplings: LuSgsUnstructuredCouplings,
    block_lusgs_preconditioner:
        Option<gmres_block_preconditioner_unstructured::UnstructuredBlockLusgsPreconditioner>,
    dual_time_state: crate::solver::time::DualTimeState<T>,
    /// 稳态 LU-SGS：RHS 装配后、隐式更新前的密度 RMS（监控 \(R(U^0)\)，与 sweep/对角无关）。
    density_rms_after_rhs: Option<Real>,
    /// 上一 GMRES 伪时间步线性迭代次数。
    gmres_inner_iterations: u32,
    /// standalone block_lusgs 内层 Richardson 迭代次数。
    block_lusgs_inner_iterations: u32,
}

pub(crate) struct UnstructuredRunEnvTyped<'a> {
    config: &'a UnstructuredDriverConfig<'a>,
}

fn allocate_unstructured_step_work_typed<T: ComputeFloat>(
    env: &UnstructuredRunEnvTyped<'_>,
) -> Result<UnstructuredStepWorkTyped<T>> {
    let n = env.config.mesh.num_cells();
    let _span = info_span!(
        "allocate_unstructured_work_typed",
        cells = n,
        precision = T::PRECISION.label(),
    )
    .entered();
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh_with_order(
        env.config.mesh,
        env.config.patches,
        env.config.cell_order,
        *env.config.freestream,
    )?;
    let interior_faces = mesh_cache.face_topology.interior.len();
    let max_bucket_faces = mesh_cache
        .face_topology
        .interior_coloring
        .max_bucket_faces();
    let mut exec_config = env.config.exec_config.clone();
    exec_config.compute_precision = T::PRECISION;
    let exec = ExecutionContext::new(
        exec_config,
        MeshExecMetrics::new(n, interior_faces, max_bucket_faces),
    )?;
    info!(
        compute_precision = ?exec.compute_precision(),
        exec_device = exec.device().label(),
        "unstructured_typed_exec_context"
    );
    Ok(UnstructuredStepWorkTyped {
        storage: Rk4StorageT::new(n)?,
        state: SolverState::default(),
        integrator: RungeKutta4Integrator::new(RungeKutta4Config {
            dt: env
                .config
                .dual_time
                .map(|d| d.dt_phys)
                .or(env.config.fixed_dt)
                .unwrap_or(0.0),
            max_steps: env.config.max_steps,
        }),
        ghosts: BoundaryGhostBuffer::with_face_capacity(env.config.mesh.num_faces()),
        primitives: PrimitiveFieldsT::zeros(n)?,
        gradients: GradientFieldsT::<T>::zeros(n)?,
        viscous_scratch: ViscousAssemblyUnstructuredScratch::new(n),
        viscous_grad_scratch_f32: UnstructuredGradientScratchF32::new(n),
        mesh_cache,
        exec,
        volumes: env.config.mesh.cell_volumes(),
        volumes_f32: env
            .config
            .mesh
            .cell_volumes()
            .iter()
            .map(|v| *v as f32)
            .collect(),
        timestep: UnstructuredTimestepBuffers {
            sigma: Vec::new(),
            cell_dts: Vec::new(),
            sigma_f32: Vec::new(),
            cell_dts_f32: Vec::new(),
        },
        lusgs_couplings: LuSgsUnstructuredCouplings::from_mesh(env.config.mesh)?,
        block_lusgs_preconditioner: None,
        dual_time_state: crate::solver::time::DualTimeState::new(n)?,
        density_rms_after_rhs: None,
        gmres_inner_iterations: 0,
        block_lusgs_inner_iterations: 0,
    })
}

fn unstructured_typed_observer_post_step<
    T: UnstructuredComputeBackend + UnstructuredCudaPrepareSync,
>(
    env: &UnstructuredRunEnvTyped<'_>,
    work: &mut UnstructuredStepWorkTyped<T>,
    fields: &mut ConservedFieldsT<T>,
    history: &[CompressibleStepInfo],
    observe_step: &mut impl FnMut(CompressibleUnstructuredStepView<'_>) -> Result<()>,
) -> Result<()> {
    let posted = history.last().expect("history");
    work.exec.sync_to_host()?;
    if super::unstructured_driver::observer_field_sync_due(
        env.config.observer_field_sync_interval,
        posted.step,
    ) {
        T::maybe_download_conserved_for_output(work, fields)?;
    }
    let fields_real = fields.cast_real()?;
    observe_step(CompressibleUnstructuredStepView {
        info: posted,
        history,
        fields: &fields_real,
    })?;
    Ok(())
}

/// typed 非结构同步推进；结束时将场转为 `f64` 供输出。
#[allow(private_bounds)]
pub fn run_unstructured_typed_with_observer<T: UnstructuredComputeBackend>(
    config: &UnstructuredDriverConfig<'_>,
    fields: &mut ConservedFieldsT<T>,
    mut observe_step: impl FnMut(CompressibleUnstructuredStepView<'_>) -> Result<()>,
) -> Result<(Vec<CompressibleStepInfo>, ConservedFields)> {
    let mut env = UnstructuredRunEnvTyped { config };
    let mut work = allocate_unstructured_step_work_typed(&env)?;
    let mut history = Vec::new();
    let control = TransientStepControl::new(env.config.residual_tolerance);
    loop {
        let step = advance_unstructured_step_typed(&mut env, fields, &mut work)?;
        let mut step = step;
        let stop = control.finalize_step(&mut step);
        if env.config.time_scheme == TimeIntegrationScheme::DualTime {
            let time_scale = env
                .config
                .reference
                .map(|reference| reference.time_scale())
                .unwrap_or(1.0);
            let dt_s = format!("{}s", format_log_sci4(step.dt * time_scale));
            let t_s = format!("{}s", format_log_sci4(step.physical_time * time_scale));
            info!(
                step = step.step,
                dt = %dt_s,
                t = %t_s,
                cfl = %format_log_fixed5(step.cfl),
            );
        } else {
            info!(
                step = step.step,
                dt = %format_log_sci4(step.dt),
                t = %format_log_sci4(step.physical_time),
                log10_residual = %format_log_fixed4(step.residual_log10),
                cfl = %format_log_fixed5(step.cfl),
            );
        }
        history.push(step);
        unstructured_typed_observer_post_step(
            &env,
            &mut work,
            fields,
            &history,
            &mut observe_step,
        )?;
        if stop {
            break;
        }
    }
    Ok((history, {
        T::maybe_download_conserved_for_output(&mut work, fields)?;
        fields.cast_real()?
    }))
}

fn resolve_unstructured_step_dt<T: UnstructuredComputeBackend + UnstructuredCudaPrepareSync>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    cfl: Real,
    p_floor: Real,
) -> Result<Real> {
    if env.config.time_scheme == TimeIntegrationScheme::DualTime {
        let dual = env
            .config
            .dual_time
            .ok_or_else(|| AsimuError::Config("dual_time 推进须设置 DualTimeConfig".to_string()))?;
        work.integrator.config.dt = dual.dt_phys;
        return Ok(dual.dt_phys);
    }
    let dt = prepare_unstructured_timestep_typed(env, fields, work, cfl, p_floor)?;
    work.integrator.config.dt = dt;
    Ok(dt)
}

fn advance_unstructured_time_integration_typed<
    T: UnstructuredComputeBackend + UnstructuredCudaPrepareSync,
>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    dt: Real,
    cfl: Real,
    p_floor: Real,
) -> Result<()> {
    match env.config.time_scheme {
        TimeIntegrationScheme::LuSgs => {
            advance_unstructured_lusgs_typed(env, fields, work, p_floor)
        }
        TimeIntegrationScheme::Gmres => {
            let outcome = advance_unstructured_gmres_typed(env, fields, work, dt, p_floor)?;
            work.gmres_inner_iterations = outcome.gmres_iterations;
            Ok(())
        }
        TimeIntegrationScheme::BlockLusgs => {
            advance_unstructured_block_lusgs_typed(env, fields, work, p_floor)
        }
        TimeIntegrationScheme::DualTime => {
            let dual = env.config.dual_time.ok_or_else(|| {
                AsimuError::Config("dual_time 推进须设置 DualTimeConfig".to_string())
            })?;
            advance_unstructured_dual_time_typed(env, fields, work, dual, cfl, p_floor).map(|_| ())
        }
        TimeIntegrationScheme::Euler | TimeIntegrationScheme::Rk4 => {
            advance_unstructured_explicit_typed(env, fields, work, dt, p_floor)
        }
        scheme => Err(AsimuError::Config(format!(
            "非结构 typed 路径暂不支持 time.scheme = \"{}\"",
            scheme.label()
        ))),
    }
}

fn advance_unstructured_step_typed<T: UnstructuredComputeBackend + UnstructuredCudaPrepareSync>(
    env: &mut UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
) -> Result<CompressibleStepInfo> {
    let step_start = Instant::now();
    work.density_rms_after_rhs = None;
    work.gmres_inner_iterations = 0;
    work.block_lusgs_inner_iterations = 0;
    #[cfg(feature = "cuda")]
    if work.exec.device() == ExecDevice::GpuCuda {
        if unstructured_prepare_timestep_typed::f32_cuda_viscous_rhs_pipeline(env, &work.exec) {
            work.exec.cuda_reset_between_timesteps()?;
        } else {
            work.exec.cuda_reset_full_pipeline_step()?;
        }
    }
    let cfl = env
        .config
        .cfl_schedule
        .at_step(work.state.time_step.saturating_add(1), env.config.max_steps);
    let p_floor = crate::field::positivity_pressure_floor(env.config.freestream.pressure);
    let compute_dt_start = Instant::now();
    let dt = resolve_unstructured_step_dt(env, fields, work, cfl, p_floor)?;
    let compute_dt_ms = elapsed_ms(compute_dt_start);
    let time_integration_start = Instant::now();
    {
        let _span = info_span!(
            "unstructured_time_integration_typed",
            scheme = env.config.time_scheme.label(),
            precision = T::PRECISION.label(),
        )
        .entered();
        advance_unstructured_time_integration_typed(env, fields, work, dt, cfl, p_floor)?;
    }
    let time_integration_ms = elapsed_ms(time_integration_start);
    T::maybe_enforce_conserved_after_integration(work, env.config.eos, p_floor)?;
    fields.enforce_positivity(env.config.eos, p_floor);
    let residual = if let Some(rms) = work.density_rms_after_rhs.take() {
        rms
    } else {
        T::step_density_residual_rms(work)?
    };
    let time_info = work.integrator.advance(&mut work.state)?;
    let step_total_ms = elapsed_ms(step_start);
    debug!(
        step = work.state.time_step,
        profile_compute_dt_ms = %format_log_fixed4(compute_dt_ms),
        profile_time_integration_ms = %format_log_fixed4(time_integration_ms),
        profile_step_total_ms = %format_log_fixed4(step_total_ms),
        precision = T::PRECISION.label(),
        "非结构 typed 时间步 profiling",
    );
    Ok(CompressibleStepInfo {
        dt: time_info.dt,
        physical_time: time_info.physical_time,
        step: time_info.step,
        residual_rms: residual,
        residual_log10: log10_positive(residual),
        cfl,
        is_final: time_info.is_final,
        converged: false,
        inner_iterations: if env.config.time_scheme == TimeIntegrationScheme::DualTime {
            work.dual_time_state.inner_iterations
        } else if env.config.time_scheme == TimeIntegrationScheme::Gmres {
            work.gmres_inner_iterations
        } else if env.config.time_scheme == TimeIntegrationScheme::BlockLusgs {
            work.block_lusgs_inner_iterations
        } else {
            0
        },
    })
}

fn advance_unstructured_lusgs_typed<T: UnstructuredComputeBackend>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    p_floor: Real,
) -> Result<()> {
    if !env.config.local_time_step {
        return Err(AsimuError::Config(
            "非结构 time.scheme = \"lu_sgs\" 须配合 local_time_step = true".to_string(),
        ));
    }
    let lu_sgs = env.config.lu_sgs;
    {
        let _span = info_span!("unstructured_lusgs_copy_base_typed").entered();
        work.storage.u0.copy_from(fields)?;
    }
    T::maybe_upload_lusgs_integration_base(work)?;
    {
        let _span = info_span!("unstructured_lusgs_rhs_typed").entered();
        let mut rhs_work = UnstructuredTypedRhsWork {
            ghosts: &mut work.ghosts,
            primitives: &mut work.primitives,
            gradients: &mut work.gradients,
            viscous_scratch: &mut work.viscous_scratch,
            viscous_grad_scratch_f32: &mut work.viscous_grad_scratch_f32,
            mesh_cache: &work.mesh_cache,
            exec: &mut work.exec,
        };
        assemble_unstructured_typed_rhs(
            env,
            &mut rhs_work,
            &work.storage.u0,
            &mut work.storage.k1,
            true,
            p_floor,
        )?;
    }
    work.density_rms_after_rhs = Some(T::step_density_residual_rms(work)?);
    if lu_sgs.sweep {
        let _span = info_span!(
            "unstructured_lusgs_sweep_typed",
            precision = T::PRECISION.label(),
        )
        .entered();
        T::run_lusgs_sweep(
            fields,
            work,
            &UnstructuredLusgsSweepContext {
                env,
                p_floor,
                sweep: true,
                omega: lu_sgs.omega,
                backward_damping: lu_sgs.sweep_backward_damping,
                inv_dt_phys: 0.0,
            },
        )?;
    } else {
        {
            let _span = info_span!("unstructured_lusgs_diagonal_update_typed").entered();
            T::assign_lusgs_diagonal_update(
                work,
                lu_sgs.omega,
                env.config.eos.gamma,
                p_floor,
                0.0,
            )?;
        }
        if !T::lusgs_skip_copy_stage_after_diagonal(work) {
            let _span = info_span!("unstructured_lusgs_copy_stage_typed").entered();
            fields.copy_from(&work.storage.stage)?;
        }
    }
    #[cfg(feature = "cuda")]
    if work.exec.device() == ExecDevice::GpuCuda {
        work.exec.mark_cuda_primitives_stale_after_integration();
    }
    Ok(())
}

pub(crate) fn assemble_unstructured_typed_rhs<T: UnstructuredRhsDispatchImpl>(
    env: &UnstructuredRunEnvTyped<'_>,
    work: &mut UnstructuredTypedRhsWork<'_, T>,
    fields: &ConservedFieldsT<T>,
    residual: &mut ConservedResidualT<T>,
    refresh_state: bool,
    p_floor: Real,
) -> Result<()> {
    T::assemble_unstructured_rhs(env, work, fields, residual, refresh_state, p_floor)
}

/// typed 非结构 RHS 装配分发（sealed，仅 f32 / f64）。
pub(crate) trait UnstructuredRhsDispatchImpl:
    ComputeFloat + rhs_dispatch::Sealed + Sized
{
    fn assemble_unstructured_rhs(
        env: &UnstructuredRunEnvTyped<'_>,
        work: &mut UnstructuredTypedRhsWork<'_, Self>,
        fields: &ConservedFieldsT<Self>,
        residual: &mut ConservedResidualT<Self>,
        refresh_state: bool,
        p_floor: Real,
    ) -> Result<()>;
}

mod rhs_dispatch {
    pub(crate) trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// 兼容别名：空标记 trait，由 [`UnstructuredRhsDispatchImpl`] 取代。
#[allow(dead_code)]
pub(crate) trait UnstructuredTypedRhsDispatch:
    ComputeFloat + rhs_dispatch::Sealed + Sized
{
}

impl UnstructuredTypedRhsDispatch for f32 {}
impl UnstructuredTypedRhsDispatch for f64 {}

impl UnstructuredRhsDispatchImpl for f32 {
    fn assemble_unstructured_rhs(
        env: &UnstructuredRunEnvTyped<'_>,
        work: &mut UnstructuredTypedRhsWork<'_, f32>,
        fields: &ConservedFieldsT<f32>,
        residual: &mut ConservedResidualT<f32>,
        refresh_state: bool,
        p_floor: Real,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        let cuda_viscous_pipeline =
            unstructured_prepare_timestep_typed::f32_cuda_viscous_rhs_pipeline(env, work.exec);
        #[cfg(feature = "cuda")]
        begin_f32_cuda_viscous_rhs_pipeline(work.exec, cuda_viscous_pipeline)?;
        #[cfg(feature = "cuda")]
        let skip_rhs_refresh = cuda_viscous_pipeline && work.exec.cuda_host_bc_primitives_synced();
        #[cfg(not(feature = "cuda"))]
        let skip_rhs_refresh = false;
        if refresh_state && !skip_rhs_refresh {
            refresh_compressible_ghosts_and_primitives_typed(RefreshCompressibleStateTypedInput {
                boundary_mesh: env.config.mesh,
                patches: env.config.patches,
                fields,
                ghosts: work.ghosts,
                eos: env.config.eos,
                freestream: env.config.freestream,
                reference: env.config.reference,
                viscous: env.config.viscous,
                min_pressure: p_floor,
                primitives: work.primitives,
            })?;
            sync_f32_rhs_primitives_after_refresh(
                work.exec,
                work.primitives,
                refresh_state,
                #[cfg(feature = "cuda")]
                cuda_viscous_pipeline,
            )?;
        } else {
            sync_f32_rhs_primitives_after_refresh(
                work.exec,
                work.primitives,
                refresh_state,
                #[cfg(feature = "cuda")]
                cuda_viscous_pipeline,
            )?;
        }
        if env.config.inviscid.reconstruction == ReconstructionKind::Muscl {
            let grad_input = UnstructuredGradientLsqInputF32 {
                mesh: env.config.mesh,
                mesh_cache: work.mesh_cache,
                primitives: work.primitives,
                eos: env.config.eos,
                ghosts: work.ghosts,
                min_pressure: p_floor,
                viscous: env.config.viscous,
            };
            compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq_f32(
                grad_input,
                work.gradients,
                work.exec,
            )?;
        }
        let muscl_gradients = match env.config.inviscid.reconstruction {
            ReconstructionKind::Muscl => Some(&*work.gradients),
            ReconstructionKind::FirstOrder => None,
        };
        let mut assembly = InviscidAssemblyUnstructuredTypedParams {
            mesh: env.config.mesh,
            eos: env.config.eos,
            config: env.config.inviscid,
            boundaries: env.config.patches,
            ghosts: work.ghosts,
            primitives: work.primitives,
            mesh_cache: work.mesh_cache,
            gradients: muscl_gradients,
            min_pressure: p_floor,
            exec: work.exec,
        };
        assemble_inviscid_residual_unstructured_typed(fields, residual, &mut assembly)?;
        if let Some(viscous) = env.config.viscous {
            let mut input = ViscousAssemblyUnstructuredF32Input {
                mesh: env.config.mesh,
                mesh_cache: work.mesh_cache,
                eos: env.config.eos,
                viscous,
                boundaries: env.config.patches,
                ghosts: work.ghosts,
                primitives: work.primitives,
                min_pressure: p_floor,
                gradient_scratch: work.gradients,
                exec: work.exec,
            };
            compute_gradients_and_assemble_viscous_unstructured_f32(
                residual,
                &mut input,
                work.viscous_scratch,
                work.viscous_grad_scratch_f32,
            )?;
        }
        Ok(())
    }
}

impl UnstructuredRhsDispatchImpl for f64 {
    fn assemble_unstructured_rhs(
        env: &UnstructuredRunEnvTyped<'_>,
        work: &mut UnstructuredTypedRhsWork<'_, f64>,
        fields: &ConservedFieldsT<f64>,
        residual: &mut ConservedResidualT<f64>,
        refresh_state: bool,
        p_floor: Real,
    ) -> Result<()> {
        if refresh_state {
            refresh_compressible_ghosts_and_primitives_typed(RefreshCompressibleStateTypedInput {
                boundary_mesh: env.config.mesh,
                patches: env.config.patches,
                fields,
                ghosts: work.ghosts,
                eos: env.config.eos,
                freestream: env.config.freestream,
                reference: env.config.reference,
                viscous: env.config.viscous,
                min_pressure: p_floor,
                primitives: work.primitives,
            })?;
        }
        if env.config.inviscid.reconstruction == ReconstructionKind::Muscl {
            let grad_input = UnstructuredGradientLsqInput {
                mesh: env.config.mesh,
                mesh_cache: work.mesh_cache,
                primitives: work.primitives,
                eos: env.config.eos,
                ghosts: work.ghosts,
                min_pressure: p_floor,
                viscous: env.config.viscous,
            };
            compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq(
                grad_input,
                work.gradients,
                &mut work.viscous_scratch.gradient,
                work.exec,
            )?;
        }
        let muscl_gradients = match env.config.inviscid.reconstruction {
            ReconstructionKind::Muscl => Some(&*work.gradients),
            ReconstructionKind::FirstOrder => None,
        };
        let mut assembly = InviscidAssemblyUnstructuredTypedParams {
            mesh: env.config.mesh,
            eos: env.config.eos,
            config: env.config.inviscid,
            boundaries: env.config.patches,
            ghosts: work.ghosts,
            primitives: work.primitives,
            mesh_cache: work.mesh_cache,
            gradients: muscl_gradients,
            min_pressure: p_floor,
            exec: work.exec,
        };
        assemble_inviscid_residual_unstructured_typed(fields, residual, &mut assembly)?;
        if let Some(viscous) = env.config.viscous {
            let mut input = ViscousAssemblyUnstructuredTypedInput {
                mesh: env.config.mesh,
                mesh_cache: work.mesh_cache,
                eos: env.config.eos,
                viscous,
                boundaries: env.config.patches,
                ghosts: work.ghosts,
                primitives: work.primitives,
                min_pressure: p_floor,
                gradient_scratch: work.gradients,
                exec: work.exec,
            };
            compute_gradients_and_assemble_viscous_unstructured_typed(
                residual,
                &mut input,
                work.viscous_scratch,
            )?;
        }
        Ok(())
    }
}

/// 非结构可压缩求解热路径所需精度后端（ADR 0018；密封于 f32 / f64）。
pub(crate) trait UnstructuredComputeBackend:
    ComputeFloat
    + LusgsDiagonalUpdateBackend
    + InviscidFaceFluxTyped
    + InviscidTypedScatterBackend
    + ViscousTypedScatterBackend
    + UnstructuredSpectralRadiusAtPrepare
    + UnstructuredTimestepFromSigma
    + UnstructuredCudaPrepareSync
    + LuSgsUnstructuredSweepTyped
    + UnstructuredRhsDispatchImpl
    + UnstructuredLusgsDiagonalUpdate
    + UnstructuredLusgsSweep
    + UnstructuredExplicitTimeAdvance
    + PrimitiveFillFromConserved
{
}

impl UnstructuredComputeBackend for f32 {}
impl UnstructuredComputeBackend for f64 {}

#[cfg(feature = "cuda")]
fn begin_f32_cuda_viscous_rhs_pipeline(
    exec: &mut ExecutionContext,
    cuda_viscous_pipeline: bool,
) -> Result<()> {
    if cuda_viscous_pipeline {
        exec.cuda_reset_pipeline_step()?;
        exec.cuda_enable_rhs_device_pipeline()?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn sync_f32_rhs_primitives_after_refresh(
    exec: &mut ExecutionContext,
    primitives: &crate::field::PrimitiveFieldsT<f32>,
    refresh_state: bool,
    cuda_viscous_pipeline: bool,
) -> Result<()> {
    if cuda_viscous_pipeline {
        return exec.sync_cuda_primitives_to_device(primitives);
    }
    if refresh_state {
        exec.mark_cuda_primitives_stale();
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn sync_f32_rhs_primitives_after_refresh(
    exec: &mut ExecutionContext,
    _primitives: &crate::field::PrimitiveFieldsT<f32>,
    refresh_state: bool,
) -> Result<()> {
    if refresh_state {
        exec.mark_cuda_primitives_stale();
    }
    Ok(())
}

#[cfg(test)]
#[path = "unstructured_driver_typed_tests.rs"]
mod unstructured_driver_typed_tests;
