//! 非结构 3D 可压缩 typed 时间推进驱动（ADR 0016 P3）。

use std::time::Instant;

use tracing::{debug, info, info_span};

use crate::core::{
    ComputeFloat, Real, elapsed_ms, format_log_fixed4, format_log_fixed5, format_log_sci4,
    log10_positive,
};
use crate::discretization::InviscidFaceFluxTyped;
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::gradient_unstructured_f32::UnstructuredGradientLsqInputF32;
use crate::discretization::residual::InviscidAssemblyUnstructuredTypedParams;
use crate::discretization::residual::{InviscidTypedScatterBackend, ViscousTypedScatterBackend};
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
use crate::field::LusgsDiagonalUpdateBackend;
use crate::field::{ConservedFields, ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};
use crate::solver::spectral_radius_unstructured::{
    SpectralRadiusUnstructuredTypedParams, UnstructuredSpectralRadiusTyped,
};
use crate::solver::time::{
    Rk4StorageT, RungeKutta4Config, RungeKutta4Integrator, TimeIntegrationScheme, TimeIntegrator,
    TransientStepControl, euler_step, euler_step_local, min_positive_dt, rk4_step, rk4_step_local,
};
use crate::solver::{
    CompressibleStepInfo, CompressibleUnstructuredStepView, LuSgsSweepUnstructuredInput,
    LuSgsSweepUnstructuredTypedParams, LuSgsUnstructuredCouplings, LuSgsUnstructuredSweepTyped,
    RefreshCompressibleStateTypedInput, SolverState, UnstructuredDriverConfig,
    finalize_cell_dts_from_sigma, lu_sgs_sweep_unstructured_typed,
    refresh_compressible_ghosts_and_primitives_typed,
};

pub(crate) struct UnstructuredTypedRhsWork<'a, T: ComputeFloat> {
    ghosts: &'a mut BoundaryGhostBuffer,
    primitives: &'a mut PrimitiveFieldsT<T>,
    gradients: &'a mut GradientFieldsT<T>,
    viscous_scratch: &'a mut ViscousAssemblyUnstructuredScratch,
    viscous_grad_scratch_f32: &'a mut UnstructuredGradientScratchF32,
    mesh_cache: &'a UnstructuredSolverMeshCache,
    exec: &'a mut ExecutionContext,
}

struct UnstructuredStepWorkTyped<T: ComputeFloat> {
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
    lusgs_couplings: LuSgsUnstructuredCouplings,
}

pub(crate) struct UnstructuredRunEnvTyped<'a> {
    config: &'a UnstructuredDriverConfig<'a>,
}

/// typed 非结构同步推进；结束时将场转为 `f64` 供输出。
#[allow(private_bounds)]
pub fn run_unstructured_typed_with_observer<T: UnstructuredComputeBackend>(
    config: &UnstructuredDriverConfig<'_>,
    fields: &mut ConservedFieldsT<T>,
    mut observe_step: impl FnMut(CompressibleUnstructuredStepView<'_>) -> Result<()>,
) -> Result<(Vec<CompressibleStepInfo>, ConservedFields)> {
    if matches!(config.time_scheme, TimeIntegrationScheme::Gmres) {
        return Err(AsimuError::Config(format!(
            "compute_precision = \"{}\" 的非结构 typed 路径暂不支持 {}",
            T::PRECISION.label(),
            config.time_scheme.label()
        )));
    }
    let mut env = UnstructuredRunEnvTyped { config };
    let n = env.config.mesh.num_cells();
    let mut work = {
        let _span = info_span!(
            "allocate_unstructured_work_typed",
            cells = n,
            precision = T::PRECISION.label(),
        )
        .entered();
        let mesh_cache =
            UnstructuredSolverMeshCache::from_mesh(env.config.mesh, env.config.patches)?;
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
        UnstructuredStepWorkTyped {
            storage: Rk4StorageT::new(n)?,
            state: SolverState::default(),
            integrator: RungeKutta4Integrator::new(RungeKutta4Config {
                dt: env.config.fixed_dt.unwrap_or(0.0),
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
            lusgs_couplings: LuSgsUnstructuredCouplings::from_mesh(env.config.mesh)?,
        }
    };
    let mut history = Vec::new();
    let control = TransientStepControl::new(env.config.residual_tolerance);
    loop {
        let step = advance_unstructured_step_typed(&mut env, fields, &mut work)?;
        let mut step = step;
        let stop = control.finalize_step(&mut step);
        info!(
            step = step.step,
            dt = %format_log_sci4(step.dt),
            t = %format_log_sci4(step.physical_time),
            log10_residual = %format_log_fixed4(step.residual_log10),
            cfl = %format_log_fixed5(step.cfl),
        );
        history.push(step);
        let fields_real = fields.cast_real()?;
        observe_step(CompressibleUnstructuredStepView {
            info: history.last().expect("history"),
            history: &history,
            fields: &fields_real,
        })?;
        if stop {
            break;
        }
    }
    Ok((history, fields.cast_real()?))
}

fn advance_unstructured_step_typed<T: UnstructuredComputeBackend>(
    env: &mut UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
) -> Result<CompressibleStepInfo> {
    let step_start = Instant::now();
    let cfl = env
        .config
        .cfl_schedule
        .at_step(work.state.time_step.saturating_add(1), env.config.max_steps);
    let p_floor = crate::field::positivity_pressure_floor(env.config.freestream.pressure);
    let compute_dt_start = Instant::now();
    let (cell_dts, sigma) = prepare_unstructured_timestep_typed(env, fields, work, cfl, p_floor)?;
    let compute_dt_ms = elapsed_ms(compute_dt_start);
    let dt = min_positive_dt(&cell_dts);
    work.integrator.config.dt = dt;
    let time_integration_start = Instant::now();
    {
        let _span = info_span!(
            "unstructured_time_integration_typed",
            scheme = env.config.time_scheme.label(),
            precision = T::PRECISION.label(),
        )
        .entered();
        match env.config.time_scheme {
            TimeIntegrationScheme::LuSgs => {
                advance_unstructured_lusgs_typed(env, fields, work, &cell_dts, &sigma, p_floor)?;
            }
            TimeIntegrationScheme::Euler | TimeIntegrationScheme::Rk4 => {
                advance_unstructured_explicit_typed(env, fields, work, dt, &cell_dts, p_floor)?;
            }
            scheme => {
                return Err(AsimuError::Config(format!(
                    "非结构 typed 路径暂不支持 time.scheme = \"{}\"",
                    scheme.label()
                )));
            }
        }
    }
    let time_integration_ms = elapsed_ms(time_integration_start);
    fields.enforce_positivity(env.config.eos, p_floor);
    let residual = work.storage.k1.density_rms_norm();
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
    })
}

fn advance_unstructured_lusgs_typed<T: UnstructuredComputeBackend>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    cell_dts: &[Real],
    sigma: &[Real],
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
    if lu_sgs.sweep {
        let mut sweep_params = LuSgsSweepUnstructuredTypedParams {
            mesh: env.config.mesh,
            eos: env.config.eos,
            primitives: &mut work.primitives,
            min_pressure: p_floor,
            backward_damping: lu_sgs.sweep_backward_damping,
        };
        let _span = info_span!(
            "unstructured_lusgs_sweep_typed",
            precision = T::PRECISION.label(),
        )
        .entered();
        lu_sgs_sweep_unstructured_typed(
            fields,
            &work.storage.k1,
            &mut sweep_params,
            LuSgsSweepUnstructuredInput {
                dt: cell_dts,
                sigma,
                volumes: &work.volumes,
                couplings: &work.lusgs_couplings,
                omega: lu_sgs.omega,
                gamma: env.config.eos.gamma,
            },
        )?;
    } else {
        {
            let _span = info_span!("unstructured_lusgs_diagonal_update_typed").entered();
            work.storage.stage.assign_lusgs_diagonal_update(
                &work.storage.u0,
                &work.storage.k1,
                sigma,
                cell_dts,
                lu_sgs.omega,
                env.config.eos.gamma,
                p_floor,
            )?;
        }
        {
            let _span = info_span!("unstructured_lusgs_copy_stage_typed").entered();
            fields.copy_from(&work.storage.stage)?;
        }
    }
    Ok(())
}

fn advance_unstructured_explicit_typed<T: UnstructuredRhsDispatchImpl>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    dt: Real,
    cell_dts: &[Real],
    p_floor: Real,
) -> Result<()> {
    let local = env.config.local_time_step;
    let scheme = env.config.time_scheme;
    let eos = env.config.eos;
    let mut reuse_current_state = true;
    let mut rhs_work = UnstructuredTypedRhsWork {
        ghosts: &mut work.ghosts,
        primitives: &mut work.primitives,
        gradients: &mut work.gradients,
        viscous_scratch: &mut work.viscous_scratch,
        viscous_grad_scratch_f32: &mut work.viscous_grad_scratch_f32,
        mesh_cache: &work.mesh_cache,
        exec: &mut work.exec,
    };
    let evaluate = |u: &ConservedFieldsT<T>, r: &mut ConservedResidualT<T>| {
        let refresh = !reuse_current_state;
        reuse_current_state = false;
        assemble_unstructured_typed_rhs(env, &mut rhs_work, u, r, refresh, p_floor)
    };
    match (scheme, local) {
        (TimeIntegrationScheme::Rk4, true) => rk4_step_local(
            fields,
            &mut work.storage,
            cell_dts,
            evaluate,
            Some(eos),
            p_floor,
        ),
        (TimeIntegrationScheme::Rk4, false) => rk4_step(fields, &mut work.storage, dt, evaluate),
        (TimeIntegrationScheme::Euler, true) => euler_step_local(
            fields,
            &mut work.storage,
            cell_dts,
            evaluate,
            Some(eos),
            p_floor,
        ),
        (TimeIntegrationScheme::Euler, false) => {
            euler_step(fields, &mut work.storage, dt, evaluate, Some(eos), p_floor)
        }
        _ => Err(AsimuError::Solver(
            "非结构 typed 显式推进收到不支持的时间格式".to_string(),
        )),
    }
}

fn assemble_unstructured_typed_rhs<T: UnstructuredRhsDispatchImpl>(
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

fn prepare_unstructured_timestep_typed<T: ComputeFloat + UnstructuredSpectralRadiusTyped>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    cfl: Real,
    p_floor: Real,
) -> Result<(Vec<Real>, Vec<Real>)> {
    fields.enforce_positivity(env.config.eos, p_floor);
    work.ghosts
        .ensure_face_capacity(env.config.mesh.num_faces());
    refresh_compressible_ghosts_and_primitives_typed(RefreshCompressibleStateTypedInput {
        boundary_mesh: env.config.mesh,
        patches: env.config.patches,
        fields,
        ghosts: &mut work.ghosts,
        eos: env.config.eos,
        freestream: env.config.freestream,
        reference: env.config.reference,
        viscous: env.config.viscous,
        min_pressure: p_floor,
        primitives: &mut work.primitives,
    })?;
    let sigma =
        T::cell_spectral_radius_unstructured_typed(&SpectralRadiusUnstructuredTypedParams {
            mesh: env.config.mesh,
            mesh_cache: &work.mesh_cache,
            boundaries: env.config.patches,
            ghosts: &work.ghosts,
            primitives: &work.primitives,
            eos: env.config.eos,
            min_pressure: p_floor,
            viscous: env.config.viscous,
        })?;
    let cell_dts = finalize_cell_dts_from_sigma(
        &work.volumes,
        &sigma,
        cfl,
        env.config.fixed_dt.filter(|dt| *dt > 0.0 && dt.is_finite()),
        env.config.local_time_step,
    )?;
    Ok((cell_dts, sigma))
}

/// 非结构可压缩求解热路径所需精度后端（ADR 0018；密封于 f32 / f64）。
pub(crate) trait UnstructuredComputeBackend:
    ComputeFloat
    + LusgsDiagonalUpdateBackend
    + InviscidFaceFluxTyped
    + InviscidTypedScatterBackend
    + ViscousTypedScatterBackend
    + UnstructuredSpectralRadiusTyped
    + LuSgsUnstructuredSweepTyped
    + UnstructuredRhsDispatchImpl
{
}

impl UnstructuredComputeBackend for f32 {}
impl UnstructuredComputeBackend for f64 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::discretization::InviscidFluxConfig;
    use crate::discretization::freestream_pair::FreestreamPairFixture;
    use crate::exec::ExecConfig;
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
    use crate::physics::{FreestreamParams, IdealGasEoS, ReferenceScales};
    use crate::solver::{
        CflSchedule, CompressibleEulerConfig, CompressibleEulerSolver,
        run_unstructured_with_observer,
    };

    fn single_tet_driver(
        side: &crate::discretization::freestream_pair::UniformFarfieldSide<'_>,
        reference: &ReferenceScales,
    ) -> (
        UnstructuredMesh3d,
        BoundarySet,
        IdealGasEoS,
        FreestreamParams,
        CompressibleEulerSolver,
        InviscidFluxConfig,
        ReferenceScales,
    ) {
        let mesh = UnstructuredMesh3d::new(
            "tet",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell")],
        )
        .expect("mesh");
        let faces = (0..mesh.num_faces())
            .map(|face| crate::core::FaceId(face as u32))
            .collect();
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces,
            BoundaryKind::Farfield {
                mach: side.fs.mach,
                pressure: side.fs.pressure,
                temperature: side.fs.temperature,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        let inviscid = InviscidFluxConfig::default();
        let solver = CompressibleEulerSolver::new(CompressibleEulerConfig::default());
        (
            mesh,
            boundary,
            *side.eos,
            *side.fs,
            solver,
            inviscid,
            reference.clone(),
        )
    }

    #[test]
    fn f32_unstructured_step_matches_f64_on_single_tet() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_side();
        let (mesh, boundary, eos, freestream, solver, inviscid, reference) =
            single_tet_driver(&side, &pair.reference);
        let driver = UnstructuredDriverConfig {
            solver: &solver,
            mesh: &mesh,
            eos: &eos,
            freestream: &freestream,
            inviscid: &inviscid,
            patches: &boundary,
            reference: Some(&reference),
            viscous: None,
            fixed_dt: Some(1.0e-4),
            local_time_step: true,
            time_scheme: TimeIntegrationScheme::Euler,
            lu_sgs: Default::default(),
            cfl_schedule: CflSchedule::constant(0.4),
            max_steps: 1,
            residual_tolerance: None,
            exec_config: ExecConfig::default(),
        };
        let base = ConservedFields::from_freestream_context(mesh.num_cells(), &side.ctx, side.fs)
            .expect("base fields");
        let mut fields_f32 = ConservedFieldsT::<f32>::from_real_fields(&base).expect("f32 fields");
        let mut fields_f64 = base;
        let (history_f32, out_f32) =
            run_unstructured_typed_with_observer::<f32>(&driver, &mut fields_f32, |_| Ok(()))
                .expect("f32 run");
        let history_f64 =
            run_unstructured_with_observer(&driver, &mut fields_f64, |_| Ok(())).expect("f64 run");
        assert_eq!(history_f32.len(), 1);
        assert_eq!(history_f64.len(), 1);
        assert!(history_f32[0].residual_rms.is_finite());
        assert!(history_f64[0].residual_rms.is_finite());
        let rel = (out_f32.density.values()[0] - fields_f64.density.values()[0]).abs()
            / fields_f64.density.values()[0].max(1.0e-12);
        assert!(rel < 1.0e-3, "rel={rel}");
    }
}
