//! 非结构 3D 可压缩时间推进驱动（不含 case 输出编排）。

use std::time::Instant;

use tracing::{info, info_span};

use crate::core::{
    Real, elapsed_ms, format_log_fixed4, format_log_sci4, log10_positive, residual_converged,
};
use crate::discretization::InviscidFluxConfig;
use crate::discretization::{BoundaryGhostBuffer, GradientFields};
use crate::error::{AsimuError, Result};
use crate::exec::{ExecConfig, ExecutionContext, MeshExecMetrics};
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{FreestreamParams, IdealGasEoS, ReferenceScales, ViscousPhysicsConfig};
use crate::solver::spectral_radius_unstructured::{
    SpectralRadiusUnstructuredParams, cell_spectral_radius_unstructured,
};
use crate::solver::time::{
    CflSchedule, LuSgsConfig, Rk4Storage, RungeKutta4Config, RungeKutta4Integrator,
    TimeIntegrationScheme, TimeIntegrator, euler_step, euler_step_local, min_positive_dt, rk4_step,
    rk4_step_local,
};
use crate::solver::{
    CompressibleEulerSolver, CompressibleStepInfo, EvaluateRhsUnstructured,
    LuSgsSweepUnstructuredInput, LuSgsSweepUnstructuredParams, LuSgsUnstructuredCouplings,
    RefreshCompressibleStateInput, SolverState, finalize_cell_dts_from_sigma,
    lu_sgs_sweep_unstructured, refresh_compressible_ghosts_and_primitives,
};

/// 非结构可压缩外层步只读视图（observer 回调参数）。
#[derive(Debug, Clone, Copy)]
pub struct CompressibleUnstructuredStepView<'a> {
    pub info: &'a CompressibleStepInfo,
    pub history: &'a [CompressibleStepInfo],
    pub fields: &'a ConservedFields,
}

/// 非结构推进配置（由 case 层从 `CaseSpec` 组装）。
pub struct UnstructuredDriverConfig<'a> {
    pub solver: &'a CompressibleEulerSolver,
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
    pub inviscid: &'a InviscidFluxConfig,
    pub patches: &'a crate::boundary::BoundarySet,
    pub reference: Option<&'a ReferenceScales>,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    pub fixed_dt: Option<Real>,
    pub local_time_step: bool,
    pub time_scheme: TimeIntegrationScheme,
    pub lu_sgs: LuSgsConfig,
    pub cfl_schedule: CflSchedule,
    pub max_steps: u64,
    pub residual_tolerance: Option<Real>,
}

struct UnstructuredRunEnv<'a> {
    config: &'a UnstructuredDriverConfig<'a>,
}

struct UnstructuredStepWork {
    storage: Rk4Storage,
    state: SolverState,
    integrator: RungeKutta4Integrator,
    ghosts: BoundaryGhostBuffer,
    primitives: PrimitiveFields,
    gradients: GradientFields,
    viscous_scratch: crate::discretization::ViscousAssemblyUnstructuredScratch,
    mesh_cache: crate::discretization::UnstructuredSolverMeshCache,
    exec: ExecutionContext,
    volumes: Vec<Real>,
    lusgs_couplings: LuSgsUnstructuredCouplings,
}

struct UnstructuredRhsWork<'a> {
    ghosts: &'a mut BoundaryGhostBuffer,
    primitives: &'a mut PrimitiveFields,
    gradients: &'a mut GradientFields,
    viscous_scratch: &'a mut crate::discretization::ViscousAssemblyUnstructuredScratch,
    mesh_cache: &'a crate::discretization::UnstructuredSolverMeshCache,
    exec: &'a mut crate::exec::ExecutionContext,
}

pub fn run_unstructured_with_observer(
    config: &UnstructuredDriverConfig<'_>,
    fields: &mut ConservedFields,
    mut observe_step: impl FnMut(CompressibleUnstructuredStepView<'_>) -> Result<()>,
) -> Result<Vec<CompressibleStepInfo>> {
    let mut env = UnstructuredRunEnv { config };
    let n = env.config.mesh.num_cells();
    let mut work = {
        let _span = info_span!("allocate_unstructured_work", cells = n).entered();
        let mesh_cache = crate::discretization::UnstructuredSolverMeshCache::from_mesh(
            env.config.mesh,
            env.config.patches,
        )?;
        let interior_faces = mesh_cache.face_topology.interior.len();
        let max_bucket_faces = mesh_cache
            .face_topology
            .interior_coloring
            .max_bucket_faces();
        let exec = ExecutionContext::new(
            ExecConfig::default(),
            MeshExecMetrics::new(n, interior_faces, max_bucket_faces),
        );
        UnstructuredStepWork {
            storage: Rk4Storage::new(n)?,
            state: SolverState::default(),
            integrator: RungeKutta4Integrator::new(RungeKutta4Config {
                dt: env.config.fixed_dt.unwrap_or(0.0),
                max_steps: env.config.max_steps,
            }),
            ghosts: BoundaryGhostBuffer::with_face_capacity(env.config.mesh.num_faces()),
            primitives: PrimitiveFields::zeros(n)?,
            gradients: GradientFields::zeros(n)?,
            viscous_scratch: crate::discretization::ViscousAssemblyUnstructuredScratch::new(n),
            mesh_cache,
            exec,
            volumes: env.config.mesh.cell_volumes(),
            lusgs_couplings: LuSgsUnstructuredCouplings::from_mesh(env.config.mesh)?,
        }
    };
    let mut history = Vec::new();
    loop {
        let step = advance_unstructured_step(&mut env, fields, &mut work)?;
        let converged = env
            .config
            .residual_tolerance
            .is_some_and(|tol| residual_converged(step.residual_rms, tol));
        let mut step = step;
        step.converged = converged;
        let stop = step.is_final || step.converged;
        info!(
            step = step.step,
            dt = %format_log_sci4(step.dt),
            t = %format_log_sci4(step.physical_time),
            log10_residual = %format_log_fixed4(step.residual_log10),
            cfl = step.cfl,
            stop,
            "非结构 3D 时间步"
        );
        history.push(step);
        observe_step(CompressibleUnstructuredStepView {
            info: history.last().expect("history"),
            history: &history,
            fields,
        })?;
        if stop {
            break;
        }
    }
    Ok(history)
}

fn advance_unstructured_step(
    env: &mut UnstructuredRunEnv<'_>,
    fields: &mut ConservedFields,
    work: &mut UnstructuredStepWork,
) -> Result<CompressibleStepInfo> {
    let step_start = Instant::now();
    let cfl = env
        .config
        .cfl_schedule
        .at_step(work.state.time_step.saturating_add(1), env.config.max_steps);
    let p_floor = crate::field::positivity_pressure_floor(env.config.freestream.pressure);
    let compute_dt_start = Instant::now();
    let (cell_dts, sigma) = {
        let _span = info_span!(
            "compute_unstructured_dt",
            cells = env.config.mesh.num_cells(),
            faces = env.config.mesh.num_faces(),
            cfl = cfl,
            viscous = env.config.viscous.is_some(),
        )
        .entered();
        prepare_unstructured_timestep(env, fields, work, cfl, p_floor)?
    };
    let compute_dt_ms = elapsed_ms(compute_dt_start);
    let dt = min_positive_dt(&cell_dts);
    work.integrator.config.dt = dt;
    let scheme = env.config.time_scheme;
    let time_integration_start = Instant::now();
    {
        let _span = info_span!(
            "unstructured_time_integration",
            scheme = scheme.label(),
            local_time_step = env.config.local_time_step,
        )
        .entered();
        match scheme {
            TimeIntegrationScheme::LuSgs => {
                advance_unstructured_lusgs(env, fields, work, &cell_dts, &sigma, p_floor)?;
            }
            TimeIntegrationScheme::Euler | TimeIntegrationScheme::Rk4 => {
                advance_unstructured_explicit(env, fields, work, dt, &cell_dts, p_floor)?;
            }
            TimeIntegrationScheme::Gmres => {
                return Err(AsimuError::Config(
                    "非结构网格暂不支持 time.scheme = \"gmres\"".to_string(),
                ));
            }
            TimeIntegrationScheme::Simplec => {
                return Err(AsimuError::Config(
                    "非结构可压缩网格不支持 time.scheme = \"simplec\"".to_string(),
                ));
            }
            TimeIntegrationScheme::Piso => {
                return Err(AsimuError::Config(
                    "非结构可压缩网格不支持 time.scheme = \"piso\"".to_string(),
                ));
            }
        }
    }
    let time_integration_ms = elapsed_ms(time_integration_start);
    {
        let _span = info_span!("unstructured_enforce_positivity_post").entered();
        fields.enforce_positivity(env.config.eos, p_floor);
    }
    let rhs_monitor_start = Instant::now();
    let residual = {
        let _span = info_span!("unstructured_rhs_monitor").entered();
        // 监控量 = ‖R(U^0)‖：显式 RK4/Euler stage1 与 LU-SGS lusgs_rhs 均已写入 k1。
        work.storage.k1.density_rms_norm()
    };
    let rhs_monitor_ms = elapsed_ms(rhs_monitor_start);
    let advance_clock_start = Instant::now();
    let time_info = {
        let _span = info_span!("unstructured_advance_clock").entered();
        work.integrator.advance(&mut work.state)?
    };
    let advance_clock_ms = elapsed_ms(advance_clock_start);
    let step_total_ms = elapsed_ms(step_start);
    info!(
        step = work.state.time_step,
        scheme = scheme.label(),
        cells = env.config.mesh.num_cells(),
        faces = env.config.mesh.num_faces(),
        profile_compute_dt_ms = %format_log_fixed4(compute_dt_ms),
        profile_time_integration_ms = %format_log_fixed4(time_integration_ms),
        profile_rhs_monitor_ms = %format_log_fixed4(rhs_monitor_ms),
        profile_advance_clock_ms = %format_log_fixed4(advance_clock_ms),
        profile_step_total_ms = %format_log_fixed4(step_total_ms),
        "非结构时间步 profiling",
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

fn advance_unstructured_explicit(
    env: &mut UnstructuredRunEnv<'_>,
    fields: &mut ConservedFields,
    work: &mut UnstructuredStepWork,
    dt: Real,
    cell_dts: &[Real],
    p_floor: Real,
) -> Result<()> {
    let local = env.config.local_time_step;
    let scheme = env.config.time_scheme;
    let eos = env.config.eos;
    let mut rhs_work = UnstructuredRhsWork {
        ghosts: &mut work.ghosts,
        primitives: &mut work.primitives,
        gradients: &mut work.gradients,
        viscous_scratch: &mut work.viscous_scratch,
        mesh_cache: &work.mesh_cache,
        exec: &mut work.exec,
    };
    let mut reuse_current_state = true;
    let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
        let _span = info_span!("unstructured_explicit_rhs").entered();
        let mut evaluator = rhs_evaluator(env, &mut rhs_work, p_floor);
        if reuse_current_state {
            reuse_current_state = false;
            evaluator.assemble_from_current_state(u, r)
        } else {
            evaluator.run(u, r)
        }
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
            "非结构显式推进收到不支持的时间格式".to_string(),
        )),
    }
}

fn advance_unstructured_lusgs(
    env: &mut UnstructuredRunEnv<'_>,
    fields: &mut ConservedFields,
    work: &mut UnstructuredStepWork,
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
        let _span = info_span!("unstructured_lusgs_copy_base").entered();
        work.storage.u0.copy_from(fields)?;
    }
    {
        let _span = info_span!("unstructured_lusgs_rhs").entered();
        let mut rhs_work = UnstructuredRhsWork {
            ghosts: &mut work.ghosts,
            primitives: &mut work.primitives,
            gradients: &mut work.gradients,
            viscous_scratch: &mut work.viscous_scratch,
            mesh_cache: &work.mesh_cache,
            exec: &mut work.exec,
        };
        assemble_unstructured_rhs_from_current_state(
            env,
            &work.storage.u0,
            &mut work.storage.k1,
            &mut rhs_work,
            p_floor,
        )?;
    }
    if lu_sgs.sweep {
        let mut sweep_params = LuSgsSweepUnstructuredParams {
            mesh: env.config.mesh,
            eos: env.config.eos,
            primitives: &mut work.primitives,
            min_pressure: p_floor,
            backward_damping: lu_sgs.sweep_backward_damping,
        };
        let _span = info_span!("unstructured_lusgs_sweep").entered();
        lu_sgs_sweep_unstructured(
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
            let _span = info_span!("unstructured_lusgs_diagonal_update").entered();
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
        let _span = info_span!("unstructured_lusgs_copy_stage").entered();
        fields.copy_from(&work.storage.stage)?;
    }
    Ok(())
}

fn prepare_unstructured_timestep(
    env: &mut UnstructuredRunEnv<'_>,
    fields: &mut ConservedFields,
    work: &mut UnstructuredStepWork,
    cfl: Real,
    p_floor: Real,
) -> Result<(Vec<Real>, Vec<Real>)> {
    {
        let _span = info_span!("unstructured_dt_enforce_positivity").entered();
        fields.enforce_positivity(env.config.eos, p_floor);
    }
    {
        let _span = info_span!("unstructured_dt_refresh_state").entered();
        work.ghosts
            .ensure_face_capacity(env.config.mesh.num_faces());
        refresh_compressible_ghosts_and_primitives(RefreshCompressibleStateInput {
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
    }
    let params = SpectralRadiusUnstructuredParams {
        mesh: env.config.mesh,
        mesh_cache: &work.mesh_cache,
        boundaries: env.config.patches,
        ghosts: &work.ghosts,
        primitives: &work.primitives,
        eos: env.config.eos,
        min_pressure: p_floor,
        viscous: env.config.viscous,
    };
    let sigma = {
        let _span = info_span!("unstructured_cell_spectral_radius").entered();
        cell_spectral_radius_unstructured(&params)?
    };
    let cell_dts = {
        let _span = info_span!("unstructured_local_dt_spectral").entered();
        finalize_cell_dts_from_sigma(
            &work.volumes,
            &sigma,
            cfl,
            env.config.fixed_dt.filter(|dt| *dt > 0.0 && dt.is_finite()),
            env.config.local_time_step,
        )?
    };
    Ok((cell_dts, sigma))
}

fn rhs_evaluator<'a>(
    env: &'a UnstructuredRunEnv<'_>,
    work: &'a mut UnstructuredRhsWork<'_>,
    p_floor: Real,
) -> EvaluateRhsUnstructured<'a> {
    EvaluateRhsUnstructured {
        mesh: env.config.mesh,
        mesh_cache: work.mesh_cache,
        patches: env.config.patches,
        ghosts: work.ghosts,
        eos: env.config.eos,
        freestream: env.config.freestream,
        reference: env.config.reference,
        inviscid: env.config.inviscid,
        viscous: env.config.viscous,
        min_pressure: p_floor,
        primitives: work.primitives,
        gradients: work.gradients,
        viscous_scratch: work.viscous_scratch,
        exec: work.exec,
    }
}

fn assemble_unstructured_rhs_from_current_state(
    env: &UnstructuredRunEnv<'_>,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    work: &mut UnstructuredRhsWork<'_>,
    p_floor: Real,
) -> Result<()> {
    rhs_evaluator(env, work, p_floor).assemble_from_current_state(fields, residual)
}
