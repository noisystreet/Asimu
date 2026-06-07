//! 3D 非结构可压缩算例编排（单域混合单元面循环）。

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use tracing::{info, info_span, warn};

use crate::core::{Real, format_log_fixed4, format_log_sci4, log10_positive, residual_converged};
use crate::discretization::{BoundaryGhostBuffer, GradientFields, ReconstructionKind};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::io::{CaseSpec, resolve_case_output_path};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{FreestreamParams, IdealGasEoS};
use crate::solver::spectral_radius_unstructured::{
    SpectralRadiusUnstructuredParams, cell_spectral_radius_unstructured,
};
use crate::solver::time::{
    Rk4Storage, RungeKutta4Config, RungeKutta4Integrator, TimeIntegrationScheme, TimeIntegrator,
    euler_step, euler_step_local, min_positive_dt, rk4_step, rk4_step_local,
};
use crate::solver::{
    CompressibleStepInfo, EvaluateRhsUnstructured, LuSgsSweepUnstructuredInput,
    LuSgsSweepUnstructuredParams, LuSgsUnstructuredCouplings, RefreshCompressibleStateInput,
    SolverState, finalize_cell_dts_from_sigma, lu_sgs_sweep_unstructured,
    refresh_compressible_ghosts_and_primitives,
};

use super::{CaseRunKind, CaseRunResult, Compressible3dRunMetrics};

pub(super) fn run(case: &CaseSpec) -> Result<CaseRunResult> {
    let mesh = case.mesh.as_unstructured_3d()?;
    run_compressible_unstructured_3d(case, mesh)
}

fn run_compressible_unstructured_3d(
    case: &CaseSpec,
    mesh: &UnstructuredMesh3d,
) -> Result<CaseRunResult> {
    let _span = info_span!(
        "run_compressible_unstructured_3d",
        cells = mesh.num_cells(),
        faces = mesh.num_faces(),
    )
    .entered();
    let (inviscid, eos, freestream) = {
        let _span = info_span!("prepare_unstructured_solver").entered();
        let disc = case.compressible_discretization()?;
        let inviscid = disc.inviscid();
        validate_unstructured_config(case, &inviscid)?;
        {
            let _span = info_span!(
                "validate_unstructured_boundary",
                patches = case.boundary.patches().len(),
                faces = mesh.num_faces(),
            )
            .entered();
            validate_boundary_coverage(mesh, &case.boundary)?;
        }
        let eos = case.physics.eos()?;
        let freestream = case
            .freestream
            .or(case.fluid_initial.freestream)
            .ok_or_else(|| AsimuError::Field("3D 可压缩算例须指定 [freestream]".to_string()))?;
        (inviscid, eos, freestream)
    };
    let mut env = UnstructuredRunEnv {
        case,
        mesh,
        eos: &eos,
        freestream: &freestream,
        inviscid,
    };
    let mut fields = {
        let _span = info_span!(
            "build_unstructured_initial_fields",
            cells = mesh.num_cells()
        )
        .entered();
        case.build_conserved_fields()?
    };
    let history = {
        let _span = info_span!("advance_unstructured_history").entered();
        advance_unstructured_history(&mut env, &mut fields)?
    };
    let last = history
        .last()
        .ok_or_else(|| AsimuError::Solver("非结构 3D 推进未产生任何时间步".to_string()))?;
    let metrics = Compressible3dRunMetrics {
        steps: last.step,
        final_time: last.physical_time,
        residual_rms: last.residual_rms,
        residual_log10: log10_positive(last.residual_rms),
        scheme: inviscid.short_label().to_string(),
        limiter: inviscid.limiter_label().to_string(),
        converged: last.converged,
    };
    let equation_label = unstructured_equation_label(case);
    log_unstructured_complete(
        &metrics,
        inviscid.short_label(),
        inviscid.limiter_label(),
        mesh,
        equation_label,
    );
    let output_paths = {
        let _span = info_span!("write_unstructured_outputs").entered();
        write_unstructured_outputs(case, mesh, &fields, &history)?
    };
    for path in output_paths {
        info!(path = %path.display(), "非结构算例输出");
    }
    Ok(CaseRunResult {
        name: case.name.clone(),
        benchmark_id: case.benchmark_id.clone(),
        kind: CaseRunKind::Compressible3dTransient,
        summary: format!(
            "3D unstructured {} {} t={} log10={} steps={} converged={} cells={}",
            equation_label,
            inviscid.short_label(),
            format_log_sci4(metrics.final_time),
            format_log_fixed4(metrics.residual_log10),
            metrics.steps,
            metrics.converged,
            mesh.num_cells()
        ),
        diffusion: None,
        sod: None,
        compressible_3d: Some(metrics),
    })
}

struct UnstructuredRunEnv<'a> {
    case: &'a CaseSpec,
    mesh: &'a UnstructuredMesh3d,
    eos: &'a IdealGasEoS,
    freestream: &'a FreestreamParams,
    inviscid: crate::discretization::InviscidFluxConfig,
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
    volumes: Vec<Real>,
    lusgs_couplings: LuSgsUnstructuredCouplings,
}

struct UnstructuredRhsWork<'a> {
    ghosts: &'a mut BoundaryGhostBuffer,
    primitives: &'a mut PrimitiveFields,
    gradients: &'a mut GradientFields,
    viscous_scratch: &'a mut crate::discretization::ViscousAssemblyUnstructuredScratch,
    mesh_cache: &'a crate::discretization::UnstructuredSolverMeshCache,
}

fn advance_unstructured_history(
    env: &mut UnstructuredRunEnv<'_>,
    fields: &mut ConservedFields,
) -> Result<Vec<CompressibleStepInfo>> {
    let n = env.mesh.num_cells();
    let mut work = {
        let _span = info_span!("allocate_unstructured_work", cells = n).entered();
        UnstructuredStepWork {
            storage: Rk4Storage::new(n)?,
            state: SolverState::default(),
            integrator: RungeKutta4Integrator::new(RungeKutta4Config {
                dt: env.case.time.dt.unwrap_or(0.0),
                max_steps: env.case.resolved_max_steps(),
            }),
            ghosts: BoundaryGhostBuffer::with_face_capacity(env.mesh.num_faces()),
            primitives: PrimitiveFields::zeros(n)?,
            gradients: GradientFields::zeros(n)?,
            viscous_scratch: crate::discretization::ViscousAssemblyUnstructuredScratch::new(n),
            mesh_cache: crate::discretization::UnstructuredSolverMeshCache::from_mesh(
                env.mesh,
                &env.case.boundary,
            )?,
            volumes: env.mesh.cell_volumes(),
            lusgs_couplings: LuSgsUnstructuredCouplings::from_mesh(env.mesh)?,
        }
    };
    let mut history = Vec::new();
    loop {
        let step = advance_unstructured_step(env, fields, &mut work)?;
        let converged = env
            .case
            .resolved_tolerance()
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
        maybe_write_unstructured_interval(
            env.case,
            env.mesh,
            fields,
            history.last().unwrap(),
            &history,
        )?;
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
    let cfl = env.case.cfl_schedule()?.at_step(
        work.state.time_step.saturating_add(1),
        env.case.resolved_max_steps(),
    );
    let p_floor = crate::field::positivity_pressure_floor(env.freestream.pressure);
    let compute_dt_start = Instant::now();
    let (cell_dts, sigma) = {
        let _span = info_span!(
            "compute_unstructured_dt",
            cells = env.mesh.num_cells(),
            faces = env.mesh.num_faces(),
            cfl = cfl,
            viscous = env.case.physics.viscous.is_some(),
        )
        .entered();
        prepare_unstructured_timestep(env, fields, work, cfl, p_floor)?
    };
    let compute_dt_ms = elapsed_ms(compute_dt_start);
    let dt = min_positive_dt(&cell_dts);
    work.integrator.config.dt = dt;
    let scheme = env.case.time.resolved_time_scheme();
    let time_integration_start = Instant::now();
    {
        let _span = info_span!(
            "unstructured_time_integration",
            scheme = scheme.label(),
            local_time_step = env.case.time.uses_local_time_step(),
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
        }
    }
    let time_integration_ms = elapsed_ms(time_integration_start);
    {
        let _span = info_span!("unstructured_enforce_positivity_post").entered();
        fields.enforce_positivity(env.eos, p_floor);
    }
    let post_rhs_start = Instant::now();
    let residual = {
        let _span = info_span!("unstructured_rhs_post").entered();
        let mut rhs_work = UnstructuredRhsWork {
            ghosts: &mut work.ghosts,
            primitives: &mut work.primitives,
            gradients: &mut work.gradients,
            viscous_scratch: &mut work.viscous_scratch,
            mesh_cache: &work.mesh_cache,
        };
        evaluate_unstructured_rhs(env, fields, &mut work.storage.k1, &mut rhs_work, p_floor)?;
        work.storage.k1.density_rms_norm()
    };
    let post_rhs_ms = elapsed_ms(post_rhs_start);
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
        cells = env.mesh.num_cells(),
        faces = env.mesh.num_faces(),
        profile_compute_dt_ms = %format_log_fixed4(compute_dt_ms),
        profile_time_integration_ms = %format_log_fixed4(time_integration_ms),
        profile_post_rhs_ms = %format_log_fixed4(post_rhs_ms),
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
    let local = env.case.time.uses_local_time_step();
    let scheme = env.case.time.resolved_time_scheme();
    let eos = env.eos;
    let mut rhs_work = UnstructuredRhsWork {
        ghosts: &mut work.ghosts,
        primitives: &mut work.primitives,
        gradients: &mut work.gradients,
        viscous_scratch: &mut work.viscous_scratch,
        mesh_cache: &work.mesh_cache,
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
    if !env.case.time.uses_local_time_step() {
        return Err(AsimuError::Config(
            "非结构 time.scheme = \"lu_sgs\" 须配合 local_time_step = true".to_string(),
        ));
    }
    let lu_sgs = env.case.time.resolved_lusgs_config()?;
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
            mesh: env.mesh,
            eos: env.eos,
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
                gamma: env.eos.gamma,
            },
        )
    } else {
        {
            let _span = info_span!("unstructured_lusgs_diagonal_update").entered();
            work.storage.stage.assign_lusgs_diagonal_update(
                &work.storage.u0,
                &work.storage.k1,
                sigma,
                cell_dts,
                lu_sgs.omega,
                env.eos.gamma,
                p_floor,
            )?;
        }
        let _span = info_span!("unstructured_lusgs_copy_stage").entered();
        fields.copy_from(&work.storage.stage)
    }
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
        fields.enforce_positivity(env.eos, p_floor);
    }
    {
        let _span = info_span!("unstructured_dt_refresh_state").entered();
        work.ghosts.ensure_face_capacity(env.mesh.num_faces());
        refresh_compressible_ghosts_and_primitives(RefreshCompressibleStateInput {
            boundary_mesh: env.mesh,
            patches: &env.case.boundary,
            fields,
            ghosts: &mut work.ghosts,
            eos: env.eos,
            freestream: env.freestream,
            reference: env.case.reference.as_ref(),
            viscous: env.case.physics.viscous.as_ref(),
            min_pressure: p_floor,
            primitives: &mut work.primitives,
        })?;
    }
    let params = SpectralRadiusUnstructuredParams {
        mesh: env.mesh,
        boundaries: &env.case.boundary,
        ghosts: &work.ghosts,
        primitives: &work.primitives,
        eos: env.eos,
        min_pressure: p_floor,
        viscous: env.case.physics.viscous.as_ref(),
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
            env.case.time.dt.filter(|dt| *dt > 0.0 && dt.is_finite()),
            env.case.time.uses_local_time_step(),
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
        mesh: env.mesh,
        mesh_cache: work.mesh_cache,
        patches: &env.case.boundary,
        ghosts: work.ghosts,
        eos: env.eos,
        freestream: env.freestream,
        reference: env.case.reference.as_ref(),
        inviscid: &env.inviscid,
        viscous: env.case.physics.viscous.as_ref(),
        min_pressure: p_floor,
        primitives: work.primitives,
        gradients: work.gradients,
        viscous_scratch: work.viscous_scratch,
    }
}

fn evaluate_unstructured_rhs(
    env: &UnstructuredRunEnv<'_>,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    work: &mut UnstructuredRhsWork<'_>,
    p_floor: Real,
) -> Result<()> {
    {
        let _span = info_span!("unstructured_rhs_refresh_state").entered();
        work.ghosts.ensure_face_capacity(env.mesh.num_faces());
        refresh_compressible_ghosts_and_primitives(RefreshCompressibleStateInput {
            boundary_mesh: env.mesh,
            patches: &env.case.boundary,
            fields,
            ghosts: work.ghosts,
            eos: env.eos,
            freestream: env.freestream,
            reference: env.case.reference.as_ref(),
            viscous: env.case.physics.viscous.as_ref(),
            min_pressure: p_floor,
            primitives: work.primitives,
        })?;
    }
    assemble_unstructured_rhs_from_current_state(env, fields, residual, work, p_floor)
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

fn validate_unstructured_config(
    case: &CaseSpec,
    inviscid: &crate::discretization::InviscidFluxConfig,
) -> Result<()> {
    let disc = case.compressible_discretization()?;
    match inviscid.reconstruction {
        ReconstructionKind::FirstOrder => {}
        ReconstructionKind::Muscl => {
            if inviscid.unstructured_gradient_limiter.is_none() {
                if disc.limiter.is_some() {
                    return Err(AsimuError::Config(
                        "非结构 MUSCL 须设置 unstructured_limiter = barth_jespersen | venkatakrishnan；\
                         结构化 limiter（minmod/van_leer/van_albada）不可在非结构 case 中复用（见 ADR 0012）"
                            .to_string(),
                    ));
                }
                return Err(AsimuError::Config(
                    "非结构 MUSCL 须设置 unstructured_limiter = barth_jespersen | venkatakrishnan"
                        .to_string(),
                ));
            }
            if disc.limiter.is_some() {
                warn!(
                    limiter = ?disc.limiter,
                    unstructured_limiter = ?disc.unstructured_limiter,
                    "非结构 MUSCL 忽略 [euler].limiter，使用 unstructured_limiter"
                );
            }
            if let Some(name) = disc.unstructured_limiter.as_deref() {
                if crate::discretization::UnstructuredGradientLimiter::parse(name).is_none() {
                    return Err(AsimuError::Config(format!(
                        "未知 unstructured_limiter \"{name}\"；可选 barth_jespersen | venkatakrishnan"
                    )));
                }
            }
        }
    }
    if case.time.residual_smoothing_config().enabled {
        warn!("非结构网格暂不支持结构化方向分裂残差光顺；本次忽略 residual_smoothing");
    }
    if case.time.resolved_time_scheme() == TimeIntegrationScheme::Gmres {
        return Err(AsimuError::Config(
            "非结构网格暂不支持 time.scheme = \"gmres\"".to_string(),
        ));
    }
    Ok(())
}

fn validate_boundary_coverage(
    mesh: &UnstructuredMesh3d,
    boundary: &crate::boundary::BoundarySet,
) -> Result<()> {
    let mut covered = HashSet::new();
    for patch in boundary.patches() {
        for &face in &patch.face_ids {
            if mesh.face_neighbor(face)?.is_some() {
                return Err(AsimuError::Boundary(format!(
                    "非结构边界 patch {} 引用了内部面 FaceId({})",
                    patch.name,
                    face.index()
                )));
            }
            covered.insert(face.index());
        }
    }
    let mut boundary_faces = 0usize;
    for face in 0..mesh.num_faces() {
        if mesh
            .face_neighbor(crate::core::FaceId(face as u32))?
            .is_none()
        {
            boundary_faces += 1;
        }
    }
    if covered.len() != boundary_faces {
        return Err(AsimuError::Boundary(format!(
            "非结构边界 patch 覆盖 {}/{} 个边界面，求解前须完整覆盖",
            covered.len(),
            boundary_faces
        )));
    }
    Ok(())
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

fn elapsed_ms(start: Instant) -> Real {
    start.elapsed().as_secs_f64() * 1000.0
}

fn maybe_write_unstructured_interval(
    case: &CaseSpec,
    mesh: &UnstructuredMesh3d,
    fields: &ConservedFields,
    step: &CompressibleStepInfo,
    history: &[CompressibleStepInfo],
) -> Result<()> {
    let Some(output) = &case.output else {
        return Ok(());
    };
    if !output.interval_output_due(step.step) {
        return Ok(());
    }
    let _ = super::output_3d::write_residual_outputs(case, history)?;
    let Some(base) = output.solution_cgns.as_ref() else {
        return Ok(());
    };
    let name = super::output_3d::flow_cgns_name_for_step(base, step.step);
    let cgns_path = resolve_case_output_path(case.case_dir.as_deref(), &output.dir, &name)?;
    write_unstructured_vtu(
        case,
        mesh,
        fields,
        step.physical_time,
        cgns_path.with_extension("vtu"),
    )?;
    Ok(())
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
    let vtu_path = cgns_path.with_extension("vtu");
    let physical_time = history.last().map(|s| s.physical_time).unwrap_or(0.0);
    write_unstructured_vtu(case, mesh, fields, physical_time, vtu_path.clone())?;
    warn!(
        requested = %cgns_path.display(),
        written = %vtu_path.display(),
        "非结构 CGNS 流场写出尚未实现，已写出 VTU"
    );
    written.push(vtu_path);
    Ok(written)
}

fn write_unstructured_vtu(
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
