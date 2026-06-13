//! 非结构 3D 可压缩 typed 时间推进驱动（ADR 0016 P3）。

use std::time::Instant;

use tracing::{info, info_span};

use crate::core::{
    ComputeFloat, Real, elapsed_ms, format_log_fixed4, format_log_sci4, log10_positive,
};
use crate::discretization::residual::InviscidAssemblyUnstructuredTypedParams;
use crate::discretization::{
    BoundaryGhostBuffer, GradientFields, ReconstructionKind, UnstructuredGradientLsqInput,
    UnstructuredSolverMeshCache, ViscousAssemblyUnstructuredScratch,
    ViscousAssemblyUnstructuredTypedInput, assemble_inviscid_residual_unstructured_typed,
    compute_gradients_and_assemble_viscous_unstructured_typed,
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq,
};
use crate::error::{AsimuError, Result};
use crate::exec::{ExecConfig, ExecutionContext, MeshExecMetrics};
use crate::field::{
    ConservedFields, ConservedFieldsT, ConservedResidualT, PrimitiveFields, PrimitiveFieldsT,
};
use crate::solver::spectral_radius_unstructured::{
    SpectralRadiusUnstructuredParams, cell_spectral_radius_unstructured,
};
use crate::solver::time::TransientStepControl;
use crate::solver::time::{
    Rk4StorageT, RungeKutta4Config, RungeKutta4Integrator, TimeIntegrationScheme, TimeIntegrator,
    euler_step, euler_step_local, min_positive_dt, rk4_step, rk4_step_local,
};
use crate::solver::{
    CompressibleStepInfo, CompressibleUnstructuredStepView, RefreshCompressibleStateTypedInput,
    SolverState, UnstructuredDriverConfig, finalize_cell_dts_from_sigma,
    refresh_compressible_ghosts_and_primitives_typed,
};

struct UnstructuredTypedRhsWork<'a, T: ComputeFloat> {
    ghosts: &'a mut BoundaryGhostBuffer,
    primitives: &'a mut PrimitiveFieldsT<T>,
    spectral_primitives: &'a mut PrimitiveFields,
    gradients: &'a mut GradientFields,
    viscous_scratch: &'a mut ViscousAssemblyUnstructuredScratch,
    mesh_cache: &'a UnstructuredSolverMeshCache,
    exec: &'a mut ExecutionContext,
}

struct UnstructuredStepWorkTyped<T: ComputeFloat> {
    storage: Rk4StorageT<T>,
    state: SolverState,
    integrator: RungeKutta4Integrator,
    ghosts: BoundaryGhostBuffer,
    primitives: PrimitiveFieldsT<T>,
    spectral_primitives: PrimitiveFields,
    gradients: GradientFields,
    viscous_scratch: ViscousAssemblyUnstructuredScratch,
    mesh_cache: UnstructuredSolverMeshCache,
    exec: ExecutionContext,
    volumes: Vec<Real>,
}

struct UnstructuredRunEnvTyped<'a> {
    config: &'a UnstructuredDriverConfig<'a>,
}

/// typed 非结构同步推进；结束时将场转为 `f64` 供输出。
pub fn run_unstructured_typed_with_observer<T: ComputeFloat>(
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
        let exec = ExecutionContext::new(
            ExecConfig {
                compute_precision: T::PRECISION,
                ..ExecConfig::default()
            },
            MeshExecMetrics::new(n, interior_faces, max_bucket_faces),
        );
        info!(
            compute_precision = ?exec.compute_precision(),
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
            spectral_primitives: PrimitiveFields::zeros(n)?,
            gradients: GradientFields::zeros(n)?,
            viscous_scratch: ViscousAssemblyUnstructuredScratch::new(n),
            mesh_cache,
            exec,
            volumes: env.config.mesh.cell_volumes(),
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
            cfl = step.cfl,
            stop,
            precision = T::PRECISION.label(),
            "非结构 3D typed 时间步"
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

fn advance_unstructured_step_typed<T: ComputeFloat>(
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
    info!(
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

fn advance_unstructured_lusgs_typed<T: ComputeFloat>(
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
    if env.config.lu_sgs.sweep {
        return Err(AsimuError::Config(
            "compute_precision f32 非结构 typed 路径暂不支持 lusgs_sweep = true".to_string(),
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
            spectral_primitives: &mut work.spectral_primitives,
            gradients: &mut work.gradients,
            viscous_scratch: &mut work.viscous_scratch,
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
    Ok(())
}

fn advance_unstructured_explicit_typed<T: ComputeFloat>(
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
        spectral_primitives: &mut work.spectral_primitives,
        gradients: &mut work.gradients,
        viscous_scratch: &mut work.viscous_scratch,
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

fn assemble_unstructured_typed_rhs<T: ComputeFloat>(
    env: &UnstructuredRunEnvTyped<'_>,
    work: &mut UnstructuredTypedRhsWork<'_, T>,
    fields: &ConservedFieldsT<T>,
    residual: &mut ConservedResidualT<T>,
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
            spectral_primitives: work.spectral_primitives,
        })?;
    }
    if env.config.inviscid.reconstruction == ReconstructionKind::Muscl {
        let grad_input = UnstructuredGradientLsqInput {
            mesh: env.config.mesh,
            mesh_cache: work.mesh_cache,
            primitives: work.spectral_primitives,
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
    let assembly = InviscidAssemblyUnstructuredTypedParams {
        mesh: env.config.mesh,
        eos: env.config.eos,
        config: env.config.inviscid,
        boundaries: env.config.patches,
        ghosts: work.ghosts,
        primitives: work.primitives,
        spectral_primitives: work.spectral_primitives,
        mesh_cache: work.mesh_cache,
        gradients: match env.config.inviscid.reconstruction {
            ReconstructionKind::Muscl => Some(&*work.gradients),
            ReconstructionKind::FirstOrder => None,
        },
        min_pressure: p_floor,
        exec: work.exec,
    };
    assemble_inviscid_residual_unstructured_typed(fields, residual, &assembly)?;
    if let Some(viscous) = env.config.viscous {
        let mut input = ViscousAssemblyUnstructuredTypedInput {
            mesh: env.config.mesh,
            mesh_cache: work.mesh_cache,
            eos: env.config.eos,
            viscous,
            boundaries: env.config.patches,
            ghosts: work.ghosts,
            primitives: work.spectral_primitives,
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

fn prepare_unstructured_timestep_typed<T: ComputeFloat>(
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
        spectral_primitives: &mut work.spectral_primitives,
    })?;
    let params = SpectralRadiusUnstructuredParams {
        mesh: env.config.mesh,
        mesh_cache: &work.mesh_cache,
        boundaries: env.config.patches,
        ghosts: &work.ghosts,
        primitives: &work.spectral_primitives,
        eos: env.config.eos,
        min_pressure: p_floor,
        viscous: env.config.viscous,
    };
    let sigma = cell_spectral_radius_unstructured(&params)?;
    let cell_dts = finalize_cell_dts_from_sigma(
        &work.volumes,
        &sigma,
        cfl,
        env.config.fixed_dt.filter(|dt| *dt > 0.0 && dt.is_finite()),
        env.config.local_time_step,
    )?;
    Ok((cell_dts, sigma))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::discretization::InviscidFluxConfig;
    use crate::discretization::freestream_pair::FreestreamPairFixture;
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
