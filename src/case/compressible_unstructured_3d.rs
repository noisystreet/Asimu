//! 3D 非结构可压缩算例编排（单域混合单元面循环）。

use std::collections::HashSet;
use std::path::PathBuf;

use tracing::{info, info_span, warn};

use crate::core::{Real, format_log_fixed4, format_log_sci4, log10_positive, residual_converged};
use crate::discretization::{
    BoundaryGhostBuffer, GradientFields, InviscidAssemblyUnstructuredParams, ReconstructionKind,
    apply_compressible_boundary_conditions, assemble_inviscid_residual_unstructured,
    compute_gradients_and_assemble_viscous_unstructured,
};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::io::{CaseSpec, resolve_case_output_path};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{FreestreamContext, FreestreamParams, IdealGasEoS};
use crate::solver::spectral_radius::cell_local_dt_spectral;
use crate::solver::spectral_radius_unstructured::{
    SpectralRadiusUnstructuredParams, cell_spectral_radius_unstructured,
};
use crate::solver::time::{
    Rk4Storage, RungeKutta4Config, RungeKutta4Integrator, TimeIntegrationScheme, TimeIntegrator,
    euler_step, euler_step_local, min_positive_dt, rk4_step, rk4_step_local,
};
use crate::solver::{
    CompressibleStepInfo, LuSgsSweepUnstructuredInput, LuSgsSweepUnstructuredParams, SolverState,
    lu_sgs_sweep_unstructured,
};

use super::{CaseRunKind, CaseRunResult, Compressible3dRunMetrics};

pub fn run(case: &CaseSpec) -> Result<CaseRunResult> {
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
    let disc = case.compressible_discretization()?;
    let inviscid = disc.inviscid();
    validate_unstructured_config(case, &inviscid)?;
    validate_boundary_coverage(mesh, &case.boundary)?;
    let eos = case.physics.eos()?;
    let freestream = case
        .freestream
        .or(case.fluid_initial.freestream)
        .ok_or_else(|| AsimuError::Field("3D 可压缩算例须指定 [freestream]".to_string()))?;
    let mut env = UnstructuredRunEnv {
        case,
        mesh,
        eos: &eos,
        freestream: &freestream,
        inviscid,
    };
    let mut fields = case.build_conserved_fields()?;
    let history = advance_unstructured_history(&mut env, &mut fields)?;
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
    let output_paths = write_unstructured_outputs(case, mesh, &fields, &history)?;
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
}

fn advance_unstructured_history(
    env: &mut UnstructuredRunEnv<'_>,
    fields: &mut ConservedFields,
) -> Result<Vec<CompressibleStepInfo>> {
    let n = env.mesh.num_cells();
    let mut work = UnstructuredStepWork {
        storage: Rk4Storage::new(n)?,
        state: SolverState::default(),
        integrator: RungeKutta4Integrator::new(RungeKutta4Config {
            dt: env.case.time.dt.unwrap_or(0.0),
            max_steps: env.case.resolved_max_steps(),
        }),
        ghosts: BoundaryGhostBuffer::new(),
        primitives: PrimitiveFields::zeros(n)?,
        gradients: GradientFields::zeros(n)?,
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
    let cfl = env.case.cfl_schedule()?.at_step(
        work.state.time_step.saturating_add(1),
        env.case.resolved_max_steps(),
    );
    let p_floor = crate::field::positivity_pressure_floor(env.freestream.pressure);
    let (cell_dts, sigma) = prepare_unstructured_timestep(env, fields, work, cfl, p_floor)?;
    let dt = min_positive_dt(&cell_dts);
    work.integrator.config.dt = dt;
    match env.case.time.resolved_time_scheme() {
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
    fields.enforce_positivity(env.eos, p_floor);
    evaluate_unstructured_rhs(
        env,
        fields,
        &mut work.storage.k1,
        &mut work.ghosts,
        &mut work.primitives,
        &mut work.gradients,
        p_floor,
    )?;
    let residual = work.storage.k1.density_rms_norm();
    let time_info = work.integrator.advance(&mut work.state)?;
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
    let ghosts = &mut work.ghosts;
    let primitives = &mut work.primitives;
    let gradients = &mut work.gradients;
    let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
        evaluate_unstructured_rhs(env, u, r, ghosts, primitives, gradients, p_floor)
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
    work.storage.u0.copy_from(fields)?;
    evaluate_unstructured_rhs(
        env,
        &work.storage.u0,
        &mut work.storage.k1,
        &mut work.ghosts,
        &mut work.primitives,
        &mut work.gradients,
        p_floor,
    )?;
    if lu_sgs.sweep {
        let volumes = env.mesh.cell_volumes();
        let mut sweep_params = LuSgsSweepUnstructuredParams {
            mesh: env.mesh,
            eos: env.eos,
            primitives: &mut work.primitives,
            min_pressure: p_floor,
            backward_damping: lu_sgs.sweep_backward_damping,
        };
        lu_sgs_sweep_unstructured(
            fields,
            &work.storage.k1,
            &mut sweep_params,
            LuSgsSweepUnstructuredInput {
                dt: cell_dts,
                sigma,
                volumes: &volumes,
                omega: lu_sgs.omega,
                gamma: env.eos.gamma,
            },
        )
    } else {
        work.storage.stage.assign_lusgs_diagonal_update(
            &work.storage.u0,
            &work.storage.k1,
            sigma,
            cell_dts,
            lu_sgs.omega,
            env.eos.gamma,
            p_floor,
        )?;
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
    fields.enforce_positivity(env.eos, p_floor);
    refresh_unstructured_ghosts_and_primitives(
        env,
        fields,
        &mut work.ghosts,
        &mut work.primitives,
        p_floor,
    )?;
    let params = SpectralRadiusUnstructuredParams {
        mesh: env.mesh,
        boundaries: &env.case.boundary,
        ghosts: &work.ghosts,
        primitives: &work.primitives,
        eos: env.eos,
        min_pressure: p_floor,
        viscous: env.case.physics.viscous.as_ref(),
    };
    let sigma = cell_spectral_radius_unstructured(&params)?;
    let volumes = env.mesh.cell_volumes();
    let mut cell_dts = cell_local_dt_spectral(&volumes, &sigma, cfl)?;
    if let Some(dt) = env.case.time.dt.filter(|dt| *dt > 0.0 && dt.is_finite()) {
        cell_dts.fill(dt);
    } else if !env.case.time.uses_local_time_step() {
        let dt = min_positive_dt(&cell_dts);
        cell_dts.fill(dt);
    }
    Ok((cell_dts, sigma))
}

fn evaluate_unstructured_rhs(
    env: &mut UnstructuredRunEnv<'_>,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    ghosts: &mut BoundaryGhostBuffer,
    primitives: &mut PrimitiveFields,
    gradients: &mut GradientFields,
    p_floor: Real,
) -> Result<()> {
    refresh_unstructured_ghosts_and_primitives(env, fields, ghosts, primitives, p_floor)?;
    let params = InviscidAssemblyUnstructuredParams {
        mesh: env.mesh,
        eos: env.eos,
        config: &env.inviscid,
        boundaries: &env.case.boundary,
        ghosts,
        primitives,
    };
    assemble_inviscid_residual_unstructured(fields, residual, &params)?;
    if let Some(viscous) = env.case.physics.viscous.as_ref() {
        let mut input = crate::discretization::ViscousAssemblyUnstructuredInput {
            mesh: env.mesh,
            eos: env.eos,
            viscous,
            boundaries: &env.case.boundary,
            ghosts,
            primitives,
            min_pressure: p_floor,
            gradient_scratch: gradients,
        };
        compute_gradients_and_assemble_viscous_unstructured(residual, &mut input)?;
    }
    Ok(())
}

fn refresh_unstructured_ghosts_and_primitives(
    env: &UnstructuredRunEnv<'_>,
    fields: &ConservedFields,
    ghosts: &mut BoundaryGhostBuffer,
    primitives: &mut PrimitiveFields,
    p_floor: Real,
) -> Result<()> {
    let viscous = env.case.physics.viscous.as_ref();
    let fs_ctx = FreestreamContext::new(env.eos, env.case.reference.as_ref(), viscous);
    apply_compressible_boundary_conditions(
        env.mesh,
        &env.case.boundary,
        fields,
        ghosts,
        &fs_ctx,
        env.freestream,
        viscous,
    )?;
    primitives.fill_from_conserved(fields, env.eos, p_floor)
}

fn validate_unstructured_config(
    case: &CaseSpec,
    inviscid: &crate::discretization::InviscidFluxConfig,
) -> Result<()> {
    if inviscid.reconstruction != ReconstructionKind::FirstOrder {
        return Err(AsimuError::Config(
            "非结构网格当前仅支持 reconstruction = \"first_order\"".to_string(),
        ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::io::{CaseMesh, parse_case_str};
    use crate::mesh::{CellKind, UnstructuredCell};

    #[test]
    fn runs_single_tet_unstructured_smoke_step() {
        let mut case = parse_case_str(
            r#"
name = "unstructured_smoke"
[mesh]
kind = "structured_3d"
nx = 1
ny = 1
nz = 1

[physics]
gamma = 1.4
gas_constant = 287.0

[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15

[euler]
flux = "hllc"
reconstruction = "first_order"

[time]
scheme = "euler"
local_time_step = true
max_steps = 1
"#,
        )
        .expect("parse");
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
            .collect::<Vec<_>>();
        let fs = case.freestream.expect("freestream");
        case.mesh = CaseMesh::Unstructured3d(mesh);
        case.boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces,
            BoundaryKind::Farfield {
                mach: fs.mach,
                pressure: fs.pressure,
                temperature: fs.temperature,
                alpha: fs.alpha,
                beta: fs.beta,
            },
        )]);
        let result = run(&case).expect("run");
        let metrics = result.compressible_3d.expect("metrics");
        assert_eq!(metrics.steps, 1);
        assert!(metrics.residual_rms.is_finite());
    }

    #[test]
    fn runs_single_tet_unstructured_lusgs_sweep_step() {
        let mut case = parse_case_str(
            r#"
name = "unstructured_lusgs_sweep"
[mesh]
kind = "structured_3d"
nx = 1
ny = 1
nz = 1

[physics]
gamma = 1.4
gas_constant = 287.0

[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15

[euler]
flux = "hllc"
reconstruction = "first_order"

[time]
scheme = "lu_sgs"
local_time_step = true
lusgs_sweep = true
lusgs_sweep_backward_damping = 0.5
max_steps = 1
"#,
        )
        .expect("parse");
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
            .collect::<Vec<_>>();
        let fs = case.freestream.expect("freestream");
        case.mesh = CaseMesh::Unstructured3d(mesh);
        case.boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces,
            BoundaryKind::Farfield {
                mach: fs.mach,
                pressure: fs.pressure,
                temperature: fs.temperature,
                alpha: fs.alpha,
                beta: fs.beta,
            },
        )]);
        let result = run(&case).expect("run");
        let metrics = result.compressible_3d.expect("metrics");
        assert_eq!(metrics.steps, 1);
        assert!(metrics.residual_rms.is_finite());
    }
}
