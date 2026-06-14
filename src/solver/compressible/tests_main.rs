use super::*;
use crate::boundary::BoundarySet;
use crate::core::approx_eq;
use crate::discretization::BoundaryGhostBuffer;
use crate::field::PrimitiveFields;
use crate::physics::{ConservedState, PrimitiveState};
#[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
use std::collections::HashSet;

#[test]
fn uniform_1d_field_remains_stationary_over_steps() {
    let mesh = StructuredMesh1d::new("line", 8, 0.0, 1.0).expect("mesh");
    let eos = IdealGasEoS::AIR_STANDARD;
    let mut fields =
        ConservedFields::from_freestream(8, &eos, &FreestreamParams::default()).expect("fields");
    let reference = fields.clone();
    let ctx = CompressibleAdvanceContext1d {
        mesh: &mesh,
        boundary: crate::discretization::InviscidBoundary1d::ZeroGradient,
        eos: &eos,
    };
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
        time: RungeKutta4Config {
            dt: 1.0e-5,
            max_steps: 2,
        },
        ..CompressibleEulerConfig::default()
    });
    solver.run_transient_1d(&ctx, &mut fields).expect("run");
    for i in 0..mesh.num_cells() {
        assert!(approx_eq(
            fields.density.values()[i],
            reference.density.values()[i],
            1.0e-8,
        ));
    }
}

#[test]
fn sod_like_disturbance_evolve_with_rk4() {
    let mesh = StructuredMesh1d::new("sod", 16, 0.0, 1.0).expect("mesh");
    let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
    let left = ConservedState::from_primitive(
        &eos,
        &PrimitiveState {
            density: 1.0,
            velocity: [0.0, 0.0, 0.0],
            pressure: 1.0,
            temperature: 1.0,
        },
    )
    .expect("left");
    let right = ConservedState::from_primitive(
        &eos,
        &PrimitiveState {
            density: 0.125,
            velocity: [0.0, 0.0, 0.0],
            pressure: 0.1,
            temperature: 1.0,
        },
    )
    .expect("right");
    let mut fields = ConservedFields::uniform(mesh.num_cells(), left).expect("fields");
    let mid = mesh.num_cells() / 2;
    for i in mid..mesh.num_cells() {
        fields.density.values_mut()[i] = right.density;
        fields.total_energy.values_mut()[i] = right.total_energy;
    }
    let rho_before = fields.density.values()[mid - 1];
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
        time: RungeKutta4Config {
            dt: 2.0e-4,
            max_steps: 5,
        },
        ..CompressibleEulerConfig::default()
    });
    solver
        .run_transient_1d(
            &CompressibleAdvanceContext1d {
                mesh: &mesh,
                boundary: crate::discretization::InviscidBoundary1d::ZeroGradient,
                eos: &eos,
            },
            &mut fields,
        )
        .expect("run");
    assert!(fields.density.values()[mid - 1] != rho_before);
}

#[test]
fn lusgs_3d_honors_fixed_dt() {
    let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
    let eos = IdealGasEoS::AIR_STANDARD;
    let fs = FreestreamParams {
        mach: 0.2,
        ..FreestreamParams::default()
    };
    let mut fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
    let mut ghosts = BoundaryGhostBuffer::new();
    let boundary = BoundarySet::default();
    let mut storage = Rk4Storage::new(mesh.num_cells()).expect("storage");
    let mut state = SolverState::default();
    let fixed_dt = 1.0e-4;
    let mut integrator = RungeKutta4Integrator::new(RungeKutta4Config {
        dt: 0.0,
        max_steps: 1,
    });
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
        time: RungeKutta4Config {
            dt: fixed_dt,
            max_steps: 1,
        },
        time_mode: CompressibleTimeMode::Steady,
        local_time_step: true,
        time_scheme: TimeIntegrationScheme::LuSgs,
        lu_sgs: LuSgsConfig {
            sweep: false,
            ..LuSgsConfig::default()
        },
        ..CompressibleEulerConfig::default()
    });
    let mut ctx = CompressibleAdvanceContext3d {
        mesh: &mesh,
        structured: &mesh,
        patches: &boundary,
        ghosts: &mut ghosts,
        eos: &eos,
        freestream: &fs,
        reference: None,
        primitive_scratch: PrimitiveFields::zeros(mesh.num_cells()).expect("primitives"),
        gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
            .expect("gradients"),
        viscous: None,
        residual_correction: None,
    };
    let info = solver
        .advance_step_3d(
            &mut ctx,
            &mut fields,
            &mut storage,
            &mut state,
            &mut integrator,
        )
        .expect("step");
    assert!((info.dt - fixed_dt).abs() < 1.0e-14);
}

/// 圆柱网格、无边界 patch：均匀来流时间推进（`--nocapture` 打印逐步指标）。
#[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
#[test]
fn cylinder_uniform_freestream_no_bc_time_advance_when_present() {
    use std::path::PathBuf;

    use crate::boundary::BoundarySet;
    use crate::discretization::{BoundaryGhostBuffer, InviscidFluxConfig};
    use crate::io::load_case;
    use crate::solver::time::{CflSchedule, RungeKutta4Integrator};

    let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
    if !case_path.is_file() {
        return;
    }
    let case = load_case(&case_path).expect("load case");
    let mesh = case.mesh.as_3d().expect("expected 3d");
    let eos = case.physics.eos().expect("eos");
    let fs = case.freestream.expect("freestream");
    let mut fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
    let rho0 = fs.pressure / (eos.gas_constant * fs.temperature);
    let empty_bc = BoundarySet::default();
    let mut ghosts = BoundaryGhostBuffer::new();
    let mut storage = Rk4Storage::new(mesh.num_cells()).expect("storage");
    let mut state = SolverState::default();
    let steps: u64 = 200;
    let mut integrator = RungeKutta4Integrator::new(RungeKutta4Config {
        dt: 0.0,
        max_steps: steps,
    });
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
        time: RungeKutta4Config {
            dt: 0.0,
            max_steps: steps,
        },
        inviscid: InviscidFluxConfig::roe_first_order(),
        cfl_schedule: CflSchedule {
            initial: 0.01,
            max: 0.1,
            ramp_steps: Some(500),
        },
        time_mode: CompressibleTimeMode::Steady,
        local_time_step: true,
        ..CompressibleEulerConfig::default()
    });
    let mut ctx = CompressibleAdvanceContext3d {
        mesh,
        structured: mesh,
        patches: &empty_bc,
        ghosts: &mut ghosts,
        eos: &eos,
        freestream: &fs,
        reference: None,
        primitive_scratch: crate::field::PrimitiveFields::zeros(mesh.num_cells())
            .expect("primitives"),
        gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
            .expect("gradients"),
        viscous: None,
        residual_correction: None,
    };

    eprintln!("=== 圆柱网格 无 BC 均匀来流 时间推进 ({} 步) ===", steps);
    eprintln!("来流 rho_ref = {rho0:.6e}");

    let report_steps = [1_u64, 10, 50, 100, 200];
    for _ in 0..steps {
        let info = solver
            .advance_step_3d(
                &mut ctx,
                &mut fields,
                &mut storage,
                &mut state,
                &mut integrator,
            )
            .expect("step");
        if report_steps.contains(&info.step) {
            let rho = fields.density.values();
            let rmin = rho.iter().copied().fold(f64::INFINITY, f64::min);
            let rmax = rho.iter().copied().fold(0.0_f64, f64::max);
            let center = rho[mesh.cell_index(mesh.nx / 2, mesh.ny / 2, 0)];
            eprintln!(
                "step {:4}: log10_res={:.4} rho=[{:.6e}, {:.6e}] center={:.6e}",
                info.step, info.residual_log10, rmin, rmax, center
            );
        }
    }

    let rho = fields.density.values();
    let rmax = rho.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        rmax < rho0 * 100.0,
        "无 BC 推进后 rho_max={rmax:.6e} 异常 (>100×来流)"
    );
}

/// 圆柱 + LU-SGS：合理 CFL 下步初 RHS 监控应随步变化（回归：勿误用步末 post RHS）。
#[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
#[test]
fn cylinder_lusgs_post_residual_changes_with_cfl1_when_present() {
    use std::path::PathBuf;

    use crate::discretization::BoundaryGhostBuffer;
    use crate::field::PrimitiveFields;
    use crate::io::load_case;
    use crate::solver::time::{
        CflSchedule, LuSgsConfig, RungeKutta4Integrator, TimeIntegrationScheme,
    };

    let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
    if !case_path.is_file() {
        return;
    }
    let case = load_case(&case_path).expect("load case");
    let mesh = case.mesh.as_3d().expect("expected 3d");
    let eos = case.physics.eos().expect("eos");
    let fs = case.freestream.expect("freestream");
    let inviscid = case.euler.as_ref().expect("euler").inviscid();
    let mut fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
    let mut ghosts = BoundaryGhostBuffer::new();
    let mut storage = Rk4Storage::new(mesh.num_cells()).expect("storage");
    let mut state = SolverState::default();
    let mut integrator = RungeKutta4Integrator::new(RungeKutta4Config {
        dt: 0.0,
        max_steps: 3,
    });
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
        time: RungeKutta4Config {
            dt: 0.0,
            max_steps: 3,
        },
        inviscid,
        cfl_schedule: CflSchedule {
            initial: 1.0,
            max: 1.0,
            ramp_steps: None,
        },
        time_mode: CompressibleTimeMode::Steady,
        local_time_step: true,
        time_scheme: TimeIntegrationScheme::LuSgs,
        lu_sgs: LuSgsConfig {
            sweep: false,
            ..LuSgsConfig::default()
        },
        ..CompressibleEulerConfig::default()
    });
    let mut ctx = CompressibleAdvanceContext3d {
        mesh,
        structured: mesh,
        patches: &case.boundary,
        ghosts: &mut ghosts,
        eos: &eos,
        freestream: &fs,
        reference: None,
        primitive_scratch: PrimitiveFields::zeros(mesh.num_cells()).expect("primitives"),
        gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
            .expect("gradients"),
        viscous: None,
        residual_correction: None,
    };
    let rho_ref = fields.density.values().to_vec();
    let info1 = solver
        .advance_step_3d(
            &mut ctx,
            &mut fields,
            &mut storage,
            &mut state,
            &mut integrator,
        )
        .expect("step1");
    let max_rho_delta1 = fields
        .density
        .values()
        .iter()
        .zip(rho_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let info2 = solver
        .advance_step_3d(
            &mut ctx,
            &mut fields,
            &mut storage,
            &mut state,
            &mut integrator,
        )
        .expect("step2");
    eprintln!(
        "lusgs cfl=1: step1 log10_res={:.6} step2 log10_res={:.6} dt1={:.6e} max|Δrho|1={:.6e}",
        info1.residual_log10, info2.residual_log10, info1.dt, max_rho_delta1
    );
    assert!(
        max_rho_delta1 > 1.0e-12,
        "一步 LU-SGS 后密度应有可观变化，max|Δrho|={max_rho_delta1:.6e}"
    );
    assert!(
        (info1.residual_rms - info2.residual_rms).abs() > 1.0e-12,
        "两步步初 RHS 监控应不同: r1={} r2={}",
        info1.residual_rms,
        info2.residual_rms
    );
}

/// 圆柱网格：仅内部面通量 + 边界单元 RHS 清零（不参与更新）。
#[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
#[test]
fn cylinder_uniform_freestream_interior_only_advance_when_present() {
    use std::path::PathBuf;

    use crate::boundary::BoundarySet;
    use crate::core::log10_positive;
    use crate::discretization::{
        BoundaryGhostBuffer, InviscidFluxConfig, assemble_inviscid_residual_3d,
    };
    use crate::field::ConservedResidual;
    use crate::io::load_case;
    use crate::solver::time::{CflSchedule, RungeKutta4Integrator};

    let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
    if !case_path.is_file() {
        return;
    }
    let case = load_case(&case_path).expect("load case");
    let mesh = case.mesh.as_3d().expect("expected 3d");
    let eos = case.physics.eos().expect("eos");
    let fs = case.freestream.expect("freestream");
    let mut fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
    let reference = fields.clone();
    let rho0 = fs.pressure / (eos.gas_constant * fs.temperature);
    let boundary_cells = structured_3d_boundary_cell_indices(mesh);
    let boundary_set: HashSet<usize> = boundary_cells.iter().copied().collect();
    let empty_bc = BoundarySet::default();
    let ghosts = BoundaryGhostBuffer::new();
    let mut storage = Rk4Storage::new(mesh.num_cells()).expect("storage");
    let mut state = SolverState::default();
    let steps: u64 = 200;
    let mut integrator = RungeKutta4Integrator::new(RungeKutta4Config {
        dt: 0.0,
        max_steps: steps,
    });
    let inviscid = InviscidFluxConfig::roe_first_order();
    let cfl_schedule = CflSchedule {
        initial: 0.01,
        max: 0.1,
        ramp_steps: Some(500),
    };

    let mut primitives = crate::field::PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
    let report_steps = [1_u64, 10, 50, 100, 200];
    for _ in 0..steps {
        let cfl = cfl_schedule.at_step(state.time_step.saturating_add(1), steps);
        let p_floor = fs.pressure * 1.0e-3;
        primitives
            .fill_from_conserved(&fields, &eos, p_floor)
            .expect("primitive for dt");
        let sigma = cell_spectral_radius_3d(&SpectralRadius3dParams {
            mesh,
            boundary_mesh: mesh,
            boundaries: &empty_bc,
            ghosts: &ghosts,
            primitives: &primitives,
            eos: &eos,
            min_pressure: p_floor,
            viscous: None,
        })
        .expect("sigma");
        let cell_dts = cell_local_dt_spectral(&mesh.cell_volumes(), &sigma, cfl).expect("dt");
        let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
            primitives.fill_from_conserved(u, &eos, p_floor)?;
            let assembly =
                crate::discretization::compressible::residual::InviscidAssembly3dParams {
                    mesh,
                    eos: &eos,
                    config: &inviscid,
                    boundaries: &empty_bc,
                    ghosts: &ghosts,
                    primitives: &primitives,
                    min_pressure: p_floor,
                };
            assemble_inviscid_residual_3d(u, r, &assembly)?;
            zero_residual_on_cells(r, &boundary_cells);
            Ok(())
        };

        rk4_step_local(
            &mut fields,
            &mut storage,
            &cell_dts,
            evaluate,
            Some(&eos),
            fs.pressure * 1.0e-3,
        )
        .expect("rk4");
        let _ = integrator.advance(&mut state).expect("advance");

        if report_steps.contains(&state.time_step) {
            let rho = fields.density.values();
            let interior_rho: Vec<f64> = rho
                .iter()
                .enumerate()
                .filter(|(i, _)| !boundary_set.contains(i))
                .map(|(_, v)| *v)
                .collect();
            let rmin = interior_rho.iter().copied().fold(f64::INFINITY, f64::min);
            let rmax = interior_rho.iter().copied().fold(0.0_f64, f64::max);
            let center = rho[mesh.cell_index(mesh.nx / 2, mesh.ny / 2, 0)];

            primitives
                .fill_from_conserved(&fields, &eos, p_floor)
                .expect("fill");
            let assembly =
                crate::discretization::compressible::residual::InviscidAssembly3dParams {
                    mesh,
                    eos: &eos,
                    config: &inviscid,
                    boundaries: &empty_bc,
                    ghosts: &ghosts,
                    primitives: &primitives,
                    min_pressure: p_floor,
                };
            assemble_inviscid_residual_3d(&fields, &mut storage.k1, &assembly).expect("rhs");
            zero_residual_on_cells(&mut storage.k1, &boundary_cells);
            let int_res = interior_density_rms(&storage.k1, &boundary_set);
            let boundary_frozen = boundary_cells
                .iter()
                .all(|&c| fields.density.values()[c] == reference.density.values()[c]);

            eprintln!(
                "step {:4}: log10_int_res={:.4} int_rho=[{:.6e}, {:.6e}] center={:.6e} boundary_frozen={boundary_frozen}",
                state.time_step,
                log10_positive(int_res),
                rmin,
                rmax,
                center
            );
        }
    }

    let interior_max = fields
        .density
        .values()
        .iter()
        .enumerate()
        .filter(|(i, _)| !boundary_set.contains(i))
        .map(|(_, v)| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        interior_max < rho0 * 1.01,
        "内部单元 rho 偏离来流: max={interior_max:.6e}"
    );
}

/// 贴体结构化网格的边界层单元（准 2D：`nz==1` 时不计 K 面）。
#[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
fn structured_3d_boundary_cell_indices(mesh: &StructuredMesh3d) -> Vec<usize> {
    let mut cells = Vec::new();
    let include_k = mesh.nz > 1;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let on_i = i == 0 || i + 1 == mesh.nx;
                let on_j = j == 0 || j + 1 == mesh.ny;
                let on_k = include_k && (k == 0 || k + 1 == mesh.nz);
                if on_i || on_j || on_k {
                    cells.push(mesh.cell_index(i, j, k));
                }
            }
        }
    }
    cells
}

#[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
fn zero_residual_on_cells(residual: &mut ConservedResidual, cells: &[usize]) {
    for &c in cells {
        residual.density.values_mut()[c] = 0.0;
        residual.momentum_x.values_mut()[c] = 0.0;
        residual.momentum_y.values_mut()[c] = 0.0;
        residual.momentum_z.values_mut()[c] = 0.0;
        residual.total_energy.values_mut()[c] = 0.0;
    }
}

#[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
fn interior_density_rms(residual: &ConservedResidual, boundary: &HashSet<usize>) -> f64 {
    let mut sum_sq = 0.0_f64;
    let mut count = 0_usize;
    for (i, &v) in residual.density.values().iter().enumerate() {
        if boundary.contains(&i) {
            continue;
        }
        sum_sq += v * v;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt()
    }
}
