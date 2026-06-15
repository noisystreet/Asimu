//! 算例编排集成测试（与 CLI 共用 `case::run_case_path`）。

use std::fs;
use std::path::Path;

use asimu::case::{CaseRunKind, run_case_path};

#[test]
fn diffusion_benchmark_via_case_runner() {
    let result = run_case_path(Path::new(
        "tests/benchmarks/1d_diffusion_analytical/case.toml",
    ))
    .expect("run");
    assert_eq!(result.kind, CaseRunKind::Diffusion1dSteady);
    assert!(result.summary.contains("扩散"));
}

#[test]
fn sod_benchmark_via_case_runner() {
    let result = run_case_path(Path::new("tests/benchmarks/sod_1d/case.toml")).expect("run");
    assert_eq!(result.kind, CaseRunKind::Sod1dTransient);
    let metrics = result.sod.expect("sod metrics");
    assert_eq!(metrics.scheme, "muscl_roe");
    assert_eq!(metrics.limiter, "van_albada");
    assert!(metrics.l1_density < 0.02);
    assert!(result.summary.contains("van_albada/muscl_roe"));
}

#[test]
fn sod_muscl_hllc_via_case_runner() {
    let result =
        run_case_path(Path::new("tests/benchmarks/sod_1d/case_muscl_hllc.toml")).expect("run");
    assert_eq!(result.kind, CaseRunKind::Sod1dTransient);
    let metrics = result.sod.expect("sod metrics");
    assert_eq!(metrics.scheme, "muscl_hllc");
    assert_eq!(metrics.limiter, "van_albada");
    assert!(metrics.l1_density < 0.02);
}

#[test]
fn channel_poiseuille_incompressible_benchmark_runs() {
    let result =
        run_case_path(Path::new("tests/benchmarks/channel_poiseuille/case.toml")).expect("run");
    assert_eq!(result.kind, CaseRunKind::Incompressible3dSteady);
    assert_eq!(result.benchmark_id.as_deref(), Some("channel_poiseuille"));
    let metrics = result.incompressible_3d.expect("incompressible metrics");
    assert!(metrics.simplec_iterations <= 2000);
    assert!(metrics.simplec_converged);
    assert!(metrics.simplec_final_residual.is_finite());
    assert!(metrics.simplec_final_residual < 1.0e-8);
    assert!(
        metrics
            .max_abs_underrelaxed_corrected_divergence
            .is_finite()
    );
    assert!(metrics.max_abs_underrelaxed_corrected_divergence < 1.0e-8);
    assert!(
        metrics
            .max_abs_corrected_field_divergence_after_boundary
            .is_finite()
    );
    assert!(metrics.pressure_correction_rhs_active_sum.is_finite());
    assert!(metrics.simplec_final_momentum_residual.is_finite());
    assert!(metrics.pressure_solve_converged);
    assert!(metrics.pressure_solve_iterations <= 500);
    let profiles = metrics.centerline_profiles.expect("poiseuille profile");
    assert_eq!(profiles.vertical_u.len(), 8);
    assert!(profiles.horizontal_v.is_empty());
    assert!(
        profiles
            .vertical_u
            .iter()
            .all(|sample| sample.coordinate.is_finite() && sample.velocity_x.is_finite())
    );
    let error = metrics
        .poiseuille_profile_error
        .expect("poiseuille profile error");
    assert!(error.max_abs.is_finite());
    assert!(error.l2.is_finite());
    assert!(error.max_abs < 0.25, "max_abs={}", error.max_abs);
    assert!(error.l2 < 0.25, "l2={}", error.l2);
}

#[test]
fn lid_driven_cavity_re100_incompressible_benchmark_runs() {
    let expected =
        std::fs::read_to_string("tests/benchmarks/lid_driven_cavity_re100/expected.json")
            .expect("expected");
    assert!(expected.contains("i2_steady_simplec_16x16"));
    assert!(expected.contains("Ghia et al. 1982"));
    let result = run_case_path(Path::new(
        "tests/benchmarks/lid_driven_cavity_re100/case.toml",
    ))
    .expect("run");
    assert_eq!(result.kind, CaseRunKind::Incompressible3dSteady);
    assert_eq!(
        result.benchmark_id.as_deref(),
        Some("lid_driven_cavity_re100")
    );
    let metrics = result.incompressible_3d.expect("incompressible metrics");
    assert_eq!(metrics.algorithm, "simplec");
    assert_eq!(metrics.pressure_correctors, 1);
    assert!(metrics.simplec_iterations >= 50);
    assert!(metrics.simplec_iterations <= 5000);
    assert!(metrics.simplec_converged);
    assert!(metrics.simplec_final_residual <= 1.0e-5);
    assert!(metrics.max_abs_corrected_divergence < 1.0e-5);
    assert!(metrics.max_abs_corrected_field_divergence_after_boundary < 1.0e-5);
    assert!(metrics.max_abs_corrected_velocity_delta_interior < 3.0e-6);
    assert!(metrics.max_abs_corrected_velocity_delta_boundary < 1.0e-12);
    assert!(metrics.pressure_correction_rhs_active_sum.abs() < 1.0e-4);
    assert!(metrics.pressure_solve_converged);
    assert!(metrics.momentum_solve_converged);
    let profiles = metrics.centerline_profiles.expect("centerline profiles");
    assert_eq!(profiles.vertical_u.len(), 16);
    assert_eq!(profiles.horizontal_v.len(), 16);
    assert!(
        profiles
            .vertical_u
            .iter()
            .all(|sample| sample.coordinate.is_finite() && sample.velocity_x.is_finite())
    );
    assert!(
        profiles
            .horizontal_v
            .iter()
            .all(|sample| sample.coordinate.is_finite() && sample.velocity_y.is_finite())
    );
    let error = metrics
        .lid_cavity_profile_error
        .expect("lid cavity profile error");
    assert!(error.vertical_u.max_abs.is_finite());
    assert!(error.vertical_u.l2.is_finite());
    assert!(error.horizontal_v.max_abs.is_finite());
    assert!(error.horizontal_v.l2.is_finite());
    assert!(
        error.vertical_u.max_abs < 0.22,
        "vertical_u max_abs={}",
        error.vertical_u.max_abs
    );
    assert!(
        error.vertical_u.l2 < 0.12,
        "vertical_u l2={}",
        error.vertical_u.l2
    );
    assert!(
        error.horizontal_v.max_abs < 0.12,
        "horizontal_v max_abs={}",
        error.horizontal_v.max_abs
    );
    assert!(
        error.horizontal_v.l2 < 0.09,
        "horizontal_v l2={}",
        error.horizontal_v.l2
    );
}

#[test]
fn taylor_green_3d_incompressible_benchmark_runs() {
    let result =
        run_case_path(Path::new("tests/benchmarks/taylor_green_3d/case.toml")).expect("run");
    assert_eq!(result.benchmark_id.as_deref(), Some("taylor_green_3d"));
    assert_eq!(
        result.kind,
        asimu::case::CaseRunKind::Incompressible3dTransient
    );
    let metrics = result.incompressible_3d.expect("incompressible metrics");
    assert_eq!(metrics.algorithm, "piso");
    assert_eq!(metrics.pressure_correctors, 2);
    assert_eq!(metrics.steps, 400);
    let initial = metrics
        .kinetic_energy_initial
        .expect("kinetic energy initial");
    let final_energy = metrics.kinetic_energy_final.expect("kinetic energy final");
    let decay = metrics
        .kinetic_energy_decay_ratio
        .expect("kinetic energy decay ratio");
    assert!((decay - final_energy / initial).abs() < 1.0e-12);
    let decay_rate = metrics
        .kinetic_energy_decay_rate
        .expect("kinetic energy decay rate");
    let analytical_rate = metrics
        .kinetic_energy_analytical_decay_rate
        .expect("kinetic energy analytical decay rate");
    assert!(decay_rate > 0.0, "decay_rate={decay_rate}");
    assert!(analytical_rate > 0.0);
    let analytical_ratio = metrics
        .kinetic_energy_analytical_ratio
        .expect("kinetic energy analytical ratio");
    assert!(
        (decay - analytical_ratio).abs() < 0.02,
        "decay={decay} analytical_ratio={analytical_ratio}"
    );
    assert!(
        final_energy < initial,
        "kinetic energy must decay: initial={initial} final={final_energy}"
    );
    assert!(decay < 1.0, "decay ratio must stay below unity: {decay}");
    assert!(
        decay_rate >= analytical_rate * 0.5,
        "decay_rate={decay_rate} analytical={analytical_rate}"
    );
    assert!(
        decay_rate <= analytical_rate * 2.0,
        "decay_rate={decay_rate} analytical={analytical_rate}"
    );
    assert!(
        metrics.max_abs_corrected_field_divergence_after_boundary < 1.0e-6,
        "face flux divergence={}",
        metrics.max_abs_corrected_field_divergence_after_boundary
    );
    assert!(
        metrics.max_abs_corrected_divergence < 1.0e-6,
        "pressure correction residual={}",
        metrics.max_abs_corrected_divergence
    );
}

struct TaylorGreenCaseConfig<'a> {
    name: &'a str,
    nx: usize,
    ny: usize,
    dt: f64,
    max_steps: u64,
    piso_correctors: usize,
    output_dir: &'a str,
}

fn write_taylor_green_case(case_path: &Path, config: TaylorGreenCaseConfig<'_>) {
    let TaylorGreenCaseConfig {
        name,
        nx,
        ny,
        dt,
        max_steps,
        piso_correctors,
        output_dir,
    } = config;
    fs::write(
        case_path,
        format!(
            r#"
name = "{name}"
benchmark_id = "taylor_green_3d"

[mesh]
kind = "structured_3d"
nx = {nx}
ny = {ny}
nz = 1
lx = 6.283185307179586
ly = 6.283185307179586
lz = 0.1

[physics]

[incompressible]
pressure = 0.0
velocity = [0.0, 0.0, 0.0]
density = 1.0
kinematic_viscosity = 0.1
convection_scheme = "central"
piso_correctors = {piso_correctors}

[incompressible.linear.momentum]
solver = "gmres"
restart = 32
max_iters = 80
tolerance = 1.0e-9

[incompressible.linear.pressure]
solver = "pcg"
max_iters = 500
tolerance = 1.0e-10

[incompressible.reference]
length = 6.283185307179586
velocity = 1.0

[boundary.i_min]
kind = "periodic"
partner = "i_max"

[boundary.i_max]
kind = "periodic"
partner = "i_min"

[boundary.j_min]
kind = "periodic"
partner = "j_max"

[boundary.j_max]
kind = "periodic"
partner = "j_min"

[boundary.k_min]
kind = "symmetry"

[boundary.k_max]
kind = "symmetry"

[time]
mode = "transient"
scheme = "bdf1"
max_steps = {max_steps}
dt = {dt}

[output]
dir = "{output_dir}"
residual_csv = "residual.csv"
"#
        ),
    )
    .expect("write taylor-green case");
}

#[test]
#[ignore = "参数敏感性对照（本地/夜间执行）"]
fn taylor_green_3d_parameter_sensitivity_baseline() {
    let root = std::env::temp_dir().join(format!("asimu_tg_sensitivity_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("temp dir");
    let out = root.join("out");
    fs::create_dir_all(&out).expect("out dir");
    let out_str = out.to_string_lossy();

    eprintln!("TG sensitivity (16x16, t*=2, PISO correctors noted):");
    eprintln!("dt\tsteps\tpiso\tE/E0\tanalytical\t|err|\tmax|div(u*)|");

    for (dt, steps) in [(0.05, 40_u64), (0.02, 100), (0.01, 200), (0.005, 400)] {
        let case_path = root.join(format!("dt_{dt}.toml"));
        write_taylor_green_case(
            &case_path,
            TaylorGreenCaseConfig {
                name: "tg_sensitivity_dt",
                nx: 16,
                ny: 16,
                dt,
                max_steps: steps,
                piso_correctors: 2,
                output_dir: &out_str,
            },
        );
        let metrics = run_case_path(&case_path)
            .expect("run dt sweep")
            .incompressible_3d
            .expect("metrics");
        let ratio = metrics.kinetic_energy_decay_ratio.expect("decay ratio");
        let analytical = metrics
            .kinetic_energy_analytical_ratio
            .expect("analytical ratio");
        let err = (ratio - analytical).abs();
        eprintln!(
            "{dt}\t{steps}\t2\t{ratio:.6}\t{analytical:.6}\t{err:.6}\t{:.3e}",
            metrics.max_abs_predicted_divergence
        );
        if (dt - 0.005).abs() < 1.0e-12 {
            assert!(
                (ratio - analytical).abs() < 0.02,
                "ratio={ratio} analytical={analytical}"
            );
            assert!(metrics.max_abs_corrected_field_divergence_after_boundary < 1.0e-6);
        }
    }

    for piso in [1_usize, 2, 3] {
        let case_path = root.join(format!("piso_{piso}.toml"));
        write_taylor_green_case(
            &case_path,
            TaylorGreenCaseConfig {
                name: "tg_sensitivity_piso",
                nx: 16,
                ny: 16,
                dt: 0.005,
                max_steps: 400,
                piso_correctors: piso,
                output_dir: &out_str,
            },
        );
        let metrics = run_case_path(&case_path)
            .expect("run piso sweep")
            .incompressible_3d
            .expect("metrics");
        let ratio = metrics.kinetic_energy_decay_ratio.expect("decay ratio");
        let analytical = metrics
            .kinetic_energy_analytical_ratio
            .expect("analytical ratio");
        let err = (ratio - analytical).abs();
        eprintln!(
            "0.005\t400\t{piso}\t{ratio:.6}\t{analytical:.6}\t{err:.6}\t{:.3e}",
            metrics.max_abs_predicted_divergence
        );
    }

    let _ = fs::remove_dir_all(&root);
}

#[test]
#[ignore = "细网格精度对比（本地/夜间执行）"]
fn taylor_green_3d_refined_grid_reduces_energy_ratio_error() {
    let root = std::env::temp_dir().join(format!("asimu_tg_refined_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("temp dir");
    let coarse_case = root.join("case_16.toml");
    let refined_case = root.join("case_32.toml");
    let baseline = TaylorGreenCaseConfig {
        name: "taylor_green_3d_refined",
        nx: 16,
        ny: 16,
        dt: 0.005,
        max_steps: 400,
        piso_correctors: 2,
        output_dir: "out",
    };
    write_taylor_green_case(
        &coarse_case,
        TaylorGreenCaseConfig {
            name: "taylor_green_3d_refined_16x16",
            ..baseline
        },
    );
    write_taylor_green_case(
        &refined_case,
        TaylorGreenCaseConfig {
            name: "taylor_green_3d_refined_32x32",
            nx: 32,
            ny: 32,
            ..baseline
        },
    );

    let coarse = run_case_path(&coarse_case).expect("run 16");
    let refined = run_case_path(&refined_case).expect("run 32");
    let coarse_metrics = coarse.incompressible_3d.expect("coarse metrics");
    let refined_metrics = refined.incompressible_3d.expect("refined metrics");

    let coarse_ratio = coarse_metrics
        .kinetic_energy_decay_ratio
        .expect("coarse kinetic ratio");
    let refined_ratio = refined_metrics
        .kinetic_energy_decay_ratio
        .expect("refined kinetic ratio");
    let coarse_analytical = coarse_metrics
        .kinetic_energy_analytical_ratio
        .expect("coarse analytical ratio");
    let refined_analytical = refined_metrics
        .kinetic_energy_analytical_ratio
        .expect("refined analytical ratio");
    let coarse_err = (coarse_ratio - coarse_analytical).abs();
    let refined_err = (refined_ratio - refined_analytical).abs();

    eprintln!(
        "TG refined baseline: coarse_ratio={coarse_ratio:.6}, refined_ratio={refined_ratio:.6}, coarse_err={coarse_err:.6}, refined_err={refined_err:.6}"
    );

    assert!(
        coarse_err < 0.02 && refined_err < 0.02,
        "coarse_err={coarse_err} refined_err={refined_err}"
    );
    assert!(
        refined_err <= coarse_err * 1.05,
        "coarse_err={coarse_err} refined_err={refined_err}"
    );
    assert!(refined_metrics.max_abs_corrected_divergence < 1.0e-6);
    assert!(refined_metrics.max_abs_corrected_field_divergence_after_boundary < 1.0e-6);
    assert!(
        (refined_ratio - coarse_ratio).abs() < 0.05,
        "coarse_ratio={coarse_ratio} refined_ratio={refined_ratio}"
    );

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn incompressible_runner_writes_residual_csv() {
    let root = std::env::temp_dir().join(format!(
        "asimu_incompressible_output_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("temp dir");
    let case_path = root.join("case.toml");
    fs::write(
        &case_path,
        r#"
name = "incompressible_output_smoke"

[mesh]
kind = "structured_3d"
nx = 4
ny = 4
nz = 1
lx = 1.0
ly = 1.0
lz = 0.1

[physics]

[incompressible]
pressure = 0.0
velocity = [0.0, 0.0, 0.0]
density = 1.0
kinematic_viscosity = 0.01
velocity_under_relaxation = 0.05
pressure_under_relaxation = 0.01
piso_correctors = 2

[incompressible.linear.momentum]
solver = "gmres"
restart = 8
max_iters = 20
tolerance = 1.0e-8

[incompressible.linear.pressure]
solver = "pcg"
max_iters = 100
tolerance = 1.0e-10

[incompressible.reference]
length = 1.0
velocity = 1.0

[boundary.i_min]
kind = "wall"
no_slip = true

[boundary.i_max]
kind = "wall"
no_slip = true

[boundary.j_min]
kind = "wall"
no_slip = true

[boundary.j_max]
kind = "moving_wall"
velocity = [1.0, 0.0, 0.0]

[boundary.k_min]
kind = "symmetry"

[boundary.k_max]
kind = "symmetry"

[time]
mode = "transient"
scheme = "piso"
max_steps = 3
min_steps = 3
dt = 0.0005
tolerance = 1.0e-4

[output]
dir = "out"
residual_csv = "residual.csv"
solution_cgns = "flow.cgns"
solution_every = 2
"#,
    )
    .expect("write case");

    let result = run_case_path(&case_path).expect("run");
    let metrics = result.incompressible_3d.expect("incompressible metrics");
    assert_eq!(metrics.simplec_iterations, 3);
    let residual = root.join("out/residual.csv");
    let interval_flow = root.join("out/flow_step000002.cgns");
    let final_flow = root.join("out/flow.cgns");
    assert!(residual.is_file(), "missing {}", residual.display());
    #[cfg(feature = "io-cgns")]
    assert!(
        interval_flow.is_file(),
        "missing {}",
        interval_flow.display()
    );
    #[cfg(feature = "io-cgns")]
    assert!(final_flow.is_file(), "missing {}", final_flow.display());
    assert!(metrics.written.contains(&residual));
    assert!(metrics.written.contains(&interval_flow));
    assert!(metrics.written.contains(&final_flow));
    let csv = fs::read_to_string(&residual).expect("read residual");
    assert!(csv.contains("face_flux_divergence"));
    assert!(csv.contains("velocity_delta_interior"));
    assert_eq!(csv.lines().count(), 4);
    let _ = fs::remove_dir_all(&root);
}
