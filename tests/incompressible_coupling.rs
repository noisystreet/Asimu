//! 不可压缩压力–速度耦合诊断（lid cavity benchmark 排查）。

use std::io::Write;
use std::path::Path;

use asimu::case::{CaseRunKind, run_case_path};

const BENCH_CASE: &str = "tests/benchmarks/lid_driven_cavity_re100/case.toml";

fn write_structured_lid_case(path: &Path, nx: usize) {
    let body = format!(
        r#"name = "lid_coupling_{nx}"
benchmark_id = "lid_driven_cavity_re100"

[mesh]
kind = "structured_3d"
nx = {nx}
ny = {nx}
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
restart = 16
max_iters = 50
tolerance = 1.0e-9

[incompressible.linear.pressure]
solver = "pcg"
max_iters = 500
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
max_steps = 100
min_steps = 100
dt = 0.0005
tolerance = 3.0e-5
"#
    );
    let mut file = std::fs::File::create(path).expect("create temp case");
    file.write_all(body.as_bytes()).expect("write temp case");
}

fn run_lid(nx: Option<usize>) -> asimu::case::Incompressible3dRunMetrics {
    let temp;
    let path: &Path = if let Some(nx) = nx {
        temp = std::env::temp_dir().join(format!("asimu_lid_coupling_{nx}.toml"));
        write_structured_lid_case(&temp, nx);
        temp.as_path()
    } else {
        Path::new(BENCH_CASE)
    };
    let result = run_case_path(path).expect("run lid cavity");
    assert_eq!(result.kind, CaseRunKind::Incompressible3dSteady);
    result.incompressible_3d.expect("incompressible metrics")
}

/// 8×8 benchmark：显式 phi corrector 让压力方程残差与 face-flux 散度同时闭合。
#[test]
fn coarse_lid_cavity_coupling_invariants() {
    let metrics = run_lid(None);
    assert!(metrics.simplec_converged);

    assert!(metrics.max_abs_corrected_divergence < 1.0e-8);
    assert!(metrics.max_abs_corrected_field_divergence_after_boundary < 1.0e-8);

    assert!(metrics.pressure_correction_rhs_active_sum.abs() < 1.0e-4);
    assert_eq!(
        metrics.pressure_corrector_residual_history.len(),
        metrics.simplec_iterations * metrics.pressure_correctors
    );
}

/// 16×16 structured：显式 phi corrector 后连续性闭合，速度仍按伪瞬态推进。
#[test]
fn refined_lid_cavity_coupling_stays_stable_on_16_grid() {
    let coarse = run_lid(Some(8));
    let fine = run_lid(Some(16));

    assert!(coarse.simplec_converged);
    assert!(fine.max_abs_corrected_divergence < 1.0e-6);
    assert!(fine.max_abs_underrelaxed_corrected_divergence < 1.0e-3);
    assert!(fine.max_abs_corrected_field_divergence_after_boundary < 1.0e-6);
    assert!(fine.pressure_correction_rhs_active_sum.abs() < 1.0e-4);
}

/// 32×32：显式 phi corrector 可闭合连续性；速度增量作为瞬态变化诊断保留。
#[test]
fn refined_lid_cavity_closes_continuity_under_benchmark_settings() {
    let fine = run_lid(Some(32));

    assert!(
        fine.simplec_converged,
        "continuity={} velocity_delta={}",
        fine.simplec_final_residual, fine.max_abs_corrected_velocity_delta_interior
    );
    assert!(fine.pressure_solve_converged);
    assert!(fine.max_abs_corrected_divergence < 1.0e-6);
    assert!(fine.max_abs_underrelaxed_corrected_divergence < 3.0e-5);
    assert!(fine.max_abs_corrected_field_divergence_after_boundary < 1.0e-6);
    assert!(fine.max_abs_corrected_velocity_delta_interior < 5.0e-2);
}
