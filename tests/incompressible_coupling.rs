//! 不可压缩压力–速度耦合快速诊断（完整 16×16 Ghia V&V 见 `case_run`）。

use std::io::Write;
use std::path::Path;

use asimu::case::{CaseRunKind, run_case_path};

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
velocity_under_relaxation = 0.3
pressure_under_relaxation = 0.3
convection_scheme = "upwind"
piso_correctors = 1

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
mode = "steady"
scheme = "simplec"
max_steps = 5000
min_steps = 50
tolerance = 1.0e-5
"#
    );
    let mut file = std::fs::File::create(path).expect("create temp case");
    file.write_all(body.as_bytes()).expect("write temp case");
}

/// 8×8 快速回归：稳态 SIMPLEC 耦合不变量（完整 16×16 benchmark 见 `case_run`）。
#[test]
fn lid_cavity_coupling_invariants_on_8_grid() {
    let path = std::env::temp_dir().join("asimu_lid_coupling_8.toml");
    write_structured_lid_case(&path, 8);
    let result = run_case_path(path.as_path()).expect("run lid cavity");
    assert_eq!(result.kind, CaseRunKind::Incompressible3dSteady);
    let metrics = result.incompressible_3d.expect("incompressible metrics");
    assert_eq!(metrics.algorithm, "simplec");
    assert!(metrics.simplec_converged);
    assert!(metrics.max_abs_corrected_field_divergence_after_boundary < 1.0e-5);
}
