use super::*;

const BENCHMARK_CASE: &str =
    include_str!("../../tests/benchmarks/1d_diffusion_analytical/case.toml");

#[test]
fn parses_diffusion_benchmark() {
    let case = parse_case_toml(BENCHMARK_CASE, None).expect("parse");
    assert_eq!(case.name, "1d_diffusion_analytical");
    assert_eq!(case.mesh.as_1d().expect("1d").num_cells(), 32);
    assert!(!case.is_compressible());
}

#[test]
fn parses_compressible_3d_case() {
    let content = r#"
name = "box_cns"
[mesh]
kind = "structured_3d"
nx = 4
ny = 4
nz = 4
lx = 1.0
ly = 1.0
lz = 1.0
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.i_min]
kind = "wall"
no_slip = true
heat = "adiabatic"
[boundary.i_max]
kind = "farfield"
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.j_min]
kind = "symmetry"
[boundary.j_max]
kind = "symmetry"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "outlet"
static_pressure = 100000.0
"#;
    let case = parse_case_toml(content, None).expect("parse");
    assert!(case.is_compressible());
    assert_eq!(case.mesh.num_cells(), 64);
    assert_eq!(case.boundary.patches().len(), 6);
    let fields = case.build_conserved_fields().expect("ic");
    assert_eq!(fields.num_cells(), 64);
}

#[test]
fn parses_lu_sgs_time_scheme() {
    let content = r#"
name = "lusgs_test"
[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
lx = 1.0
ly = 1.0
lz = 1.0
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.i_min]
kind = "wall"
no_slip = true
heat = "adiabatic"
[boundary.i_max]
kind = "farfield"
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.j_min]
kind = "symmetry"
[boundary.j_max]
kind = "symmetry"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "outlet"
static_pressure = 100000.0
[time]
scheme = "lu_sgs"
local_time_step = true
lusgs_omega = 0.8
max_steps = 10
"#;
    let case = parse_case_toml(content, None).expect("parse");
    assert_eq!(
        case.time.resolved_time_scheme(),
        crate::solver::time::TimeIntegrationScheme::LuSgs
    );
    assert!(case.time.uses_local_time_step());
    let cfg = case.time.resolved_lusgs_config().expect("lusgs");
    assert!((cfg.omega - 0.8).abs() < 1.0e-12);
    assert!(!cfg.sweep);
}

#[test]
fn parses_lusgs_diagonal_only() {
    let content = r#"
name = "lusgs_diag"
[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
lx = 1.0
ly = 1.0
lz = 1.0
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.i_min]
kind = "wall"
no_slip = true
heat = "adiabatic"
[boundary.i_max]
kind = "farfield"
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.j_min]
kind = "symmetry"
[boundary.j_max]
kind = "symmetry"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "outlet"
static_pressure = 100000.0
[time]
scheme = "lu_sgs"
local_time_step = true
lusgs_sweep = false
"#;
    let case = parse_case_toml(content, None).expect("parse");
    assert!(!case.time.resolved_lusgs_config().expect("cfg").sweep);
}

#[test]
fn parses_lusgs_sweep_backward_damping() {
    let case = parse_case_toml(
        r#"
name = "lusgs_sweep_damp"
[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
lx = 1.0
ly = 1.0
lz = 1.0
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.i_min]
kind = "wall"
no_slip = true
heat = "adiabatic"
[boundary.i_max]
kind = "farfield"
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.j_min]
kind = "symmetry"
[boundary.j_max]
kind = "symmetry"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "outlet"
static_pressure = 100000.0
[time]
mode = "steady"
scheme = "lu_sgs"
local_time_step = true
lusgs_sweep = true
lusgs_sweep_backward_damping = 0.35
max_steps = 10
"#,
        None,
    )
    .expect("case");
    let cfg = case.time.resolved_lusgs_config().expect("lusgs");
    assert!(cfg.sweep);
    assert!((cfg.sweep_backward_damping - 0.35).abs() < 1.0e-12);
}

#[test]
fn parses_gmres_time_scheme() {
    let content = r#"
name = "gmres_test"
[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.i_min]
kind = "wall"
[boundary.i_max]
kind = "farfield"
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.j_min]
kind = "symmetry"
[boundary.j_max]
kind = "symmetry"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "outlet"
static_pressure = 100000.0
[time]
scheme = "gmres"
local_time_step = true
gmres_preconditioner = "cell_block_diagonal"
max_steps = 3
"#;
    let case = parse_case_toml(content, None).expect("parse");
    assert_eq!(
        case.time.resolved_time_scheme(),
        crate::solver::time::TimeIntegrationScheme::Gmres
    );
    assert!(case.time.uses_local_time_step());
    assert_eq!(
        case.time.resolved_gmres_config().preconditioner,
        crate::solver::GmresPreconditionerKind::CellBlockDiagonal
    );
}

#[test]
fn parses_residual_smoothing_config() {
    let content = r#"
name = "smooth_test"
[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.i_min]
kind = "farfield"
[boundary.i_max]
kind = "farfield"
[boundary.j_min]
kind = "symmetry"
[boundary.j_max]
kind = "symmetry"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "outlet"
static_pressure = 100000.0
[time]
mode = "steady"
residual_smoothing = true
residual_smoothing_epsilon = 0.25
residual_smoothing_sweeps = 2
"#;
    let case = parse_case_toml(content, None).expect("parse");
    let cfg = case.time.residual_smoothing_config();
    assert!(cfg.enabled);
    assert!((cfg.epsilon - 0.25).abs() < 1.0e-12);
    assert_eq!(cfg.sweeps, 2);
}

#[test]
fn parses_sod_benchmark_case() {
    let content = include_str!("../../tests/benchmarks/sod_1d/case.toml");
    let case = parse_case_toml(content, None).expect("parse");
    assert_eq!(case.benchmark_id.as_deref(), Some("sod_1d"));
    assert_eq!(case.mesh.as_1d().expect("1d").num_cells(), 100);
    assert!(case.is_compressible());
    let sod = case.sod.expect("sod");
    assert_eq!(case.time.mode, CaseTimeMode::Transient);
    let inviscid = sod.inviscid();
    assert_eq!(inviscid.short_label(), "muscl_roe");
    assert_eq!(inviscid.limiter_label(), "van_albada");
}

#[test]
fn parses_sod_muscl_hllc_case() {
    let content = include_str!("../../tests/benchmarks/sod_1d/case_muscl_hllc.toml");
    let case = parse_case_toml(content, None).expect("parse");
    let sod = case.sod.expect("sod");
    let inviscid = sod.inviscid();
    assert_eq!(inviscid.short_label(), "muscl_hllc");
    assert_eq!(inviscid.limiter_label(), "van_albada");
}

#[test]
fn parses_inlet_and_turbulent_inlet() {
    let content = r#"
name = "inlet_test"
[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.1
[boundary.i_min]
kind = "turbulent_inlet"
total_pressure = 110000.0
total_temperature = 320.0
turbulent_k = 1.0
turbulent_omega = 100.0
velocity_direction = [1.0, 0.0, 0.0]
[boundary.i_max]
kind = "inlet"
total_pressure = 105000.0
total_temperature = 300.0
[boundary.j_min]
kind = "wall"
[boundary.j_max]
kind = "wall"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "wall"
"#;
    let case = parse_case_toml(content, None).expect("parse");
    assert!(case.boundary.find("i_min").is_some());
}
