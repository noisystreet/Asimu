use super::*;
use crate::error::AsimuError;

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
        case.time
            .resolved_gmres_config()
            .expect("gmres")
            .preconditioner,
        crate::solver::GmresPreconditionerKind::CellBlockDiagonal
    );
}

#[test]
fn parses_gmres_linear_solver_config() {
    let content = r#"
name = "gmres_linear_test"
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
gmres_tolerance = 1.0e-5
gmres_max_iters = 40
gmres_restart = 15
max_steps = 3
"#;
    let case = parse_case_toml(content, None).expect("parse");
    let gmres = case.time.resolved_gmres_config().expect("gmres");
    assert!((gmres.gmres.tolerance - 1.0e-5).abs() < 1.0e-15);
    assert_eq!(gmres.gmres.max_iters, 40);
    assert_eq!(gmres.gmres.restart, 15);
}

#[test]
fn rejects_gmres_linear_solver_config_outside_gmres_scheme() {
    let content = r#"
name = "gmres_linear_invalid"
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
scheme = "lu_sgs"
local_time_step = true
gmres_max_iters = 10
"#;
    let case = parse_case_toml(content, None).expect("parse");
    let err = case
        .time
        .resolved_gmres_config()
        .expect_err("invalid gmres config");
    assert!(matches!(err, AsimuError::Config(_)));
}
