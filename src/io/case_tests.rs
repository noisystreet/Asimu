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
    let CaseMesh::MultiBlockStructured3d(mesh) = &case.mesh else {
        panic!("structured_3d 应读入为 1-block MultiBlockStructured3d");
    };
    assert_eq!(mesh.num_blocks(), 1);
    assert!(mesh.interfaces().is_empty());
    assert_eq!(case.mesh.num_cells(), 64);
    assert_eq!(case.boundary.patches().len(), 6);
    let fields = case.build_conserved_fields().expect("ic");
    assert_eq!(fields.num_cells(), 64);
}

#[test]
fn parses_multiblock_structured_3d_mesh() {
    let case = parse_case_toml(
        r#"
name = "multi"
[mesh]
kind = "multi_block_structured_3d"

[[mesh.blocks]]
name = "inlet"
nx = 2
ny = 1
nz = 1
lx = 2.0

[[mesh.blocks]]
name = "outlet"
nx = 3
ny = 1
nz = 1
lx = 3.0

[physics]
diffusivity = 1.0
"#,
        None,
    )
    .expect("parse");

    let CaseMesh::MultiBlockStructured3d(mesh) = &case.mesh else {
        panic!("expected multiblock mesh");
    };
    assert_eq!(mesh.num_blocks(), 2);
    assert_eq!(mesh.num_cells(), 5);
    assert_eq!(mesh.blocks()[1].cell_offset, 2);
    assert_eq!(case.mesh.num_cells(), 5);
}

#[test]
fn rejects_duplicate_multiblock_names() {
    let err = parse_case_toml(
        r#"
name = "multi"
[mesh]
kind = "multi_block_structured_3d"

[[mesh.blocks]]
name = "block"
nx = 1
ny = 1
nz = 1

[[mesh.blocks]]
name = "block"
nx = 1
ny = 1
nz = 1

[physics]
diffusivity = 1.0
"#,
        None,
    )
    .expect_err("duplicate");

    assert!(matches!(err, AsimuError::Mesh(_)));
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
fn parses_dual_time_time_scheme() {
    let content = r#"
name = "dual_time_test"
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
[euler]
flux = "hllc"
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
mode = "transient"
scheme = "dual_time"
dt = 1.0e-4
local_time_step = true
max_inner_steps = 25
inner_tolerance = -3.0
max_steps = 100
"#;
    let case = parse_case_toml(content, None).expect("parse");
    assert_eq!(
        case.time.resolved_time_scheme(),
        crate::solver::time::TimeIntegrationScheme::DualTime
    );
    let dual = case
        .time
        .resolved_dual_time_config()
        .expect("dual cfg")
        .expect("some");
    let reference = case.reference.expect("reference");
    let expected_dt = 1.0e-4 / reference.time_scale();
    assert!((dual.dt_phys - expected_dt).abs() < 1.0e-9);
    assert_eq!(dual.max_inner_steps, 25);
    assert_eq!(dual.inner_log10_tolerance, Some(-3.0));
}

#[test]
fn parses_low_mach_preconditioning_time_fields() {
    let content = r#"
name = "low_mach_time_test"
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
mach = 0.1
pressure = 101325.0
temperature = 288.15
[euler]
flux = "hllc"
[boundary.i_min]
kind = "wall"
no_slip = true
heat = "adiabatic"
[boundary.i_max]
kind = "farfield"
mach = 0.1
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
low_mach_preconditioning = true
low_mach_mach_cutoff = 0.08
max_steps = 10
"#;
    let case = parse_case_toml(content, None).expect("parse");
    let cfg = case.time.low_mach_preconditioning.expect("low mach cfg");
    assert!((cfg.mach_cutoff - 0.08).abs() < 1.0e-12);
}

#[test]
fn rejects_invalid_low_mach_cutoff() {
    let content = r#"
name = "low_mach_invalid"
[mesh]
kind = "structured_1d"
cells = 8
length = 1.0
[physics]
diffusivity = 1.0
[time]
low_mach_preconditioning = true
low_mach_mach_cutoff = 2.0
"#;
    let err = parse_case_toml(content, None).expect_err("invalid cutoff");
    assert!(err.to_string().contains("low_mach_mach_cutoff"));
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

#[test]
fn rejects_gmres_for_connected_multiblock_at_parse() {
    use crate::io::nondimensional;
    use crate::mesh::{
        MultiBlockStructuredMesh3d, StructuredBlockInterface3d, StructuredIndexRange3d,
        StructuredMesh3d,
    };
    use crate::physics::IdealGasEoS;
    use crate::solver::time::{LuSgsConfig, ResidualSmoothingConfig, TimeIntegrationScheme};

    let mesh = MultiBlockStructuredMesh3d::with_interfaces(
        "connected",
        vec![
            StructuredMesh3d::uniform_box("a", 2, 1, 1, 1.0, 1.0, 1.0).expect("a"),
            StructuredMesh3d::uniform_box("b", 2, 1, 1, 1.0, 1.0, 1.0).expect("b"),
        ],
        vec![StructuredBlockInterface3d {
            owner_block: "a".to_string(),
            donor_block: "b".to_string(),
            owner_range: StructuredIndexRange3d {
                imin: 2,
                imax: 2,
                jmin: 1,
                jmax: 2,
                kmin: 1,
                kmax: 2,
            },
            donor_range: StructuredIndexRange3d {
                imin: 1,
                imax: 1,
                jmin: 1,
                jmax: 2,
                kmin: 1,
                kmax: 2,
            },
            transform: [1, 2, 3],
        }],
    )
    .expect("mesh");

    let mut case = CaseSpec {
        name: "connected".to_string(),
        benchmark_id: None,
        mesh: CaseMesh::MultiBlockStructured3d(mesh),
        physics: crate::physics::PhysicsConfig {
            diffusivity: None,
            eos: Some(IdealGasEoS::new(1.4, 287.0).expect("eos")),
            viscous: None,
        },
        boundary: BoundarySet::default(),
        initial: InitialSet::default(),
        fluid_initial: FluidInitialConfig {
            freestream: Some(FreestreamParams {
                mach: 0.3,
                pressure: 101_325.0,
                temperature: 288.15,
                velocity_direction: [1.0, 0.0, 0.0],
                alpha: 0.0,
                beta: 0.0,
            }),
            scalars: InitialSet::default(),
        },
        freestream: Some(FreestreamParams {
            mach: 0.3,
            pressure: 101_325.0,
            temperature: 288.15,
            velocity_direction: [1.0, 0.0, 0.0],
            alpha: 0.0,
            beta: 0.0,
        }),
        restart: None,
        time: CaseTimeConfig {
            mode: CaseTimeMode::Steady,
            scheme: Some(TimeIntegrationScheme::Gmres),
            lusgs_sweep: None,
            lusgs_omega: None,
            lusgs_sweep_backward_damping: None,
            residual_smoothing: ResidualSmoothingConfig {
                enabled: false,
                epsilon: 0.0,
                sweeps: 0,
            },
            gmres_preconditioner: None,
            dt: None,
            cfl: Some(0.5),
            cfl_max: None,
            final_time: None,
            max_steps: Some(10),
            min_steps: None,
            tolerance: None,
            local_time_step: false,
            cfl_ramp_steps: None,
            max_inner_steps: None,
            inner_tolerance: None,
            low_mach_preconditioning: None,
        },
        sod: None,
        euler: Some(EulerCaseConfig {
            final_time: None,
            max_steps: None,
            reconstruction: None,
            flux: None,
            limiter: None,
            unstructured_limiter: None,
        }),
        navier_stokes: None,
        incompressible: None,
        output: None,
        observability: None,
        case_dir: None,
        reference: None,
        incompressible_reference: None,
        numerics: CaseNumericsConfig::default(),
    };
    nondimensional::apply_nondimensionalization_for_compressible(&mut case).expect("nd");

    let err = case.validate_multiblock_compressible().expect_err("gmres");
    assert!(matches!(err, AsimuError::Config(_)));
    assert!(err.to_string().contains("lu_sgs"), "unexpected: {err}");

    case.time.scheme = Some(TimeIntegrationScheme::LuSgs);
    case.time.lusgs_sweep = Some(true);
    let _ = LuSgsConfig::parse(
        case.time.lusgs_omega,
        case.time.lusgs_sweep,
        case.time.lusgs_sweep_backward_damping,
    )
    .expect("lusgs cfg");
    let err = case.validate_multiblock_compressible().expect_err("sweep");
    assert!(err.to_string().contains("lusgs_sweep"));
}

#[test]
fn numerics_defaults_to_f64() {
    let case = parse_case_toml(BENCHMARK_CASE, None).expect("parse");
    assert_eq!(
        case.numerics.compute_precision,
        crate::core::ComputePrecision::F64
    );
}

#[test]
fn parses_numerics_compute_precision() {
    let content = r#"
name = "precision_case"
benchmark_id = "1d_diffusion_analytical"

[mesh]
kind = "structured_1d"
cells = 4
length = 1.0
origin = 0.0

[physics]
diffusivity = 1.0

[boundary.left]
kind = "dirichlet"
value = 0.0

[boundary.right]
kind = "dirichlet"
value = 1.0

[numerics]
compute_precision = "f32"
"#;
    let case = parse_case_toml(content, None).expect("parse");
    assert_eq!(
        case.numerics.compute_precision,
        crate::core::ComputePrecision::F32
    );
}

#[test]
fn rejects_unknown_numerics_compute_precision() {
    let content = r#"
name = "bad_precision"
benchmark_id = "1d_diffusion_analytical"

[mesh]
kind = "structured_1d"
cells = 4
length = 1.0
origin = 0.0

[physics]
diffusivity = 1.0

[boundary.left]
kind = "dirichlet"
value = 0.0

[boundary.right]
kind = "dirichlet"
value = 1.0

[numerics]
compute_precision = "mixed"
"#;
    let err = parse_case_toml(content, None).expect_err("parse");
    assert!(err.to_string().contains("compute_precision"));
}
