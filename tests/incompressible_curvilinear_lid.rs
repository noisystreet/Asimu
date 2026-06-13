//! 非均匀、非正交结构网格上的不可压 lid cavity 验证。

use asimu::boundary::{BoundaryKind, BoundaryPatch, BoundarySet, WallHeat};
use asimu::case::{CaseRunKind, IncompressibleLineSample, run_case};
use asimu::field::{FluidInitialConfig, InitialSet};
use asimu::io::{CaseMesh, CaseSpec, CaseTimeConfig, CaseTimeMode, IncompressibleCaseConfig};
use asimu::linalg::GmresConfig;
use asimu::mesh::{BoundaryMesh, MeshMetricMode, MultiBlockStructuredMesh3d, StructuredMesh3d};
use asimu::physics::PhysicsConfig;
use asimu::solver::{
    IncompressibleLinearSolverConfig, IncompressiblePressureLinearSolverConfig,
    IncompressiblePressureLinearSolverKind,
};

#[test]
fn curvilinear_lid_cavity_preserves_unit_cavity_and_stays_physical() {
    let mesh = curvilinear_unit_cavity_mesh(12, 12);
    assert_unit_cavity_boundary(&mesh);
    assert!(mesh.cell_metric(5, 5, 0).volume > 0.0);

    let case = curvilinear_lid_case(mesh, 80, 0.02, 0.01, 1.0e-5, 0.0005);
    let result = run_case(&case).expect("run curvilinear lid cavity");
    assert_eq!(result.kind, CaseRunKind::Incompressible3dSteady);
    let metrics = result.incompressible_3d.expect("metrics");

    assert!(metrics.pressure_solve_converged);
    assert!(metrics.momentum_solve_converged);
    assert!(metrics.max_abs_corrected_divergence < 1.0e-5);
    assert!(metrics.max_abs_corrected_field_divergence_after_boundary < 1.0e-5);
    assert!(metrics.pressure_correction_rhs_active_sum.abs() < 1.0e-4);
    assert!(
        metrics
            .max_abs_corrected_velocity_delta_interior
            .is_finite()
    );
    assert!(metrics.max_abs_corrected_velocity_delta_interior < 5.0e-2);
    assert!(metrics.max_abs_pressure_correction.is_finite());

    let profiles = metrics.centerline_profiles.expect("centerline profiles");
    assert_eq!(profiles.vertical_u.len(), 12);
    assert_eq!(profiles.horizontal_v.len(), 12);
    assert!(profiles.vertical_u.iter().all(|sample| {
        sample.coordinate.is_finite()
            && sample.velocity_x.is_finite()
            && (-0.2..=1.05).contains(&sample.velocity_x)
    }));
    assert!(profiles.horizontal_v.iter().all(|sample| {
        sample.coordinate.is_finite()
            && sample.velocity_y.is_finite()
            && sample.velocity_y.abs() < 0.25
    }));
}

/// 慢速 V&V：非均匀、非正交但边界仍为单位方腔的 Re=100 lid cavity，
/// 对 Ghia et al. (1982) 中心线表格做剖面误差回归。
#[test]
#[ignore = "slow Ghia profile V&V; run explicitly with --ignored"]
fn curvilinear_lid_cavity_regresses_ghia_profile_error() {
    let mesh = curvilinear_unit_cavity_mesh(24, 24);
    assert_unit_cavity_boundary(&mesh);

    let case = curvilinear_lid_case(mesh, 1000, 0.005, 0.001, 2.0e-5, 0.02);
    let result = run_case(&case).expect("run curvilinear lid cavity Ghia V&V");
    let metrics = result.incompressible_3d.expect("metrics");
    assert!(metrics.pressure_solve_converged);
    assert!(metrics.momentum_solve_converged);
    assert!(metrics.max_abs_corrected_field_divergence_after_boundary < 2.0e-5);

    let error = metrics
        .lid_cavity_profile_error
        .expect("lid cavity profile error");
    println!(
        "curvilinear Ghia error: u max={} u l2={} v max={} v l2={}",
        error.vertical_u.max_abs,
        error.vertical_u.l2,
        error.horizontal_v.max_abs,
        error.horizontal_v.l2
    );
    assert!(
        error.vertical_u.max_abs < 0.60,
        "vertical u max_abs={}",
        error.vertical_u.max_abs
    );
    assert!(
        error.vertical_u.l2 < 0.30,
        "vertical u l2={}",
        error.vertical_u.l2
    );
    assert!(
        error.horizontal_v.max_abs < 0.25,
        "horizontal v max_abs={}",
        error.horizontal_v.max_abs
    );
    assert!(
        error.horizontal_v.l2 < 0.17,
        "horizontal v l2={}",
        error.horizontal_v.l2
    );
}

#[test]
#[ignore = "diagnostic parameter scan; run explicitly with --ignored --nocapture"]
fn scan_curvilinear_lid_time_window() {
    println!(
        "grid,steps,dt,vel_urf,p_urf,cont,field_cont,vel_delta,max_p,max_u,pressure_conv,momentum_conv,u_max,u_l2,v_max,v_l2"
    );
    for case in [
        ScanCase {
            grid: "skew24",
            steps: 100,
            dt: 0.0005,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "skew24",
            steps: 300,
            dt: 0.0005,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "skew24",
            steps: 1000,
            dt: 0.0005,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "skew24",
            steps: 1000,
            dt: 0.005,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "skew24",
            steps: 1000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
    ] {
        print_scan_metrics(case, run_scan_case(case));
    }
}

#[test]
#[ignore = "diagnostic drift onset scan; run explicitly with --ignored --nocapture"]
fn scan_curvilinear_lid_drift_onset() {
    println!(
        "grid,steps,dt,vel_urf,p_urf,cont,field_cont,vel_delta,max_p,max_u,pressure_conv,momentum_conv,u_max,u_l2,v_max,v_l2"
    );
    for case in [
        ScanCase {
            grid: "skew24",
            steps: 1000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "skew24",
            steps: 1500,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "skew24",
            steps: 2000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
    ] {
        print_scan_metrics(case, run_scan_case(case));
    }
}

#[test]
#[ignore = "diagnostic relaxation scan; run explicitly with --ignored --nocapture"]
fn scan_curvilinear_lid_pressure_relaxation() {
    println!(
        "grid,steps,dt,vel_urf,p_urf,cont,field_cont,vel_delta,max_p,max_u,pressure_conv,momentum_conv,u_max,u_l2,v_max,v_l2"
    );
    for case in [
        ScanCase {
            grid: "skew24",
            steps: 2000,
            dt: 0.005,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "skew24",
            steps: 2000,
            dt: 0.005,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.0005,
        },
        ScanCase {
            grid: "skew24",
            steps: 2000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "skew24",
            steps: 2000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.0005,
        },
    ] {
        print_scan_metrics(case, run_scan_case(case));
    }
}

#[test]
#[ignore = "diagnostic resolution scan; run explicitly with --ignored --nocapture"]
fn scan_lid_uniform_resolution_window() {
    println!(
        "grid,steps,dt,vel_urf,p_urf,cont,field_cont,vel_delta,max_p,max_u,pressure_conv,momentum_conv,u_max,u_l2,v_max,v_l2"
    );
    for case in [
        ScanCase {
            grid: "uniform24",
            steps: 1000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "uniform32",
            steps: 1000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "uniform48",
            steps: 1000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
    ] {
        print_scan_metrics(case, run_scan_case(case));
    }
}

#[test]
#[ignore = "diagnostic profile point errors; run explicitly with --ignored --nocapture"]
fn scan_lid_profile_point_errors() {
    println!("profile,grid,coordinate,expected,actual,error");
    for case in [
        ScanCase {
            grid: "uniform24",
            steps: 1000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "skew24",
            steps: 1000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
    ] {
        let metrics = run_scan_case(case);
        let profiles = metrics.centerline_profiles.expect("centerline profiles");
        print_profile_point_errors(
            "u",
            case.grid,
            &profiles.vertical_u,
            &GHIA_RE100_VERTICAL_U,
            |sample| sample.velocity_x,
        );
        print_profile_point_errors(
            "v",
            case.grid,
            &profiles.horizontal_v,
            &GHIA_RE100_HORIZONTAL_V,
            |sample| sample.velocity_y,
        );
    }
}

#[test]
#[ignore = "diagnostic grid-factor scan; run explicitly with --ignored --nocapture"]
fn scan_lid_grid_factor_window() {
    println!(
        "grid,steps,dt,vel_urf,p_urf,cont,field_cont,vel_delta,max_p,max_u,pressure_conv,momentum_conv,u_max,u_l2,v_max,v_l2"
    );
    for case in [
        ScanCase {
            grid: "uniform24",
            steps: 1000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "stretch24",
            steps: 1000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
        ScanCase {
            grid: "skew24",
            steps: 1000,
            dt: 0.02,
            velocity_under_relaxation: 0.005,
            pressure_under_relaxation: 0.001,
        },
    ] {
        print_scan_metrics(case, run_scan_case(case));
    }
}

#[derive(Debug, Clone, Copy)]
struct ScanCase {
    grid: &'static str,
    steps: u64,
    dt: f64,
    velocity_under_relaxation: f64,
    pressure_under_relaxation: f64,
}

fn run_scan_case(case: ScanCase) -> asimu::case::Incompressible3dRunMetrics {
    let mesh = match case.grid {
        "uniform24" => unit_cavity_mesh(24, 24, CoordinateMap::Uniform, 0.0),
        "uniform32" => unit_cavity_mesh(32, 32, CoordinateMap::Uniform, 0.0),
        "uniform48" => unit_cavity_mesh(48, 48, CoordinateMap::Uniform, 0.0),
        "stretch24" => unit_cavity_mesh(24, 24, CoordinateMap::Stretched, 0.0),
        "skew24" => curvilinear_unit_cavity_mesh(24, 24),
        _ => unreachable!("unknown scan grid"),
    };
    let case = curvilinear_lid_case(
        mesh,
        case.steps,
        case.velocity_under_relaxation,
        case.pressure_under_relaxation,
        2.0e-5,
        case.dt,
    );
    let result = run_case(&case).expect("run scan case");
    result.incompressible_3d.expect("metrics")
}

fn print_scan_metrics(case: ScanCase, metrics: asimu::case::Incompressible3dRunMetrics) {
    let error = metrics
        .lid_cavity_profile_error
        .expect("lid cavity profile error");
    println!(
        "{},{},{},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{},{},{:.6e},{:.6e},{:.6e},{:.6e}",
        case.grid,
        case.steps,
        case.dt,
        case.velocity_under_relaxation,
        case.pressure_under_relaxation,
        metrics.max_abs_corrected_divergence,
        metrics.max_abs_corrected_field_divergence_after_boundary,
        metrics.max_abs_corrected_velocity_delta_interior,
        metrics.max_abs_pressure,
        metrics.max_abs_velocity,
        metrics.pressure_solve_converged,
        metrics.momentum_solve_converged,
        error.vertical_u.max_abs,
        error.vertical_u.l2,
        error.horizontal_v.max_abs,
        error.horizontal_v.l2
    );
}

fn print_profile_point_errors(
    profile: &str,
    grid: &str,
    samples: &[IncompressibleLineSample],
    reference: &[(f64, f64)],
    value: impl Fn(&IncompressibleLineSample) -> f64,
) {
    for &(coordinate, expected) in reference {
        if let Some(actual) = interpolate_profile_value(samples, coordinate, &value) {
            println!(
                "{},{},{:.6},{:.6e},{:.6e},{:.6e}",
                profile,
                grid,
                coordinate,
                expected,
                actual,
                actual - expected
            );
        }
    }
}

fn interpolate_profile_value(
    samples: &[IncompressibleLineSample],
    coordinate: f64,
    value: &impl Fn(&IncompressibleLineSample) -> f64,
) -> Option<f64> {
    let mut sorted = samples
        .iter()
        .map(|sample| (sample.coordinate, value(sample)))
        .collect::<Vec<_>>();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    for pair in sorted.windows(2) {
        let (x0, y0) = pair[0];
        let (x1, y1) = pair[1];
        if coordinate >= x0 && coordinate <= x1 {
            let t = if (x1 - x0).abs() <= f64::EPSILON {
                0.0
            } else {
                (coordinate - x0) / (x1 - x0)
            };
            return Some(y0 + t * (y1 - y0));
        }
    }
    None
}

const GHIA_RE100_VERTICAL_U: [(f64, f64); 17] = [
    (1.0, 1.0),
    (0.9766, 0.84123),
    (0.9688, 0.78871),
    (0.9609, 0.73722),
    (0.9531, 0.68717),
    (0.8516, 0.23151),
    (0.7344, 0.00332),
    (0.6172, -0.13641),
    (0.5, -0.20581),
    (0.4531, -0.2109),
    (0.2813, -0.15662),
    (0.1719, -0.1015),
    (0.1016, -0.06434),
    (0.0703, -0.04775),
    (0.0625, -0.04192),
    (0.0547, -0.03717),
    (0.0, 0.0),
];

const GHIA_RE100_HORIZONTAL_V: [(f64, f64); 17] = [
    (1.0, 0.0),
    (0.9688, -0.05906),
    (0.9609, -0.07391),
    (0.9531, -0.08864),
    (0.9453, -0.10313),
    (0.9063, -0.16914),
    (0.8594, -0.22445),
    (0.8047, -0.24533),
    (0.5, 0.05454),
    (0.2344, 0.17527),
    (0.2266, 0.17507),
    (0.1563, 0.16077),
    (0.0938, 0.12317),
    (0.0781, 0.1089),
    (0.0703, 0.10091),
    (0.0625, 0.09233),
    (0.0, 0.0),
];

fn curvilinear_unit_cavity_mesh(nx: usize, ny: usize) -> StructuredMesh3d {
    unit_cavity_mesh(nx, ny, CoordinateMap::Stretched, 1.0)
}

#[derive(Debug, Clone, Copy)]
enum CoordinateMap {
    Uniform,
    Stretched,
}

fn unit_cavity_mesh(
    nx: usize,
    ny: usize,
    coordinate_map: CoordinateMap,
    skew_strength: f64,
) -> StructuredMesh3d {
    let nz = 1;
    let mut points_x = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    let mut points_y = Vec::with_capacity(points_x.capacity());
    let mut points_z = Vec::with_capacity(points_x.capacity());
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                let xi = unit_coordinate(i, nx, coordinate_map);
                let eta = unit_coordinate(j, ny, coordinate_map);
                let skew = skew_strength
                    * interior_weight(i, nx)
                    * interior_weight(j, ny)
                    * (std::f64::consts::PI * xi).sin()
                    * (std::f64::consts::PI * eta).sin();
                points_x.push(xi + 0.045 * skew);
                points_y.push(eta + 0.035 * skew);
                points_z.push(k as f64 * 0.1);
            }
        }
    }
    let mut mesh =
        StructuredMesh3d::new("curvilinear_lid", nx, ny, nz, points_x, points_y, points_z)
            .expect("mesh");
    mesh.set_metric_mode(MeshMetricMode::Curvilinear);
    mesh.build_metric_cache().expect("metric cache");
    mesh
}

fn unit_coordinate(index: usize, n: usize, coordinate_map: CoordinateMap) -> f64 {
    match coordinate_map {
        CoordinateMap::Uniform => index as f64 / n as f64,
        CoordinateMap::Stretched => stretched_unit_coordinate(index, n),
    }
}

fn stretched_unit_coordinate(index: usize, n: usize) -> f64 {
    let s = index as f64 / n as f64;
    0.5 * (1.0 - (std::f64::consts::PI * s).cos())
}

fn interior_weight(index: usize, n: usize) -> f64 {
    let s = index as f64 / n as f64;
    (4.0 * s * (1.0 - s)).max(0.0)
}

fn assert_unit_cavity_boundary(mesh: &StructuredMesh3d) {
    let tol = 1.0e-12;
    for j in 0..=mesh.ny {
        for k in 0..=mesh.nz {
            assert!((mesh.node_x(0, j, k) - 0.0).abs() < tol);
            assert!((mesh.node_x(mesh.nx, j, k) - 1.0).abs() < tol);
        }
    }
    for i in 0..=mesh.nx {
        for k in 0..=mesh.nz {
            assert!((mesh.node_y(i, 0, k) - 0.0).abs() < tol);
            assert!((mesh.node_y(i, mesh.ny, k) - 1.0).abs() < tol);
        }
    }
}

fn curvilinear_lid_case(
    mesh: StructuredMesh3d,
    steps: u64,
    velocity_under_relaxation: f64,
    pressure_under_relaxation: f64,
    tolerance: f64,
    dt: f64,
) -> CaseSpec {
    let boundary = BoundarySet::new(vec![
        BoundaryPatch::new(
            "i_min",
            mesh.resolve_logical_boundary("i_min").expect("i_min"),
            wall(),
        ),
        BoundaryPatch::new(
            "i_max",
            mesh.resolve_logical_boundary("i_max").expect("i_max"),
            wall(),
        ),
        BoundaryPatch::new(
            "j_min",
            mesh.resolve_logical_boundary("j_min").expect("j_min"),
            wall(),
        ),
        BoundaryPatch::new(
            "j_max",
            mesh.resolve_logical_boundary("j_max").expect("j_max"),
            BoundaryKind::MovingWall {
                velocity: [1.0, 0.0, 0.0],
            },
        ),
        BoundaryPatch::new(
            "k_min",
            mesh.resolve_logical_boundary("k_min").expect("k_min"),
            BoundaryKind::Symmetry,
        ),
        BoundaryPatch::new(
            "k_max",
            mesh.resolve_logical_boundary("k_max").expect("k_max"),
            BoundaryKind::Symmetry,
        ),
    ]);
    CaseSpec {
        name: "curvilinear_lid_cavity_re100".to_string(),
        benchmark_id: Some("lid_driven_cavity_re100".to_string()),
        mesh: CaseMesh::MultiBlockStructured3d(
            MultiBlockStructuredMesh3d::from_single_mesh(mesh).expect("multiblock"),
        ),
        physics: PhysicsConfig {
            diffusivity: None,
            eos: None,
            viscous: None,
        },
        boundary,
        initial: InitialSet::default(),
        fluid_initial: FluidInitialConfig::default(),
        freestream: None,
        restart: None,
        time: CaseTimeConfig {
            mode: CaseTimeMode::Transient,
            dt: Some(dt),
            max_steps: Some(steps),
            min_steps: Some(steps),
            tolerance: Some(tolerance),
            ..CaseTimeConfig::default()
        },
        sod: None,
        euler: None,
        navier_stokes: None,
        incompressible: Some(IncompressibleCaseConfig {
            pressure: 0.0,
            velocity: [0.0, 0.0, 0.0],
            body_force: [0.0, 0.0, 0.0],
            density: 1.0,
            kinematic_viscosity: 0.01,
            velocity_under_relaxation,
            pressure_under_relaxation,
            convection_scheme: asimu::discretization::IncompressibleConvectionScheme::Upwind,
            piso_correctors: 2,
            linear_solvers: IncompressibleLinearSolverConfig {
                momentum: GmresConfig {
                    restart: 32,
                    max_iters: 300,
                    tolerance: 1.0e-9,
                },
                pressure: IncompressiblePressureLinearSolverConfig {
                    kind: IncompressiblePressureLinearSolverKind::Pcg,
                    max_iters: 1200,
                    tolerance: 1.0e-10,
                    gmres_restart: 64,
                },
            },
            reference: asimu::io::IncompressibleReferenceConfig {
                length: 1.0,
                velocity: 1.0,
            },
        }),
        output: None,
        observability: None,
        case_dir: None,
        reference: None,
        incompressible_reference: None,
    }
}

fn wall() -> BoundaryKind {
    BoundaryKind::Wall {
        no_slip: true,
        heat: WallHeat::Adiabatic,
    }
}
