use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::io::{CaseMesh, parse_case_str};
use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

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
    attach_single_tet_farfield(&mut case);
    let result = super::compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 1);
    assert!(metrics.residual_rms.is_finite());
}

#[test]
fn runs_single_tet_unstructured_f32_smoke_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_smoke_f32"
[numerics]
compute_precision = "f32"
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
    attach_single_tet_farfield(&mut case);
    let result = super::compressible_unstructured_3d::run(&case).expect("run");
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
    attach_single_tet_farfield(&mut case);
    let result = super::compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 1);
    assert!(metrics.residual_rms.is_finite());
}

#[test]
fn runs_single_tet_unstructured_second_order_smoke_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_second_order_smoke"
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
reconstruction = "muscl"
unstructured_limiter = "barth_jespersen"

[time]
scheme = "euler"
local_time_step = true
max_steps = 1
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let result = super::compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 1);
    assert_eq!(metrics.limiter, "barth_jespersen");
    assert!(metrics.residual_rms.is_finite());
}

#[test]
fn rejects_second_order_without_unstructured_limiter() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_second_order_bad"
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
reconstruction = "muscl"
limiter = "van_albada"

[time]
max_steps = 1
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let err = super::compressible_unstructured_3d::run(&case).expect_err("config");
    assert!(err.to_string().contains("unstructured_limiter"));
}

fn attach_single_tet_farfield(case: &mut crate::io::CaseSpec) {
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
}

#[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
fn dual_ellipsoid_benchmark_case() -> Option<std::path::PathBuf> {
    let output_case = std::path::PathBuf::from("output/case_dualellipsoid/case.toml");
    let mesh = std::env::var("ASIMU_MIX_CGNS_PATH")
        .map(std::path::PathBuf::from)
        .ok()
        .filter(|p| p.is_file())
        .or_else(|| {
            let p = std::path::PathBuf::from("output/case_dualellipsoid/mix.cgns");
            p.is_file().then_some(p)
        })?;
    let _ = mesh;
    if output_case.is_file() {
        return Some(output_case);
    }
    let bench_case = std::path::PathBuf::from("tests/benchmarks/dual_ellipsoid/case.toml");
    bench_case.is_file().then_some(bench_case)
}

#[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
#[test]
fn dual_ellipsoid_smoke_when_cgns_present() {
    let Some(case_path) = dual_ellipsoid_benchmark_case() else {
        return;
    };
    let result = super::run_case_path(&case_path).expect("run");
    assert_eq!(result.benchmark_id.as_deref(), Some("dual_ellipsoid"));
    let metrics = result.compressible_3d.expect("metrics");
    assert!(metrics.steps >= 1);
    assert!(metrics.residual_rms.is_finite() && metrics.residual_rms > 0.0);
}
