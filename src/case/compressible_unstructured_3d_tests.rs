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

#[cfg(feature = "cuda")]
#[test]
fn cuda_backend_single_tet_passes_validate() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_cuda_validate"
[numerics]
compute_precision = "f32"
backend = "cuda"
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
flux = "roe"
reconstruction = "first_order"

[time]
scheme = "rk4"
local_time_step = true
max_steps = 3
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    super::validate::exec_backend(&case).expect("cuda validate");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "gpu"]
fn runs_single_tet_unstructured_cuda_smoke_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_cuda_smoke"
[numerics]
compute_precision = "f32"
backend = "cuda"
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
flux = "roe"
reconstruction = "first_order"

[time]
scheme = "rk4"
local_time_step = true
max_steps = 3
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let result = super::compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 3);
    assert!(metrics.residual_rms.is_finite());
    assert!(metrics.residual_rms < 1.0e-3);
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
fn runs_single_tet_unstructured_lusgs_sweep_f32_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_lusgs_sweep_f32"
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
fn runs_single_tet_unstructured_dual_time_freestream_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_dual_time_freestream"
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
mode = "transient"
scheme = "dual_time"
dt = 1.0e-4
local_time_step = true
cfl = 0.4
max_inner_steps = 10
inner_tolerance = -2.0
max_steps = 1
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let result = super::compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 1);
    assert!(
        metrics.residual_log10 <= -2.0,
        "uniform freestream R_eff should converge: log10={}",
        metrics.residual_log10
    );
}

#[test]
fn runs_single_tet_unstructured_dual_time_freestream_f32_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_dual_time_freestream_f32"
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
mode = "transient"
scheme = "dual_time"
dt = 1.0e-4
local_time_step = true
cfl = 0.4
max_inner_steps = 10
inner_tolerance = -2.0
max_steps = 1
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let result = super::compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 1);
    assert!(
        metrics.residual_log10 <= -1.5,
        "f32 uniform freestream R_eff should converge: log10={}",
        metrics.residual_log10
    );
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

#[cfg(all(feature = "cuda", feature = "io-cgns", feature = "slow-tests"))]
fn dual_ellipsoid_cuda_benchmark_case() -> Option<std::path::PathBuf> {
    let mesh = std::env::var("ASIMU_MIX_CGNS_PATH")
        .map(std::path::PathBuf::from)
        .ok()
        .filter(|p| p.is_file())
        .or_else(|| {
            let p = std::path::PathBuf::from("output/case_dualellipsoid/mix.cgns");
            p.is_file().then_some(p)
        })?;
    let _ = mesh;
    let bench_case = std::path::PathBuf::from("tests/benchmarks/dual_ellipsoid/case_cuda_f32.toml");
    bench_case.is_file().then_some(bench_case)
}

#[cfg(all(feature = "cuda", feature = "io-cgns", feature = "slow-tests"))]
#[test]
#[ignore = "gpu"]
fn dual_ellipsoid_cuda_smoke_when_cgns_present() {
    let Some(case_path) = dual_ellipsoid_cuda_benchmark_case() else {
        return;
    };
    let result = super::run_case_path(&case_path).expect("cuda run");
    assert_eq!(result.benchmark_id.as_deref(), Some("dual_ellipsoid"));
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 2);
    assert!(metrics.residual_rms.is_finite() && metrics.residual_rms > 0.0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_backend_viscous_single_tet_passes_validate() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_cuda_viscous_validate"
[numerics]
compute_precision = "f32"
backend = "cuda"
[mesh]
kind = "structured_3d"
nx = 1
ny = 1
nz = 1

[physics]
gamma = 1.4
gas_constant = 287.0
prandtl = 0.72

[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15

[navier_stokes]
flux = "roe"
reconstruction = "first_order"

[time]
scheme = "euler"
local_time_step = true
max_steps = 1
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    super::validate::exec_backend(&case).expect("cuda viscous validate");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "gpu"]
fn runs_single_tet_unstructured_cuda_viscous_smoke_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_cuda_viscous_smoke"
[numerics]
compute_precision = "f32"
backend = "cuda"
[mesh]
kind = "structured_3d"
nx = 1
ny = 1
nz = 1

[physics]
gamma = 1.4
gas_constant = 287.0
prandtl = 0.72

[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15

[navier_stokes]
flux = "roe"
reconstruction = "first_order"

[time]
scheme = "euler"
local_time_step = true
max_steps = 2
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let result = super::compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 2);
    assert!(metrics.residual_rms.is_finite());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_backend_lusgs_single_tet_passes_validate() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_cuda_lusgs_validate"
[numerics]
compute_precision = "f32"
backend = "cuda"
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
flux = "roe"
reconstruction = "first_order"

[time]
scheme = "lu_sgs"
local_time_step = true
max_steps = 2
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    super::validate::exec_backend(&case).expect("cuda lu_sgs validate");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "gpu"]
fn runs_single_tet_unstructured_cuda_lusgs_smoke_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_cuda_lusgs_smoke"
[numerics]
compute_precision = "f32"
backend = "cuda"
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
flux = "roe"
reconstruction = "first_order"

[time]
scheme = "lu_sgs"
local_time_step = true
cfl = 0.001
max_steps = 2
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let result = super::compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 2);
    assert!(metrics.residual_rms.is_finite());
}
