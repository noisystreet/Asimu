//! 非结构 3D 可压缩 CUDA case 集成测试（`#[cfg(feature = "cuda")]` 子模块）。

use crate::case::{compressible_unstructured_3d, run_case_path, validate};
use crate::io::parse_case_str;

use super::attach_single_tet_farfield;

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
    validate::exec_backend(&case).expect("cuda validate");
}

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
    let result = compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 3);
    assert!(metrics.residual_rms.is_finite());
    assert!(metrics.residual_rms < 1.0e-3);
}
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

#[test]
#[ignore = "gpu"]
fn dual_ellipsoid_cuda_smoke_when_cgns_present() {
    let Some(case_path) = dual_ellipsoid_cuda_benchmark_case() else {
        return;
    };
    let result = run_case_path(&case_path).expect("cuda run");
    assert_eq!(result.benchmark_id.as_deref(), Some("dual_ellipsoid"));
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 2);
    assert!(metrics.residual_rms.is_finite() && metrics.residual_rms > 0.0);
}

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
    validate::exec_backend(&case).expect("cuda viscous validate");
}

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
    let result = compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 2);
    assert!(metrics.residual_rms.is_finite());
}

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
    validate::exec_backend(&case).expect("cuda lu_sgs validate");
}

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
    let result = compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 2);
    assert!(metrics.residual_rms.is_finite());
}

#[test]
fn cuda_backend_dual_time_single_tet_passes_validate() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_cuda_dual_time_validate"
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
mode = "transient"
scheme = "dual_time"
dt = 1.0e-4
local_time_step = true
cfl = 0.4
max_inner_steps = 10
inner_tolerance = -2.0
lusgs_sweep = false
max_steps = 1
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    validate::exec_backend(&case).expect("cuda dual_time validate");
}

#[test]
#[ignore = "gpu"]
fn runs_single_tet_unstructured_cuda_dual_time_smoke_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_cuda_dual_time_smoke"
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
mode = "transient"
scheme = "dual_time"
dt = 1.0e-4
local_time_step = true
cfl = 0.001
max_inner_steps = 20
inner_tolerance = -2.0
lusgs_sweep = false
max_steps = 1
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let result = compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 1);
    assert!(
        metrics.residual_log10 <= -2.0,
        "cuda dual_time freestream should converge: log10={}",
        metrics.residual_log10
    );
    assert!(metrics.inner_iterations > 0);
}

#[test]
fn cuda_backend_dual_time_navier_stokes_single_tet_passes_validate() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_cuda_dual_time_viscous_validate"
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
mode = "transient"
scheme = "dual_time"
dt = 1.0e-4
local_time_step = true
cfl = 0.4
max_inner_steps = 10
inner_tolerance = -2.0
lusgs_sweep = false
max_steps = 1
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    validate::exec_backend(&case).expect("cuda dual_time navier_stokes validate");
}

#[test]
#[ignore = "gpu"]
fn runs_single_tet_unstructured_cuda_dual_time_viscous_smoke_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_cuda_dual_time_viscous_smoke"
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
mode = "transient"
scheme = "dual_time"
dt = 1.0e-4
local_time_step = true
cfl = 0.001
max_inner_steps = 20
inner_tolerance = -2.0
lusgs_sweep = false
max_steps = 1
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let result = compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 1);
    assert!(
        metrics.residual_log10 <= -2.0,
        "cuda dual_time viscous freestream should converge: log10={}",
        metrics.residual_log10
    );
    assert!(metrics.inner_iterations > 0);
}
