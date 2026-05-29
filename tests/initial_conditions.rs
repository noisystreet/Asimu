//! 初始条件解析与场构建集成测试。

use std::path::Path;

use asimu::core::approx_eq;
use asimu::io::load_case;

const CASE_PATH: &str = "tests/benchmarks/1d_diffusion_analytical/case.toml";

#[test]
fn benchmark_case_has_linear_phi_initial() {
    let case = load_case(Path::new(CASE_PATH)).expect("load case");
    let phi = case.initial_scalar("phi").expect("phi");
    assert_eq!(phi.len(), case.mesh.as_1d().expect("1d").num_cells());

    let mesh = case.mesh.as_1d().expect("1d");
    let dx = mesh.dx();
    let x0 = mesh.origin + 0.5 * dx;
    let expected = x0 / mesh.length;
    assert!(approx_eq(phi.values()[0], expected, 1.0e-12));

    let fields = case.build_initial_fields().expect("fields");
    let stored = fields.get("phi").expect("phi in fields");
    assert_eq!(stored.values(), phi.values());
}
