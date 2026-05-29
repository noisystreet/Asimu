//! 1D 扩散 benchmark 端到端：case.toml → 装配 → BC → 求解。

use std::path::Path;

use asimu::discretization::{apply_boundary_conditions, assemble_diffusion_1d};
use asimu::io::load_case;
use asimu::linalg::LinearSystem;

const CASE_PATH: &str = "tests/benchmarks/1d_diffusion_analytical/case.toml";

fn analytical_solution(x: f64, length: f64) -> f64 {
    x / length
}

#[test]
fn one_d_diffusion_with_dirichlet_matches_analytical() {
    let case = load_case(Path::new(CASE_PATH)).expect("load case");
    let mesh = case.mesh.as_1d().expect("1d mesh");
    let n = mesh.num_cells();
    let mut system = LinearSystem::zeros(n).expect("system");
    assemble_diffusion_1d(mesh, &mut system, case.diffusivity()).expect("assemble");
    apply_boundary_conditions(
        mesh,
        &case.boundary,
        &mut system,
        case.diffusivity(),
    )
    .expect("bc");

    let solution = system.solve_tridiagonal().expect("solve");
    let dx = mesh.dx();
    let origin = mesh.origin;

    let mut max_err = 0.0_f64;
    for (i, &phi) in solution.iter().enumerate() {
        let x = origin + (i as f64 + 0.5) * dx;
        let exact = analytical_solution(x, mesh.length);
        max_err = max_err.max((phi - exact).abs());
    }
    assert!(
        max_err < 0.25 / n as f64,
        "最大误差 {max_err} 超过 O(1/n) 容差（n={n}）"
    );
}

#[test]
fn case_boundary_patches_are_dirichlet() {
    let case = load_case(Path::new(CASE_PATH)).expect("load case");
    let left = case.boundary.find("left").expect("left patch");
    let right = case.boundary.find("right").expect("right patch");
    assert!(matches!(left.kind, asimu::boundary::BoundaryKind::Dirichlet { value: 0.0 }));
    assert!(matches!(
        right.kind,
        asimu::boundary::BoundaryKind::Dirichlet { value: 1.0 }
    ));
}
