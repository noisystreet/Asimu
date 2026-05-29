//! Sod 激波管 benchmark：数值解 vs 精确 Riemann 解。

use asimu::solver::{SodBenchmarkConfig, run_sod_benchmark};

const L1_REF: f64 = 0.025;
const L1_TOL: f64 = 0.01;
const L2_REF: f64 = 0.04;
const L2_TOL: f64 = 0.02;

#[test]
fn sod_1d_matches_exact_riemann_solution() {
    let result = run_sod_benchmark(&SodBenchmarkConfig::default()).expect("sod benchmark");
    assert!(
        result.l1_density <= L1_REF + L1_TOL,
        "L1 density error {} exceeds {} + {}",
        result.l1_density,
        L1_REF,
        L1_TOL
    );
    assert!(
        result.l2_density <= L2_REF + L2_TOL,
        "L2 density error {} exceeds {} + {}",
        result.l2_density,
        L2_REF,
        L2_TOL
    );
    assert!((result.final_time - 0.2).abs() < 1.0e-5);
    assert!(result.steps > 0);
}

#[test]
fn sod_midpoint_density_between_shock_and_expansion() {
    let result = run_sod_benchmark(&SodBenchmarkConfig::default()).expect("sod benchmark");
    let mid = result.density_numeric.len() / 2;
    let rho_mid = result.density_numeric[mid];
    let rho_left = result.density_numeric[mid - 10];
    let rho_right = result.density_numeric[mid + 10];
    assert!(rho_left > rho_mid);
    assert!(rho_right < rho_mid);
}
