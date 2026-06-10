use super::validate_simplec_step;
use crate::core::Real;

#[test]
fn simplec_step_validation_rejects_divergence() {
    let err = validate_simplec_step(1.0e60, 1.0, 1.0).expect_err("divergence");
    assert!(err.to_string().contains("SIMPLEC 发散"));
}

#[test]
fn simplec_step_validation_rejects_non_finite_values() {
    let err = validate_simplec_step(1.0, Real::INFINITY, 1.0).expect_err("non-finite");
    assert!(err.to_string().contains("非有限值"));
}
