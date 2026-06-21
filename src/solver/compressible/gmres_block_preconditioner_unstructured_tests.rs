use super::super::gmres_block_preconditioner_unstructured_viscous::{
    PARABOLIC_SPECTRAL_FACTOR_3D, ViscousCellDiffusivity, add_viscous_off_diagonal,
    viscous_component_sigma,
};
use super::*;

fn diag_index(component: usize) -> usize {
    component * CONSERVED_COMPONENTS_3D + component
}

#[test]
fn viscous_component_sigma_skips_density_and_splits_diffusivity() {
    let sigma = viscous_component_sigma(
        ViscousCellDiffusivity {
            momentum: 2.0,
            energy: 3.0,
        },
        0.5,
        2.0,
    );
    let scale = PARABOLIC_SPECTRAL_FACTOR_3D * 0.5 * 0.5 / (2.0 * 2.0);
    assert_eq!(sigma[0], 0.0);
    assert_eq!(sigma[1], scale * 2.0);
    assert_eq!(sigma[2], scale * 2.0);
    assert_eq!(sigma[3], scale * 2.0);
    assert_eq!(sigma[4], scale * 3.0);
}

#[test]
fn add_viscous_off_diagonal_uses_component_coefficients() {
    use crate::core::Vector3;

    let mut block = [0.0; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    add_viscous_off_diagonal(
        &mut block,
        ViscousCellDiffusivity {
            momentum: 4.0,
            energy: 12.0,
        },
        1.0,
        Vector3::new(1.0, 0.0, 0.0),
    );
    assert_eq!(block[diag_index(0)], 0.0);
    assert_eq!(block[diag_index(1)], -4.0);
    assert_eq!(block[diag_index(2)], -4.0);
    assert_eq!(block[diag_index(3)], -4.0);
    assert_eq!(block[diag_index(4)], -12.0);
    assert_eq!(block[CONSERVED_COMPONENTS_3D + 2], 0.0);
}

#[test]
fn add_viscous_off_diagonal_adds_momentum_cross_terms() {
    use crate::core::Vector3;

    let mut block = [0.0; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    add_viscous_off_diagonal(
        &mut block,
        ViscousCellDiffusivity {
            momentum: 4.0,
            energy: 0.0,
        },
        2.0,
        Vector3::new(0.6, 0.8, 0.0),
    );
    let shear = 2.0 * (4.0 / (4.0 / 3.0));
    assert!((block[CONSERVED_COMPONENTS_3D + 2] + shear * 0.6 * 0.8).abs() < 1.0e-12);
    assert!((block[2 * CONSERVED_COMPONENTS_3D + 1] + shear * 0.6 * 0.8).abs() < 1.0e-12);
    assert_eq!(block[CONSERVED_COMPONENTS_3D + 1], -8.0);
}

#[test]
fn finite_difference_perturbation_accepts_backward_direction() {
    use crate::field::max_physical_increment_scale;
    use crate::physics::ConservedState;
    use crate::solver::compressible::gmres_implicit_3d::CONSERVED_COMPONENTS_3D;

    let base = ConservedState {
        density: 1.0,
        momentum: [0.0, 0.0, 0.0],
        total_energy: 1.0,
    };
    let mut increment = [0.0; CONSERVED_COMPONENTS_3D];
    increment[1] = 10.0;
    let forward_eps = max_physical_increment_scale(&base, increment, 1.0, 1.4, 0.0);
    assert!(forward_eps > 0.0 && forward_eps < 1.0);

    increment[1] = -10.0;
    let backward_eps = max_physical_increment_scale(&base, increment, 1.0, 1.4, 0.0);
    assert!(backward_eps > 0.0);
}
