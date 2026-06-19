use super::super::gmres_block_preconditioner_unstructured_viscous::PARABOLIC_SPECTRAL_FACTOR_3D;
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
    let mut block = [0.0; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    add_viscous_off_diagonal(&mut block, [0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(block[diag_index(0)], 0.0);
    assert_eq!(block[diag_index(1)], -1.0);
    assert_eq!(block[diag_index(2)], -2.0);
    assert_eq!(block[diag_index(3)], -3.0);
    assert_eq!(block[diag_index(4)], -4.0);
}
