use crate::core::{Real, Vector3};
use crate::error::Result;
use crate::field::PrimitiveFields;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};
use crate::solver::compressible::gmres_implicit_3d::CONSERVED_COMPONENTS_3D;

pub(super) const PARABOLIC_SPECTRAL_FACTOR_3D: Real = 6.0;
const MOMENTUM_VISCOUS_FACTOR_3D: Real = 4.0 / 3.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ViscousCellDiffusivity {
    pub(super) momentum: Real,
    pub(super) energy: Real,
}

pub(super) fn local_viscous_diffusivity(
    primitives: &PrimitiveFields,
    eos: &IdealGasEoS,
    viscous: Option<&ViscousPhysicsConfig>,
) -> Result<Option<Vec<ViscousCellDiffusivity>>> {
    let Some(config) = viscous else {
        return Ok(None);
    };
    let n = primitives.num_cells();
    let mut diff = Vec::with_capacity(n);
    for i in 0..n {
        let rho = primitives.density.values()[i].max(1.0e-30);
        let pressure = primitives.pressure.values()[i].max(1.0e-30);
        let t_star = config.static_temperature(pressure, rho, eos);
        let (mu_eff, _lambda) = config.face_transport_coefficients(t_star, t_star, eos)?;
        let nu = mu_eff / rho;
        diff.push(ViscousCellDiffusivity {
            momentum: MOMENTUM_VISCOUS_FACTOR_3D * nu,
            energy: eos.gamma * nu / config.prandtl,
        });
    }
    Ok(Some(diff))
}

pub(super) fn add_component_sigma(
    out: &mut [Real; CONSERVED_COMPONENTS_3D],
    sigma: [Real; CONSERVED_COMPONENTS_3D],
) {
    for (dst, src) in out.iter_mut().zip(sigma) {
        *dst += src;
    }
}

pub(super) fn viscous_coupling_from_scale(
    diffusivity: ViscousCellDiffusivity,
    parabolic_scale: Real,
) -> [Real; CONSERVED_COMPONENTS_3D] {
    if parabolic_scale <= 0.0 {
        return [0.0; CONSERVED_COMPONENTS_3D];
    }
    let mut sigma = [0.0; CONSERVED_COMPONENTS_3D];
    sigma[1] = parabolic_scale * diffusivity.momentum.max(0.0);
    sigma[2] = sigma[1];
    sigma[3] = sigma[1];
    sigma[4] = parabolic_scale * diffusivity.energy.max(0.0);
    sigma
}

pub(super) fn viscous_component_sigma(
    diffusivity: ViscousCellDiffusivity,
    area: Real,
    volume: Real,
) -> [Real; CONSERVED_COMPONENTS_3D] {
    let mut sigma = [0.0; CONSERVED_COMPONENTS_3D];
    if area <= Real::EPSILON || volume <= 1.0e-30 {
        return sigma;
    }
    let scale = PARABOLIC_SPECTRAL_FACTOR_3D * area * area / (volume * volume);
    sigma[1] = scale * diffusivity.momentum.max(0.0);
    sigma[2] = sigma[1];
    sigma[3] = sigma[1];
    sigma[4] = scale * diffusivity.energy.max(0.0);
    sigma
}

pub(super) fn add_viscous_off_diagonal(
    block: &mut [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D],
    diffusivity: ViscousCellDiffusivity,
    parabolic_scale: Real,
    normal: Vector3,
) {
    let coupling = viscous_coupling_from_scale(diffusivity, parabolic_scale);
    for (component, &value) in coupling.iter().enumerate() {
        if value > 0.0 {
            block[component * CONSERVED_COMPONENTS_3D + component] -= value;
        }
    }
    if parabolic_scale <= 0.0 {
        return;
    }
    let shear = parabolic_scale * (diffusivity.momentum / MOMENTUM_VISCOUS_FACTOR_3D).max(0.0);
    if shear <= Real::EPSILON {
        return;
    }
    let n = [normal.x, normal.y, normal.z];
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                block[(1 + i) * CONSERVED_COMPONENTS_3D + (1 + j)] -= shear * n[i] * n[j];
            }
        }
    }
}
