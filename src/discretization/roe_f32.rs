//! Roe 近似 Riemann 求解器 f32 热路径（语义对齐 `roe.rs` / CUDA kernel）。

use crate::core::Vector3;
use crate::discretization::inviscid::InviscidFlux;
use crate::discretization::inviscid_f32::{
    ConservedStateF32, InviscidFluxF32, conserved_from_primitive_f32, face_tangent_basis_f32,
    inviscid_flux_f32_to_real, normalize_face_normal_f32, physical_inviscid_flux_f32,
};
use crate::discretization::roe::RoeFluxConfig;
use crate::discretization::viscous_boundary_f32::PrimitiveStateF32;
use crate::error::{AsimuError, Result};
use crate::physics::IdealGasEoS;

/// f32 Roe 数值通量（理想气体 Euler）。
pub fn roe_flux_with_primitives_f32(
    prim_l: &PrimitiveStateF32,
    prim_r: &PrimitiveStateF32,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &RoeFluxConfig,
) -> Result<InviscidFlux> {
    let n = normalize_face_normal_f32(normal)?;
    let left = conserved_from_primitive_f32(eos, prim_l)?;
    let right = conserved_from_primitive_f32(eos, prim_r)?;
    let flux_l = physical_inviscid_flux_f32(&left, prim_l, n);
    let flux_r = physical_inviscid_flux_f32(&right, prim_r, n);
    let gamma = eos.gamma as f32;
    let roe = roe_averages_f32(prim_l, prim_r, &left, &right, gamma, n)?;
    let (t1, t2) = face_tangent_basis_f32(n);
    let waves = wave_strengths_f32(prim_l, prim_r, &roe, n, t1, t2)?;
    let delta = entropy_delta_f32(&roe, config);
    let l1 = fixed_eigenvalue_f32(roe.un - roe.a, delta, config.entropy_fix);
    let l5 = fixed_eigenvalue_f32(roe.un + roe.a, delta, config.entropy_fix);
    let l_mid = roe.un.abs();
    let diss = dissipation_vector_f32(
        n,
        t1,
        t2,
        &roe,
        &waves,
        RoeDissipationCoeffsF32 { l1, l_mid, l5 },
    );
    Ok(inviscid_flux_f32_to_real(combine_fluxes_f32(
        flux_l, flux_r, diss,
    )))
}

struct RoeAveragesF32 {
    rho: f32,
    velocity: [f32; 3],
    enthalpy: f32,
    un: f32,
    a: f32,
}

struct WaveStrengthsF32 {
    alpha1: f32,
    alpha2: f32,
    alpha3: f32,
    alpha4: f32,
    alpha5: f32,
}

fn specific_enthalpy_f32(cons: &ConservedStateF32, prim: &PrimitiveStateF32) -> f32 {
    (cons.total_energy + prim.pressure) / prim.density
}

fn roe_averages_f32(
    prim_l: &PrimitiveStateF32,
    prim_r: &PrimitiveStateF32,
    left: &ConservedStateF32,
    right: &ConservedStateF32,
    gamma: f32,
    normal: Vector3,
) -> Result<RoeAveragesF32> {
    let sqrt_dl = prim_l.density.sqrt();
    let sqrt_dr = prim_r.density.sqrt();
    let inv = 1.0 / (sqrt_dl + sqrt_dr);
    let h_l = specific_enthalpy_f32(left, prim_l);
    let h_r = specific_enthalpy_f32(right, prim_r);
    let velocity = [
        (sqrt_dl * prim_l.velocity[0] + sqrt_dr * prim_r.velocity[0]) * inv,
        (sqrt_dl * prim_l.velocity[1] + sqrt_dr * prim_r.velocity[1]) * inv,
        (sqrt_dl * prim_l.velocity[2] + sqrt_dr * prim_r.velocity[2]) * inv,
    ];
    let enthalpy = (sqrt_dl * h_l + sqrt_dr * h_r) * inv;
    let vel2 = velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2];
    let gamma_term = (enthalpy - 0.5 * vel2).max(1.0e-6);
    let a = ((gamma - 1.0) * gamma_term).sqrt();
    let rho = sqrt_dl * sqrt_dr;
    let nx = normal.x as f32;
    let ny = normal.y as f32;
    let nz = normal.z as f32;
    let un = velocity[0] * nx + velocity[1] * ny + velocity[2] * nz;
    Ok(RoeAveragesF32 {
        rho,
        velocity,
        enthalpy,
        un,
        a,
    })
}

fn wave_strengths_f32(
    prim_l: &PrimitiveStateF32,
    prim_r: &PrimitiveStateF32,
    roe: &RoeAveragesF32,
    normal: Vector3,
    t1: Vector3,
    t2: Vector3,
) -> Result<WaveStrengthsF32> {
    let nx = normal.x as f32;
    let ny = normal.y as f32;
    let nz = normal.z as f32;
    let un_l = dot3_f32(prim_l.velocity, [nx, ny, nz]);
    let un_r = dot3_f32(prim_r.velocity, [nx, ny, nz]);
    let ut1_l = dot3_f32(prim_l.velocity, [t1.x as f32, t1.y as f32, t1.z as f32]);
    let ut1_r = dot3_f32(prim_r.velocity, [t1.x as f32, t1.y as f32, t1.z as f32]);
    let ut2_l = dot3_f32(prim_l.velocity, [t2.x as f32, t2.y as f32, t2.z as f32]);
    let ut2_r = dot3_f32(prim_r.velocity, [t2.x as f32, t2.y as f32, t2.z as f32]);
    let dp = prim_r.pressure - prim_l.pressure;
    let drho = prim_r.density - prim_l.density;
    let dun = un_r - un_l;
    let a2 = roe.a * roe.a;
    if a2 < 1.0e-30 {
        return Err(AsimuError::Field("Roe f32 声速过小".to_string()));
    }
    Ok(WaveStrengthsF32 {
        alpha1: 0.5 * (dp - roe.rho * roe.a * dun) / a2,
        alpha5: 0.5 * (dp + roe.rho * roe.a * dun) / a2,
        alpha2: drho - dp / a2,
        alpha3: roe.rho * (ut1_r - ut1_l),
        alpha4: roe.rho * (ut2_r - ut2_l),
    })
}

fn entropy_delta_f32(roe: &RoeAveragesF32, config: &RoeFluxConfig) -> f32 {
    let auto = 0.2 * (roe.un.abs() + roe.a);
    config
        .entropy_delta
        .map(|d| d as f32)
        .unwrap_or(auto)
        .max(1.0e-30)
}

fn fixed_eigenvalue_f32(lambda: f32, delta: f32, fix: bool) -> f32 {
    if !fix {
        return lambda.abs();
    }
    harten_entropy_fix_f32(lambda, delta)
}

fn harten_entropy_fix_f32(lambda: f32, delta: f32) -> f32 {
    let abs_l = lambda.abs();
    if abs_l >= delta {
        abs_l
    } else {
        (lambda * lambda + delta * delta) / (2.0 * delta)
    }
}

struct RoeDissipationCoeffsF32 {
    l1: f32,
    l_mid: f32,
    l5: f32,
}

fn dissipation_vector_f32(
    normal: Vector3,
    t1: Vector3,
    t2: Vector3,
    roe: &RoeAveragesF32,
    waves: &WaveStrengthsF32,
    coeffs: RoeDissipationCoeffsF32,
) -> InviscidFluxF32 {
    let RoeDissipationCoeffsF32 { l1, l_mid, l5 } = coeffs;
    let u = roe.velocity;
    let h = roe.enthalpy;
    let a = roe.a;
    let un = roe.un;
    let n = [normal.x as f32, normal.y as f32, normal.z as f32];
    let t1v = [t1.x as f32, t1.y as f32, t1.z as f32];
    let t2v = [t2.x as f32, t2.y as f32, t2.z as f32];

    let mut diss = scale_eigenvector_f32(
        &eigenvector_acoustic_f32(u, h, a, un, n, -1.0),
        l1 * waves.alpha1,
    );
    add_scaled_f32(&mut diss, &eigenvector_contact_f32(u), l_mid * waves.alpha2);
    add_scaled_f32(
        &mut diss,
        &eigenvector_shear_f32(t1v, dot3_f32(u, t1v)),
        l_mid * waves.alpha3,
    );
    add_scaled_f32(
        &mut diss,
        &eigenvector_shear_f32(t2v, dot3_f32(u, t2v)),
        l_mid * waves.alpha4,
    );
    add_scaled_f32(
        &mut diss,
        &eigenvector_acoustic_f32(u, h, a, un, n, 1.0),
        l5 * waves.alpha5,
    );
    diss
}

fn eigenvector_acoustic_f32(
    u: [f32; 3],
    h: f32,
    a: f32,
    un: f32,
    n: [f32; 3],
    sign: f32,
) -> InviscidFluxF32 {
    InviscidFluxF32 {
        mass: 1.0,
        momentum: [
            u[0] + sign * a * n[0],
            u[1] + sign * a * n[1],
            u[2] + sign * a * n[2],
        ],
        energy: h + sign * a * un,
    }
}

fn eigenvector_contact_f32(u: [f32; 3]) -> InviscidFluxF32 {
    let vel2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    InviscidFluxF32 {
        mass: 1.0,
        momentum: u,
        energy: 0.5 * vel2,
    }
}

fn eigenvector_shear_f32(t: [f32; 3], ut: f32) -> InviscidFluxF32 {
    InviscidFluxF32 {
        mass: 0.0,
        momentum: t,
        energy: ut,
    }
}

fn scale_eigenvector_f32(v: &InviscidFluxF32, scale: f32) -> InviscidFluxF32 {
    InviscidFluxF32 {
        mass: scale * v.mass,
        momentum: [
            scale * v.momentum[0],
            scale * v.momentum[1],
            scale * v.momentum[2],
        ],
        energy: scale * v.energy,
    }
}

fn add_scaled_f32(target: &mut InviscidFluxF32, v: &InviscidFluxF32, scale: f32) {
    target.mass += scale * v.mass;
    target.momentum[0] += scale * v.momentum[0];
    target.momentum[1] += scale * v.momentum[1];
    target.momentum[2] += scale * v.momentum[2];
    target.energy += scale * v.energy;
}

fn combine_fluxes_f32(
    left: InviscidFluxF32,
    right: InviscidFluxF32,
    diss: InviscidFluxF32,
) -> InviscidFluxF32 {
    let half = 0.5;
    InviscidFluxF32 {
        mass: half * (left.mass + right.mass) - half * diss.mass,
        momentum: [
            half * (left.momentum[0] + right.momentum[0]) - half * diss.momentum[0],
            half * (left.momentum[1] + right.momentum[1]) - half * diss.momentum[1],
            half * (left.momentum[2] + right.momentum[2]) - half * diss.momentum[2],
        ],
        energy: half * (left.energy + right.energy) - half * diss.energy,
    }
}

fn dot3_f32(v: [f32; 3], n: [f32; 3]) -> f32 {
    v[0] * n[0] + v[1] * n[1] + v[2] * n[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::discretization::roe::roe_flux_with_primitives;
    use crate::discretization::viscous_boundary_f32::primitive_state_f32_from_real;
    use crate::physics::{ConservedState, PrimitiveState};

    #[test]
    fn roe_f32_matches_f64_on_sod_interface() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let left = PrimitiveState {
            density: 1.0,
            velocity: [0.0, 0.0, 0.0],
            pressure: 1.0,
            temperature: 1.0,
        };
        let right = PrimitiveState {
            density: 0.125,
            velocity: [0.0, 0.0, 0.0],
            pressure: 0.1,
            temperature: 1.0,
        };
        let cons_l = ConservedState::from_primitive(&eos, &left).expect("left");
        let cons_r = ConservedState::from_primitive(&eos, &right).expect("right");
        let normal = Vector3::new(1.0, 0.0, 0.0);
        let config = RoeFluxConfig::default();
        let f64_flux =
            roe_flux_with_primitives(&cons_l, &cons_r, &left, &right, normal, &eos, &config)
                .expect("f64");
        let f32_flux = roe_flux_with_primitives_f32(
            &primitive_state_f32_from_real(left),
            &primitive_state_f32_from_real(right),
            normal,
            &eos,
            &config,
        )
        .expect("f32");
        assert!(approx_eq(f32_flux.mass, f64_flux.mass, 1.0e-3));
        assert!(approx_eq(f32_flux.energy, f64_flux.energy, 1.0e-2));
    }
}
