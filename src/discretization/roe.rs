//! Roe 近似 Riemann 求解器 + Harten 熵修正。
//!
//! 理论：[`docs/theory/inviscid_flux.md`](../../docs/theory/inviscid_flux.md) §4

use crate::core::{Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::field::primitive_from_conserved;
use crate::physics::{ConservedState, IdealGasEoS, PrimitiveState};

use super::inviscid::{InviscidFlux, physical_inviscid_flux, velocity_dot_normal};

/// Roe 通量选项。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoeFluxConfig {
    /// 是否对声学波特征值做 Harten 熵修正。
    pub entropy_fix: bool,
    /// 熵修正宽度 \(\delta\)；`None` 时按 \(0.2(|u_n|+a)\) 自动估计。
    pub entropy_delta: Option<Real>,
}

impl Default for RoeFluxConfig {
    fn default() -> Self {
        Self {
            entropy_fix: true,
            entropy_delta: None,
        }
    }
}

/// Roe 数值通量 \( \hat{\mathbf{F}} \cdot \mathbf{n} \)（理想气体 Euler）。
pub fn roe_flux(
    left: &ConservedState,
    right: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &RoeFluxConfig,
) -> Result<InviscidFlux> {
    let n = normalize_or_error(normal)?;
    let prim_l = primitive_from_conserved(eos, left)?;
    let prim_r = primitive_from_conserved(eos, right)?;
    let flux_l = physical_inviscid_flux(left, &prim_l, n);
    let flux_r = physical_inviscid_flux(right, &prim_r, n);
    let roe = roe_averages(&prim_l, &prim_r, left, right, eos.gamma, n)?;
    let (t1, t2) = face_tangent_basis(n);
    let waves = wave_strengths(&prim_l, &prim_r, &roe, n, t1, t2)?;
    let delta = entropy_delta(&roe, config);
    let l1 = fixed_eigenvalue(roe.un - roe.a, delta, config.entropy_fix);
    let l5 = fixed_eigenvalue(roe.un + roe.a, delta, config.entropy_fix);
    let l_mid = roe.un.abs();
    let diss = dissipation_vector(
        n,
        t1,
        t2,
        &RoeDissipation {
            roe: &roe,
            waves: &waves,
            l1,
            l_mid,
            l5,
        },
    );
    Ok(combine_fluxes(flux_l, flux_r, diss))
}

struct RoeAverages {
    rho: Real,
    velocity: [Real; 3],
    enthalpy: Real,
    un: Real,
    a: Real,
}

struct WaveStrengths {
    alpha1: Real,
    alpha2: Real,
    alpha3: Real,
    alpha4: Real,
    alpha5: Real,
}

struct Dissipation {
    mass: Real,
    momentum: [Real; 3],
    energy: Real,
}

fn normalize_or_error(normal: Vector3) -> Result<Vector3> {
    let mag = (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z).sqrt();
    if mag < Real::EPSILON {
        return Err(AsimuError::Mesh("面法向不能为零向量".to_string()));
    }
    Ok(Vector3::new(normal.x / mag, normal.y / mag, normal.z / mag))
}

fn specific_enthalpy(cons: &ConservedState, prim: &PrimitiveState) -> Real {
    (cons.total_energy + prim.pressure) / prim.density
}

fn roe_averages(
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    left: &ConservedState,
    right: &ConservedState,
    gamma: Real,
    normal: Vector3,
) -> Result<RoeAverages> {
    let sqrt_dl = prim_l.density.sqrt();
    let sqrt_dr = prim_r.density.sqrt();
    let inv = 1.0 / (sqrt_dl + sqrt_dr);
    let h_l = specific_enthalpy(left, prim_l);
    let h_r = specific_enthalpy(right, prim_r);
    let velocity = [
        (sqrt_dl * prim_l.velocity[0] + sqrt_dr * prim_r.velocity[0]) * inv,
        (sqrt_dl * prim_l.velocity[1] + sqrt_dr * prim_r.velocity[1]) * inv,
        (sqrt_dl * prim_l.velocity[2] + sqrt_dr * prim_r.velocity[2]) * inv,
    ];
    let enthalpy = (sqrt_dl * h_l + sqrt_dr * h_r) * inv;
    let vel2 = velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2];
    let gamma_term = enthalpy - 0.5 * vel2;
    if gamma_term <= 0.0 {
        return Err(AsimuError::Field("Roe 平均焓导致非物理解".to_string()));
    }
    let a = ((gamma - 1.0) * gamma_term).sqrt();
    let rho = sqrt_dl * sqrt_dr;
    let un = velocity_dot_normal(velocity, normal);
    Ok(RoeAverages {
        rho,
        velocity,
        enthalpy,
        un,
        a,
    })
}

fn face_tangent_basis(normal: Vector3) -> (Vector3, Vector3) {
    let ref_vec = if normal.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let t1 = cross(normal, ref_vec);
    let t1 = normalize_unchecked(t1);
    let t2 = cross(normal, t1);
    (t1, normalize_unchecked(t2))
}

fn cross(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn normalize_unchecked(v: Vector3) -> Vector3 {
    let mag = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    Vector3::new(v.x / mag, v.y / mag, v.z / mag)
}

fn tangential_velocity(velocity: [Real; 3], tangent: Vector3) -> Real {
    velocity[0] * tangent.x + velocity[1] * tangent.y + velocity[2] * tangent.z
}

fn wave_strengths(
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    roe: &RoeAverages,
    normal: Vector3,
    t1: Vector3,
    t2: Vector3,
) -> Result<WaveStrengths> {
    let un_l = velocity_dot_normal(prim_l.velocity, normal);
    let un_r = velocity_dot_normal(prim_r.velocity, normal);
    let ut1_l = tangential_velocity(prim_l.velocity, t1);
    let ut1_r = tangential_velocity(prim_r.velocity, t1);
    let ut2_l = tangential_velocity(prim_l.velocity, t2);
    let ut2_r = tangential_velocity(prim_r.velocity, t2);
    let dp = prim_r.pressure - prim_l.pressure;
    let drho = prim_r.density - prim_l.density;
    let dun = un_r - un_l;
    let a2 = roe.a * roe.a;
    if a2 < Real::EPSILON {
        return Err(AsimuError::Field("Roe 声速过小".to_string()));
    }
    let alpha1 = 0.5 * (dp - roe.rho * roe.a * dun) / a2;
    let alpha5 = 0.5 * (dp + roe.rho * roe.a * dun) / a2;
    let alpha2 = drho - dp / a2;
    let alpha3 = roe.rho * (ut1_r - ut1_l);
    let alpha4 = roe.rho * (ut2_r - ut2_l);
    Ok(WaveStrengths {
        alpha1,
        alpha2,
        alpha3,
        alpha4,
        alpha5,
    })
}

fn entropy_delta(roe: &RoeAverages, config: &RoeFluxConfig) -> Real {
    config
        .entropy_delta
        .unwrap_or(0.2 * (roe.un.abs() + roe.a))
        .max(Real::EPSILON)
}

fn fixed_eigenvalue(lambda: Real, delta: Real, fix: bool) -> Real {
    if !fix {
        return lambda.abs();
    }
    harten_entropy_fix(lambda, delta)
}

/// Harten 熵修正：\(|\lambda| \to (\lambda^2 + \delta^2) / (2\delta)\) 当 \(|\lambda| < \delta\)。
fn harten_entropy_fix(lambda: Real, delta: Real) -> Real {
    let abs_l = lambda.abs();
    if abs_l >= delta {
        abs_l
    } else {
        (lambda * lambda + delta * delta) / (2.0 * delta)
    }
}

struct RoeDissipation<'a> {
    roe: &'a RoeAverages,
    waves: &'a WaveStrengths,
    l1: Real,
    l_mid: Real,
    l5: Real,
}

fn dissipation_vector(
    normal: Vector3,
    t1: Vector3,
    t2: Vector3,
    input: &RoeDissipation<'_>,
) -> Dissipation {
    let roe = input.roe;
    let waves = input.waves;
    let u = roe.velocity;
    let h = roe.enthalpy;
    let a = roe.a;
    let un = roe.un;
    let n = [normal.x, normal.y, normal.z];
    let t1v = [t1.x, t1.y, t1.z];
    let t2v = [t2.x, t2.y, t2.z];

    let r1 = eigenvector_acoustic(u, h, a, un, n, -1.0);
    let r2 = eigenvector_contact(u);
    let r3 = eigenvector_shear(t1v, tangential_velocity(u, t1));
    let r4 = eigenvector_shear(t2v, tangential_velocity(u, t2));
    let r5 = eigenvector_acoustic(u, h, a, un, n, 1.0);

    let mut diss = scale_eigenvector(&r1, input.l1 * waves.alpha1);
    add_scaled(&mut diss, &r2, input.l_mid * waves.alpha2);
    add_scaled(&mut diss, &r3, input.l_mid * waves.alpha3);
    add_scaled(&mut diss, &r4, input.l_mid * waves.alpha4);
    add_scaled(&mut diss, &r5, input.l5 * waves.alpha5);
    diss
}

fn eigenvector_acoustic(
    u: [Real; 3],
    h: Real,
    a: Real,
    un: Real,
    n: [Real; 3],
    sign: Real,
) -> Dissipation {
    Dissipation {
        mass: 1.0,
        momentum: [
            u[0] + sign * a * n[0],
            u[1] + sign * a * n[1],
            u[2] + sign * a * n[2],
        ],
        energy: h + sign * a * un,
    }
}

fn eigenvector_contact(u: [Real; 3]) -> Dissipation {
    let vel2 = 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    Dissipation {
        mass: 1.0,
        momentum: u,
        energy: vel2,
    }
}

fn eigenvector_shear(tangent: [Real; 3], ut: Real) -> Dissipation {
    Dissipation {
        mass: 0.0,
        momentum: tangent,
        energy: ut,
    }
}

fn scale_eigenvector(v: &Dissipation, scale: Real) -> Dissipation {
    Dissipation {
        mass: scale * v.mass,
        momentum: [
            scale * v.momentum[0],
            scale * v.momentum[1],
            scale * v.momentum[2],
        ],
        energy: scale * v.energy,
    }
}

fn add_scaled(target: &mut Dissipation, v: &Dissipation, scale: Real) {
    target.mass += scale * v.mass;
    target.momentum[0] += scale * v.momentum[0];
    target.momentum[1] += scale * v.momentum[1];
    target.momentum[2] += scale * v.momentum[2];
    target.energy += scale * v.energy;
}

fn combine_fluxes(flux_l: InviscidFlux, flux_r: InviscidFlux, diss: Dissipation) -> InviscidFlux {
    let half = 0.5;
    InviscidFlux {
        mass: half * (flux_l.mass + flux_r.mass) - half * diss.mass,
        momentum: [
            half * (flux_l.momentum[0] + flux_r.momentum[0]) - half * diss.momentum[0],
            half * (flux_l.momentum[1] + flux_r.momentum[1]) - half * diss.momentum[1],
            half * (flux_l.momentum[2] + flux_r.momentum[2]) - half * diss.momentum[2],
        ],
        energy: half * (flux_l.energy + flux_r.energy) - half * diss.energy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::discretization::inviscid::physical_inviscid_flux;
    use crate::discretization::reconstruction::reconstruct_first_order;

    fn sod_left_state(eos: &IdealGasEoS) -> ConservedState {
        let prim = PrimitiveState {
            density: 1.0,
            velocity: [0.0, 0.0, 0.0],
            pressure: 1.0,
            temperature: 1.0,
        };
        ConservedState::from_primitive(eos, &prim).expect("left")
    }

    fn sod_right_state(eos: &IdealGasEoS) -> ConservedState {
        let prim = PrimitiveState {
            density: 0.125,
            velocity: [0.0, 0.0, 0.0],
            pressure: 0.1,
            temperature: 1.0,
        };
        ConservedState::from_primitive(eos, &prim).expect("right")
    }

    #[test]
    fn identical_states_match_physical_flux() {
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let prim = eos
            .freestream_primitive(0.3, 1.0, 1.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let cons = ConservedState::from_primitive(&eos, &prim).expect("cons");
        let n = Vector3::new(1.0, 0.0, 0.0);
        let config = RoeFluxConfig::default();
        let roe = roe_flux(&cons, &cons, n, &eos, &config).expect("roe");
        let phys = physical_inviscid_flux(&cons, &prim, n);
        assert!(approx_eq(roe.mass, phys.mass, 1.0e-10));
        assert!(approx_eq(roe.momentum[0], phys.momentum[0], 1.0e-10));
        assert!(approx_eq(roe.energy, phys.energy, 1.0e-10));
    }

    #[test]
    fn sod_interface_roe_flux_matches_reference_values() {
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let left = sod_left_state(&eos);
        let right = sod_right_state(&eos);
        let n = Vector3::new(1.0, 0.0, 0.0);
        let config = RoeFluxConfig::default();
        let iface = reconstruct_first_order(left, right);
        let flux = roe_flux(&iface.left, &iface.right, n, &eos, &config).expect("flux");
        assert!(approx_eq(flux.momentum[0], 0.55, 1.0e-10));
        assert!(approx_eq(flux.mass, 0.390_660_485_785_962_96, 1.0e-10));
        assert!(approx_eq(flux.energy, 1.295_882_277_373_113, 1.0e-10));
    }

    #[test]
    fn entropy_fix_keeps_expansion_flux_realizable() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let left_prim = PrimitiveState {
            density: 1.0,
            velocity: [100.0, 0.0, 0.0],
            pressure: 1.0e5,
            temperature: 300.0,
        };
        let right_prim = PrimitiveState {
            density: 0.2,
            velocity: [50.0, 0.0, 0.0],
            pressure: 1.0e4,
            temperature: 300.0,
        };
        let left = ConservedState::from_primitive(&eos, &left_prim).expect("left");
        let right = ConservedState::from_primitive(&eos, &right_prim).expect("right");
        let n = Vector3::new(1.0, 0.0, 0.0);
        let with_fix = RoeFluxConfig {
            entropy_fix: true,
            ..RoeFluxConfig::default()
        };
        let without_fix = RoeFluxConfig {
            entropy_fix: false,
            ..RoeFluxConfig::default()
        };
        let flux_fix = roe_flux(&left, &right, n, &eos, &with_fix).expect("fix");
        let flux_raw = roe_flux(&left, &right, n, &eos, &without_fix).expect("raw");
        assert!(flux_fix.mass.is_finite());
        assert!(flux_raw.mass.is_finite());
        assert!(flux_fix.energy.is_finite());
    }

    #[test]
    fn harten_fix_smooths_small_eigenvalues() {
        let delta = 0.5;
        assert!(approx_eq(harten_entropy_fix(0.1, delta), 0.26, 1.0e-12));
        assert!(approx_eq(harten_entropy_fix(1.0, delta), 1.0, 1.0e-12));
    }
}
