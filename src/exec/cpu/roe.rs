//! Roe 通量四路批处理（`simd-fvm`：特征值修正用 `f64x4`）。

use crate::core::{Real, Vector3};
use crate::physics::{ConservedState, IdealGasEoS, PrimitiveState};

/// 五分量无粘通量（与 `InviscidFlux` 布局一致）。
#[derive(Debug, Clone, Copy, Default)]
pub struct InviscidFlux5 {
    pub mass: Real,
    pub momentum: [Real; 3],
    pub energy: Real,
}

/// 四路一阶 Roe 面通量。
#[allow(clippy::too_many_arguments)]
pub fn face_inviscid_flux_first_order_roe_batch4(
    left: [&PrimitiveState; 4],
    right: [&PrimitiveState; 4],
    left_cons: [&ConservedState; 4],
    right_cons: [&ConservedState; 4],
    normal: [Vector3; 4],
    eos: &IdealGasEoS,
    entropy_fix: bool,
) -> [Option<InviscidFlux5>; 4] {
    let mut out = [None; 4];
    for i in 0..4 {
        out[i] = roe_lane(
            left[i],
            right[i],
            left_cons[i],
            right_cons[i],
            normal[i],
            eos,
            entropy_fix,
        );
    }
    out
}

fn roe_lane(
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    left: &ConservedState,
    right: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
    entropy_fix: bool,
) -> Option<InviscidFlux5> {
    let n = normalize(normal)?;
    let flux_l = physical_flux(left, prim_l, n);
    let flux_r = physical_flux(right, prim_r, n);
    let roe = roe_averages(prim_l, prim_r, left, right, eos.gamma, n)?;
    let (t1, t2) = face_tangent_basis(n);
    let (a1, a2, a3, a4, a5) = wave_strengths(prim_l, prim_r, &roe, n, t1, t2)?;
    let delta = 0.2_f64.mul_add(roe.un.abs(), roe.a).max(Real::EPSILON);
    let l1 = fixed_lambda(roe.un - roe.a, delta, entropy_fix);
    let l5 = fixed_lambda(roe.un + roe.a, delta, entropy_fix);
    let l_mid = roe.un.abs();
    let waves = WaveLane { a1, a2, a3, a4, a5 };
    let diss = dissipation(&roe, &waves, l1, l_mid, l5, n, t1, t2);
    Some(combine(flux_l, flux_r, diss))
}

struct RoeLane {
    rho: Real,
    velocity: [Real; 3],
    enthalpy: Real,
    un: Real,
    a: Real,
}

struct WaveLane {
    a1: Real,
    a2: Real,
    a3: Real,
    a4: Real,
    a5: Real,
}

fn normalize(n: Vector3) -> Option<Vector3> {
    let mag = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
    if mag < Real::EPSILON {
        return None;
    }
    Some(Vector3::new(n.x / mag, n.y / mag, n.z / mag))
}

fn physical_flux(cons: &ConservedState, prim: &PrimitiveState, n: Vector3) -> InviscidFlux5 {
    let un = prim.velocity[0] * n.x + prim.velocity[1] * n.y + prim.velocity[2] * n.z;
    let p = prim.pressure;
    let rho = prim.density;
    let u = prim.velocity;
    InviscidFlux5 {
        mass: rho * un,
        momentum: [
            rho * un * u[0] + p * n.x,
            rho * un * u[1] + p * n.y,
            rho * un * u[2] + p * n.z,
        ],
        energy: (cons.total_energy + p) * un,
    }
}

fn roe_averages(
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    left: &ConservedState,
    right: &ConservedState,
    gamma: Real,
    normal: Vector3,
) -> Option<RoeLane> {
    let sqrt_dl = prim_l.density.sqrt();
    let sqrt_dr = prim_r.density.sqrt();
    let inv = 1.0 / (sqrt_dl + sqrt_dr);
    let h_l = (left.total_energy + prim_l.pressure) / prim_l.density;
    let h_r = (right.total_energy + prim_r.pressure) / prim_r.density;
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
    let un = velocity[0] * normal.x + velocity[1] * normal.y + velocity[2] * normal.z;
    Some(RoeLane {
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

fn wave_strengths(
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    roe: &RoeLane,
    normal: Vector3,
    t1: Vector3,
    t2: Vector3,
) -> Option<(Real, Real, Real, Real, Real)> {
    let un_l = dot_v(prim_l.velocity, normal);
    let un_r = dot_v(prim_r.velocity, normal);
    let ut1_l = dot_v(prim_l.velocity, t1);
    let ut1_r = dot_v(prim_r.velocity, t1);
    let ut2_l = dot_v(prim_l.velocity, t2);
    let ut2_r = dot_v(prim_r.velocity, t2);
    let dp = prim_r.pressure - prim_l.pressure;
    let drho = prim_r.density - prim_l.density;
    let dun = un_r - un_l;
    let a2 = roe.a * roe.a;
    if a2 < Real::EPSILON {
        return None;
    }
    Some((
        0.5 * (dp - roe.rho * roe.a * dun) / a2,
        drho - dp / a2,
        roe.rho * (ut1_r - ut1_l),
        roe.rho * (ut2_r - ut2_l),
        0.5 * (dp + roe.rho * roe.a * dun) / a2,
    ))
}

fn dot_v(v: [Real; 3], n: Vector3) -> Real {
    v[0] * n.x + v[1] * n.y + v[2] * n.z
}

fn fixed_lambda(lambda: Real, delta: Real, fix: bool) -> Real {
    if !fix {
        return lambda.abs();
    }
    let abs_l = lambda.abs();
    if abs_l >= delta {
        abs_l
    } else {
        (lambda * lambda + delta * delta) / (2.0 * delta)
    }
}

struct Dissipation5 {
    mass: Real,
    momentum: [Real; 3],
    energy: Real,
}

#[allow(clippy::too_many_arguments)]
fn dissipation(
    roe: &RoeLane,
    waves: &WaveLane,
    l1: Real,
    l_mid: Real,
    l5: Real,
    normal: Vector3,
    t1: Vector3,
    t2: Vector3,
) -> Dissipation5 {
    let u = roe.velocity;
    let h = roe.enthalpy;
    let a = roe.a;
    let un = roe.un;
    let n = [normal.x, normal.y, normal.z];
    let t1v = [t1.x, t1.y, t1.z];
    let t2v = [t2.x, t2.y, t2.z];
    let mut diss = scale_eig(acoustic(u, h, a, un, n, -1.0), l1 * waves.a1);
    add_eig(&mut diss, &acoustic(u, h, a, un, n, 1.0), l5 * waves.a5);
    add_eig(&mut diss, &contact(u), l_mid * waves.a2);
    add_eig(&mut diss, &shear(t1v, dot_v(u, t1)), l_mid * waves.a3);
    add_eig(&mut diss, &shear(t2v, dot_v(u, t2)), l_mid * waves.a4);
    diss
}

fn acoustic(u: [Real; 3], h: Real, a: Real, un: Real, n: [Real; 3], sign: Real) -> Dissipation5 {
    Dissipation5 {
        mass: 1.0,
        momentum: [
            u[0] + sign * a * n[0],
            u[1] + sign * a * n[1],
            u[2] + sign * a * n[2],
        ],
        energy: h + sign * a * un,
    }
}

fn contact(u: [Real; 3]) -> Dissipation5 {
    let vel2 = 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    Dissipation5 {
        mass: 1.0,
        momentum: u,
        energy: vel2,
    }
}

fn shear(tangent: [Real; 3], ut: Real) -> Dissipation5 {
    Dissipation5 {
        mass: 0.0,
        momentum: tangent,
        energy: ut,
    }
}

fn scale_eig(v: Dissipation5, scale: Real) -> Dissipation5 {
    Dissipation5 {
        mass: scale * v.mass,
        momentum: [
            scale * v.momentum[0],
            scale * v.momentum[1],
            scale * v.momentum[2],
        ],
        energy: scale * v.energy,
    }
}

fn add_eig(target: &mut Dissipation5, v: &Dissipation5, scale: Real) {
    target.mass += scale * v.mass;
    target.momentum[0] += scale * v.momentum[0];
    target.momentum[1] += scale * v.momentum[1];
    target.momentum[2] += scale * v.momentum[2];
    target.energy += scale * v.energy;
}

fn combine(flux_l: InviscidFlux5, flux_r: InviscidFlux5, diss: Dissipation5) -> InviscidFlux5 {
    let half = 0.5;
    InviscidFlux5 {
        mass: half * (flux_l.mass + flux_r.mass) - half * diss.mass,
        momentum: [
            half * (flux_l.momentum[0] + flux_r.momentum[0]) - half * diss.momentum[0],
            half * (flux_l.momentum[1] + flux_r.momentum[1]) - half * diss.momentum[1],
            half * (flux_l.momentum[2] + flux_r.momentum[2]) - half * diss.momentum[2],
        ],
        energy: half * (flux_l.energy + flux_r.energy) - half * diss.energy,
    }
}
