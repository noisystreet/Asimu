//! IDWLS：RHS 三分量累加与对称 3×3 求解。

use crate::core::{Real, Vector3};

/// 对称 3×3 矩阵（与 `LsqPrecomputedCell` 布局一致）。
#[derive(Debug, Clone, Copy, Default)]
pub struct Symmetric3x3 {
    pub a_xx: Real,
    pub a_xy: Real,
    pub a_xz: Real,
    pub a_yy: Real,
    pub a_yz: Real,
    pub a_zz: Real,
}

/// `rhs += w * delta * dr`。
#[inline]
pub fn accumulate_lsq_rhs_component(rhs: &mut Vector3, dr: Vector3, w: Real, delta: Real) {
    if w <= 0.0 {
        return;
    }
    let coeff = w * delta;
    rhs.x += coeff * dr.x;
    rhs.y += coeff * dr.y;
    rhs.z += coeff * dr.z;
}

/// 求解 \(A x = b\)。
#[must_use]
pub fn solve_symmetric_3x3(a: &Symmetric3x3, rhs: Vector3) -> Option<Vector3> {
    let c_xx = a.a_yy * a.a_zz - a.a_yz * a.a_yz;
    let c_xy = a.a_xz * a.a_yz - a.a_xy * a.a_zz;
    let c_xz = a.a_xy * a.a_yz - a.a_xz * a.a_yy;
    let c_yy = a.a_xx * a.a_zz - a.a_xz * a.a_xz;
    let c_yz = a.a_xy * a.a_xz - a.a_xx * a.a_yz;
    let c_zz = a.a_xx * a.a_yy - a.a_xy * a.a_xy;
    let det = a.a_xx * c_xx + a.a_xy * c_xy + a.a_xz * c_xz;
    if det.abs() <= Real::EPSILON {
        return None;
    }
    let inv_det = 1.0 / det;
    Some(Vector3::new(
        (c_xx * rhs.x + c_xy * rhs.y + c_xz * rhs.z) * inv_det,
        (c_xy * rhs.x + c_yy * rhs.y + c_yz * rhs.z) * inv_det,
        (c_xz * rhs.x + c_yz * rhs.y + c_zz * rhs.z) * inv_det,
    ))
}

/// 四单元一批求解（`simd-fvm` 下用 `f64x4` 并行算子式；退化单元回退标量）。
pub fn solve_symmetric_3x3_batch4(
    a: [&Symmetric3x3; 4],
    rhs: [Vector3; 4],
) -> [Option<Vector3>; 4] {
    #[cfg(feature = "simd-fvm")]
    {
        return solve_symmetric_3x3_batch4_simd(a, rhs);
    }
    #[cfg(not(feature = "simd-fvm"))]
    {
        [
            solve_symmetric_3x3(a[0], rhs[0]),
            solve_symmetric_3x3(a[1], rhs[1]),
            solve_symmetric_3x3(a[2], rhs[2]),
            solve_symmetric_3x3(a[3], rhs[3]),
        ]
    }
}

#[cfg(feature = "simd-fvm")]
fn solve_symmetric_3x3_batch4_simd(
    a: [&Symmetric3x3; 4],
    rhs: [Vector3; 4],
) -> [Option<Vector3>; 4] {
    use wide::f64x4;

    let lane = |i: usize| -> f64x4 {
        f64x4::new([a[0].lane(i), a[1].lane(i), a[2].lane(i), a[3].lane(i)])
    };
    let a_xx = lane(0);
    let a_xy = lane(1);
    let a_xz = lane(2);
    let a_yy = lane(3);
    let a_yz = lane(4);
    let a_zz = lane(5);

    let c_xx = a_yy * a_zz - a_yz * a_yz;
    let c_xy = a_xz * a_yz - a_xy * a_zz;
    let c_xz = a_xy * a_yz - a_xz * a_yy;
    let c_yy = a_xx * a_zz - a_xz * a_xz;
    let c_yz = a_xy * a_xz - a_xx * a_yz;
    let c_zz = a_xx * a_yy - a_xy * a_xy;
    let det = a_xx * c_xx + a_xy * c_xy + a_xz * c_xz;

    let bx = f64x4::new([rhs[0].x, rhs[1].x, rhs[2].x, rhs[3].x]);
    let by = f64x4::new([rhs[0].y, rhs[1].y, rhs[2].y, rhs[3].y]);
    let bz = f64x4::new([rhs[0].z, rhs[1].z, rhs[2].z, rhs[3].z]);

    let inv_det = f64x4::from(1.0) / det;
    let x = (c_xx * bx + c_xy * by + c_xz * bz) * inv_det;
    let y = (c_xy * bx + c_yy * by + c_yz * bz) * inv_det;
    let z = (c_xz * bx + c_yz * by + c_zz * bz) * inv_det;

    let mut out = [None; 4];
    let eps = Real::EPSILON;
    for i in 0..4 {
        let d = det.to_array()[i];
        if d.abs() <= eps {
            out[i] = solve_symmetric_3x3(a[i], rhs[i]);
        } else {
            out[i] = Some(Vector3::new(
                x.to_array()[i],
                y.to_array()[i],
                z.to_array()[i],
            ));
        }
    }
    out
}

#[cfg(feature = "simd-fvm")]
impl Symmetric3x3 {
    fn lane(self, idx: usize) -> Real {
        match idx {
            0 => self.a_xx,
            1 => self.a_xy,
            2 => self.a_xz,
            3 => self.a_yy,
            4 => self.a_yz,
            5 => self.a_zz,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    fn sample_matrix() -> Symmetric3x3 {
        Symmetric3x3 {
            a_xx: 4.0,
            a_xy: 1.0,
            a_xz: 0.0,
            a_yy: 3.0,
            a_yz: 0.0,
            a_zz: 2.0,
        }
    }

    #[test]
    fn batch4_matches_scalar() {
        let a = sample_matrix();
        let mats = [&a, &a, &a, &a];
        let rhs = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(0.0, 0.0, 3.0),
            Vector3::new(1.0, 1.0, 1.0),
        ];
        let batch = solve_symmetric_3x3_batch4(mats, rhs);
        for i in 0..4 {
            let scalar = solve_symmetric_3x3(&a, rhs[i]).expect("solve");
            let b = batch[i].expect("batch");
            assert!(approx_eq(b.x, scalar.x, 1.0e-11));
            assert!(approx_eq(b.y, scalar.y, 1.0e-11));
            assert!(approx_eq(b.z, scalar.z, 1.0e-11));
        }
    }
}
