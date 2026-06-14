//! IDWLS f32：RHS 累加与对称 3×3 求解（几何矩阵仍 f64）。

use crate::core::{Real, Vector3};

/// `rhs += w * delta * dr`（f32 RHS，f64 几何）。
#[inline]
pub fn accumulate_lsq_rhs_component_f32(rhs: &mut [f32; 3], dr: Vector3, w: Real, delta: f32) {
    if w <= 0.0 {
        return;
    }
    let coeff = w as f32 * delta;
    rhs[0] += coeff * dr.x as f32;
    rhs[1] += coeff * dr.y as f32;
    rhs[2] += coeff * dr.z as f32;
}

/// 求解 \(A x = b\)（A 从 f64 几何 cast 到 f32）。
#[must_use]
pub fn solve_symmetric_3x3_f32(a: &super::Symmetric3x3, rhs: [f32; 3]) -> Option<[f32; 3]> {
    let a_xx = a.a_xx as f32;
    let a_xy = a.a_xy as f32;
    let a_xz = a.a_xz as f32;
    let a_yy = a.a_yy as f32;
    let a_yz = a.a_yz as f32;
    let a_zz = a.a_zz as f32;

    let c_xx = a_yy * a_zz - a_yz * a_yz;
    let c_xy = a_xz * a_yz - a_xy * a_zz;
    let c_xz = a_xy * a_yz - a_xz * a_yy;
    let c_yy = a_xx * a_zz - a_xz * a_xz;
    let c_yz = a_xy * a_xz - a_xx * a_yz;
    let c_zz = a_xx * a_yy - a_xy * a_xy;
    let det = a_xx * c_xx + a_xy * c_xy + a_xz * c_xz;
    if det.abs() <= f32::EPSILON {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (c_xx * rhs[0] + c_xy * rhs[1] + c_xz * rhs[2]) * inv_det,
        (c_xy * rhs[0] + c_yy * rhs[1] + c_yz * rhs[2]) * inv_det,
        (c_xz * rhs[0] + c_yz * rhs[1] + c_zz * rhs[2]) * inv_det,
    ])
}
