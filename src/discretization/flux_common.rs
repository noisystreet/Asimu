//! 无粘通量公共几何/限制器工具。

use crate::core::{Real, Vector3};
use crate::error::{AsimuError, Result};

use super::flux_config::SlopeLimiter;

pub(crate) fn normalize_face_normal(normal: Vector3) -> Result<Vector3> {
    let mag = (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z).sqrt();
    if mag < Real::EPSILON {
        return Err(AsimuError::Mesh("面法向不能为零向量".to_string()));
    }
    Ok(Vector3::new(normal.x / mag, normal.y / mag, normal.z / mag))
}

pub(crate) fn face_tangent_basis(normal: Vector3) -> (Vector3, Vector3) {
    let reference = if normal.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let t1 = cross(normal, reference);
    let t1 = normalize_unchecked(t1);
    let t2 = cross(normal, t1);
    (t1, normalize_unchecked(t2))
}

pub(crate) fn cross(a: Vector3, b: Vector3) -> Vector3 {
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

pub(crate) fn limited_slope(d_minus: Real, d_plus: Real, limiter: SlopeLimiter) -> Real {
    if d_minus * d_plus <= 0.0 {
        return 0.0;
    }
    match limiter {
        SlopeLimiter::Minmod => {
            if d_minus.abs() < d_plus.abs() {
                d_minus
            } else {
                d_plus
            }
        }
        SlopeLimiter::VanLeer => {
            if d_plus.abs() < Real::EPSILON {
                return 0.0;
            }
            let r = d_minus / d_plus;
            0.5 * d_plus * (r + r.abs()) / (1.0 + r.abs())
        }
        SlopeLimiter::VanAlbada => {
            if d_plus.abs() < Real::EPSILON {
                return 0.0;
            }
            let r = d_minus / d_plus;
            d_plus * (r * r + r) / (r * r + 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    #[test]
    fn van_albada_is_second_order_on_uniform_slope() {
        let slope = limited_slope(2.0, 2.0, SlopeLimiter::VanAlbada);
        assert!(approx_eq(slope, 2.0, 1.0e-12));
    }

    #[test]
    fn van_albada_zeros_out_downwind_increase() {
        assert!(approx_eq(
            limited_slope(1.0, -1.0, SlopeLimiter::VanAlbada),
            0.0,
            1.0e-12
        ));
    }

    #[test]
    fn van_albada_is_less_diffusive_than_minmod_on_smooth_data() {
        let minmod = limited_slope(1.0, 3.0, SlopeLimiter::Minmod);
        let van_albada = limited_slope(1.0, 3.0, SlopeLimiter::VanAlbada);
        assert!(van_albada.abs() > minmod.abs());
    }
}
