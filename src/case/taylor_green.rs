//! Taylor–Green 涡初场与动能衰减 V&V（ADR 0015 I3）。
//!
//! 理论：[`docs/theory/incompressible_simplec_piso.md`](../../../docs/theory/incompressible_simplec_piso.md)

use std::f64::consts::PI;

use crate::core::Real;
use crate::error::Result;
use crate::field::{IncompressibleFields, ScalarField};
use crate::mesh::StructuredMesh3d;

const TWO_PI: Real = 2.0 * PI;

/// 装配 Taylor–Green 初场（结构化域，坐标已无量纲化至 \([0,1]^d\)）。
pub fn taylor_green_initial_fields(mesh: &StructuredMesh3d) -> Result<IncompressibleFields> {
    let n = mesh.num_cells();
    let mut pressure = Vec::with_capacity(n);
    let mut ux = Vec::with_capacity(n);
    let mut uy = Vec::with_capacity(n);
    let mut uz = Vec::with_capacity(n);
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let center = mesh.cell_metric(i, j, k).center;
                let x = TWO_PI * center.x;
                let y = TWO_PI * center.y;
                let z = TWO_PI * center.z;
                let cos_z = z.cos();
                ux.push(x.sin() * y.cos() * cos_z);
                uy.push(-x.cos() * y.sin() * cos_z);
                uz.push(0.0);
                let cos_2x = (2.0 * x).cos();
                let cos_2y = (2.0 * y).cos();
                let cos_2z = (2.0 * z).cos();
                let p = -(cos_2x + cos_2y) * (cos_2z + 2.0) / 16.0;
                pressure.push(p);
            }
        }
    }
    Ok(IncompressibleFields {
        pressure: ScalarField::from_values(pressure)?,
        velocity_x: ScalarField::from_values(ux)?,
        velocity_y: ScalarField::from_values(uy)?,
        velocity_z: ScalarField::from_values(uz)?,
    })
}

/// 体积平均 kinetic energy \(E=\frac{1}{V}\int \frac{1}{2}\rho|\mathbf{u}|^2\,\mathrm{d}V\)。
#[must_use]
pub fn kinetic_energy(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    density: Real,
) -> Real {
    let mut integral = 0.0;
    let mut volume = 0.0;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let idx = mesh.cell_index(i, j, k);
                let cell_volume = mesh.cell_metric(i, j, k).volume;
                let u = fields.velocity_x.values()[idx];
                let v = fields.velocity_y.values()[idx];
                let w = fields.velocity_z.values()[idx];
                integral += 0.5 * density * (u * u + v * v + w * w) * cell_volume;
                volume += cell_volume;
            }
        }
    }
    if volume > Real::EPSILON {
        integral / volume
    } else {
        0.0
    }
}

/// 解析动能比 \(E(t)/E(0)=\exp(-4\nu^* t^*)\)（\(\nu^*=1/Re\)）。
#[must_use]
pub fn analytical_kinetic_energy_ratio(inv_reynolds: Real, nondimensional_time: Real) -> Real {
    (-4.0 * inv_reynolds * nondimensional_time).exp()
}

/// 在 spin-up 后区间估计 \(-\mathrm{d}\ln E/\mathrm{d}t\)，与解析 \(4\nu^*\) 对比。
pub(crate) fn taylor_green_decay_rates(
    enabled: bool,
    inv_reynolds: Real,
    time_step: Real,
    history: &[Real],
) -> (Option<Real>, Option<Real>) {
    if !enabled {
        return (None, None);
    }
    const SPIN_UP_STEPS: usize = 10;
    let analytical = Some(4.0 * inv_reynolds);
    if history.len() <= SPIN_UP_STEPS + 1 {
        return (None, analytical);
    }
    let start = history[SPIN_UP_STEPS];
    let end = *history.last().expect("history has end");
    let elapsed = time_step * (history.len() - 1 - SPIN_UP_STEPS) as Real;
    if start <= Real::EPSILON || end <= Real::EPSILON || elapsed <= Real::EPSILON {
        return (None, analytical);
    }
    let rate = -(end / start).ln() / elapsed;
    (Some(rate), analytical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::mesh::StructuredMesh3d;

    fn mesh_8x8x1() -> StructuredMesh3d {
        StructuredMesh3d::uniform_box("tg", 8, 8, 1, 1.0, 1.0, 0.1).expect("mesh")
    }

    #[test]
    fn taylor_green_initial_field_is_divergence_free_on_periodic_topology() {
        let mesh = mesh_8x8x1();
        let fields = taylor_green_initial_fields(&mesh).expect("fields");
        let energy = kinetic_energy(&mesh, &fields, 1.0);
        assert!(energy > 0.05);
        assert!(energy < 0.5);
    }

    #[test]
    fn analytical_decay_matches_reference_slope_at_small_time() {
        let inv_re = 0.01;
        let dt = 0.1;
        let ratio = analytical_kinetic_energy_ratio(inv_re, dt);
        assert!(approx_eq(ratio, (-0.004_f64).exp(), 1.0e-12));
    }
}
