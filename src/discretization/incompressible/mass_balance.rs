//! 不可压缩边界质量通量守恒诊断（I4 V&V）。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::discretization::incompressible::face_boundary::incompressible_boundary_mass_flux_3d;
use crate::error::Result;
use crate::field::IncompressibleFields;
use crate::mesh::{BoundaryMesh, BoundaryMesh3d, StructuredMesh3d};

/// 边界 face 质量通量汇总（\(\dot m = \rho\,\mathbf{u}\cdot\mathbf{S}\)，不可压 \(\rho=1\)）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressibleBoundaryMassBalance {
    /// 非周期边界 face 通量代数和（稳态守恒时应 \(\approx 0\)）。
    pub net_flux: Real,
    /// 速度入口 patch 通量代数和（进入域为负，见 face_flux 单测）。
    pub inlet_flux: Real,
    /// 入口流入量 \(\sum_{\mathrm{inlet}} \max(-\dot m, 0)\)。
    pub inlet_magnitude: Real,
    /// \(|\dot m_{\mathrm{net}}| / \max(\dot m_{\mathrm{in,mag}}, \varepsilon)\)（ADR 0015 I4）。
    pub imbalance_ratio: Real,
}

/// 由修正后速度场计算边界质量守恒指标。
pub fn compute_incompressible_boundary_mass_balance_3d(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    boundary: &BoundarySet,
) -> Result<IncompressibleBoundaryMassBalance> {
    fields.validate_len(mesh.num_cells())?;
    let mut net_flux = 0.0;
    let mut inlet_flux = 0.0;
    for patch in boundary.patches() {
        if matches!(patch.kind, BoundaryKind::Periodic { .. }) {
            continue;
        }
        let mut patch_flux = 0.0;
        for &face in &patch.face_ids {
            let owner = mesh.face_owner(face)?.index() as usize;
            let geom = mesh.face_geometry_3d(face)?;
            patch_flux += incompressible_boundary_mass_flux_3d(
                mesh,
                owner,
                &patch.kind,
                fields,
                geom.normal,
                geom.area,
            );
        }
        net_flux += patch_flux;
        if matches!(
            patch.kind,
            BoundaryKind::IncompressibleVelocityInlet { .. } | BoundaryKind::Inlet { .. }
        ) {
            inlet_flux += patch_flux;
        }
    }
    let inlet_magnitude = if inlet_flux < 0.0 { -inlet_flux } else { 0.0 };
    let scale = inlet_magnitude.max(Real::EPSILON);
    Ok(IncompressibleBoundaryMassBalance {
        net_flux,
        inlet_flux,
        inlet_magnitude,
        imbalance_ratio: net_flux.abs() / scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundaryPatch;
    use crate::core::approx_eq;
    use crate::mesh::BoundaryMesh;

    #[test]
    fn uniform_channel_inlet_outlet_balances() {
        let mesh = StructuredMesh3d::uniform_box("channel", 4, 2, 1, 4.0, 1.0, 0.1).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [1.0, 0.0, 0.0]).expect("fields");
        let boundary = BoundarySet::new(vec![
            BoundaryPatch::new(
                "i_min",
                mesh.resolve_logical_boundary("i_min").expect("inlet"),
                BoundaryKind::IncompressibleVelocityInlet {
                    velocity: [1.0, 0.0, 0.0],
                },
            ),
            BoundaryPatch::new(
                "i_max",
                mesh.resolve_logical_boundary("i_max").expect("outlet"),
                BoundaryKind::IncompressiblePressureOutlet { pressure: 0.0 },
            ),
        ]);
        let balance = compute_incompressible_boundary_mass_balance_3d(&mesh, &fields, &boundary)
            .expect("balance");
        assert!(balance.inlet_magnitude > 0.0);
        assert!(approx_eq(balance.imbalance_ratio, 0.0, 1.0e-12));
    }
}
