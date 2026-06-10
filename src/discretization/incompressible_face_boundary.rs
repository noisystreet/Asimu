//! 不可压缩边界 face 状态刷新。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::field::IncompressibleFields;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompressibleMassFluxBoundaryKind {
    NoPenetration,
    PrescribedVelocity,
    OwnerExtrapolated,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressibleBoundaryFaceState {
    pub velocity: [Real; 3],
    pub pressure: Option<Real>,
    pub pressure_correction_dirichlet: bool,
    pub mass_flux_kind: IncompressibleMassFluxBoundaryKind,
}

/// 返回不可压缩边界 face 的速度状态。
///
/// `owner` 是边界面的 owner 单元索引。墙面和对称面使用无穿透面速度；
/// 动壁与速度入口使用给定 face 速度；压力出口使用 owner 零梯度外推。
#[must_use]
pub fn incompressible_boundary_face_velocity(
    owner: usize,
    kind: &BoundaryKind,
    fields: &IncompressibleFields,
) -> [Real; 3] {
    incompressible_boundary_face_state(owner, kind, fields).velocity
}

#[must_use]
pub fn incompressible_boundary_face_state(
    owner: usize,
    kind: &BoundaryKind,
    fields: &IncompressibleFields,
) -> IncompressibleBoundaryFaceState {
    let velocity = match kind {
        BoundaryKind::Wall { .. } | BoundaryKind::Symmetry => [0.0, 0.0, 0.0],
        BoundaryKind::MovingWall { velocity } => *velocity,
        BoundaryKind::IncompressibleVelocityInlet { velocity } => *velocity,
        BoundaryKind::IncompressiblePressureOutlet { .. } | BoundaryKind::Outlet { .. } => {
            owner_velocity(fields, owner)
        }
        _ => owner_velocity(fields, owner),
    };
    let pressure = match kind {
        BoundaryKind::IncompressiblePressureOutlet { pressure } => Some(*pressure),
        BoundaryKind::Outlet {
            static_pressure, ..
        } => Some(*static_pressure),
        _ => None,
    };
    let mass_flux_kind = match kind {
        BoundaryKind::Wall { .. } | BoundaryKind::Symmetry | BoundaryKind::MovingWall { .. } => {
            IncompressibleMassFluxBoundaryKind::NoPenetration
        }
        BoundaryKind::IncompressibleVelocityInlet { .. } | BoundaryKind::Inlet { .. } => {
            IncompressibleMassFluxBoundaryKind::PrescribedVelocity
        }
        _ => IncompressibleMassFluxBoundaryKind::OwnerExtrapolated,
    };
    IncompressibleBoundaryFaceState {
        velocity,
        pressure,
        pressure_correction_dirichlet: incompressible_pressure_correction_dirichlet(kind),
        mass_flux_kind,
    }
}

#[must_use]
pub fn incompressible_pressure_correction_dirichlet(kind: &BoundaryKind) -> bool {
    matches!(
        kind,
        BoundaryKind::Wall { .. }
            | BoundaryKind::MovingWall { .. }
            | BoundaryKind::IncompressibleVelocityInlet { .. }
            | BoundaryKind::IncompressiblePressureOutlet { .. }
            | BoundaryKind::Outlet { .. }
            | BoundaryKind::Inlet { .. }
    )
}

/// `BoundarySet` 是否包含成对的 i 向周期边界。
#[must_use]
pub fn has_periodic_x(boundary: &BoundarySet) -> bool {
    boundary.has_periodic_pair("i_min", "i_max")
}

fn owner_velocity(fields: &IncompressibleFields, cell: usize) -> [Real; 3] {
    [
        fields.velocity_x.values()[cell],
        fields.velocity_y.values()[cell],
        fields.velocity_z.values()[cell],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::WallHeat;

    #[test]
    fn wall_face_velocity_is_no_penetration() {
        let fields = IncompressibleFields::uniform(1, 0.0, [1.0, 2.0, 3.0]).expect("fields");

        let state = incompressible_boundary_face_state(
            0,
            &BoundaryKind::Wall {
                no_slip: true,
                heat: WallHeat::Adiabatic,
            },
            &fields,
        );

        assert_eq!(state.velocity, [0.0, 0.0, 0.0]);
        assert!(state.pressure_correction_dirichlet);
        assert_eq!(
            state.mass_flux_kind,
            IncompressibleMassFluxBoundaryKind::NoPenetration
        );
    }

    #[test]
    fn pressure_outlet_uses_owner_velocity() {
        let fields = IncompressibleFields::uniform(1, 0.0, [1.0, 2.0, 3.0]).expect("fields");

        let state = incompressible_boundary_face_state(
            0,
            &BoundaryKind::IncompressiblePressureOutlet { pressure: 0.0 },
            &fields,
        );

        assert_eq!(state.velocity, [1.0, 2.0, 3.0]);
        assert_eq!(state.pressure, Some(0.0));
    }
}
