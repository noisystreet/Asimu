//! 不可压缩边界 face 状态刷新。

use super::boundary_flux::interior_face_velocity;
use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::core::Vector3;
use crate::field::IncompressibleFields;
use crate::mesh::StructuredMesh3d;

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

/// 将速度投影到面切平面（去除法向分量）。
#[must_use]
pub(crate) fn tangential_velocity(velocity: [Real; 3], normal: [Real; 3]) -> [Real; 3] {
    let un = velocity[0] * normal[0] + velocity[1] * normal[1] + velocity[2] * normal[2];
    [
        velocity[0] - un * normal[0],
        velocity[1] - un * normal[1],
        velocity[2] - un * normal[2],
    ]
}

#[must_use]
pub(crate) fn cell_velocity(fields: &IncompressibleFields, cell: usize) -> [Real; 3] {
    [
        fields.velocity_x.values()[cell],
        fields.velocity_y.values()[cell],
        fields.velocity_z.values()[cell],
    ]
}

/// 边界质量通量（owner 单元净入通量，已含面积）。
#[must_use]
pub(crate) fn incompressible_boundary_mass_flux(
    owner: usize,
    kind: &BoundaryKind,
    fields: &IncompressibleFields,
    normal: Vector3,
    area: Real,
) -> Real {
    let state = incompressible_boundary_face_state(owner, kind, fields);
    boundary_mass_flux_from_state(state, normal, area)
}

/// 结构化网格边界质量通量；压力出口用相邻内部面速度（零梯度外推）。
#[must_use]
pub(crate) fn incompressible_boundary_mass_flux_3d(
    mesh: &StructuredMesh3d,
    owner: usize,
    kind: &BoundaryKind,
    fields: &IncompressibleFields,
    normal: Vector3,
    area: Real,
) -> Real {
    if matches!(
        kind,
        BoundaryKind::IncompressiblePressureOutlet { .. } | BoundaryKind::Outlet { .. }
    ) && let Some(flux) = structured_pressure_outlet_mass_flux(mesh, owner, fields, normal, area)
    {
        return flux;
    }
    incompressible_boundary_mass_flux(owner, kind, fields, normal, area)
}

fn boundary_mass_flux_from_state(
    state: IncompressibleBoundaryFaceState,
    normal: Vector3,
    area: Real,
) -> Real {
    match state.mass_flux_kind {
        IncompressibleMassFluxBoundaryKind::NoPenetration => 0.0,
        IncompressibleMassFluxBoundaryKind::PrescribedVelocity
        | IncompressibleMassFluxBoundaryKind::OwnerExtrapolated => {
            (state.velocity[0] * normal.x
                + state.velocity[1] * normal.y
                + state.velocity[2] * normal.z)
                * area
        }
    }
}

enum StructuredBoundaryAxis {
    XMax,
    XMin,
    YMax,
    YMin,
    ZMax,
    ZMin,
}

fn structured_boundary_axis(normal: Vector3) -> Option<StructuredBoundaryAxis> {
    if normal.x > 0.5 {
        Some(StructuredBoundaryAxis::XMax)
    } else if normal.x < -0.5 {
        Some(StructuredBoundaryAxis::XMin)
    } else if normal.y > 0.5 {
        Some(StructuredBoundaryAxis::YMax)
    } else if normal.y < -0.5 {
        Some(StructuredBoundaryAxis::YMin)
    } else if normal.z > 0.5 {
        Some(StructuredBoundaryAxis::ZMax)
    } else if normal.z < -0.5 {
        Some(StructuredBoundaryAxis::ZMin)
    } else {
        None
    }
}

fn structured_outlet_face_velocity(
    mesh: &StructuredMesh3d,
    owner: usize,
    fields: &IncompressibleFields,
    axis: StructuredBoundaryAxis,
) -> Option<[Real; 3]> {
    let (i, j, k) = structured_cell_ijk(mesh, owner);
    let (left, right) = match axis {
        StructuredBoundaryAxis::XMax if i + 1 == mesh.nx && i > 0 => {
            (mesh.cell_index(i - 1, j, k), owner)
        }
        StructuredBoundaryAxis::XMin if i == 0 && mesh.nx > 1 => {
            (owner, mesh.cell_index(i + 1, j, k))
        }
        StructuredBoundaryAxis::YMax if j + 1 == mesh.ny && j > 0 => {
            (mesh.cell_index(i, j - 1, k), owner)
        }
        StructuredBoundaryAxis::YMin if j == 0 && mesh.ny > 1 => {
            (owner, mesh.cell_index(i, j + 1, k))
        }
        StructuredBoundaryAxis::ZMax if k + 1 == mesh.nz && k > 0 => {
            (mesh.cell_index(i, j, k - 1), owner)
        }
        StructuredBoundaryAxis::ZMin if k == 0 && mesh.nz > 1 => {
            (owner, mesh.cell_index(i, j, k + 1))
        }
        _ => return None,
    };
    Some(face_velocity_between(fields, left, right))
}

fn structured_pressure_outlet_mass_flux(
    mesh: &StructuredMesh3d,
    owner: usize,
    fields: &IncompressibleFields,
    normal: Vector3,
    area: Real,
) -> Option<Real> {
    let axis = structured_boundary_axis(normal)?;
    let velocity = structured_outlet_face_velocity(mesh, owner, fields, axis)?;
    Some((velocity[0] * normal.x + velocity[1] * normal.y + velocity[2] * normal.z) * area)
}

fn face_velocity_between(fields: &IncompressibleFields, left: usize, right: usize) -> [Real; 3] {
    [
        interior_face_velocity(fields, left, right, 0),
        interior_face_velocity(fields, left, right, 1),
        interior_face_velocity(fields, left, right, 2),
    ]
}

fn structured_cell_ijk(mesh: &StructuredMesh3d, cell: usize) -> (usize, usize, usize) {
    let cells_per_layer = mesh.nx * mesh.ny;
    let k = cell / cells_per_layer;
    let rem = cell % cells_per_layer;
    let j = rem / mesh.nx;
    let i = rem % mesh.nx;
    (i, j, k)
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
        BoundaryKind::IncompressiblePressureOutlet { .. } | BoundaryKind::Outlet { .. }
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
    use crate::core::approx_eq;
    use crate::field::ScalarField;
    use crate::mesh::{BoundaryMesh, BoundaryMesh3d};

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
        assert!(!state.pressure_correction_dirichlet);
        assert_eq!(
            state.mass_flux_kind,
            IncompressibleMassFluxBoundaryKind::NoPenetration
        );
    }

    #[test]
    fn velocity_inlet_is_not_pressure_correction_dirichlet() {
        assert!(!incompressible_pressure_correction_dirichlet(
            &BoundaryKind::IncompressibleVelocityInlet {
                velocity: [1.0, 0.0, 0.0],
            }
        ));
    }

    #[test]
    fn pressure_outlet_is_pressure_correction_dirichlet() {
        assert!(incompressible_pressure_correction_dirichlet(
            &BoundaryKind::IncompressiblePressureOutlet { pressure: 0.0 }
        ));
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

    #[test]
    fn structured_outlet_uses_internal_face_velocity_for_mass_flux() {
        let mesh = StructuredMesh3d::uniform_box("channel", 4, 2, 1, 4.0, 1.0, 0.1).expect("mesh");
        let mut values_x = vec![0.0; mesh.num_cells()];
        values_x[mesh.cell_index(2, 0, 0)] = 1.0;
        values_x[mesh.cell_index(2, 1, 0)] = 1.0;
        values_x[mesh.cell_index(3, 0, 0)] = 0.2;
        values_x[mesh.cell_index(3, 1, 0)] = 0.2;
        let fields = IncompressibleFields {
            pressure: ScalarField::uniform(mesh.num_cells(), 0.0).expect("p"),
            velocity_x: ScalarField::from_values(values_x).expect("u"),
            velocity_y: ScalarField::uniform(mesh.num_cells(), 0.0).expect("v"),
            velocity_z: ScalarField::uniform(mesh.num_cells(), 0.0).expect("w"),
        };
        let owner = mesh.cell_index(3, 0, 0);
        let geom = mesh
            .face_geometry_3d(mesh.resolve_logical_boundary("i_max").expect("outlet")[0])
            .expect("geom");
        let cell_flux = incompressible_boundary_mass_flux(
            owner,
            &BoundaryKind::IncompressiblePressureOutlet { pressure: 0.0 },
            &fields,
            geom.normal,
            geom.area,
        );
        let face_flux = incompressible_boundary_mass_flux_3d(
            &mesh,
            owner,
            &BoundaryKind::IncompressiblePressureOutlet { pressure: 0.0 },
            &fields,
            geom.normal,
            geom.area,
        );
        assert!(face_flux.abs() > cell_flux.abs());
        assert!(approx_eq(face_flux, 0.6 * geom.area, 1.0e-12));
    }
}
