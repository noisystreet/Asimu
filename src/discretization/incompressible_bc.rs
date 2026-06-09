//! 不可压缩 cell-centered 边界条件应用。
//!
//! I2 首版把边界条件施加到边界 owner 单元，供结构化 skeleton runner 使用；
//! 后续完整 FVM 会把同一语义下沉为面 ghost / 面通量。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::IncompressibleFields;
use crate::mesh::{BoundaryMesh, BoundaryMesh3d, StructuredMesh3d};

/// 对结构化 3D 不可压缩场施加边界 owner 单元约束。
pub fn apply_incompressible_boundary_conditions_3d(
    mesh: &StructuredMesh3d,
    fields: &mut IncompressibleFields,
    boundary: &BoundarySet,
) -> Result<IncompressibleBoundaryApplyStats> {
    fields.validate_len(mesh.num_cells())?;
    let mut stats = IncompressibleBoundaryApplyStats::default();
    for patch in boundary.patches() {
        for &face in &patch.face_ids {
            let owner = mesh.face_owner(face)?.index() as usize;
            let normal = mesh.face_normal_3d(face)?;
            match &patch.kind {
                BoundaryKind::Wall { no_slip, .. } => {
                    if *no_slip {
                        set_velocity(fields, owner, [0.0, 0.0, 0.0]);
                    } else {
                        zero_normal_velocity(fields, owner, [normal.x, normal.y, normal.z]);
                    }
                    stats.velocity_cells += 1;
                }
                BoundaryKind::MovingWall { velocity } => {
                    set_velocity(fields, owner, *velocity);
                    stats.velocity_cells += 1;
                }
                BoundaryKind::IncompressibleVelocityInlet { velocity } => {
                    set_velocity(fields, owner, *velocity);
                    stats.velocity_cells += 1;
                }
                BoundaryKind::IncompressiblePressureOutlet { pressure } => {
                    fields.pressure.values_mut()[owner] = *pressure;
                    stats.pressure_cells += 1;
                }
                BoundaryKind::Symmetry => {
                    zero_normal_velocity(fields, owner, [normal.x, normal.y, normal.z]);
                    stats.velocity_cells += 1;
                }
                BoundaryKind::Outlet {
                    static_pressure, ..
                } => {
                    fields.pressure.values_mut()[owner] = *static_pressure;
                    stats.pressure_cells += 1;
                }
                BoundaryKind::Inlet {
                    velocity_direction, ..
                } => {
                    set_velocity(fields, owner, normalized(*velocity_direction)?);
                    stats.velocity_cells += 1;
                }
                BoundaryKind::Dirichlet { .. }
                | BoundaryKind::Neumann { .. }
                | BoundaryKind::Farfield { .. }
                | BoundaryKind::Periodic { .. }
                | BoundaryKind::TurbulentInlet { .. } => {
                    stats.ignored_faces += 1;
                }
            }
        }
    }
    Ok(stats)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IncompressibleBoundaryApplyStats {
    pub velocity_cells: usize,
    pub pressure_cells: usize,
    pub ignored_faces: usize,
}

fn set_velocity(fields: &mut IncompressibleFields, cell: usize, velocity: [Real; 3]) {
    fields.velocity_x.values_mut()[cell] = velocity[0];
    fields.velocity_y.values_mut()[cell] = velocity[1];
    fields.velocity_z.values_mut()[cell] = velocity[2];
}

fn zero_normal_velocity(fields: &mut IncompressibleFields, cell: usize, normal: [Real; 3]) {
    let velocity = [
        fields.velocity_x.values()[cell],
        fields.velocity_y.values()[cell],
        fields.velocity_z.values()[cell],
    ];
    let un = velocity[0] * normal[0] + velocity[1] * normal[1] + velocity[2] * normal[2];
    set_velocity(
        fields,
        cell,
        [
            velocity[0] - un * normal[0],
            velocity[1] - un * normal[1],
            velocity[2] - un * normal[2],
        ],
    );
}

fn normalized(velocity: [Real; 3]) -> Result<[Real; 3]> {
    let mag =
        (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]).sqrt();
    if mag <= Real::EPSILON {
        return Err(AsimuError::Boundary(
            "不可压缩入口速度方向不能为零".to_string(),
        ));
    }
    Ok([velocity[0] / mag, velocity[1] / mag, velocity[2] / mag])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundaryPatch;

    #[test]
    fn velocity_inlet_sets_owner_cell_velocity() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 1, 1.0, 1.0, 1.0).expect("mesh");
        let mut fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "i_min",
            mesh.resolve_logical_boundary("i_min").expect("faces"),
            BoundaryKind::IncompressibleVelocityInlet {
                velocity: [2.0, 0.0, 0.0],
            },
        )]);

        let stats =
            apply_incompressible_boundary_conditions_3d(&mesh, &mut fields, &boundary).expect("bc");

        assert_eq!(stats.velocity_cells, 2);
        assert_eq!(fields.velocity_x.values()[mesh.cell_index(0, 0, 0)], 2.0);
        assert_eq!(fields.velocity_x.values()[mesh.cell_index(0, 1, 0)], 2.0);
    }

    #[test]
    fn pressure_outlet_sets_owner_cell_pressure() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
        let mut fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [1.0, 0.0, 0.0]).expect("fields");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "i_max",
            mesh.resolve_logical_boundary("i_max").expect("faces"),
            BoundaryKind::IncompressiblePressureOutlet { pressure: 3.0 },
        )]);

        let stats =
            apply_incompressible_boundary_conditions_3d(&mesh, &mut fields, &boundary).expect("bc");

        assert_eq!(stats.pressure_cells, 1);
        assert_eq!(fields.pressure.values()[mesh.cell_index(1, 0, 0)], 3.0);
    }

    #[test]
    fn symmetry_removes_normal_velocity() {
        let mesh = StructuredMesh3d::uniform_box("box", 1, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
        let mut fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [1.0, 2.0, 3.0]).expect("fields");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "j_min",
            mesh.resolve_logical_boundary("j_min").expect("faces"),
            BoundaryKind::Symmetry,
        )]);

        apply_incompressible_boundary_conditions_3d(&mesh, &mut fields, &boundary).expect("bc");

        assert_eq!(fields.velocity_x.values()[0], 1.0);
        assert_eq!(fields.velocity_y.values()[0], 0.0);
        assert_eq!(fields.velocity_z.values()[0], 3.0);
    }
}
