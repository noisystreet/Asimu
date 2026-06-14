//! 不可压缩 cell-centered 边界条件应用。
//!
//! I2 首版把边界条件施加到边界 owner 单元，供结构化 skeleton runner 使用；
//! 后续完整 FVM 会把同一语义下沉为面 ghost / 面通量。

use super::face_boundary::tangential_velocity;
use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::{FaceId, Real};
use crate::error::{AsimuError, Result};
use crate::field::IncompressibleFields;
use crate::mesh::{BoundaryMesh, BoundaryMesh3d, LogicalFace3d, StructuredMesh3d};

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
            let normal_arr = [normal.x, normal.y, normal.z];
            match &patch.kind {
                BoundaryKind::Wall { no_slip: true, .. } => {
                    set_velocity(fields, owner, [0.0, 0.0, 0.0]);
                    stats.velocity_cells += 1;
                }
                BoundaryKind::Wall { no_slip: false, .. } => {
                    zero_normal_velocity(fields, owner, normal_arr);
                    stats.velocity_cells += 1;
                }
                BoundaryKind::MovingWall { velocity } => {
                    set_velocity(fields, owner, tangential_velocity(*velocity, normal_arr));
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
                    if interior_neighbor_index(mesh, face)?.is_some() {
                        apply_symmetry_mirror(fields, owner, face, mesh)?;
                    } else {
                        zero_normal_velocity(fields, owner, normal_arr);
                    }
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

/// 动量边界装配共用的 owner 目标速度（与 cell BC 一致）。
#[must_use]
pub(crate) fn incompressible_boundary_owner_velocity_target(
    kind: &BoundaryKind,
    normal: [Real; 3],
    _fields: &IncompressibleFields,
    _interior: Option<usize>,
) -> Option<[Real; 3]> {
    match kind {
        BoundaryKind::Wall { no_slip: true, .. } => Some([0.0, 0.0, 0.0]),
        BoundaryKind::MovingWall { velocity } => Some(tangential_velocity(*velocity, normal)),
        BoundaryKind::IncompressibleVelocityInlet { velocity } => Some(*velocity),
        _ => None,
    }
}

/// 边界面对应的域内邻居单元（结构化轴对齐网格）。
pub(crate) fn interior_neighbor_index(
    mesh: &StructuredMesh3d,
    face: FaceId,
) -> Result<Option<usize>> {
    let (logical, local) = LogicalFace3d::decode(face)?;
    let (i, j, k) = mesh.face_ij(logical, local)?;
    let index = match logical {
        LogicalFace3d::IMin if mesh.nx > 1 => Some(mesh.cell_index(1, j, k)),
        LogicalFace3d::IMax if mesh.nx > 1 => Some(mesh.cell_index(mesh.nx - 2, j, k)),
        LogicalFace3d::JMin if mesh.ny > 1 => Some(mesh.cell_index(i, 1, k)),
        LogicalFace3d::JMax if mesh.ny > 1 => Some(mesh.cell_index(i, mesh.ny - 2, k)),
        LogicalFace3d::KMin if mesh.nz > 1 => Some(mesh.cell_index(i, j, 1)),
        LogicalFace3d::KMax if mesh.nz > 1 => Some(mesh.cell_index(i, j, mesh.nz - 2)),
        _ => None,
    };
    Ok(index)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IncompressibleBoundaryApplyStats {
    pub velocity_cells: usize,
    pub pressure_cells: usize,
    pub ignored_faces: usize,
}

fn apply_symmetry_mirror(
    fields: &mut IncompressibleFields,
    owner: usize,
    face: FaceId,
    mesh: &StructuredMesh3d,
) -> Result<()> {
    let interior = interior_neighbor_index(mesh, face)?
        .ok_or_else(|| AsimuError::Boundary("对称边界缺少域内邻居".to_string()))?;
    let normal = mesh.face_normal_3d(face)?;
    let normal_arr = [normal.x, normal.y, normal.z];
    let interior_v = [
        fields.velocity_x.values()[interior],
        fields.velocity_y.values()[interior],
        fields.velocity_z.values()[interior],
    ];
    let mut owner_v = [
        fields.velocity_x.values()[owner],
        fields.velocity_y.values()[owner],
        fields.velocity_z.values()[owner],
    ];
    for component in 0..3 {
        if normal_arr[component].abs() > 0.5 {
            owner_v[component] = -interior_v[component];
        }
    }
    set_velocity(fields, owner, owner_v);
    Ok(())
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
    }

    #[test]
    fn no_slip_wall_zeros_owner_velocity() {
        let mesh = StructuredMesh3d::uniform_box("box", 1, 2, 1, 1.0, 1.0, 1.0).expect("mesh");
        let mut fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.4, -0.2, 0.1]).expect("fields");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "j_min",
            mesh.resolve_logical_boundary("j_min").expect("faces"),
            BoundaryKind::Wall {
                no_slip: true,
                heat: crate::boundary::WallHeat::Adiabatic,
            },
        )]);

        let stats =
            apply_incompressible_boundary_conditions_3d(&mesh, &mut fields, &boundary).expect("bc");

        let wall = mesh.cell_index(0, 0, 0);
        assert_eq!(stats.velocity_cells, 1);
        assert_eq!(fields.velocity_x.values()[wall], 0.0);
        assert_eq!(fields.velocity_y.values()[wall], 0.0);
        assert_eq!(fields.velocity_z.values()[wall], 0.0);
    }

    #[test]
    fn moving_wall_sets_tangential_owner_velocity() {
        let mesh = StructuredMesh3d::uniform_box("box", 1, 2, 1, 1.0, 1.0, 1.0).expect("mesh");
        let mut fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.2, 0.3, 0.0]).expect("fields");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "j_max",
            mesh.resolve_logical_boundary("j_max").expect("faces"),
            BoundaryKind::MovingWall {
                velocity: [1.0, 0.0, 0.0],
            },
        )]);

        let stats =
            apply_incompressible_boundary_conditions_3d(&mesh, &mut fields, &boundary).expect("bc");

        let lid = mesh.cell_index(0, 1, 0);
        assert_eq!(stats.velocity_cells, 1);
        assert_eq!(fields.velocity_x.values()[lid], 1.0);
        assert_eq!(fields.velocity_y.values()[lid], 0.0);
    }

    #[test]
    fn symmetry_preserves_uniform_field() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 1, 1.0, 1.0, 1.0).expect("mesh");
        let mut fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.2, 0.1, 0.0]).expect("fields");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "k_min",
            mesh.resolve_logical_boundary("k_min").expect("faces"),
            BoundaryKind::Symmetry,
        )]);

        let stats =
            apply_incompressible_boundary_conditions_3d(&mesh, &mut fields, &boundary).expect("bc");

        assert_eq!(stats.ignored_faces, 0);
        for cell in 0..mesh.num_cells() {
            assert_eq!(fields.velocity_x.values()[cell], 0.2);
            assert_eq!(fields.velocity_y.values()[cell], 0.1);
            assert_eq!(fields.velocity_z.values()[cell], 0.0);
        }
    }

    #[test]
    fn lid_cavity_wall_configuration_preserves_corner_constraints() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 1, 1.0, 1.0, 0.1).expect("mesh");
        let mut fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let boundary = BoundarySet::new(vec![
            BoundaryPatch::new(
                "i_min",
                mesh.resolve_logical_boundary("i_min").expect("faces"),
                BoundaryKind::Wall {
                    no_slip: true,
                    heat: crate::boundary::WallHeat::Adiabatic,
                },
            ),
            BoundaryPatch::new(
                "i_max",
                mesh.resolve_logical_boundary("i_max").expect("faces"),
                BoundaryKind::Wall {
                    no_slip: true,
                    heat: crate::boundary::WallHeat::Adiabatic,
                },
            ),
            BoundaryPatch::new(
                "j_min",
                mesh.resolve_logical_boundary("j_min").expect("faces"),
                BoundaryKind::Wall {
                    no_slip: true,
                    heat: crate::boundary::WallHeat::Adiabatic,
                },
            ),
            BoundaryPatch::new(
                "j_max",
                mesh.resolve_logical_boundary("j_max").expect("faces"),
                BoundaryKind::MovingWall {
                    velocity: [1.0, 0.0, 0.0],
                },
            ),
            BoundaryPatch::new(
                "k_min",
                mesh.resolve_logical_boundary("k_min").expect("faces"),
                BoundaryKind::Symmetry,
            ),
            BoundaryPatch::new(
                "k_max",
                mesh.resolve_logical_boundary("k_max").expect("faces"),
                BoundaryKind::Symmetry,
            ),
        ]);

        let stats =
            apply_incompressible_boundary_conditions_3d(&mesh, &mut fields, &boundary).expect("bc");

        assert!(stats.velocity_cells > 0);
        let lid = mesh.cell_index(0, 1, 0);
        assert_eq!(fields.velocity_x.values()[lid], 1.0);
        assert_eq!(fields.velocity_y.values()[lid], 0.0);
        let bottom = mesh.cell_index(0, 0, 0);
        assert_eq!(fields.velocity_x.values()[bottom], 0.0);
        assert_eq!(fields.velocity_y.values()[bottom], 0.0);
    }
}
