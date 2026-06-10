//! 不可压缩结构化 3D 边界感知 face-flux 散度诊断。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::discretization::incompressible_face_boundary::incompressible_boundary_face_velocity;
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::mesh::{BoundaryMesh, BoundaryMesh3d, StructuredMesh3d};

/// 用面速度净通量计算不可压缩散度诊断。
///
/// 内部面使用相邻 cell-centered 速度算术平均；边界面使用 `BoundarySet` 给定的
/// face 语义，墙面/对称面无穿透、动壁/速度入口使用给定面速度、压力出口使用 owner 外推。
pub fn compute_incompressible_face_flux_divergence_3d(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    boundary: &BoundarySet,
) -> Result<ScalarField> {
    fields.validate_len(mesh.num_cells())?;
    let spacing = CartesianSpacing::from_mesh(mesh)?;
    let mut net = vec![0.0; mesh.num_cells()];
    let periodic_x = boundary.has_periodic_pair("i_min", "i_max");
    add_internal_fluxes(mesh, fields, spacing, periodic_x, &mut net);
    add_boundary_fluxes(mesh, fields, boundary, &mut net)?;
    let volume = spacing.volume();
    for value in &mut net {
        *value /= volume;
    }
    ScalarField::from_values(net)
}

#[derive(Debug, Clone, Copy)]
struct CartesianSpacing {
    dx: Real,
    dy: Real,
    dz: Real,
}

impl CartesianSpacing {
    fn from_mesh(mesh: &StructuredMesh3d) -> Result<Self> {
        let dx = mesh.node_x(1, 0, 0) - mesh.node_x(0, 0, 0);
        let dy = mesh.node_y(0, 1, 0) - mesh.node_y(0, 0, 0);
        let dz = mesh.node_z(0, 0, 1) - mesh.node_z(0, 0, 0);
        if dx.abs() <= Real::EPSILON || dy.abs() <= Real::EPSILON || dz.abs() <= Real::EPSILON {
            return Err(AsimuError::Mesh(
                "不可压缩 face-flux 散度要求正的 Cartesian 网格间距".to_string(),
            ));
        }
        Ok(Self {
            dx: dx.abs(),
            dy: dy.abs(),
            dz: dz.abs(),
        })
    }

    fn volume(self) -> Real {
        self.dx * self.dy * self.dz
    }
}

fn add_internal_fluxes(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    spacing: CartesianSpacing,
    periodic_x: bool,
    net: &mut [Real],
) {
    let ax = spacing.dy * spacing.dz;
    let ay = spacing.dx * spacing.dz;
    let az = spacing.dx * spacing.dy;
    add_x_fluxes(mesh, fields, ax, periodic_x, net);
    add_y_fluxes(mesh, fields, ay, net);
    add_z_fluxes(mesh, fields, az, net);
}

fn add_x_fluxes(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    area: Real,
    periodic_x: bool,
    net: &mut [Real],
) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i + 1, j, k);
                let u_face =
                    0.5 * (fields.velocity_x.values()[left] + fields.velocity_x.values()[right]);
                scatter_flux(net, left, right, u_face * area);
            }
        }
    }
    if periodic_x && mesh.nx > 1 {
        for k in 0..mesh.nz {
            for j in 0..mesh.ny {
                let left = mesh.cell_index(mesh.nx - 1, j, k);
                let right = mesh.cell_index(0, j, k);
                let u_face =
                    0.5 * (fields.velocity_x.values()[left] + fields.velocity_x.values()[right]);
                scatter_flux(net, left, right, u_face * area);
            }
        }
    }
}

fn add_y_fluxes(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    area: Real,
    net: &mut [Real],
) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny.saturating_sub(1) {
            for i in 0..mesh.nx {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i, j + 1, k);
                let v_face =
                    0.5 * (fields.velocity_y.values()[left] + fields.velocity_y.values()[right]);
                scatter_flux(net, left, right, v_face * area);
            }
        }
    }
}

fn add_z_fluxes(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    area: Real,
    net: &mut [Real],
) {
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i, j, k + 1);
                let w_face =
                    0.5 * (fields.velocity_z.values()[left] + fields.velocity_z.values()[right]);
                scatter_flux(net, left, right, w_face * area);
            }
        }
    }
}

fn add_boundary_fluxes(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    boundary: &BoundarySet,
    net: &mut [Real],
) -> Result<()> {
    for patch in boundary.patches() {
        if matches!(patch.kind, BoundaryKind::Periodic { .. }) {
            continue;
        }
        for &face in &patch.face_ids {
            let owner = mesh.face_owner(face)?.index() as usize;
            let geom = mesh.face_geometry_3d(face)?;
            let velocity = incompressible_boundary_face_velocity(owner, &patch.kind, fields);
            net[owner] += (velocity[0] * geom.normal.x
                + velocity[1] * geom.normal.y
                + velocity[2] * geom.normal.z)
                * geom.area;
        }
    }
    Ok(())
}

fn scatter_flux(net: &mut [Real], owner: usize, neighbor: usize, flux_owner_to_neighbor: Real) {
    net[owner] += flux_owner_to_neighbor;
    net[neighbor] -= flux_owner_to_neighbor;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryPatch, WallHeat};
    use crate::core::approx_eq;

    #[test]
    fn wall_ignores_owner_normal_velocity() {
        let mesh = StructuredMesh3d::uniform_box("box", 1, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, -3.0, 0.0]).expect("fields");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "j_min",
            mesh.resolve_logical_boundary("j_min").expect("faces"),
            BoundaryKind::Wall {
                no_slip: true,
                heat: WallHeat::Adiabatic,
            },
        )]);

        let div = compute_incompressible_face_flux_divergence_3d(&mesh, &fields, &boundary)
            .expect("divergence");

        assert!(approx_eq(div.values()[0], 0.0, 1.0e-12));
    }

    #[test]
    fn velocity_inlet_contributes_boundary_flux() {
        let mesh = StructuredMesh3d::uniform_box("box", 1, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "i_min",
            mesh.resolve_logical_boundary("i_min").expect("faces"),
            BoundaryKind::IncompressibleVelocityInlet {
                velocity: [1.0, 0.0, 0.0],
            },
        )]);

        let div = compute_incompressible_face_flux_divergence_3d(&mesh, &fields, &boundary)
            .expect("divergence");

        assert!(approx_eq(div.values()[0], -1.0, 1.0e-12));
    }
}
