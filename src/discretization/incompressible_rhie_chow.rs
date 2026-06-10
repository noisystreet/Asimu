//! 不可压缩结构化 3D Rhie-Chow 面质量通量。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::mesh::{BoundaryMesh, BoundaryMesh3d, StructuredMesh3d};

/// 用 Rhie-Chow 面通量计算连续性残差。
///
/// 内部面通量使用中心插值速度，并减去 \(d_f \nabla p\) 的面法向贡献；
/// 边界面按不可压缩边界类型给定法向通量。返回值为每个单元的净体积通量除以体积。
pub fn compute_incompressible_rhie_chow_divergence_3d(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    d_coefficient: &ScalarField,
    boundary: &BoundarySet,
) -> Result<ScalarField> {
    fields.validate_len(mesh.num_cells())?;
    if d_coefficient.len() != mesh.num_cells() {
        return Err(AsimuError::Field(format!(
            "Rhie-Chow d_P 长度 {} 与网格单元数 {} 不一致",
            d_coefficient.len(),
            mesh.num_cells()
        )));
    }
    let spacing = CartesianSpacing::from_mesh(mesh)?;
    let mut net = vec![0.0; mesh.num_cells()];
    let periodic_x = boundary.has_periodic_pair("i_min", "i_max");
    add_interior_fluxes(
        mesh,
        fields,
        d_coefficient.values(),
        spacing,
        periodic_x,
        &mut net,
    );
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
                "Rhie-Chow 通量要求正的 Cartesian 网格间距".to_string(),
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

fn add_interior_fluxes(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    d: &[Real],
    spacing: CartesianSpacing,
    periodic_x: bool,
    net: &mut [Real],
) {
    let ax = spacing.dy * spacing.dz;
    let ay = spacing.dx * spacing.dz;
    let az = spacing.dx * spacing.dy;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i + 1, j, k);
                let u_face =
                    0.5 * (fields.velocity_x.values()[left] + fields.velocity_x.values()[right]);
                let d_face = 0.5 * (d[left] + d[right]);
                let dp = fields.pressure.values()[right] - fields.pressure.values()[left];
                scatter_pair(net, left, right, (u_face - d_face * dp / spacing.dx) * ax);
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
                let d_face = 0.5 * (d[left] + d[right]);
                let dp = fields.pressure.values()[right] - fields.pressure.values()[left];
                scatter_pair(net, left, right, (u_face - d_face * dp / spacing.dx) * ax);
            }
        }
    }
    for k in 0..mesh.nz {
        for j in 0..mesh.ny.saturating_sub(1) {
            for i in 0..mesh.nx {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i, j + 1, k);
                let v_face =
                    0.5 * (fields.velocity_y.values()[left] + fields.velocity_y.values()[right]);
                let d_face = 0.5 * (d[left] + d[right]);
                let dp = fields.pressure.values()[right] - fields.pressure.values()[left];
                scatter_pair(net, left, right, (v_face - d_face * dp / spacing.dy) * ay);
            }
        }
    }
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i, j, k + 1);
                let w_face =
                    0.5 * (fields.velocity_z.values()[left] + fields.velocity_z.values()[right]);
                let d_face = 0.5 * (d[left] + d[right]);
                let dp = fields.pressure.values()[right] - fields.pressure.values()[left];
                scatter_pair(net, left, right, (w_face - d_face * dp / spacing.dz) * az);
            }
        }
    }
}

fn scatter_pair(net: &mut [Real], owner: usize, neighbor: usize, flux_owner_to_neighbor: Real) {
    net[owner] += flux_owner_to_neighbor;
    net[neighbor] -= flux_owner_to_neighbor;
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
            let velocity = boundary_face_velocity(owner, &patch.kind, fields);
            let flux = (velocity[0] * geom.normal.x
                + velocity[1] * geom.normal.y
                + velocity[2] * geom.normal.z)
                * geom.area;
            net[owner] += flux;
        }
    }
    Ok(())
}

fn boundary_face_velocity(
    owner: usize,
    kind: &BoundaryKind,
    fields: &IncompressibleFields,
) -> [Real; 3] {
    match kind {
        BoundaryKind::Wall { .. } | BoundaryKind::Symmetry => [0.0, 0.0, 0.0],
        BoundaryKind::MovingWall { velocity } => *velocity,
        BoundaryKind::IncompressibleVelocityInlet { velocity } => *velocity,
        BoundaryKind::IncompressiblePressureOutlet { .. } | BoundaryKind::Outlet { .. } => {
            cell_velocity(fields, owner)
        }
        _ => cell_velocity(fields, owner),
    }
}

fn cell_velocity(fields: &IncompressibleFields, cell: usize) -> [Real; 3] {
    [
        fields.velocity_x.values()[cell],
        fields.velocity_y.values()[cell],
        fields.velocity_z.values()[cell],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundaryPatch;
    use crate::core::approx_eq;

    #[test]
    fn uniform_field_has_zero_rhie_chow_divergence() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let d = ScalarField::uniform(mesh.num_cells(), 1.0).expect("d");

        let div = compute_incompressible_rhie_chow_divergence_3d(
            &mesh,
            &fields,
            &d,
            &BoundarySet::default(),
        )
        .expect("div");

        assert!(
            div.values()
                .iter()
                .all(|value| approx_eq(*value, 0.0, 1.0e-12))
        );
    }

    #[test]
    fn pressure_gradient_drives_rhie_chow_flux() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 1, 1, 2.0, 1.0, 1.0).expect("mesh");
        let fields = IncompressibleFields {
            pressure: ScalarField::from_values(vec![0.0, 2.0]).expect("p"),
            velocity_x: ScalarField::uniform(mesh.num_cells(), 0.0).expect("u"),
            velocity_y: ScalarField::uniform(mesh.num_cells(), 0.0).expect("v"),
            velocity_z: ScalarField::uniform(mesh.num_cells(), 0.0).expect("w"),
        };
        let d = ScalarField::uniform(mesh.num_cells(), 0.5).expect("d");

        let div = compute_incompressible_rhie_chow_divergence_3d(
            &mesh,
            &fields,
            &d,
            &BoundarySet::default(),
        )
        .expect("div");

        assert!(approx_eq(div.values()[0], -1.0, 1.0e-12));
        assert!(approx_eq(div.values()[1], 1.0, 1.0e-12));
    }

    #[test]
    fn velocity_inlet_boundary_contributes_flux() {
        let mesh = StructuredMesh3d::uniform_box("box", 1, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let d = ScalarField::uniform(mesh.num_cells(), 1.0).expect("d");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "i_min",
            mesh.resolve_logical_boundary("i_min").expect("faces"),
            BoundaryKind::IncompressibleVelocityInlet {
                velocity: [1.0, 0.0, 0.0],
            },
        )]);

        let div = compute_incompressible_rhie_chow_divergence_3d(&mesh, &fields, &d, &boundary)
            .expect("div");

        assert!(approx_eq(div.values()[0], -1.0, 1.0e-12));
    }
}
