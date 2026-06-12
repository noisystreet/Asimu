//! 不可压缩结构化 3D 边界感知 face-flux 散度诊断。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::discretization::incompressible_boundary_flux::{
    IncompressibleBoundaryOwnerMap, interior_face_velocity,
};
use crate::discretization::incompressible_face_boundary::incompressible_boundary_mass_flux;
use crate::error::Result;
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
    let boundary_map = IncompressibleBoundaryOwnerMap::build(mesh, boundary);
    let mut net = vec![0.0; mesh.num_cells()];
    let periodic_x = boundary.has_periodic_pair("i_min", "i_max");
    add_internal_fluxes(mesh, fields, periodic_x, &boundary_map, &mut net);
    add_boundary_fluxes(mesh, fields, boundary, &mut net)?;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let cell = mesh.cell_index(i, j, k);
                net[cell] /= mesh.cell_metric(i, j, k).volume;
            }
        }
    }
    ScalarField::from_values(net)
}

fn add_internal_fluxes(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    periodic_x: bool,
    boundary: &IncompressibleBoundaryOwnerMap,
    net: &mut [Real],
) {
    add_x_fluxes(mesh, fields, periodic_x, boundary, net);
    add_y_fluxes(mesh, fields, boundary, net);
    add_z_fluxes(mesh, fields, boundary, net);
}

fn add_x_fluxes(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    periodic_x: bool,
    boundary: &IncompressibleBoundaryOwnerMap,
    net: &mut [Real],
) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i + 1, j, k);
                let metric = mesh.i_face_metric(i, j, k);
                let flux = interior_face_normal_flux(fields, left, right, &metric, boundary);
                scatter_flux(net, left, right, flux);
            }
        }
    }
    if periodic_x && mesh.nx > 1 {
        for k in 0..mesh.nz {
            for j in 0..mesh.ny {
                let left = mesh.cell_index(mesh.nx - 1, j, k);
                let right = mesh.cell_index(0, j, k);
                let metric = mesh.i_face_metric(mesh.nx.saturating_sub(2), j, k);
                let flux = interior_face_normal_flux(fields, left, right, &metric, boundary);
                scatter_flux(net, left, right, flux);
            }
        }
    }
}

fn add_y_fluxes(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    boundary: &IncompressibleBoundaryOwnerMap,
    net: &mut [Real],
) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny.saturating_sub(1) {
            for i in 0..mesh.nx {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i, j + 1, k);
                let metric = mesh.j_face_metric(i, j, k);
                let flux = interior_face_normal_flux(fields, left, right, &metric, boundary);
                scatter_flux(net, left, right, flux);
            }
        }
    }
}

fn add_z_fluxes(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    boundary: &IncompressibleBoundaryOwnerMap,
    net: &mut [Real],
) {
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i, j, k + 1);
                let metric = mesh.k_face_metric(i, j, k);
                let flux = interior_face_normal_flux(fields, left, right, &metric, boundary);
                scatter_flux(net, left, right, flux);
            }
        }
    }
}

fn interior_face_normal_flux(
    fields: &IncompressibleFields,
    left: usize,
    right: usize,
    metric: &crate::mesh::FaceMetric,
    boundary: &IncompressibleBoundaryOwnerMap,
) -> Real {
    let velocity = [
        interior_face_velocity(fields, left, right, 0, boundary),
        interior_face_velocity(fields, left, right, 1, boundary),
        interior_face_velocity(fields, left, right, 2, boundary),
    ];
    (velocity[0] * metric.normal.x + velocity[1] * metric.normal.y + velocity[2] * metric.normal.z)
        * metric.area
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
            let flux = incompressible_boundary_mass_flux(
                owner,
                &patch.kind,
                fields,
                geom.normal,
                geom.area,
            );
            net[owner] += flux;
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
    use crate::mesh::MeshMetricMode;

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

    #[test]
    fn curvilinear_internal_flux_uses_face_normal_projection() {
        let mut mesh = sheared_two_cell_mesh();
        mesh.set_metric_mode(MeshMetricMode::Curvilinear);
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 1.0, 0.0]).expect("fields");

        let div =
            compute_incompressible_face_flux_divergence_3d(&mesh, &fields, &BoundarySet::default())
                .expect("divergence");

        assert!(div.values()[0].abs() > 0.1);
        assert!(approx_eq(
            div.values()[0] * mesh.cell_metric(0, 0, 0).volume
                + div.values()[1] * mesh.cell_metric(1, 0, 0).volume,
            0.0,
            1.0e-12
        ));
    }

    fn sheared_two_cell_mesh() -> StructuredMesh3d {
        let nx = 2;
        let ny = 1;
        let nz = 1;
        let shear = 0.5;
        let mut px = Vec::new();
        let mut py = Vec::new();
        let mut pz = Vec::new();
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    px.push(i as Real + shear * j as Real);
                    py.push(j as Real);
                    pz.push(k as Real);
                }
            }
        }
        StructuredMesh3d::new("sheared", nx, ny, nz, px, py, pz).expect("mesh")
    }
}
