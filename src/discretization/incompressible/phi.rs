//! 不可压缩压力-速度耦合的显式面通量状态。
//!
//! 参考 OpenFOAM 的 `phiHbyA -> pEqn -> phi -= pEqn.flux()` 流程，本模块在
//! 结构化网格上保存内部面体积通量，并在压力校正后直接更新面通量。

use super::boundary_flux::interior_face_velocity;
use super::face_boundary::incompressible_boundary_mass_flux;
use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::mesh::{BoundaryMesh, BoundaryMesh3d, StructuredMesh3d};

#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleFaceFluxField {
    phi_x: Vec<Real>,
    phi_x_periodic: Option<Vec<Real>>,
    phi_y: Vec<Real>,
    phi_z: Vec<Real>,
    boundary_net: Vec<Real>,
}

impl IncompressibleFaceFluxField {
    pub fn from_rhie_chow(
        mesh: &StructuredMesh3d,
        fields: &IncompressibleFields,
        d_coefficient: &ScalarField,
        boundary: &BoundarySet,
    ) -> Result<Self> {
        fields.validate_len(mesh.num_cells())?;
        if d_coefficient.len() != mesh.num_cells() {
            return Err(AsimuError::Field(
                "面通量 d_P 长度与网格单元数不一致".to_string(),
            ));
        }
        let periodic_x = boundary.has_periodic_pair("i_min", "i_max");
        let mut flux = Self::zeros(mesh, periodic_x);
        fill_internal_rhie_chow_fluxes(mesh, fields, d_coefficient.values(), &mut flux);
        fill_boundary_net(mesh, fields, boundary, &mut flux.boundary_net)?;
        Ok(flux)
    }

    pub fn apply_pressure_correction(
        &mut self,
        mesh: &StructuredMesh3d,
        d_coefficient: &[Real],
        pressure_correction: &[Real],
        scale: Real,
    ) -> Result<()> {
        if d_coefficient.len() != mesh.num_cells() || pressure_correction.len() != mesh.num_cells()
        {
            return Err(AsimuError::Field(
                "面通量压力校正长度与网格单元数不一致".to_string(),
            ));
        }
        update_x_fluxes(mesh, d_coefficient, pressure_correction, scale, self);
        update_y_fluxes(mesh, d_coefficient, pressure_correction, scale, self);
        update_z_fluxes(mesh, d_coefficient, pressure_correction, scale, self);
        Ok(())
    }

    pub fn divergence(&self, mesh: &StructuredMesh3d) -> Result<ScalarField> {
        let mut net = self.boundary_net.clone();
        scatter_x_fluxes(mesh, self, &mut net);
        scatter_y_fluxes(mesh, self, &mut net);
        scatter_z_fluxes(mesh, self, &mut net);
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

    pub fn cell_face_flux(
        &self,
        mesh: &StructuredMesh3d,
        axis: usize,
        cell: (usize, usize, usize),
        upper: bool,
    ) -> Option<Real> {
        let (i, j, k) = cell;
        match (axis, upper) {
            (0, true) if i + 1 < mesh.nx => Some(self.phi_x[x_index(mesh, i, j, k)]),
            (0, true) => self
                .phi_x_periodic
                .as_ref()
                .map(|values| values[x_periodic_index(mesh, j, k)]),
            (0, false) if i > 0 => Some(-self.phi_x[x_index(mesh, i - 1, j, k)]),
            (0, false) => self
                .phi_x_periodic
                .as_ref()
                .map(|values| -values[x_periodic_index(mesh, j, k)]),
            (1, true) if j + 1 < mesh.ny => Some(self.phi_y[y_index(mesh, i, j, k)]),
            (1, false) if j > 0 => Some(-self.phi_y[y_index(mesh, i, j - 1, k)]),
            (2, true) if k + 1 < mesh.nz => Some(self.phi_z[z_index(mesh, i, j, k)]),
            (2, false) if k > 0 => Some(-self.phi_z[z_index(mesh, i, j, k - 1)]),
            _ => None,
        }
    }

    fn zeros(mesh: &StructuredMesh3d, periodic_x: bool) -> Self {
        Self {
            phi_x: vec![0.0; mesh.nx.saturating_sub(1) * mesh.ny * mesh.nz],
            phi_x_periodic: periodic_x.then(|| vec![0.0; mesh.ny * mesh.nz]),
            phi_y: vec![0.0; mesh.nx * mesh.ny.saturating_sub(1) * mesh.nz],
            phi_z: vec![0.0; mesh.nx * mesh.ny * mesh.nz.saturating_sub(1)],
            boundary_net: vec![0.0; mesh.num_cells()],
        }
    }
}

fn fill_internal_rhie_chow_fluxes(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    d: &[Real],
    flux: &mut IncompressibleFaceFluxField,
) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i + 1, j, k);
                let metric = mesh.i_face_metric(i, j, k);
                let spacing = owner_neighbor_distance(mesh, (i, j, k), (i + 1, j, k), &metric);
                flux.phi_x[x_index(mesh, i, j, k)] =
                    rhie_chow_face_flux(fields, d, left, right, spacing, &metric) * metric.area;
            }
            if let Some(phi_x_periodic) = flux.phi_x_periodic.as_mut() {
                let left = mesh.cell_index(mesh.nx - 1, j, k);
                let right = mesh.cell_index(0, j, k);
                let metric = mesh.i_face_metric(mesh.nx.saturating_sub(2), j, k);
                let spacing =
                    owner_neighbor_distance(mesh, (mesh.nx - 1, j, k), (0, j, k), &metric);
                phi_x_periodic[x_periodic_index(mesh, j, k)] =
                    rhie_chow_face_flux(fields, d, left, right, spacing, &metric) * metric.area;
            }
        }
    }
    for k in 0..mesh.nz {
        for j in 0..mesh.ny.saturating_sub(1) {
            for i in 0..mesh.nx {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i, j + 1, k);
                let metric = mesh.j_face_metric(i, j, k);
                let spacing = owner_neighbor_distance(mesh, (i, j, k), (i, j + 1, k), &metric);
                flux.phi_y[y_index(mesh, i, j, k)] =
                    rhie_chow_face_flux(fields, d, left, right, spacing, &metric) * metric.area;
            }
        }
    }
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i, j, k + 1);
                let metric = mesh.k_face_metric(i, j, k);
                let spacing = owner_neighbor_distance(mesh, (i, j, k), (i, j, k + 1), &metric);
                flux.phi_z[z_index(mesh, i, j, k)] =
                    rhie_chow_face_flux(fields, d, left, right, spacing, &metric) * metric.area;
            }
        }
    }
}

fn rhie_chow_face_flux(
    fields: &IncompressibleFields,
    d: &[Real],
    left: usize,
    right: usize,
    spacing: Real,
    metric: &crate::mesh::FaceMetric,
) -> Real {
    let u_face = interior_face_velocity(fields, left, right, 0) * metric.normal.x
        + interior_face_velocity(fields, left, right, 1) * metric.normal.y
        + interior_face_velocity(fields, left, right, 2) * metric.normal.z;
    let d_face = 0.5 * (d[left] + d[right]);
    let dp = fields.pressure.values()[right] - fields.pressure.values()[left];
    u_face - d_face * dp / spacing
}

fn fill_boundary_net(
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
            net[owner] += incompressible_boundary_mass_flux(
                owner,
                &patch.kind,
                fields,
                geom.normal,
                geom.area,
            );
        }
    }
    Ok(())
}

fn update_x_fluxes(
    mesh: &StructuredMesh3d,
    d: &[Real],
    p_corr: &[Real],
    scale: Real,
    flux: &mut IncompressibleFaceFluxField,
) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i + 1, j, k);
                let idx = x_index(mesh, i, j, k);
                let metric = mesh.i_face_metric(i, j, k);
                let spacing = owner_neighbor_distance(mesh, (i, j, k), (i + 1, j, k), &metric);
                flux.phi_x[idx] +=
                    pressure_flux_delta(d, p_corr, left, right, spacing) * metric.area * scale;
            }
            if let Some(phi_x_periodic) = flux.phi_x_periodic.as_mut() {
                let left = mesh.cell_index(mesh.nx - 1, j, k);
                let right = mesh.cell_index(0, j, k);
                let idx = x_periodic_index(mesh, j, k);
                let metric = mesh.i_face_metric(mesh.nx.saturating_sub(2), j, k);
                let spacing =
                    owner_neighbor_distance(mesh, (mesh.nx - 1, j, k), (0, j, k), &metric);
                phi_x_periodic[idx] +=
                    pressure_flux_delta(d, p_corr, left, right, spacing) * metric.area * scale;
            }
        }
    }
}

fn update_y_fluxes(
    mesh: &StructuredMesh3d,
    d: &[Real],
    p_corr: &[Real],
    scale: Real,
    flux: &mut IncompressibleFaceFluxField,
) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny.saturating_sub(1) {
            for i in 0..mesh.nx {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i, j + 1, k);
                let idx = y_index(mesh, i, j, k);
                let metric = mesh.j_face_metric(i, j, k);
                let spacing = owner_neighbor_distance(mesh, (i, j, k), (i, j + 1, k), &metric);
                flux.phi_y[idx] +=
                    pressure_flux_delta(d, p_corr, left, right, spacing) * metric.area * scale;
            }
        }
    }
}

fn update_z_fluxes(
    mesh: &StructuredMesh3d,
    d: &[Real],
    p_corr: &[Real],
    scale: Real,
    flux: &mut IncompressibleFaceFluxField,
) {
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let left = mesh.cell_index(i, j, k);
                let right = mesh.cell_index(i, j, k + 1);
                let idx = z_index(mesh, i, j, k);
                let metric = mesh.k_face_metric(i, j, k);
                let spacing = owner_neighbor_distance(mesh, (i, j, k), (i, j, k + 1), &metric);
                flux.phi_z[idx] +=
                    pressure_flux_delta(d, p_corr, left, right, spacing) * metric.area * scale;
            }
        }
    }
}

fn pressure_flux_delta(
    d: &[Real],
    pressure_correction: &[Real],
    left: usize,
    right: usize,
    spacing: Real,
) -> Real {
    0.5 * (d[left] + d[right]) * (pressure_correction[right] - pressure_correction[left]) / spacing
}

fn owner_neighbor_distance(
    mesh: &StructuredMesh3d,
    owner: (usize, usize, usize),
    neighbor: (usize, usize, usize),
    face: &crate::mesh::FaceMetric,
) -> Real {
    let owner_center = mesh.cell_metric(owner.0, owner.1, owner.2).center;
    let neighbor_center = mesh.cell_metric(neighbor.0, neighbor.1, neighbor.2).center;
    let dx = neighbor_center.x - owner_center.x;
    let dy = neighbor_center.y - owner_center.y;
    let dz = neighbor_center.z - owner_center.z;
    let projected = (dx * face.normal.x + dy * face.normal.y + dz * face.normal.z).abs();
    projected.max(Real::EPSILON)
}

fn scatter_x_fluxes(mesh: &StructuredMesh3d, flux: &IncompressibleFaceFluxField, net: &mut [Real]) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                scatter_pair(
                    net,
                    mesh.cell_index(i, j, k),
                    mesh.cell_index(i + 1, j, k),
                    flux.phi_x[x_index(mesh, i, j, k)],
                );
            }
            if let Some(phi_x_periodic) = &flux.phi_x_periodic {
                scatter_pair(
                    net,
                    mesh.cell_index(mesh.nx - 1, j, k),
                    mesh.cell_index(0, j, k),
                    phi_x_periodic[x_periodic_index(mesh, j, k)],
                );
            }
        }
    }
}

fn scatter_y_fluxes(mesh: &StructuredMesh3d, flux: &IncompressibleFaceFluxField, net: &mut [Real]) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny.saturating_sub(1) {
            for i in 0..mesh.nx {
                scatter_pair(
                    net,
                    mesh.cell_index(i, j, k),
                    mesh.cell_index(i, j + 1, k),
                    flux.phi_y[y_index(mesh, i, j, k)],
                );
            }
        }
    }
}

fn scatter_z_fluxes(mesh: &StructuredMesh3d, flux: &IncompressibleFaceFluxField, net: &mut [Real]) {
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                scatter_pair(
                    net,
                    mesh.cell_index(i, j, k),
                    mesh.cell_index(i, j, k + 1),
                    flux.phi_z[z_index(mesh, i, j, k)],
                );
            }
        }
    }
}

fn scatter_pair(net: &mut [Real], owner: usize, neighbor: usize, flux_owner_to_neighbor: Real) {
    net[owner] += flux_owner_to_neighbor;
    net[neighbor] -= flux_owner_to_neighbor;
}

fn x_index(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> usize {
    (k * mesh.ny + j) * mesh.nx.saturating_sub(1) + i
}

fn x_periodic_index(mesh: &StructuredMesh3d, j: usize, k: usize) -> usize {
    k * mesh.ny + j
}

fn y_index(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> usize {
    (k * mesh.ny.saturating_sub(1) + j) * mesh.nx + i
}

fn z_index(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> usize {
    (k * mesh.ny + j) * mesh.nx + i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::discretization::{
        IncompressiblePressureCorrectionConfig, assemble_incompressible_pressure_correction_3d,
    };

    #[test]
    fn pressure_correction_reduces_two_cell_flux_divergence() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 1, 1, 2.0, 1.0, 1.0).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [1.0, 0.0, 0.0]).expect("fields");
        let d = ScalarField::uniform(mesh.num_cells(), 1.0).expect("d");
        let mut flux = IncompressibleFaceFluxField::from_rhie_chow(
            &mesh,
            &fields,
            &d,
            &BoundarySet::default(),
        )
        .expect("flux");
        flux.apply_pressure_correction(&mesh, d.values(), &[1.0, 0.0], 1.0)
            .expect("correct");

        let div = flux.divergence(&mesh).expect("div");

        assert!(
            div.values()
                .iter()
                .all(|value| approx_eq(*value, 0.0, 1.0e-12))
        );
    }

    #[test]
    fn pressure_matrix_and_phi_correction_match_on_nonuniform_mesh() {
        let mesh = nonuniform_two_cell_mesh();
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [1.0, 0.0, 0.0]).expect("fields");
        let d = ScalarField::uniform(mesh.num_cells(), 1.0).expect("d");
        let boundary = BoundarySet::default();
        let mut flux = IncompressibleFaceFluxField::from_rhie_chow(&mesh, &fields, &d, &boundary)
            .expect("flux");
        let predicted = flux.divergence(&mesh).expect("predicted");
        let system = assemble_incompressible_pressure_correction_3d(
            &mesh,
            &predicted,
            &d,
            &boundary,
            IncompressiblePressureCorrectionConfig::new(1.0, 0, 0.0).expect("config"),
        )
        .expect("system");
        let correction = [0.0, -0.75];

        flux.apply_pressure_correction(&mesh, d.values(), &correction, 1.0)
            .expect("correct");
        let corrected = flux.divergence(&mesh).expect("corrected");

        let cell = 1;
        let ax = system
            .matrix
            .row_entries(cell)
            .map(|(col, value)| value * correction[col])
            .sum::<Real>();
        let expected = (system.rhs[cell] - ax) / mesh.cell_metric(1, 0, 0).volume;
        assert!(
            approx_eq(corrected.values()[cell], expected, 1.0e-12),
            "corrected={} expected={expected} rhs={} ax={} volume={}",
            corrected.values()[cell],
            system.rhs[cell],
            ax,
            mesh.cell_metric(1, 0, 0).volume
        );
    }

    fn nonuniform_two_cell_mesh() -> StructuredMesh3d {
        let nx = 2;
        let ny = 1;
        let nz = 1;
        let xs = [0.0, 0.5, 2.0];
        let mut px = Vec::new();
        let mut py = Vec::new();
        let mut pz = Vec::new();
        for k in 0..=nz {
            for j in 0..=ny {
                for &x in &xs {
                    px.push(x);
                    py.push(j as Real);
                    pz.push(k as Real);
                }
            }
        }
        StructuredMesh3d::new("nonuniform", nx, ny, nz, px, py, pz).expect("mesh")
    }
}
