//! 网格读入后的诊断报告（几何 + 边界 patch 摘要）。

use std::fmt;

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::mesh::{
    MeshDiagnostics, StructuredMesh, StructuredMesh1d, StructuredMesh3d, mesh1d_diagnostics,
    mesh3d_diagnostics, structured_mesh_diagnostics,
};

use super::CaseMesh;

/// 边界 patch 摘要行。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryPatchSummary {
    pub name: String,
    pub kind: String,
    pub faces: usize,
}

/// 带来源说明的网格诊断报告。
#[derive(Debug, Clone, PartialEq)]
pub struct MeshReport {
    pub source: String,
    pub mesh: MeshDiagnostics,
    pub boundary: Vec<BoundaryPatchSummary>,
}

impl fmt::Display for MeshReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_mesh_header(f, self)?;
        write_mesh_geometry(f, &self.mesh)?;
        write_mesh_warnings(f, &self.mesh)?;
        write_boundary_section(f, &self.boundary)
    }
}

fn write_mesh_header(f: &mut fmt::Formatter<'_>, report: &MeshReport) -> fmt::Result {
    let m = &report.mesh;
    writeln!(f, "source: {}", report.source)?;
    writeln!(f, "mesh: {}", m.name)?;
    writeln!(
        f,
        "  dim={}  cells={} ({})  nodes={}",
        m.dimension,
        m.num_cells,
        m.cell_dims_label(),
        m.num_nodes
    )
}

fn write_mesh_geometry(f: &mut fmt::Formatter<'_>, mesh: &MeshDiagnostics) -> fmt::Result {
    write_axis(f, "x", mesh.bounds.x)?;
    if mesh.dimension >= 2 {
        write_axis(f, "y", mesh.bounds.y)?;
    }
    if mesh.dimension >= 3 {
        write_axis(f, "z", mesh.bounds.z)?;
    }
    if let Some(spacing) = mesh.spacing {
        write_spacing(f, "Δx", spacing.dx)?;
        if mesh.dimension >= 2 {
            write_spacing(f, "Δy", spacing.dy)?;
        }
        if mesh.dimension >= 3 {
            write_spacing(f, "Δz", spacing.dz)?;
        }
    }
    Ok(())
}

fn write_mesh_warnings(f: &mut fmt::Formatter<'_>, mesh: &MeshDiagnostics) -> fmt::Result {
    for warning in &mesh.warnings {
        writeln!(f, "  warn: {warning}")?;
    }
    Ok(())
}

fn write_boundary_section(
    f: &mut fmt::Formatter<'_>,
    boundary: &[BoundaryPatchSummary],
) -> fmt::Result {
    if boundary.is_empty() {
        writeln!(f, "boundary: (none)")
    } else {
        writeln!(f, "boundary patches ({}):", boundary.len())?;
        for patch in boundary {
            writeln!(
                f,
                "  {:<16} {:<8} faces={}",
                patch.name, patch.kind, patch.faces
            )?;
        }
        Ok(())
    }
}

#[must_use]
pub fn report_structured_mesh(
    source: impl Into<String>,
    mesh: &StructuredMesh,
    boundary: Option<&BoundarySet>,
) -> MeshReport {
    MeshReport {
        source: source.into(),
        mesh: structured_mesh_diagnostics(mesh),
        boundary: summarize_boundary(boundary),
    }
}

#[must_use]
pub fn report_mesh1d(
    source: impl Into<String>,
    mesh: &StructuredMesh1d,
    boundary: Option<&BoundarySet>,
) -> MeshReport {
    MeshReport {
        source: source.into(),
        mesh: mesh1d_diagnostics(mesh),
        boundary: summarize_boundary(boundary),
    }
}

#[must_use]
pub fn report_mesh3d(
    source: impl Into<String>,
    mesh: &StructuredMesh3d,
    boundary: Option<&BoundarySet>,
) -> MeshReport {
    MeshReport {
        source: source.into(),
        mesh: mesh3d_diagnostics(mesh),
        boundary: summarize_boundary(boundary),
    }
}

#[must_use]
pub fn report_case_mesh(
    source: impl Into<String>,
    mesh: &CaseMesh,
    boundary: &BoundarySet,
) -> MeshReport {
    match mesh {
        CaseMesh::Structured1d(m) => report_mesh1d(source, m, Some(boundary)),
        CaseMesh::Structured3d(m) => report_mesh3d(source, m, Some(boundary)),
    }
}

#[cfg(feature = "io-cgns")]
#[must_use]
pub fn report_cgns_zone(loaded: &super::CgnsLoadResult) -> MeshReport {
    let source = format!("CGNS zone {}/{}", loaded.zone.index, loaded.zone.name);
    MeshReport {
        source,
        mesh: structured_mesh_diagnostics(&loaded.mesh),
        boundary: summarize_boundary(Some(&loaded.boundary)),
    }
}

#[cfg(feature = "io-vtk")]
#[must_use]
pub fn report_vts(loaded: &super::VtsLoadResult, path: &std::path::Path) -> MeshReport {
    MeshReport {
        source: format!("VTS {}", path.display()),
        mesh: structured_mesh_diagnostics(&loaded.mesh),
        boundary: Vec::new(),
    }
}

fn summarize_boundary(boundary: Option<&BoundarySet>) -> Vec<BoundaryPatchSummary> {
    boundary
        .map(|set| {
            set.patches()
                .iter()
                .map(|patch| BoundaryPatchSummary {
                    name: patch.name.clone(),
                    kind: boundary_kind_label(&patch.kind).to_string(),
                    faces: patch.face_ids.len(),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn boundary_kind_label(kind: &BoundaryKind) -> &'static str {
    match kind {
        BoundaryKind::Dirichlet { .. } => "Dirichlet",
        BoundaryKind::Neumann { .. } => "Neumann",
        BoundaryKind::Farfield { .. } => "Farfield",
        BoundaryKind::Inlet { .. } => "Inlet",
        BoundaryKind::Outlet { .. } => "Outlet",
        BoundaryKind::Wall { .. } => "Wall",
        BoundaryKind::Symmetry => "Symmetry",
        BoundaryKind::Periodic { .. } => "Periodic",
        BoundaryKind::TurbulentInlet { .. } => "TurbInlet",
    }
}

fn write_axis(
    f: &mut fmt::Formatter<'_>,
    axis: &str,
    range: crate::mesh::CoordRange,
) -> fmt::Result {
    writeln!(f, "  {axis} ∈ [{:.6}, {:.6}]", range.min, range.max)
}

fn write_spacing(
    f: &mut fmt::Formatter<'_>,
    label: &str,
    range: crate::mesh::CoordRange,
) -> fmt::Result {
    if range.max <= 0.0 {
        return Ok(());
    }
    if (range.max - range.min).abs() < 1.0e-12 {
        writeln!(f, "  {label} ≈ {:.6}", range.min)
    } else {
        writeln!(f, "  {label} ∈ [{:.6}, {:.6}]", range.min, range.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::mesh::StructuredMesh3d;

    #[test]
    fn report_formats_bounds_and_patches() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 1, 1.0, 2.0, 0.5).expect("mesh");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "wall",
            Vec::new(),
            BoundaryKind::Wall {
                no_slip: true,
                heat: crate::boundary::WallHeat::Adiabatic,
            },
        )]);
        let report = report_mesh3d("test", &mesh, Some(&boundary));
        let text = report.to_string();
        assert!(text.contains("cells=4 (2×2×1)"));
        assert!(text.contains("x ∈"));
        assert!(text.contains("nz=1"));
        assert!(text.contains("wall"));
    }
}
