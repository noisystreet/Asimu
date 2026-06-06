//! 结构化与非结构混合单元 3D FVM 网格。

mod boundary;
mod check;
mod check_unstructured;
mod diagnostics;
mod metrics;
mod multiblock;
mod structured;
mod structured_1d;
mod structured_3d_boundary;
mod unstructured;

pub use boundary::BoundaryMesh;
pub use check::{
    BoundaryPatchReport, CheckFinding, CheckSeverity, MeshCheckOptions, MeshCheckReport,
    MeshCheckReportDisplay, check_mesh1d, check_mesh2d, check_mesh3d, check_multiblock_mesh3d,
    write_mesh_check_report,
};
pub use check_unstructured::check_unstructured_mesh3d;
pub use diagnostics::{
    CoordRange, MeshBounds, MeshDiagnostics, SpacingStats, mesh1d_diagnostics, mesh2d_diagnostics,
    mesh3d_diagnostics, multiblock_mesh3d_diagnostics, structured_mesh_diagnostics,
    unstructured_mesh3d_diagnostics,
};
pub use metrics::{CellMetric, FaceMetric, MeshMetricMode, MetricCache3d, boundary_cell_spacing};
pub use multiblock::{
    MultiBlockStructuredMesh3d, StructuredBlock3d, StructuredBlockInterface3d,
    StructuredIndexRange3d,
};
pub use structured::{StructuredMesh, StructuredMesh2d, StructuredMesh3d};
pub use structured_1d::StructuredMesh1d;
pub use structured_3d_boundary::{BoundaryMesh3d, FaceGeometry3d, LogicalFace3d};
pub use unstructured::{CellKind, UnstructuredCell, UnstructuredMesh3d};

use crate::error::{AsimuError, Result};

/// 最小网格描述，用于骨架验证与集成测试。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mesh {
    pub name: String,
    pub cell_count: usize,
}

impl Mesh {
    pub fn new(name: impl Into<String>, cell_count: usize) -> Result<Self> {
        if cell_count == 0 {
            return Err(AsimuError::Mesh("cell_count 必须大于 0".to_string()));
        }
        Ok(Self {
            name: name.into(),
            cell_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_mesh() {
        let err = Mesh::new("empty", 0).unwrap_err();
        assert!(matches!(err, AsimuError::Mesh(_)));
    }
}
