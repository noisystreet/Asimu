//! 结构化网格（1D / 2D / 3D FVM）。

mod boundary;
mod structured;
mod structured_1d;
mod structured_3d_boundary;

pub use boundary::BoundaryMesh;
pub use structured::{StructuredMesh, StructuredMesh2d, StructuredMesh3d};
pub use structured_1d::StructuredMesh1d;
pub use structured_3d_boundary::{BoundaryMesh3d, FaceGeometry3d, LogicalFace3d};

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
