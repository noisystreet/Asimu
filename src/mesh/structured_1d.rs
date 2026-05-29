//! 1D 均匀结构化网格（扩散 benchmark 首版）。

use crate::core::{CellId, FaceId, Real};
use crate::error::{AsimuError, Result};

use super::boundary::BoundaryMesh;

/// 1D 均匀单元网格：`[origin, origin + length]` 划分为 `ncells` 个单元。
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredMesh1d {
    pub name: String,
    pub ncells: usize,
    pub origin: Real,
    pub length: Real,
}

impl StructuredMesh1d {
    pub fn new(name: impl Into<String>, ncells: usize, origin: Real, length: Real) -> Result<Self> {
        if ncells == 0 {
            return Err(AsimuError::Mesh("ncells 必须大于 0".to_string()));
        }
        if length <= 0.0 {
            return Err(AsimuError::Mesh("length 必须大于 0".to_string()));
        }
        Ok(Self {
            name: name.into(),
            ncells,
            origin,
            length,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.ncells
    }

    #[must_use]
    pub fn dx(&self) -> Real {
        self.length / self.ncells as Real
    }

    /// 1D 单元体积（单位展宽）。
    #[must_use]
    pub fn cell_volume(&self) -> Real {
        self.dx()
    }

    /// 1D 面面积（单位展宽）。
    #[must_use]
    pub const fn face_area(&self) -> Real {
        1.0
    }

    /// 左域边界 `FaceId(0)`，右域边界 `FaceId(1)`。
    #[must_use]
    pub const fn left_face() -> FaceId {
        FaceId(0)
    }

    #[must_use]
    pub const fn right_face() -> FaceId {
        FaceId(1)
    }
}

impl BoundaryMesh for StructuredMesh1d {
    fn num_cells(&self) -> usize {
        self.ncells
    }

    fn face_owner(&self, face: FaceId) -> Result<CellId> {
        match face.index() {
            0 => Ok(CellId(0)),
            1 => Ok(CellId((self.ncells - 1) as u32)),
            _ => Err(AsimuError::Mesh(format!(
                "1D 网格无 FaceId({})",
                face.index()
            ))),
        }
    }

    fn face_spacing(&self, face: FaceId) -> Result<Real> {
        match face.index() {
            0 | 1 => Ok(self.dx() * 0.5),
            _ => Err(AsimuError::Mesh(format!(
                "1D 网格无 FaceId({})",
                face.index()
            ))),
        }
    }

    fn face_outward_normal(&self, face: FaceId) -> Result<Real> {
        match face.index() {
            0 => Ok(-1.0),
            1 => Ok(1.0),
            _ => Err(AsimuError::Mesh(format!(
                "1D 网格无 FaceId({})",
                face.index()
            ))),
        }
    }

    fn resolve_logical_boundary(&self, name: &str) -> Result<Vec<FaceId>> {
        match name {
            "left" => Ok(vec![Self::left_face()]),
            "right" => Ok(vec![Self::right_face()]),
            _ => Err(AsimuError::Mesh(format!(
                "1D 网格不识别逻辑边界 \"{name}\"（支持 left / right）"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_logical_boundaries() {
        let mesh = StructuredMesh1d::new("line", 8, 0.0, 1.0).expect("mesh");
        let left = mesh.resolve_logical_boundary("left").expect("left");
        assert_eq!(left, vec![FaceId(0)]);
        assert_eq!(mesh.face_owner(left[0]).expect("owner"), CellId(0));

        let right = mesh.resolve_logical_boundary("right").expect("right");
        assert_eq!(right, vec![FaceId(1)]);
        assert_eq!(mesh.face_owner(right[0]).expect("owner"), CellId(7));
    }
}
