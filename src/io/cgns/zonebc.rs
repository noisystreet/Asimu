//! CGNS ZoneBC → `BoundaryPatch` 映射。

use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::core::FaceId;
use crate::error::{AsimuError, Result};
use crate::mesh::{LogicalFace3d, StructuredMesh3d};

/// CGNS PointRange 为 1-based 顶点索引，顺序 `(imin,jmin,kmin,imax,jmax,kmax)`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CgnsPointRange {
    pub imin: i32,
    pub imax: i32,
    pub jmin: i32,
    pub jmax: i32,
    pub kmin: i32,
    pub kmax: i32,
}

impl CgnsPointRange {
    /// 将 PointRange 转为逻辑面 + 局部面 ID 列表（结构化 zone）。
    pub fn to_face_ids(&self, mesh: &StructuredMesh3d) -> Result<Vec<FaceId>> {
        let nx = mesh.nx as i32;
        let ny = mesh.ny as i32;
        let nz = mesh.nz as i32;

        let face = detect_logical_face(self, nx, ny, nz)?;
        let mut faces = Vec::new();
        match face {
            LogicalFace3d::IMin | LogicalFace3d::IMax => {
                let j_end = cell_index_end(self.jmax, ny);
                let k_end = cell_index_end(self.kmax, nz);
                for k in self.kmin..=k_end {
                    for j in self.jmin..=j_end {
                        let local = (j - 1) + (k - 1) * ny;
                        faces.push(face.encode(local as u32));
                    }
                }
            }
            LogicalFace3d::JMin | LogicalFace3d::JMax => {
                let i_end = cell_index_end(self.imax, nx);
                let k_end = cell_index_end(self.kmax, nz);
                for k in self.kmin..=k_end {
                    for i in self.imin..=i_end {
                        let local = (i - 1) + (k - 1) * nx;
                        faces.push(face.encode(local as u32));
                    }
                }
            }
            LogicalFace3d::KMin | LogicalFace3d::KMax => {
                let i_end = cell_index_end(self.imax, nx);
                let j_end = cell_index_end(self.jmax, ny);
                for j in self.jmin..=j_end {
                    for i in self.imin..=i_end {
                        let local = (i - 1) + (j - 1) * nx;
                        faces.push(face.encode(local as u32));
                    }
                }
            }
        }
        if faces.is_empty() {
            return Err(AsimuError::Boundary(
                "CGNS PointRange 未产生任何面".to_string(),
            ));
        }
        Ok(faces)
    }
}

/// CGNS PointRange 上界可能是 `n+1`（顶点）或 `n`（单元）；统一为单元索引上界。
fn cell_index_end(max_index: i32, n_cells: i32) -> i32 {
    if max_index == n_cells + 1 {
        n_cells
    } else {
        max_index.min(n_cells)
    }
}

fn detect_logical_face(range: &CgnsPointRange, nx: i32, ny: i32, nz: i32) -> Result<LogicalFace3d> {
    if range.imin == 1 && range.imax == 1 {
        return Ok(LogicalFace3d::IMin);
    }
    if range.imin == nx + 1 && range.imax == nx + 1 {
        return Ok(LogicalFace3d::IMax);
    }
    if range.jmin == 1 && range.jmax == 1 {
        return Ok(LogicalFace3d::JMin);
    }
    if range.jmin == ny + 1 && range.jmax == ny + 1 {
        return Ok(LogicalFace3d::JMax);
    }
    if range.kmin == 1 && range.kmax == 1 {
        return Ok(LogicalFace3d::KMin);
    }
    if range.kmin == nz + 1 && range.kmax == nz + 1 {
        return Ok(LogicalFace3d::KMax);
    }
    Err(AsimuError::Boundary(format!(
        "无法识别 CGNS PointRange 对应逻辑面: {:?}",
        (
            range.imin, range.imax, range.jmin, range.jmax, range.kmin, range.kmax
        )
    )))
}

/// 由 CGNS BC 元数据构造 patch。
pub fn patch_from_cgns(
    name: impl Into<String>,
    bctype: i32,
    bctype_label: &str,
    range: CgnsPointRange,
    mesh: &StructuredMesh3d,
) -> Result<BoundaryPatch> {
    let name = name.into();
    let kind = BoundaryKind::from_cgns_bctype(bctype, bctype_label);
    let face_ids = range.to_face_ids(mesh)?;
    Ok(BoundaryPatch::new(name, face_ids, kind))
}

/// 合并多个 CGNS BC 为 `BoundarySet`。
pub fn boundary_set_from_cgns(patches: Vec<BoundaryPatch>) -> BoundarySet {
    BoundarySet::new(patches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imin_range_maps_faces() {
        let mesh = StructuredMesh3d::uniform_box("b", 2, 3, 4, 1.0, 1.0, 1.0).expect("mesh");
        let range = CgnsPointRange {
            imin: 1,
            imax: 1,
            jmin: 1,
            jmax: 3,
            kmin: 1,
            kmax: 4,
        };
        let faces = range.to_face_ids(&mesh).expect("faces");
        assert_eq!(faces.len(), 3 * 4);
    }
}
