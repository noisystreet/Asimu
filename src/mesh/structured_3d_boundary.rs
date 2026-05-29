//! 3D 结构化网格边界面拓扑与 `BoundaryMesh` 实现。

use crate::core::{CellId, FaceId, Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::mesh::boundary::BoundaryMesh;

use super::StructuredMesh3d;

/// 3D 逻辑边界面（CGNS / CFL3D 命名）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LogicalFace3d {
    IMin = 0,
    IMax = 1,
    JMin = 2,
    JMax = 3,
    KMin = 4,
    KMax = 5,
}

impl LogicalFace3d {
    pub const COUNT: u32 = 6;

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "i_min" | "imin" | "left" => Some(Self::IMin),
            "i_max" | "imax" | "right" => Some(Self::IMax),
            "j_min" | "jmin" | "bottom" => Some(Self::JMin),
            "j_max" | "jmax" | "top" => Some(Self::JMax),
            "k_min" | "kmin" | "front" => Some(Self::KMin),
            "k_max" | "kmax" | "back" => Some(Self::KMax),
            _ => None,
        }
    }

    #[must_use]
    pub const fn tag(self) -> u32 {
        self as u32
    }

    pub fn encode(self, local_index: u32) -> FaceId {
        FaceId(self.tag() * 1_000_000 + local_index)
    }

    pub fn decode(face: FaceId) -> Result<(Self, u32)> {
        let raw = face.index();
        let tag = raw / 1_000_000;
        let local = raw % 1_000_000;
        let logical = match tag {
            0 => LogicalFace3d::IMin,
            1 => LogicalFace3d::IMax,
            2 => LogicalFace3d::JMin,
            3 => LogicalFace3d::JMax,
            4 => LogicalFace3d::KMin,
            5 => LogicalFace3d::KMax,
            _ => {
                return Err(AsimuError::Mesh(format!("无效 3D FaceId({raw})")));
            }
        };
        Ok((logical, local))
    }
}

/// 3D 边界面几何。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceGeometry3d {
    pub normal: Vector3,
    pub spacing: Real,
    pub area: Real,
}

/// 3D 边界网格接口。
pub trait BoundaryMesh3d: BoundaryMesh {
    fn structured_3d(&self) -> Result<&StructuredMesh3d>;
    fn face_geometry_3d(&self, face: FaceId) -> Result<FaceGeometry3d>;
    fn face_normal_3d(&self, face: FaceId) -> Result<Vector3> {
        Ok(self.face_geometry_3d(face)?.normal)
    }
}

impl StructuredMesh3d {
    /// 均匀盒子网格 \([0,lx]×[0,ly]×[0,lz]\)。
    pub fn uniform_box(
        name: impl Into<String>,
        nx: usize,
        ny: usize,
        nz: usize,
        lx: Real,
        ly: Real,
        lz: Real,
    ) -> Result<Self> {
        if lx <= 0.0 || ly <= 0.0 || lz <= 0.0 {
            return Err(AsimuError::Mesh("盒子尺寸必须大于 0".to_string()));
        }
        let mut points_x = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
        let mut points_y = Vec::with_capacity(points_x.capacity());
        let mut points_z = Vec::with_capacity(points_x.capacity());
        let dx = lx / nx as Real;
        let dy = ly / ny as Real;
        let dz = lz / nz as Real;
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    points_x.push(i as Real * dx);
                    points_y.push(j as Real * dy);
                    points_z.push(k as Real * dz);
                }
            }
        }
        Self::new(name, nx, ny, nz, points_x, points_y, points_z)
    }

    #[must_use]
    pub fn cell_dx(&self) -> Real {
        let i0 = self.node_index(0, 0, 0);
        let i1 = self.node_index(1, 0, 0);
        (self.points_x[i1] - self.points_x[i0]).abs()
    }

    #[must_use]
    pub fn cell_dy(&self) -> Real {
        let i0 = self.node_index(0, 0, 0);
        let i1 = self.node_index(0, 1, 0);
        (self.points_y[i1] - self.points_y[i0]).abs()
    }

    #[must_use]
    pub fn cell_dz(&self) -> Real {
        let i0 = self.node_index(0, 0, 0);
        let i1 = self.node_index(0, 0, 1);
        (self.points_z[i1] - self.points_z[i0]).abs()
    }

    fn face_count(&self, face: LogicalFace3d) -> usize {
        match face {
            LogicalFace3d::IMin | LogicalFace3d::IMax => self.ny * self.nz,
            LogicalFace3d::JMin | LogicalFace3d::JMax => self.nx * self.nz,
            LogicalFace3d::KMin | LogicalFace3d::KMax => self.nx * self.ny,
        }
    }

    fn face_ij(&self, face: LogicalFace3d, local: u32) -> Result<(usize, usize, usize)> {
        let local = local as usize;
        match face {
            LogicalFace3d::IMin | LogicalFace3d::IMax => {
                let ny = self.ny;
                let j = local % ny;
                let k = local / ny;
                if k >= self.nz {
                    return Err(AsimuError::Mesh("面局部索引越界".to_string()));
                }
                let i = if face == LogicalFace3d::IMin { 0 } else { self.nx - 1 };
                Ok((i, j, k))
            }
            LogicalFace3d::JMin | LogicalFace3d::JMax => {
                let nx = self.nx;
                let i = local % nx;
                let k = local / nx;
                if k >= self.nz {
                    return Err(AsimuError::Mesh("面局部索引越界".to_string()));
                }
                let j = if face == LogicalFace3d::JMin { 0 } else { self.ny - 1 };
                Ok((i, j, k))
            }
            LogicalFace3d::KMin | LogicalFace3d::KMax => {
                let nx = self.nx;
                let i = local % nx;
                let j = local / nx;
                if j >= self.ny {
                    return Err(AsimuError::Mesh("面局部索引越界".to_string()));
                }
                let k = if face == LogicalFace3d::KMin { 0 } else { self.nz - 1 };
                Ok((i, j, k))
            }
        }
    }

    fn cell_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + j * self.nx + k * self.nx * self.ny
    }
}

impl BoundaryMesh for StructuredMesh3d {
    fn num_cells(&self) -> usize {
        self.num_cells()
    }

    fn face_owner(&self, face: FaceId) -> Result<CellId> {
        let (logical, local) = LogicalFace3d::decode(face)?;
        let (i, j, k) = self.face_ij(logical, local)?;
        Ok(CellId(self.cell_index(i, j, k) as u32))
    }

    fn face_spacing(&self, face: FaceId) -> Result<Real> {
        let (logical, _) = LogicalFace3d::decode(face)?;
        Ok(match logical {
            LogicalFace3d::IMin | LogicalFace3d::IMax => self.cell_dx() * 0.5,
            LogicalFace3d::JMin | LogicalFace3d::JMax => self.cell_dy() * 0.5,
            LogicalFace3d::KMin | LogicalFace3d::KMax => self.cell_dz() * 0.5,
        })
    }

    fn face_outward_normal(&self, face: FaceId) -> Result<Real> {
        let n = self.face_normal_3d(face)?;
        Ok(n.x)
    }

    fn resolve_logical_boundary(&self, name: &str) -> Result<Vec<FaceId>> {
        let logical = LogicalFace3d::from_name(name).ok_or_else(|| {
            AsimuError::Mesh(format!(
                "3D 网格不识别逻辑边界 \"{name}\"（支持 i_min/i_max/j_min/j_max/k_min/k_max）"
            ))
        })?;
        let count = self.face_count(logical);
        Ok((0..count as u32)
            .map(|local| logical.encode(local))
            .collect())
    }
}

impl BoundaryMesh3d for StructuredMesh3d {
    fn structured_3d(&self) -> Result<&StructuredMesh3d> {
        Ok(self)
    }

    fn face_geometry_3d(&self, face: FaceId) -> Result<FaceGeometry3d> {
        let (logical, _) = LogicalFace3d::decode(face)?;
        let (normal, area, spacing) = match logical {
            LogicalFace3d::IMin => (
                Vector3::new(-1.0, 0.0, 0.0),
                self.cell_dy() * self.cell_dz(),
                self.cell_dx() * 0.5,
            ),
            LogicalFace3d::IMax => (
                Vector3::new(1.0, 0.0, 0.0),
                self.cell_dy() * self.cell_dz(),
                self.cell_dx() * 0.5,
            ),
            LogicalFace3d::JMin => (
                Vector3::new(0.0, -1.0, 0.0),
                self.cell_dx() * self.cell_dz(),
                self.cell_dy() * 0.5,
            ),
            LogicalFace3d::JMax => (
                Vector3::new(0.0, 1.0, 0.0),
                self.cell_dx() * self.cell_dz(),
                self.cell_dy() * 0.5,
            ),
            LogicalFace3d::KMin => (
                Vector3::new(0.0, 0.0, -1.0),
                self.cell_dx() * self.cell_dy(),
                self.cell_dz() * 0.5,
            ),
            LogicalFace3d::KMax => (
                Vector3::new(0.0, 0.0, 1.0),
                self.cell_dx() * self.cell_dy(),
                self.cell_dz() * 0.5,
            ),
        };
        Ok(FaceGeometry3d {
            normal,
            spacing,
            area,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_imin_faces() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 3, 4, 1.0, 1.0, 1.0).expect("mesh");
        let faces = mesh.resolve_logical_boundary("i_min").expect("faces");
        assert_eq!(faces.len(), 3 * 4);
        assert_eq!(mesh.face_owner(faces[0]).expect("owner"), CellId(0));
    }
}
