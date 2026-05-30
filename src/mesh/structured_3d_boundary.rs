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

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::IMin => "i_min",
            Self::IMax => "i_max",
            Self::JMin => "j_min",
            Self::JMax => "j_max",
            Self::KMin => "k_min",
            Self::KMax => "k_max",
        }
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

    #[must_use]
    pub fn cell_dx_at(&self, i: usize, j: usize, k: usize) -> Real {
        (self.node_x(i + 1, j, k) - self.node_x(i, j, k)).abs()
    }

    #[must_use]
    pub fn cell_dy_at(&self, i: usize, j: usize, k: usize) -> Real {
        (self.node_y(i, j + 1, k) - self.node_y(i, j, k)).abs()
    }

    #[must_use]
    pub fn cell_dz_at(&self, i: usize, j: usize, k: usize) -> Real {
        (self.node_z(i, j, k + 1) - self.node_z(i, j, k)).abs()
    }

    #[must_use]
    pub fn cell_volume_at(&self, i: usize, j: usize, k: usize) -> Real {
        self.cell_dx_at(i, j, k) * self.cell_dy_at(i, j, k) * self.cell_dz_at(i, j, k)
    }

    /// i-方向内界面（cell `i` 与 `i+1` 之间）面积。
    #[must_use]
    pub fn i_face_area_between(&self, i: usize, j: usize, k: usize) -> Real {
        let x = i + 1;
        let dy = (self.node_y(x, j + 1, k) - self.node_y(x, j, k)).abs();
        let dz = (self.node_z(x, j, k + 1) - self.node_z(x, j, k)).abs();
        dy * dz
    }

    /// j-方向内界面（cell `j` 与 `j+1` 之间）面积。
    #[must_use]
    pub fn j_face_area_between(&self, i: usize, j: usize, k: usize) -> Real {
        let y = j + 1;
        let dx = (self.node_x(i + 1, y, k) - self.node_x(i, y, k)).abs();
        let dz = (self.node_z(i, y, k + 1) - self.node_z(i, y, k)).abs();
        dx * dz
    }

    /// k-方向内界面（cell `k` 与 `k+1` 之间）面积。
    #[must_use]
    pub fn k_face_area_between(&self, i: usize, j: usize, k: usize) -> Real {
        let z = k + 1;
        let dx = (self.node_x(i + 1, j, z) - self.node_x(i, j, z)).abs();
        let dy = (self.node_y(i, j + 1, z) - self.node_y(i, j, z)).abs();
        dx * dy
    }

    /// 逻辑边界面面积与法向间距（用于 BC ghost / 边界通量）。
    pub fn boundary_face_geometry(
        &self,
        face: LogicalFace3d,
        i: usize,
        j: usize,
        k: usize,
    ) -> (Vector3, Real, Real) {
        match face {
            LogicalFace3d::IMin => (
                Vector3::new(-1.0, 0.0, 0.0),
                {
                    let dy = (self.node_y(0, j + 1, k) - self.node_y(0, j, k)).abs();
                    let dz = (self.node_z(0, j, k + 1) - self.node_z(0, j, k)).abs();
                    dy * dz
                },
                self.cell_dx_at(i, j, k) * 0.5,
            ),
            LogicalFace3d::IMax => (
                Vector3::new(1.0, 0.0, 0.0),
                {
                    let x = self.nx;
                    let dy = (self.node_y(x, j + 1, k) - self.node_y(x, j, k)).abs();
                    let dz = (self.node_z(x, j, k + 1) - self.node_z(x, j, k)).abs();
                    dy * dz
                },
                self.cell_dx_at(i, j, k) * 0.5,
            ),
            LogicalFace3d::JMin => (
                Vector3::new(0.0, -1.0, 0.0),
                {
                    let dx = (self.node_x(i + 1, 0, k) - self.node_x(i, 0, k)).abs();
                    let dz = (self.node_z(i, 0, k + 1) - self.node_z(i, 0, k)).abs();
                    dx * dz
                },
                self.cell_dy_at(i, j, k) * 0.5,
            ),
            LogicalFace3d::JMax => (
                Vector3::new(0.0, 1.0, 0.0),
                {
                    let y = self.ny;
                    let dx = (self.node_x(i + 1, y, k) - self.node_x(i, y, k)).abs();
                    let dz = (self.node_z(i, y, k + 1) - self.node_z(i, y, k)).abs();
                    dx * dz
                },
                self.cell_dy_at(i, j, k) * 0.5,
            ),
            LogicalFace3d::KMin => (
                Vector3::new(0.0, 0.0, -1.0),
                {
                    let dx = (self.node_x(i + 1, j, 0) - self.node_x(i, j, 0)).abs();
                    let dy = (self.node_y(i, j + 1, 0) - self.node_y(i, j, 0)).abs();
                    dx * dy
                },
                self.cell_dz_at(i, j, k) * 0.5,
            ),
            LogicalFace3d::KMax => (
                Vector3::new(0.0, 0.0, 1.0),
                {
                    let z = self.nz;
                    let dx = (self.node_x(i + 1, j, z) - self.node_x(i, j, z)).abs();
                    let dy = (self.node_y(i, j + 1, z) - self.node_y(i, j, z)).abs();
                    dx * dy
                },
                self.cell_dz_at(i, j, k) * 0.5,
            ),
        }
    }

    /// 各单元 CFL 特征长度：相邻面间距的最小值。
    pub fn cell_cfl_lengths(&self) -> Result<Vec<Real>> {
        let n = self.num_cells();
        let mut lengths = vec![Real::INFINITY; n];
        if self.uses_curvilinear_metrics() {
            self.accumulate_curvilinear_face_lengths(&mut lengths)?;
        } else {
            self.accumulate_cartesian_face_lengths(&mut lengths);
        }
        for h in &mut lengths {
            if !h.is_finite() || *h <= Real::EPSILON {
                return Err(crate::error::AsimuError::Mesh(
                    "单元 CFL 特征长度无效".to_string(),
                ));
            }
        }
        Ok(lengths)
    }

    fn accumulate_cartesian_face_lengths(&self, lengths: &mut [Real]) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let idx = self.cell_index(i, j, k);
                    let hx = self.cell_dx_at(i, j, k);
                    let hy = self.cell_dy_at(i, j, k);
                    let hz = self.cell_dz_at(i, j, k);
                    lengths[idx] = lengths[idx].min(hx).min(hy).min(hz);
                }
            }
        }
    }

    fn accumulate_curvilinear_face_lengths(&self, lengths: &mut [Real]) -> Result<()> {
        let mut track = |owner: usize, neighbor: usize, h: Real| {
            if h > Real::EPSILON && h.is_finite() {
                lengths[owner] = lengths[owner].min(h);
                lengths[neighbor] = lengths[neighbor].min(h);
            }
        };
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx.saturating_sub(1) {
                    let face = self.i_face_metric(i, j, k);
                    let h = face_spacing(
                        self.cell_metric(i, j, k).volume,
                        self.cell_metric(i + 1, j, k).volume,
                        face.area,
                    );
                    track(self.cell_index(i, j, k), self.cell_index(i + 1, j, k), h);
                }
            }
        }
        for k in 0..self.nz {
            for j in 0..self.ny.saturating_sub(1) {
                for i in 0..self.nx {
                    let face = self.j_face_metric(i, j, k);
                    let h = face_spacing(
                        self.cell_metric(i, j, k).volume,
                        self.cell_metric(i, j + 1, k).volume,
                        face.area,
                    );
                    track(self.cell_index(i, j, k), self.cell_index(i, j + 1, k), h);
                }
            }
        }
        for k in 0..self.nz.saturating_sub(1) {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let face = self.k_face_metric(i, j, k);
                    let h = face_spacing(
                        self.cell_metric(i, j, k).volume,
                        self.cell_metric(i, j, k + 1).volume,
                        face.area,
                    );
                    track(self.cell_index(i, j, k), self.cell_index(i, j, k + 1), h);
                }
            }
        }
        Ok(())
    }

    /// 全场最小正单元间距（CFL 用；忽略数值零间距）。
    pub fn min_positive_spacing(&self) -> Result<Real> {
        if self.uses_curvilinear_metrics() {
            return self.min_positive_face_spacing();
        }
        let max_sp = self.cartesian_max_axis_step()?;
        let floor = (max_sp * 1.0e-6).max(1.0e-12);
        let min_sp = self
            .min_cartesian_dx_above(floor)?
            .min(self.min_cartesian_dy_above(floor)?)
            .min(self.min_cartesian_dz_above(floor)?);
        if !min_sp.is_finite() || min_sp <= 0.0 {
            return Err(crate::error::AsimuError::Mesh(
                "网格不存在正单元间距".to_string(),
            ));
        }
        Ok(min_sp)
    }

    fn cartesian_max_axis_step(&self) -> Result<Real> {
        let mut max_sp = 0.0_f64;
        for k in 0..=self.nz {
            for j in 0..=self.ny {
                for i in 0..self.nx {
                    max_sp = max_sp.max(self.cell_dx_at(i, j, k));
                }
            }
        }
        for k in 0..=self.nz {
            for j in 0..self.ny {
                for i in 0..=self.nx {
                    let step = (self.node_y(i, j + 1, k) - self.node_y(i, j, k)).abs();
                    max_sp = max_sp.max(step);
                }
            }
        }
        for k in 0..self.nz {
            for j in 0..=self.ny {
                for i in 0..self.nx {
                    max_sp = max_sp.max(self.cell_dz_at(i, j, k));
                }
            }
        }
        Ok(max_sp)
    }

    fn min_cartesian_dx_above(&self, floor: Real) -> Result<Real> {
        let mut min_sp = Real::INFINITY;
        for k in 0..=self.nz {
            for j in 0..=self.ny {
                for i in 0..self.nx {
                    let step = self.cell_dx_at(i, j, k);
                    if step >= floor {
                        min_sp = min_sp.min(step);
                    }
                }
            }
        }
        Ok(min_sp)
    }

    fn min_cartesian_dy_above(&self, floor: Real) -> Result<Real> {
        let mut min_sp = Real::INFINITY;
        for k in 0..=self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let step = (self.node_y(i, j + 1, k) - self.node_y(i, j, k)).abs();
                    if step >= floor {
                        min_sp = min_sp.min(step);
                    }
                }
            }
        }
        Ok(min_sp)
    }

    fn min_cartesian_dz_above(&self, floor: Real) -> Result<Real> {
        let mut min_sp = Real::INFINITY;
        for k in 0..self.nz {
            for j in 0..=self.ny {
                for i in 0..self.nx {
                    let step = self.cell_dz_at(i, j, k);
                    if step >= floor {
                        min_sp = min_sp.min(step);
                    }
                }
            }
        }
        Ok(min_sp)
    }

    /// 曲线网格：用 \(h_f = (V_{\mathrm{ow}}+V_{\mathrm{nb}})/(2A)\) 估计最小正面间距。
    pub fn min_positive_face_spacing(&self) -> Result<Real> {
        if let Some(spacing) = self.cached_min_face_spacing() {
            return Ok(spacing);
        }
        let hs = self.collect_internal_face_spacings()?;
        let max_h = hs.iter().copied().fold(0.0_f64, Real::max);
        let floor = (max_h * 1.0e-6).max(1.0e-12);
        let min_h = hs
            .into_iter()
            .filter(|h| *h >= floor)
            .fold(Real::INFINITY, Real::min);
        if !min_h.is_finite() || min_h <= 0.0 {
            return Err(crate::error::AsimuError::Mesh(
                "曲线网格不存在正面间距".to_string(),
            ));
        }
        Ok(min_h)
    }

    fn collect_internal_face_spacings(&self) -> Result<Vec<Real>> {
        let mut hs = Vec::new();
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx.saturating_sub(1) {
                    let face = self.i_face_metric(i, j, k);
                    hs.push(face_spacing(
                        self.cell_metric(i, j, k).volume,
                        self.cell_metric(i + 1, j, k).volume,
                        face.area,
                    ));
                }
            }
        }
        for k in 0..self.nz {
            for j in 0..self.ny.saturating_sub(1) {
                for i in 0..self.nx {
                    let face = self.j_face_metric(i, j, k);
                    hs.push(face_spacing(
                        self.cell_metric(i, j, k).volume,
                        self.cell_metric(i, j + 1, k).volume,
                        face.area,
                    ));
                }
            }
        }
        for k in 0..self.nz.saturating_sub(1) {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let face = self.k_face_metric(i, j, k);
                    hs.push(face_spacing(
                        self.cell_metric(i, j, k).volume,
                        self.cell_metric(i, j, k + 1).volume,
                        face.area,
                    ));
                }
            }
        }
        Ok(hs)
    }

    fn face_count(&self, face: LogicalFace3d) -> usize {
        match face {
            LogicalFace3d::IMin | LogicalFace3d::IMax => self.ny * self.nz,
            LogicalFace3d::JMin | LogicalFace3d::JMax => self.nx * self.nz,
            LogicalFace3d::KMin | LogicalFace3d::KMax => self.nx * self.ny,
        }
    }

    pub(crate) fn face_ij(&self, face: LogicalFace3d, local: u32) -> Result<(usize, usize, usize)> {
        let local = local as usize;
        match face {
            LogicalFace3d::IMin | LogicalFace3d::IMax => {
                let ny = self.ny;
                let j = local % ny;
                let k = local / ny;
                if k >= self.nz {
                    return Err(AsimuError::Mesh("面局部索引越界".to_string()));
                }
                let i = if face == LogicalFace3d::IMin {
                    0
                } else {
                    self.nx - 1
                };
                Ok((i, j, k))
            }
            LogicalFace3d::JMin | LogicalFace3d::JMax => {
                let nx = self.nx;
                let i = local % nx;
                let k = local / nx;
                if k >= self.nz {
                    return Err(AsimuError::Mesh("面局部索引越界".to_string()));
                }
                let j = if face == LogicalFace3d::JMin {
                    0
                } else {
                    self.ny - 1
                };
                Ok((i, j, k))
            }
            LogicalFace3d::KMin | LogicalFace3d::KMax => {
                let nx = self.nx;
                let i = local % nx;
                let j = local / nx;
                if j >= self.ny {
                    return Err(AsimuError::Mesh("面局部索引越界".to_string()));
                }
                let k = if face == LogicalFace3d::KMin {
                    0
                } else {
                    self.nz - 1
                };
                Ok((i, j, k))
            }
        }
    }
}

#[must_use]
fn face_spacing(owner_volume: Real, neighbor_volume: Real, area: Real) -> Real {
    if area <= Real::EPSILON {
        Real::INFINITY
    } else {
        (owner_volume + neighbor_volume) / (2.0 * area)
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
        let (logical, local) = LogicalFace3d::decode(face)?;
        let (i, j, k) = self.face_ij(logical, local)?;
        let metric = self.boundary_face_metric(logical, i, j, k);
        let spacing = match logical {
            LogicalFace3d::IMin | LogicalFace3d::IMax => self.cell_dx_at(i, j, k) * 0.5,
            LogicalFace3d::JMin | LogicalFace3d::JMax => self.cell_dy_at(i, j, k) * 0.5,
            LogicalFace3d::KMin | LogicalFace3d::KMax => self.cell_dz_at(i, j, k) * 0.5,
        };
        Ok(FaceGeometry3d {
            normal: metric.normal,
            spacing,
            area: metric.area,
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
