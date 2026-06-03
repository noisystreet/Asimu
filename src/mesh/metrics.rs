//! 3D 贴体 / 曲线坐标网格几何度量（单元体积、面面积向量）。
//!
//! 理论：[`docs/theory/curvilinear_metrics.md`](../../docs/theory/curvilinear_metrics.md) §3

#[path = "metrics_cache.rs"]
mod metrics_cache;

use crate::core::{Real, Vector3};
use crate::error::{AsimuError, Result};

use super::LogicalFace3d;
use super::StructuredMesh3d;

/// 网格几何度量模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshMetricMode {
    /// 逻辑 Δx·Δy·Δz 与坐标轴法向（均匀笛卡尔网格）。
    #[default]
    Cartesian,
    /// 由节点坐标计算 \(V\)、\(\mathbf{S}\)（CGNS 贴体网格）。
    Curvilinear,
}

/// 控制体几何度量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellMetric {
    pub volume: Real,
    pub center: Vector3,
}

/// 面几何度量（owner → neighbor 或边界面 outward）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceMetric {
    pub area_vector: Vector3,
    pub area: Real,
    pub normal: Vector3,
    /// 四边形面四顶点算术平均（几何面心）。
    pub center: Vector3,
}

impl FaceMetric {
    #[must_use]
    pub fn from_area_vector(area_vector: Vector3) -> Self {
        Self::from_area_vector_and_center(area_vector, Vector3::new(0.0, 0.0, 0.0))
    }

    #[must_use]
    pub fn from_area_vector_and_center(area_vector: Vector3, center: Vector3) -> Self {
        let area = area_vector.magnitude();
        let normal = if area > Real::EPSILON {
            Vector3::new(
                area_vector.x / area,
                area_vector.y / area,
                area_vector.z / area,
            )
        } else {
            Vector3::new(0.0, 0.0, 0.0)
        };
        Self {
            area_vector,
            area,
            normal,
            center,
        }
    }
}

/// 边界面 owner 单元中心到面心的法向距离（BC ghost / 粘性壁面 \(\delta\)）。
#[must_use]
pub fn boundary_cell_spacing(cell_center: Vector3, face: &FaceMetric, cell_volume: Real) -> Real {
    let dr = vec_sub(face.center, cell_center);
    let proj = (dr.x * face.normal.x + dr.y * face.normal.y + dr.z * face.normal.z).abs();
    if proj > Real::EPSILON {
        return proj;
    }
    if face.area > Real::EPSILON {
        return cell_volume / (2.0 * face.area);
    }
    Real::EPSILON
}

/// 预计算的 3D 曲线网格度量（加载或 `scale` 后构建，求解循环内只读）。
#[derive(Debug, Clone, PartialEq)]
pub struct MetricCache3d {
    cells: Vec<CellMetric>,
    i_faces: Vec<FaceMetric>,
    j_faces: Vec<FaceMetric>,
    k_faces: Vec<FaceMetric>,
    boundary_imin: Vec<FaceMetric>,
    boundary_imax: Vec<FaceMetric>,
    boundary_jmin: Vec<FaceMetric>,
    boundary_jmax: Vec<FaceMetric>,
    boundary_kmin: Vec<FaceMetric>,
    boundary_kmax: Vec<FaceMetric>,
    min_face_spacing: Real,
}

impl StructuredMesh3d {
    #[must_use]
    pub const fn metric_mode(&self) -> MeshMetricMode {
        self.metric_mode
    }

    #[must_use]
    pub fn metric_cache(&self) -> Option<&MetricCache3d> {
        self.metric_cache.as_ref()
    }

    pub fn set_metric_mode(&mut self, mode: MeshMetricMode) {
        if self.metric_mode != mode {
            self.metric_mode = mode;
            self.metric_cache = None;
        }
    }

    /// 曲线模式下预计算全部单元/面度量；笛卡尔模式清除缓存。
    pub fn rebuild_metric_cache_if_needed(&mut self) -> Result<()> {
        if self.uses_curvilinear_metrics() {
            self.build_metric_cache()?;
        } else {
            self.metric_cache = None;
        }
        Ok(())
    }

    /// 强制构建曲线 metric 缓存（`metric_mode` 须为 `Curvilinear`）。
    pub fn build_metric_cache(&mut self) -> Result<()> {
        if !self.uses_curvilinear_metrics() {
            return Err(AsimuError::Mesh(
                "仅 Curvilinear 模式可构建 MetricCache".to_string(),
            ));
        }
        self.metric_cache = Some(metrics_cache::build_curvilinear_metric_cache(self)?);
        Ok(())
    }

    #[must_use]
    pub fn uses_curvilinear_metrics(&self) -> bool {
        self.metric_mode == MeshMetricMode::Curvilinear
    }

    /// 曲线网格预计算的最小面间距（须已 `build_metric_cache`）。
    #[must_use]
    pub fn cached_min_face_spacing(&self) -> Option<Real> {
        self.metric_cache
            .as_ref()
            .map(|cache| cache.min_face_spacing)
    }

    /// 单元 `(i,j,k)` 体积与中心。
    #[must_use]
    pub fn cell_metric(&self, i: usize, j: usize, k: usize) -> CellMetric {
        if let Some(cache) = &self.metric_cache {
            return cache.cells[self.cell_index(i, j, k)];
        }
        if self.uses_curvilinear_metrics() {
            curvilinear_cell_metric(self, i, j, k)
        } else {
            CellMetric {
                volume: self.cell_volume_at(i, j, k),
                center: cartesian_cell_center(self, i, j, k),
            }
        }
    }

    /// i 内界面（cell `i` 与 `i+1` 之间，法向 owner → neighbor 为 +i）。
    #[must_use]
    pub fn i_face_metric(&self, i: usize, j: usize, k: usize) -> FaceMetric {
        if let Some(cache) = &self.metric_cache {
            return cache.i_faces[i_face_cache_index(self.nx, self.ny, i, j, k)];
        }
        if self.uses_curvilinear_metrics() {
            compute_i_face_metric(self, i, j, k)
        } else {
            let x = i + 1;
            FaceMetric::from_area_vector_and_center(
                Vector3::new(self.i_face_area_between(i, j, k), 0.0, 0.0),
                quad_vertex_center(
                    node_vec(self, x, j, k),
                    node_vec(self, x, j + 1, k),
                    node_vec(self, x, j + 1, k + 1),
                    node_vec(self, x, j, k + 1),
                ),
            )
        }
    }

    /// j 内界面（cell `j` 与 `j+1` 之间，法向 +j）。
    #[must_use]
    pub fn j_face_metric(&self, i: usize, j: usize, k: usize) -> FaceMetric {
        if let Some(cache) = &self.metric_cache {
            return cache.j_faces[j_face_cache_index(self.nx, self.ny, i, j, k)];
        }
        if self.uses_curvilinear_metrics() {
            compute_j_face_metric(self, i, j, k)
        } else {
            let y = j + 1;
            FaceMetric::from_area_vector_and_center(
                Vector3::new(0.0, self.j_face_area_between(i, j, k), 0.0),
                quad_vertex_center(
                    node_vec(self, i, y, k),
                    node_vec(self, i + 1, y, k),
                    node_vec(self, i + 1, y, k + 1),
                    node_vec(self, i, y, k + 1),
                ),
            )
        }
    }

    /// k 内界面（cell `k` 与 `k+1` 之间，法向 +k）。
    #[must_use]
    pub fn k_face_metric(&self, i: usize, j: usize, k: usize) -> FaceMetric {
        if let Some(cache) = &self.metric_cache {
            return cache.k_faces[k_face_cache_index(self.nx, self.ny, i, j, k)];
        }
        if self.uses_curvilinear_metrics() {
            compute_k_face_metric(self, i, j, k)
        } else {
            let z = k + 1;
            FaceMetric::from_area_vector_and_center(
                Vector3::new(0.0, 0.0, self.k_face_area_between(i, j, k)),
                quad_vertex_center(
                    node_vec(self, i, j, z),
                    node_vec(self, i + 1, j, z),
                    node_vec(self, i + 1, j + 1, z),
                    node_vec(self, i, j + 1, z),
                ),
            )
        }
    }

    /// 逻辑边界面度量（法向指向网格外侧）。
    #[must_use]
    pub fn boundary_face_metric(
        &self,
        face: LogicalFace3d,
        i: usize,
        j: usize,
        k: usize,
    ) -> FaceMetric {
        if let Some(cache) = &self.metric_cache {
            return cache.boundary_face(face, self.nx, self.ny, i, j, k);
        }
        if self.uses_curvilinear_metrics() {
            curvilinear_boundary_face_metric(self, face, i, j, k)
        } else {
            let (normal, area, _spacing) = self.boundary_face_geometry(face, i, j, k);
            let [v0, v1, v2, v3] = boundary_quad_vertices(self, face, i, j, k);
            FaceMetric::from_area_vector_and_center(
                Vector3::new(normal.x * area, normal.y * area, normal.z * area),
                quad_vertex_center(v0, v1, v2, v3),
            )
        }
    }
}

impl MetricCache3d {
    fn boundary_face(
        &self,
        face: LogicalFace3d,
        nx: usize,
        ny: usize,
        i: usize,
        j: usize,
        k: usize,
    ) -> FaceMetric {
        let _ = (i, j, k);
        match face {
            LogicalFace3d::IMin => self.boundary_imin[j + k * ny],
            LogicalFace3d::IMax => self.boundary_imax[j + k * ny],
            LogicalFace3d::JMin => self.boundary_jmin[i + k * nx],
            LogicalFace3d::JMax => self.boundary_jmax[i + k * nx],
            LogicalFace3d::KMin => self.boundary_kmin[i + j * nx],
            LogicalFace3d::KMax => self.boundary_kmax[i + j * nx],
        }
    }
}

#[must_use]
fn cell_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
    i + j * nx + k * nx * ny
}

#[must_use]
fn i_face_cache_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
    i + j * nx.saturating_sub(1) + k * nx.saturating_sub(1) * ny
}

#[must_use]
fn j_face_cache_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
    i + j * nx + k * nx * ny.saturating_sub(1)
}

#[must_use]
fn k_face_cache_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
    i + j * nx + k * nx * ny
}

#[must_use]
fn face_spacing(owner_volume: Real, neighbor_volume: Real, area: Real) -> Real {
    if area <= Real::EPSILON {
        Real::INFINITY
    } else {
        (owner_volume + neighbor_volume) / (2.0 * area)
    }
}

#[must_use]
fn compute_i_face_metric(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> FaceMetric {
    let x = i + 1;
    let v00 = node_vec(mesh, x, j, k);
    let v10 = node_vec(mesh, x, j + 1, k);
    let v11 = node_vec(mesh, x, j + 1, k + 1);
    let v01 = node_vec(mesh, x, j, k + 1);
    let area_vector = orient_internal_face_area_vector(
        mesh,
        i,
        j,
        k,
        i + 1,
        j,
        k,
        quad_area_vector(v00, v10, v11, v01),
    );
    FaceMetric::from_area_vector_and_center(area_vector, quad_vertex_center(v00, v10, v11, v01))
}

#[must_use]
fn compute_j_face_metric(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> FaceMetric {
    let y = j + 1;
    let v00 = node_vec(mesh, i, y, k);
    let v10 = node_vec(mesh, i + 1, y, k);
    let v11 = node_vec(mesh, i + 1, y, k + 1);
    let v01 = node_vec(mesh, i, y, k + 1);
    let area_vector = orient_internal_face_area_vector(
        mesh,
        i,
        j,
        k,
        i,
        j + 1,
        k,
        quad_area_vector(v00, v10, v11, v01),
    );
    FaceMetric::from_area_vector_and_center(area_vector, quad_vertex_center(v00, v10, v11, v01))
}

#[must_use]
fn compute_k_face_metric(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> FaceMetric {
    let z = k + 1;
    let v00 = node_vec(mesh, i, j, z);
    let v10 = node_vec(mesh, i + 1, j, z);
    let v11 = node_vec(mesh, i + 1, j + 1, z);
    let v01 = node_vec(mesh, i, j + 1, z);
    let area_vector = orient_internal_face_area_vector(
        mesh,
        i,
        j,
        k,
        i,
        j,
        k + 1,
        quad_area_vector(v00, v10, v11, v01),
    );
    FaceMetric::from_area_vector_and_center(area_vector, quad_vertex_center(v00, v10, v11, v01))
}

/// 内界面面积向量与 owner→neighbor 方向对齐（贴体网格顶点顺序可能反号）。
#[allow(clippy::too_many_arguments)]
fn orient_internal_face_area_vector(
    mesh: &StructuredMesh3d,
    owner_i: usize,
    owner_j: usize,
    owner_k: usize,
    neighbor_i: usize,
    neighbor_j: usize,
    neighbor_k: usize,
    mut area_vector: Vector3,
) -> Vector3 {
    let owner_center = curvilinear_cell_metric(mesh, owner_i, owner_j, owner_k).center;
    let neighbor_center = curvilinear_cell_metric(mesh, neighbor_i, neighbor_j, neighbor_k).center;
    let to_neighbor = vec_sub(neighbor_center, owner_center);
    if to_neighbor.x * area_vector.x + to_neighbor.y * area_vector.y + to_neighbor.z * area_vector.z
        < 0.0
    {
        area_vector = Vector3::new(-area_vector.x, -area_vector.y, -area_vector.z);
    }
    area_vector
}

fn node_vec(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> Vector3 {
    Vector3::new(
        mesh.node_x(i, j, k),
        mesh.node_y(i, j, k),
        mesh.node_z(i, j, k),
    )
}

fn cartesian_cell_center(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> Vector3 {
    let mut center = Vector3::new(0.0, 0.0, 0.0);
    for di in 0..=1 {
        for dj in 0..=1 {
            for dk in 0..=1 {
                let p = node_vec(mesh, i + di, j + dj, k + dk);
                center.x += p.x;
                center.y += p.y;
                center.z += p.z;
            }
        }
    }
    let scale = 1.0 / 8.0;
    Vector3::new(center.x * scale, center.y * scale, center.z * scale)
}

fn curvilinear_cell_metric(mesh: &StructuredMesh3d, i: usize, j: usize, k: usize) -> CellMetric {
    let c000 = node_vec(mesh, i, j, k);
    let c100 = node_vec(mesh, i + 1, j, k);
    let c110 = node_vec(mesh, i + 1, j + 1, k);
    let c010 = node_vec(mesh, i, j + 1, k);
    let c001 = node_vec(mesh, i, j, k + 1);
    let c101 = node_vec(mesh, i + 1, j, k + 1);
    let c111 = node_vec(mesh, i + 1, j + 1, k + 1);
    let c011 = node_vec(mesh, i, j + 1, k + 1);

    let center = Vector3::new(
        (c000.x + c100.x + c110.x + c010.x + c001.x + c101.x + c111.x + c011.x) / 8.0,
        (c000.y + c100.y + c110.y + c010.y + c001.y + c101.y + c111.y + c011.y) / 8.0,
        (c000.z + c100.z + c110.z + c010.z + c001.z + c101.z + c111.z + c011.z) / 8.0,
    );

    let mut volume = 0.0;
    volume += tet_volume(center, c000, c010, c011);
    volume += tet_volume(center, c000, c011, c001);
    volume += tet_volume(center, c100, c101, c111);
    volume += tet_volume(center, c100, c111, c110);
    volume += tet_volume(center, c000, c001, c101);
    volume += tet_volume(center, c000, c101, c100);
    volume += tet_volume(center, c010, c110, c111);
    volume += tet_volume(center, c010, c111, c011);
    volume += tet_volume(center, c001, c011, c111);
    volume += tet_volume(center, c001, c111, c101);
    volume += tet_volume(center, c000, c100, c110);
    volume += tet_volume(center, c000, c110, c010);

    CellMetric { volume, center }
}

/// 边界面 owner 单元在域内的唯一邻居 `(i,j,k)`。
fn boundary_interior_neighbor(
    face: LogicalFace3d,
    i: usize,
    j: usize,
    k: usize,
) -> (usize, usize, usize) {
    match face {
        LogicalFace3d::IMin => (i + 1, j, k),
        LogicalFace3d::IMax => (i - 1, j, k),
        LogicalFace3d::JMin => (i, j + 1, k),
        LogicalFace3d::JMax => (i, j - 1, k),
        LogicalFace3d::KMin => (i, j, k + 1),
        LogicalFace3d::KMax => (i, j, k - 1),
    }
}

fn curvilinear_boundary_face_metric(
    mesh: &StructuredMesh3d,
    face: LogicalFace3d,
    i: usize,
    j: usize,
    k: usize,
) -> FaceMetric {
    let [v0, v1, v2, v3] = boundary_quad_vertices(mesh, face, i, j, k);
    let mut area_vector = quad_area_vector(v0, v1, v2, v3);
    // 贴体网格上逻辑边界面可能物理倾斜；用 owner→域内邻居修正外向法向。
    // 准 2D（`nz==1`）时 K 面无域内邻居，跳过 K 面修正。
    let include_k = mesh.nz > 1;
    if matches!(
        face,
        LogicalFace3d::IMin | LogicalFace3d::IMax | LogicalFace3d::JMin | LogicalFace3d::JMax
    ) || (include_k && matches!(face, LogicalFace3d::KMin | LogicalFace3d::KMax))
    {
        let owner_center = curvilinear_cell_metric(mesh, i, j, k).center;
        let (ni, nj, nk) = boundary_interior_neighbor(face, i, j, k);
        let to_interior = vec_sub(
            curvilinear_cell_metric(mesh, ni, nj, nk).center,
            owner_center,
        );
        if to_interior.x * area_vector.x
            + to_interior.y * area_vector.y
            + to_interior.z * area_vector.z
            > 0.0
        {
            area_vector = Vector3::new(-area_vector.x, -area_vector.y, -area_vector.z);
        }
    }
    FaceMetric::from_area_vector_and_center(area_vector, quad_vertex_center(v0, v1, v2, v3))
}

fn boundary_quad_vertices(
    mesh: &StructuredMesh3d,
    face: LogicalFace3d,
    i: usize,
    j: usize,
    k: usize,
) -> [Vector3; 4] {
    let _ = (i, j, k);
    match face {
        LogicalFace3d::IMin => {
            let x = 0;
            [
                node_vec(mesh, x, j + 1, k),
                node_vec(mesh, x, j, k),
                node_vec(mesh, x, j, k + 1),
                node_vec(mesh, x, j + 1, k + 1),
            ]
        }
        LogicalFace3d::IMax => {
            let x = mesh.nx;
            [
                node_vec(mesh, x, j, k),
                node_vec(mesh, x, j + 1, k),
                node_vec(mesh, x, j + 1, k + 1),
                node_vec(mesh, x, j, k + 1),
            ]
        }
        LogicalFace3d::JMin => {
            let y = 0;
            [
                node_vec(mesh, i + 1, y, k),
                node_vec(mesh, i, y, k),
                node_vec(mesh, i, y, k + 1),
                node_vec(mesh, i + 1, y, k + 1),
            ]
        }
        LogicalFace3d::JMax => {
            let y = mesh.ny;
            [
                node_vec(mesh, i, y, k),
                node_vec(mesh, i + 1, y, k),
                node_vec(mesh, i + 1, y, k + 1),
                node_vec(mesh, i, y, k + 1),
            ]
        }
        LogicalFace3d::KMin => {
            let z = 0;
            [
                node_vec(mesh, i + 1, j, z),
                node_vec(mesh, i, j, z),
                node_vec(mesh, i, j + 1, z),
                node_vec(mesh, i + 1, j + 1, z),
            ]
        }
        LogicalFace3d::KMax => {
            let z = mesh.nz;
            [
                node_vec(mesh, i, j, z),
                node_vec(mesh, i + 1, j, z),
                node_vec(mesh, i + 1, j + 1, z),
                node_vec(mesh, i, j + 1, z),
            ]
        }
    }
}

#[must_use]
fn quad_vertex_center(v00: Vector3, v10: Vector3, v11: Vector3, v01: Vector3) -> Vector3 {
    Vector3::new(
        (v00.x + v10.x + v11.x + v01.x) / 4.0,
        (v00.y + v10.y + v11.y + v01.y) / 4.0,
        (v00.z + v10.z + v11.z + v01.z) / 4.0,
    )
}

/// 四边形面两三角分解（式 (3)）。
#[must_use]
fn quad_area_vector(v00: Vector3, v10: Vector3, v11: Vector3, v01: Vector3) -> Vector3 {
    let s1 = tri_area_vector(v00, v10, v01);
    let s2 = tri_area_vector(v10, v11, v01);
    Vector3::new(s1.x + s2.x, s1.y + s2.y, s1.z + s2.z)
}

#[must_use]
fn tri_area_vector(v0: Vector3, v1: Vector3, v2: Vector3) -> Vector3 {
    let a = vec_sub(v1, v0);
    let b = vec_sub(v2, v0);
    let c = vec_cross(a, b);
    Vector3::new(0.5 * c.x, 0.5 * c.y, 0.5 * c.z)
}

#[must_use]
fn tet_volume(v0: Vector3, v1: Vector3, v2: Vector3, v3: Vector3) -> Real {
    let a = vec_sub(v1, v0);
    let b = vec_sub(v2, v0);
    let c = vec_sub(v3, v0);
    scalar_triple(a, b, c).abs() / 6.0
}

#[must_use]
fn vec_sub(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

#[must_use]
fn vec_cross(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

#[must_use]
fn scalar_triple(a: Vector3, b: Vector3, c: Vector3) -> Real {
    let cross = vec_cross(b, c);
    a.x * cross.x + a.y * cross.y + a.z * cross.z
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod tests;
