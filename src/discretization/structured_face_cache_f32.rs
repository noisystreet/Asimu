//! 结构化 3D 内面几何 f32 预打包（节点坐标仍 f64；ADR 0016 §4 / ADR 0019 S1-a）。

use crate::core::{Real, Vector3};
use crate::discretization::inviscid_f32::FaceNormalF32;
use crate::mesh::{FaceMetric, StructuredMesh3d};

/// f32 结构化内面几何（与 `StructuredMesh3d::{i,j,k}_face_metric` 索引顺序一致）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuredInteriorFaceF32 {
    pub owner: usize,
    pub neighbor: usize,
    pub area: f32,
    pub normal: FaceNormalF32,
    pub owner_volume: f32,
    pub neighbor_volume: f32,
}

/// 结构化 3D 内面 + 单元体积 f32 缓存。
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredFaceCacheF32 {
    pub i_faces: Vec<StructuredInteriorFaceF32>,
    pub j_faces: Vec<StructuredInteriorFaceF32>,
    pub k_faces: Vec<StructuredInteriorFaceF32>,
    pub cell_volumes: Vec<f32>,
}

impl StructuredFaceCacheF32 {
    /// 自 f64 网格度量构建 f32 缓存（求解循环外调用一次）。
    #[must_use]
    pub fn from_mesh(mesh: &StructuredMesh3d) -> Self {
        let nx = mesh.nx;
        let ny = mesh.ny;
        let nz = mesh.nz;
        let mut i_faces = Vec::with_capacity(nx.saturating_sub(1) * ny * nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx.saturating_sub(1) {
                    let owner = mesh.cell_index(i, j, k);
                    let neighbor = mesh.cell_index(i + 1, j, k);
                    let face = mesh.i_face_metric(i, j, k);
                    let owner_volume = mesh.cell_metric(i, j, k).volume;
                    let neighbor_volume = mesh.cell_metric(i + 1, j, k).volume;
                    i_faces.push(interior_face_f32(
                        owner,
                        neighbor,
                        &face,
                        owner_volume,
                        neighbor_volume,
                    ));
                }
            }
        }
        let mut j_faces = Vec::with_capacity(nx * ny.saturating_sub(1) * nz);
        for k in 0..nz {
            for j in 0..ny.saturating_sub(1) {
                for i in 0..nx {
                    let owner = mesh.cell_index(i, j, k);
                    let neighbor = mesh.cell_index(i, j + 1, k);
                    let face = mesh.j_face_metric(i, j, k);
                    let owner_volume = mesh.cell_metric(i, j, k).volume;
                    let neighbor_volume = mesh.cell_metric(i, j + 1, k).volume;
                    j_faces.push(interior_face_f32(
                        owner,
                        neighbor,
                        &face,
                        owner_volume,
                        neighbor_volume,
                    ));
                }
            }
        }
        let mut k_faces = Vec::with_capacity(nx * ny * nz.saturating_sub(1));
        for k in 0..nz.saturating_sub(1) {
            for j in 0..ny {
                for i in 0..nx {
                    let owner = mesh.cell_index(i, j, k);
                    let neighbor = mesh.cell_index(i, j, k + 1);
                    let face = mesh.k_face_metric(i, j, k);
                    let owner_volume = mesh.cell_metric(i, j, k).volume;
                    let neighbor_volume = mesh.cell_metric(i, j, k + 1).volume;
                    k_faces.push(interior_face_f32(
                        owner,
                        neighbor,
                        &face,
                        owner_volume,
                        neighbor_volume,
                    ));
                }
            }
        }
        let mut cell_volumes = Vec::with_capacity(mesh.num_cells());
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    cell_volumes.push(mesh.cell_metric(i, j, k).volume as f32);
                }
            }
        }
        Self {
            i_faces,
            j_faces,
            k_faces,
            cell_volumes,
        }
    }
}

fn interior_face_f32(
    owner: usize,
    neighbor: usize,
    face: &FaceMetric,
    owner_volume: Real,
    neighbor_volume: Real,
) -> StructuredInteriorFaceF32 {
    StructuredInteriorFaceF32 {
        owner,
        neighbor,
        area: face.area as f32,
        normal: [
            face.normal.x as f32,
            face.normal.y as f32,
            face.normal.z as f32,
        ],
        owner_volume: owner_volume as f32,
        neighbor_volume: neighbor_volume as f32,
    }
}

#[must_use]
pub(crate) fn i_face_cache_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
    i + j * nx.saturating_sub(1) + k * nx.saturating_sub(1) * ny
}

#[must_use]
pub(crate) fn j_face_cache_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
    i + j * nx + k * nx * ny.saturating_sub(1)
}

#[must_use]
pub(crate) fn k_face_cache_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
    i + j * nx + k * nx * ny
}

#[must_use]
pub fn vec3_from_f32(n: FaceNormalF32) -> Vector3 {
    Vector3::new(n[0] as Real, n[1] as Real, n[2] as Real)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::discretization::freestream_pair::uniform_farfield_box;
    use crate::mesh::MeshMetricMode;

    #[test]
    fn uniform_box_face_cache_f32_matches_f64_geometry() {
        let pair =
            crate::discretization::freestream_pair::FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_side();
        let (mut mesh, _, _, _) = uniform_farfield_box(4, 5, 6, 1.0, 1.0, 1.0, &side);
        mesh.set_metric_mode(MeshMetricMode::Cartesian);
        let cache = StructuredFaceCacheF32::from_mesh(&mesh);
        let nx = mesh.nx;
        let ny = mesh.ny;
        let nz = mesh.nz;
        assert_eq!(cache.i_faces.len(), nx.saturating_sub(1) * ny * nz);
        assert_eq!(cache.j_faces.len(), nx * ny.saturating_sub(1) * nz);
        assert_eq!(cache.k_faces.len(), nx * ny * nz.saturating_sub(1));
        assert_eq!(cache.cell_volumes.len(), mesh.num_cells());

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx.saturating_sub(1) {
                    let idx = i_face_cache_index(nx, ny, i, j, k);
                    let f64_face = mesh.i_face_metric(i, j, k);
                    let f32_face = &cache.i_faces[idx];
                    assert_face_metric_match(&f64_face, f32_face);
                    assert_eq!(f32_face.owner, mesh.cell_index(i, j, k));
                    assert_eq!(f32_face.neighbor, mesh.cell_index(i + 1, j, k));
                }
            }
        }
        for k in 0..nz {
            for j in 0..ny.saturating_sub(1) {
                for i in 0..nx {
                    let idx = j_face_cache_index(nx, ny, i, j, k);
                    assert_face_metric_match(&mesh.j_face_metric(i, j, k), &cache.j_faces[idx]);
                }
            }
        }
        for k in 0..nz.saturating_sub(1) {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = k_face_cache_index(nx, ny, i, j, k);
                    assert_face_metric_match(&mesh.k_face_metric(i, j, k), &cache.k_faces[idx]);
                }
            }
        }
        for (cell, &volume_f32) in cache.cell_volumes.iter().enumerate() {
            let i = cell % nx;
            let j = (cell / nx) % ny;
            let k = cell / (nx * ny);
            let volume_f64 = mesh.cell_metric(i, j, k).volume;
            assert!(
                approx_eq(f64::from(volume_f32), volume_f64, 1.0e-6),
                "cell ({i},{j},{k}) volume"
            );
        }
    }

    fn assert_face_metric_match(f64_face: &FaceMetric, f32_face: &StructuredInteriorFaceF32) {
        assert!(approx_eq(f64::from(f32_face.area), f64_face.area, 1.0e-6));
        assert!(approx_eq(
            f64::from(f32_face.normal[0]),
            f64_face.normal.x,
            1.0e-6
        ));
        assert!(approx_eq(
            f64::from(f32_face.normal[1]),
            f64_face.normal.y,
            1.0e-6
        ));
        assert!(approx_eq(
            f64::from(f32_face.normal[2]),
            f64_face.normal.z,
            1.0e-6
        ));
    }
}
