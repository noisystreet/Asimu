//! 曲线网格 `MetricCache` 构建。

use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::{
    CellMetric, FaceMetric, LogicalFace3d, MetricCache3d, StructuredMesh3d, cell_index,
    compute_i_face_metric, compute_j_face_metric, compute_k_face_metric,
    curvilinear_boundary_face_metric, curvilinear_cell_metric, face_spacing, i_face_cache_index,
    j_face_cache_index, k_face_cache_index,
};

type BoundaryFaceMetricSets = (
    Vec<FaceMetric>,
    Vec<FaceMetric>,
    Vec<FaceMetric>,
    Vec<FaceMetric>,
    Vec<FaceMetric>,
    Vec<FaceMetric>,
);

pub(super) fn build_curvilinear_metric_cache(mesh: &StructuredMesh3d) -> Result<MetricCache3d> {
    let cells = build_cell_metrics(mesh);
    let (i_faces, j_faces, k_faces, max_h) = build_internal_face_metrics(mesh, &cells);
    let min_h = min_internal_face_spacing(mesh, &cells, &i_faces, &j_faces, &k_faces, max_h)?;
    let (boundary_imin, boundary_imax, boundary_jmin, boundary_jmax, boundary_kmin, boundary_kmax) =
        build_boundary_face_metrics(mesh);
    Ok(MetricCache3d {
        cells,
        i_faces,
        j_faces,
        k_faces,
        boundary_imin,
        boundary_imax,
        boundary_jmin,
        boundary_jmax,
        boundary_kmin,
        boundary_kmax,
        min_face_spacing: min_h,
    })
}

fn build_cell_metrics(mesh: &StructuredMesh3d) -> Vec<CellMetric> {
    let mut cells = Vec::with_capacity(mesh.num_cells());
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                cells.push(curvilinear_cell_metric(mesh, i, j, k));
            }
        }
    }
    cells
}

fn build_internal_face_metrics(
    mesh: &StructuredMesh3d,
    cells: &[CellMetric],
) -> (Vec<FaceMetric>, Vec<FaceMetric>, Vec<FaceMetric>, f64) {
    let mut i_faces = Vec::with_capacity(mesh.nx.saturating_sub(1) * mesh.ny * mesh.nz);
    let mut j_faces = Vec::with_capacity(mesh.nx * mesh.ny.saturating_sub(1) * mesh.nz);
    let mut k_faces = Vec::with_capacity(mesh.nx * mesh.ny * mesh.nz.saturating_sub(1));
    let mut max_h = 0.0_f64;
    push_i_faces(mesh, cells, &mut i_faces, &mut max_h);
    push_j_faces(mesh, cells, &mut j_faces, &mut max_h);
    push_k_faces(mesh, cells, &mut k_faces, &mut max_h);
    (i_faces, j_faces, k_faces, max_h)
}

fn push_i_faces(
    mesh: &StructuredMesh3d,
    cells: &[CellMetric],
    faces: &mut Vec<FaceMetric>,
    max_h: &mut f64,
) {
    let nx = mesh.nx;
    let ny = mesh.ny;
    for k in 0..mesh.nz {
        for j in 0..ny {
            for i in 0..nx.saturating_sub(1) {
                let face = compute_i_face_metric(mesh, i, j, k);
                track_max_h(
                    max_h,
                    face_spacing(
                        cells[cell_index(nx, ny, i, j, k)].volume,
                        cells[cell_index(nx, ny, i + 1, j, k)].volume,
                        face.area,
                    ),
                );
                faces.push(face);
            }
        }
    }
}

fn push_j_faces(
    mesh: &StructuredMesh3d,
    cells: &[CellMetric],
    faces: &mut Vec<FaceMetric>,
    max_h: &mut f64,
) {
    let nx = mesh.nx;
    let ny = mesh.ny;
    for k in 0..mesh.nz {
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx {
                let face = compute_j_face_metric(mesh, i, j, k);
                track_max_h(
                    max_h,
                    face_spacing(
                        cells[cell_index(nx, ny, i, j, k)].volume,
                        cells[cell_index(nx, ny, i, j + 1, k)].volume,
                        face.area,
                    ),
                );
                faces.push(face);
            }
        }
    }
}

fn push_k_faces(
    mesh: &StructuredMesh3d,
    cells: &[CellMetric],
    faces: &mut Vec<FaceMetric>,
    max_h: &mut f64,
) {
    let nx = mesh.nx;
    let ny = mesh.ny;
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..ny {
            for i in 0..nx {
                let face = compute_k_face_metric(mesh, i, j, k);
                track_max_h(
                    max_h,
                    face_spacing(
                        cells[cell_index(nx, ny, i, j, k)].volume,
                        cells[cell_index(nx, ny, i, j, k + 1)].volume,
                        face.area,
                    ),
                );
                faces.push(face);
            }
        }
    }
}

fn track_max_h(max_h: &mut f64, h: Real) {
    if h.is_finite() && h > Real::EPSILON {
        *max_h = max_h.max(h);
    }
}

fn min_internal_face_spacing(
    mesh: &StructuredMesh3d,
    cells: &[CellMetric],
    i_faces: &[FaceMetric],
    j_faces: &[FaceMetric],
    k_faces: &[FaceMetric],
    max_h: f64,
) -> Result<Real> {
    let floor = (max_h * 1.0e-6).max(1.0e-12);
    let mut min_h = Real::INFINITY;
    collect_min_h_i(mesh, cells, i_faces, floor, &mut min_h);
    collect_min_h_j(mesh, cells, j_faces, floor, &mut min_h);
    collect_min_h_k(mesh, cells, k_faces, floor, &mut min_h);
    if !min_h.is_finite() || min_h <= 0.0 {
        return Err(AsimuError::Mesh(
            "曲线网格 metric 缓存：不存在正面间距".to_string(),
        ));
    }
    Ok(min_h)
}

fn collect_min_h_i(
    mesh: &StructuredMesh3d,
    cells: &[CellMetric],
    faces: &[FaceMetric],
    floor: Real,
    min_h: &mut Real,
) {
    let nx = mesh.nx;
    let ny = mesh.ny;
    for k in 0..mesh.nz {
        for j in 0..ny {
            for i in 0..nx.saturating_sub(1) {
                let face = &faces[i_face_cache_index(nx, ny, i, j, k)];
                let h = face_spacing(
                    cells[cell_index(nx, ny, i, j, k)].volume,
                    cells[cell_index(nx, ny, i + 1, j, k)].volume,
                    face.area,
                );
                if h.is_finite() && h >= floor {
                    *min_h = min_h.min(h);
                }
            }
        }
    }
}

fn collect_min_h_j(
    mesh: &StructuredMesh3d,
    cells: &[CellMetric],
    faces: &[FaceMetric],
    floor: Real,
    min_h: &mut Real,
) {
    let nx = mesh.nx;
    let ny = mesh.ny;
    for k in 0..mesh.nz {
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx {
                let face = &faces[j_face_cache_index(nx, ny, i, j, k)];
                let h = face_spacing(
                    cells[cell_index(nx, ny, i, j, k)].volume,
                    cells[cell_index(nx, ny, i, j + 1, k)].volume,
                    face.area,
                );
                if h.is_finite() && h >= floor {
                    *min_h = min_h.min(h);
                }
            }
        }
    }
}

fn collect_min_h_k(
    mesh: &StructuredMesh3d,
    cells: &[CellMetric],
    faces: &[FaceMetric],
    floor: Real,
    min_h: &mut Real,
) {
    let nx = mesh.nx;
    let ny = mesh.ny;
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..ny {
            for i in 0..nx {
                let face = &faces[k_face_cache_index(nx, ny, i, j, k)];
                let h = face_spacing(
                    cells[cell_index(nx, ny, i, j, k)].volume,
                    cells[cell_index(nx, ny, i, j, k + 1)].volume,
                    face.area,
                );
                if h.is_finite() && h >= floor {
                    *min_h = min_h.min(h);
                }
            }
        }
    }
}

#[allow(clippy::type_complexity)]
fn build_boundary_face_metrics(mesh: &StructuredMesh3d) -> BoundaryFaceMetricSets {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    let mut boundary_imin = Vec::with_capacity(ny * nz);
    let mut boundary_imax = Vec::with_capacity(ny * nz);
    let mut boundary_jmin = Vec::with_capacity(nx * nz);
    let mut boundary_jmax = Vec::with_capacity(nx * nz);
    let mut boundary_kmin = Vec::with_capacity(nx * ny);
    let mut boundary_kmax = Vec::with_capacity(nx * ny);
    for k in 0..nz {
        for j in 0..ny {
            boundary_imin.push(curvilinear_boundary_face_metric(
                mesh,
                LogicalFace3d::IMin,
                0,
                j,
                k,
            ));
            boundary_imax.push(curvilinear_boundary_face_metric(
                mesh,
                LogicalFace3d::IMax,
                nx - 1,
                j,
                k,
            ));
        }
    }
    for k in 0..nz {
        for i in 0..nx {
            boundary_jmin.push(curvilinear_boundary_face_metric(
                mesh,
                LogicalFace3d::JMin,
                i,
                0,
                k,
            ));
            boundary_jmax.push(curvilinear_boundary_face_metric(
                mesh,
                LogicalFace3d::JMax,
                i,
                ny - 1,
                k,
            ));
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            boundary_kmin.push(curvilinear_boundary_face_metric(
                mesh,
                LogicalFace3d::KMin,
                i,
                j,
                0,
            ));
            boundary_kmax.push(curvilinear_boundary_face_metric(
                mesh,
                LogicalFace3d::KMax,
                i,
                j,
                nz - 1,
            ));
        }
    }
    (
        boundary_imin,
        boundary_imax,
        boundary_jmin,
        boundary_jmax,
        boundary_kmin,
        boundary_kmax,
    )
}
