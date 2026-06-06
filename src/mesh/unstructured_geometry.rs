//! 非结构单元几何：面面积向量、单元体积（凸多面体散度公式）。

use crate::core::{Real, Vector3};

use crate::mesh::metrics::FaceMetric;

#[must_use]
pub(super) fn point(points: &[[Real; 3]], index: usize) -> Vector3 {
    let p = points[index];
    Vector3::new(p[0], p[1], p[2])
}

#[must_use]
pub(super) fn tri_face_metric(points: &[[Real; 3]], nodes: [usize; 3]) -> FaceMetric {
    let v0 = point(points, nodes[0]);
    let v1 = point(points, nodes[1]);
    let v2 = point(points, nodes[2]);
    let area_vector = tri_area_vector(v0, v1, v2);
    let center = Vector3::new(
        (v0.x + v1.x + v2.x) / 3.0,
        (v0.y + v1.y + v2.y) / 3.0,
        (v0.z + v1.z + v2.z) / 3.0,
    );
    FaceMetric::from_area_vector_and_center(area_vector, center)
}

#[must_use]
pub(super) fn quad_face_metric(points: &[[Real; 3]], nodes: [usize; 4]) -> FaceMetric {
    let v0 = point(points, nodes[0]);
    let v1 = point(points, nodes[1]);
    let v2 = point(points, nodes[2]);
    let v3 = point(points, nodes[3]);
    let a0 = tri_area_vector(v0, v1, v2);
    let a1 = tri_area_vector(v0, v2, v3);
    let area_vector = Vector3::new(a0.x + a1.x, a0.y + a1.y, a0.z + a1.z);
    let center = Vector3::new(
        (v0.x + v1.x + v2.x + v3.x) * 0.25,
        (v0.y + v1.y + v2.y + v3.y) * 0.25,
        (v0.z + v1.z + v2.z + v3.z) * 0.25,
    );
    FaceMetric::from_area_vector_and_center(area_vector, center)
}

/// 凸单元：\(V = \frac{1}{3}\sum_f \mathbf{S}_f \cdot \mathbf{x}_f\)。
#[must_use]
pub(super) fn volume_from_outward_faces(faces: &[FaceMetric]) -> Real {
    let mut volume = 0.0;
    for face in faces {
        volume += scalar_dot(face.area_vector, face.center) / 3.0;
    }
    volume
}

#[must_use]
pub(super) fn cell_center(points: &[[Real; 3]], node_indices: &[usize]) -> Vector3 {
    let mut center = Vector3::new(0.0, 0.0, 0.0);
    let n = node_indices.len() as Real;
    for &idx in node_indices {
        let p = point(points, idx);
        center.x += p.x;
        center.y += p.y;
        center.z += p.z;
    }
    Vector3::new(center.x / n, center.y / n, center.z / n)
}

#[must_use]
fn tri_area_vector(v0: Vector3, v1: Vector3, v2: Vector3) -> Vector3 {
    let a = vec_sub(v1, v0);
    let b = vec_sub(v2, v0);
    let c = vec_cross(a, b);
    Vector3::new(0.5 * c.x, 0.5 * c.y, 0.5 * c.z)
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
pub(super) fn orient_metric_outward_from(metric: FaceMetric, cell_center: Vector3) -> FaceMetric {
    let to_face = vec_sub(metric.center, cell_center);
    if scalar_dot(metric.area_vector, to_face) < 0.0 {
        flip_face_metric(metric)
    } else {
        metric
    }
}

#[must_use]
pub(super) fn flip_face_metric(metric: FaceMetric) -> FaceMetric {
    FaceMetric::from_area_vector_and_center(
        Vector3::new(
            -metric.area_vector.x,
            -metric.area_vector.y,
            -metric.area_vector.z,
        ),
        metric.center,
    )
}

#[must_use]
pub(super) fn reverse_face_nodes(nodes: &[usize]) -> Vec<usize> {
    let mut reversed = nodes.to_vec();
    reversed.reverse();
    reversed
}

#[must_use]
fn scalar_dot(a: Vector3, b: Vector3) -> Real {
    a.x * b.x + a.y * b.y + a.z * b.z
}
