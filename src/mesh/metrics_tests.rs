use super::*;
use crate::core::approx_eq;

#[test]
fn uniform_box_curvilinear_volume_matches_cartesian() {
    let mut mesh = StructuredMesh3d::uniform_box("box", 4, 5, 6, 2.0, 3.0, 4.0).expect("mesh");
    mesh.set_metric_mode(MeshMetricMode::Curvilinear);
    let expected = 2.0 * 3.0 * 4.0 / (4.0 * 5.0 * 6.0);
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let cart = mesh.cell_volume_at(i, j, k);
                let curv = mesh.cell_metric(i, j, k).volume;
                assert!(approx_eq(cart, curv, 1.0e-12));
                assert!(approx_eq(curv, expected, 1.0e-12));
            }
        }
    }
}

#[test]
fn uniform_box_curvilinear_i_face_matches_cartesian() {
    let mut mesh = StructuredMesh3d::uniform_box("box", 3, 3, 3, 1.0, 1.0, 1.0).expect("mesh");
    mesh.set_metric_mode(MeshMetricMode::Curvilinear);
    let dy = 1.0 / 3.0;
    let dz = 1.0 / 3.0;
    let expected_area = dy * dz;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                let face = mesh.i_face_metric(i, j, k);
                assert!(approx_eq(face.area, expected_area, 1.0e-12));
                assert!(approx_eq(face.normal.x, 1.0, 1.0e-12));
                assert!(face.normal.y.abs() < 1.0e-12);
                assert!(face.normal.z.abs() < 1.0e-12);
            }
        }
    }
}

#[test]
fn uniform_box_boundary_imin_normal_points_outward() {
    let mut mesh = StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
    mesh.set_metric_mode(MeshMetricMode::Curvilinear);
    let face = mesh.boundary_face_metric(LogicalFace3d::IMin, 0, 0, 0);
    assert!(approx_eq(face.normal.x, -1.0, 1.0e-12));
    assert!(approx_eq(face.area, 0.25, 1.0e-12));
}

#[test]
fn metric_cache_matches_on_the_fly_computation() {
    let mut mesh = StructuredMesh3d::uniform_box("box", 3, 4, 5, 2.0, 3.0, 4.0).expect("mesh");
    mesh.set_metric_mode(MeshMetricMode::Curvilinear);
    let cell = mesh.cell_metric(1, 2, 3);
    let i_face = mesh.i_face_metric(0, 1, 2);
    let boundary = mesh.boundary_face_metric(LogicalFace3d::JMax, 1, mesh.ny - 1, 2);
    let spacing = mesh.min_positive_face_spacing().expect("spacing");

    mesh.build_metric_cache().expect("cache");
    assert!(mesh.metric_cache().is_some());
    assert_eq!(mesh.cell_metric(1, 2, 3), cell);
    assert_eq!(mesh.i_face_metric(0, 1, 2), i_face);
    assert_eq!(
        mesh.boundary_face_metric(LogicalFace3d::JMax, 1, mesh.ny - 1, 2),
        boundary
    );
    assert!(approx_eq(
        mesh.cached_min_face_spacing().expect("cached"),
        spacing,
        1.0e-12
    ));
}

#[test]
fn scale_clears_cache_until_rebuild() {
    let mut mesh = StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
    mesh.set_metric_mode(MeshMetricMode::Curvilinear);
    mesh.build_metric_cache().expect("cache");
    let before = mesh.cell_metric(0, 0, 0).volume;
    mesh.scale_coordinates(2.0);
    assert!(mesh.metric_cache().is_none());
    mesh.rebuild_metric_cache_if_needed().expect("rebuild");
    assert!(approx_eq(
        mesh.cell_metric(0, 0, 0).volume,
        before * 8.0,
        1.0e-10
    ));
}
