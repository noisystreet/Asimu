use super::{CellKind, UnstructuredCell, UnstructuredMesh3d};
use crate::core::{CellId, FaceId, approx_eq};
use crate::error::AsimuError;

fn unit_tet() -> UnstructuredMesh3d {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let cells = vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("tet")];
    UnstructuredMesh3d::new("tet", points, cells).expect("mesh")
}

fn unit_hex() -> UnstructuredMesh3d {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let cells =
        vec![UnstructuredCell::new(CellKind::Hex, vec![0, 1, 2, 3, 4, 5, 6, 7]).expect("hex")];
    UnstructuredMesh3d::new("hex", points, cells).expect("mesh")
}

fn unit_pyramid() -> UnstructuredMesh3d {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ];
    let cells = vec![UnstructuredCell::new(CellKind::Pyramid, vec![0, 1, 2, 3, 4]).expect("pyr")];
    UnstructuredMesh3d::new("pyramid", points, cells).expect("mesh")
}

fn unit_prism() -> UnstructuredMesh3d {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let cells =
        vec![UnstructuredCell::new(CellKind::Prism, vec![0, 1, 2, 3, 4, 5]).expect("prism")];
    UnstructuredMesh3d::new("prism", points, cells).expect("mesh")
}

#[test]
fn tet_unit_volume_and_boundary_faces() {
    let mesh = unit_tet();
    assert_eq!(mesh.num_cells(), 1);
    assert_eq!(mesh.num_faces(), 4);
    assert!(approx_eq(
        mesh.cell_metric(CellId(0)).volume,
        1.0 / 6.0,
        1.0e-12
    ));
    for face in 0..4 {
        assert!(mesh.face_neighbor(FaceId(face)).expect("face").is_none());
    }
}

#[test]
fn hex_unit_volume_and_six_boundary_faces() {
    let mesh = unit_hex();
    assert_eq!(mesh.num_faces(), 6);
    assert!(approx_eq(mesh.cell_metric(CellId(0)).volume, 1.0, 1.0e-12));
}

#[test]
fn pyramid_unit_volume() {
    let mesh = unit_pyramid();
    assert_eq!(mesh.num_faces(), 5);
    assert!(approx_eq(
        mesh.cell_metric(CellId(0)).volume,
        1.0 / 3.0,
        1.0e-12
    ));
}

#[test]
fn prism_unit_volume() {
    let mesh = unit_prism();
    assert_eq!(mesh.num_faces(), 5);
    assert!(approx_eq(mesh.cell_metric(CellId(0)).volume, 0.5, 1.0e-12));
}

#[test]
fn mixed_hex_and_tet_only_share_tri_faces_between_tets() {
    // 六面体顶面为四边形，四面体底面为三角形；M1 仅合并节点集完全相同的面。
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 1.5],
    ];
    let cells = vec![
        UnstructuredCell::new(CellKind::Hex, vec![0, 1, 2, 3, 4, 5, 6, 7]).expect("hex"),
        UnstructuredCell::new(CellKind::Tet, vec![4, 5, 6, 8]).expect("tet"),
        UnstructuredCell::new(CellKind::Tet, vec![4, 6, 7, 8]).expect("tet"),
    ];
    let mesh = UnstructuredMesh3d::new("mixed", points, cells).expect("mixed");
    assert_eq!(mesh.num_cells(), 3);
    let interior = (0..mesh.num_faces())
        .filter(|face| {
            mesh.face_neighbor(FaceId(*face as u32))
                .expect("n")
                .is_some()
        })
        .count();
    assert_eq!(interior, 1, "两四面体共享三角形 4-6-8");
    assert!(mesh.cell_metric(CellId(0)).volume > 0.0);
}

#[test]
fn two_hexes_share_interior_quad_face() {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
    ];
    let cells = vec![
        UnstructuredCell::new(CellKind::Hex, vec![0, 1, 2, 3, 4, 5, 6, 7]).expect("left"),
        UnstructuredCell::new(CellKind::Hex, vec![1, 8, 9, 2, 5, 10, 11, 6]).expect("right"),
    ];
    let mesh = UnstructuredMesh3d::new("hex_pair", points, cells).expect("pair");
    assert_eq!(mesh.num_cells(), 2);
    let interior = (0..mesh.num_faces())
        .filter(|face| {
            mesh.face_neighbor(FaceId(*face as u32))
                .expect("n")
                .is_some()
        })
        .count();
    assert_eq!(interior, 1);
    assert!(approx_eq(mesh.cell_metric(CellId(0)).volume, 1.0, 1.0e-12));
    assert!(approx_eq(mesh.cell_metric(CellId(1)).volume, 1.0, 1.0e-12));
}

#[test]
fn rejects_non_manifold_face() {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.25, 0.25, 1.0],
        [0.25, 0.25, 2.0],
        [0.25, 0.25, 3.0],
    ];
    let cells = vec![
        UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("a"),
        UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 4]).expect("b"),
        UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 5]).expect("c"),
    ];
    let err = UnstructuredMesh3d::new("non-manifold", points, cells).expect_err("non-manifold");
    assert!(matches!(err, AsimuError::Mesh(_)));
}

#[test]
fn cell_kind_from_vtk_type() {
    assert_eq!(CellKind::from_vtk_type(10).expect("tet"), CellKind::Tet);
    assert_eq!(CellKind::from_vtk_type(12).expect("hex"), CellKind::Hex);
    assert_eq!(CellKind::from_vtk_type(13).expect("prism"), CellKind::Prism);
    assert_eq!(CellKind::from_vtk_type(14).expect("pyr"), CellKind::Pyramid);
    assert!(CellKind::from_vtk_type(99).is_err());
}

#[test]
fn rejects_wrong_node_count() {
    let err = UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2]).expect_err("nodes");
    assert!(matches!(err, AsimuError::Mesh(_)));
}
