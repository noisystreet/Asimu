use super::*;
use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::core::FaceId;
use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

fn unit_hex_mesh() -> UnstructuredMesh3d {
    UnstructuredMesh3d::new(
        "hex",
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        vec![UnstructuredCell::new(CellKind::Hex, vec![0, 1, 2, 3, 4, 5, 6, 7]).expect("cell")],
    )
    .expect("mesh")
}

#[test]
fn face_topology_counts_match_closed_hex() {
    let mesh = unit_hex_mesh();
    let faces = (0..mesh.num_faces())
        .map(|face| FaceId(face as u32))
        .collect::<Vec<_>>();
    let boundaries = BoundarySet::new(vec![BoundaryPatch::new(
        "all",
        faces,
        BoundaryKind::Farfield {
            mach: 0.0,
            pressure: 101_325.0,
            temperature: 300.0,
            alpha: 0.0,
            beta: 0.0,
        },
    )]);
    let cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundaries).expect("cache");
    assert!(cache.face_topology.interior.is_empty());
    assert!(cache.face_topology.interior_coloring.is_empty());
    assert_eq!(cache.face_topology.boundary.len(), mesh.num_faces());
    assert_eq!(cache.lsq_geometry.len(), mesh.num_cells());
}

#[test]
fn precomputed_lsq_geometry_is_positive_definite_on_hex_samples() {
    let mesh = unit_hex_mesh();
    let faces = (0..mesh.num_faces())
        .map(|face| FaceId(face as u32))
        .collect::<Vec<_>>();
    let boundaries = BoundarySet::new(vec![BoundaryPatch::new(
        "all",
        faces,
        BoundaryKind::Farfield {
            mach: 0.0,
            pressure: 101_325.0,
            temperature: 300.0,
            alpha: 0.0,
            beta: 0.0,
        },
    )]);
    let cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundaries).expect("cache");
    let g = &cache.lsq_geometry[0];
    assert!(g.a_xx > 0.0);
    assert!(g.a_yy > 0.0);
    assert!(g.a_zz > 0.0);
}

fn two_tet_mesh() -> UnstructuredMesh3d {
    UnstructuredMesh3d::new(
        "two_tets",
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        vec![
            UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell"),
            UnstructuredCell::new(CellKind::Tet, vec![1, 2, 3, 4]).expect("cell"),
        ],
    )
    .expect("mesh")
}

#[test]
fn lsq_rhs_incidence_covers_all_interior_faces() {
    let mesh = two_tet_mesh();
    let faces = (0..mesh.num_faces())
        .map(|face| FaceId(face as u32))
        .collect::<Vec<_>>();
    let boundaries = BoundarySet::new(vec![BoundaryPatch::new(
        "farfield",
        faces,
        BoundaryKind::Farfield {
            mach: 0.0,
            pressure: 101_325.0,
            temperature: 300.0,
            alpha: 0.0,
            beta: 0.0,
        },
    )]);
    let cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundaries).expect("cache");
    let topology = &cache.face_topology;
    let inc = &cache.lsq_rhs_incidence;
    assert_eq!(inc.interior_as_owner.len(), mesh.num_cells());
    let owner_count: usize = inc.interior_as_owner.iter().map(Vec::len).sum();
    let neighbor_count: usize = inc.interior_as_neighbor.iter().map(Vec::len).sum();
    assert_eq!(owner_count, topology.interior.len());
    assert_eq!(neighbor_count, topology.interior.len());
    assert_eq!(
        cache.block_lusgs_topology.num_off_diagonal_blocks(),
        topology.interior.len() * 2
    );
}

#[test]
fn interior_face_coloring_has_no_same_color_cell_conflicts() {
    let mesh = two_tet_mesh();
    let faces = (0..mesh.num_faces())
        .map(|face| FaceId(face as u32))
        .collect::<Vec<_>>();
    let boundaries = BoundarySet::new(vec![BoundaryPatch::new(
        "farfield",
        faces,
        BoundaryKind::Farfield {
            mach: 0.0,
            pressure: 101_325.0,
            temperature: 300.0,
            alpha: 0.0,
            beta: 0.0,
        },
    )]);
    let cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundaries).expect("cache");
    let topology = &cache.face_topology;
    assert!(!topology.interior.is_empty());
    assert_eq!(
        topology.interior_coloring.num_colors,
        topology.interior_coloring.buckets.len()
    );
    for bucket in &topology.interior_coloring.buckets {
        let mut cells = std::collections::HashSet::new();
        for &face_idx in bucket {
            let face = &topology.interior[face_idx];
            assert!(cells.insert(face.owner));
            assert!(cells.insert(face.neighbor));
        }
    }
    assert_eq!(
        topology.interior_coloring.buckets.len(),
        topology.interior_coloring.bucket_batch_layouts.len()
    );
    for (bucket, layout) in topology
        .interior_coloring
        .buckets
        .iter()
        .zip(&topology.interior_coloring.bucket_batch_layouts)
    {
        assert_eq!(layout.num_faces(), bucket.len());
        let mut recovered = Vec::with_capacity(bucket.len());
        for batch in &layout.full_batches {
            assert_eq!(batch.face_indices.len(), 4);
            recovered.extend_from_slice(&batch.face_indices);
        }
        recovered.extend_from_slice(&layout.remainder);
        assert_eq!(recovered, *bucket);
    }
}
