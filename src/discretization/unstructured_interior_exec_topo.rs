//! 非结构内面 CUDA exec 拓扑（init 一次；P0 消除每步 host `collect`）。

#[cfg(feature = "cuda")]
use crate::discretization::unstructured_face_cache::UnstructuredFaceTopology;
#[cfg(feature = "cuda")]
use crate::discretization::unstructured_face_cache_f32::UnstructuredFaceTopologyF32;
#[cfg(feature = "cuda")]
use crate::exec::gpu::cuda::{
    DeviceViscousFaceGeom, ExecInteriorColorBucket, ExecInteriorFaceStatic,
    ExecInteriorFaceTopology, ExecViscousInteriorTopology,
};

#[cfg(feature = "cuda")]
fn unit_normal(nx: f32, ny: f32, nz: f32) -> (f32, f32, f32) {
    let mag = (nx * nx + ny * ny + nz * nz).sqrt();
    if mag > 1.0e-30 {
        let inv = 1.0 / mag;
        (nx * inv, ny * inv, nz * inv)
    } else {
        (nx, ny, nz)
    }
}

#[cfg(feature = "cuda")]
fn build_color_buckets(coloring: &UnstructuredFaceTopology) -> Vec<ExecInteriorColorBucket> {
    coloring
        .interior_coloring
        .buckets
        .iter()
        .map(|bucket| ExecInteriorColorBucket {
            face_indices: bucket.iter().map(|&i| i as u32).collect(),
        })
        .collect()
}

/// 构建无粘内面 CUDA 拓扑（静态几何 + 着色桶；init 一次 H2D）。
#[cfg(feature = "cuda")]
#[must_use]
pub fn build_cuda_inviscid_interior_topology(
    topology_f32: &UnstructuredFaceTopologyF32,
    coloring: &UnstructuredFaceTopology,
) -> ExecInteriorFaceTopology {
    let faces = topology_f32
        .interior
        .iter()
        .map(|face| {
            let (nx, ny, nz) = unit_normal(face.normal[0], face.normal[1], face.normal[2]);
            ExecInteriorFaceStatic {
                owner: face.owner as u32,
                neighbor: face.neighbor as u32,
                nx,
                ny,
                nz,
                owner_scale: face.owner_rhs_scale,
                neighbor_scale: face.neighbor_rhs_scale,
            }
        })
        .collect();
    let color_buckets = build_color_buckets(coloring);
    ExecInteriorFaceTopology {
        faces,
        color_buckets,
    }
}

/// 构建粘性内面 CUDA 拓扑（静态几何 + 着色桶；\(\mu,\lambda\) 每步刷新）。
#[cfg(feature = "cuda")]
#[must_use]
pub fn build_cuda_viscous_interior_topology(
    topology_f32: &UnstructuredFaceTopologyF32,
    coloring: &UnstructuredFaceTopology,
) -> ExecViscousInteriorTopology {
    let faces = topology_f32
        .interior
        .iter()
        .map(|face| {
            let (nx, ny, nz) = unit_normal(face.normal[0], face.normal[1], face.normal[2]);
            DeviceViscousFaceGeom {
                owner: face.owner as u32,
                neighbor: face.neighbor as u32,
                nx,
                ny,
                nz,
                mu: 0.0,
                lambda: 0.0,
                owner_scale: face.owner_rhs_scale,
                neighbor_scale: face.neighbor_rhs_scale,
            }
        })
        .collect();
    let color_buckets = build_color_buckets(coloring);
    ExecViscousInteriorTopology {
        faces,
        color_buckets,
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet, WallHeat};
    use crate::core::FaceId;
    use crate::discretization::UnstructuredSolverMeshCache;
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

    fn closed_tet_mesh() -> (UnstructuredMesh3d, BoundarySet) {
        let mesh = UnstructuredMesh3d::new(
            "tet",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell")],
        )
        .expect("mesh");
        let faces = (0..mesh.num_faces())
            .map(|face| FaceId(face as u32))
            .collect::<Vec<_>>();
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "wall",
            faces,
            BoundaryKind::Wall {
                no_slip: true,
                heat: WallHeat::Adiabatic,
            },
        )]);
        (mesh, boundary)
    }

    #[test]
    fn cuda_interior_topo_matches_mesh_cache_face_counts() {
        let (mesh, boundary) = closed_tet_mesh();
        let cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let inviscid = &cache.cuda_inviscid_interior_topo;
        let viscous = &cache.cuda_viscous_interior_topo;
        assert_eq!(
            inviscid.num_interior_faces(),
            cache.face_topology.interior.len()
        );
        assert_eq!(
            viscous.num_interior_faces(),
            cache.face_topology.interior.len()
        );
        assert_eq!(
            inviscid.num_colors(),
            cache.face_topology.interior_coloring.num_colors
        );
        assert_eq!(
            viscous.num_colors(),
            cache.face_topology.interior_coloring.num_colors
        );
        assert_eq!(inviscid.color_buckets.len(), viscous.color_buckets.len());
        for (inv, vis) in inviscid.faces.iter().zip(viscous.faces.iter()) {
            assert_eq!(inv.owner, vis.owner);
            assert_eq!(inv.neighbor, vis.neighbor);
            assert_eq!(inv.nx, vis.nx);
            assert_eq!(inv.ny, vis.ny);
            assert_eq!(inv.nz, vis.nz);
            assert_eq!(inv.owner_scale, vis.owner_scale);
            assert_eq!(inv.neighbor_scale, vis.neighbor_scale);
        }
    }
}
