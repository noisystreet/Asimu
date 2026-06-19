use super::*;
use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::discretization::GhostCellState;
use crate::discretization::unstructured_face_cache::UnstructuredSolverMeshCache;
use crate::exec::{ExecBackend, ExecConfig, ExecutionContext, MeshExecMetrics};
use crate::mesh::{CellKind, UnstructuredCell};
use crate::physics::{ConservedState, PrimitiveState};

#[test]
fn linear_field_recovers_constant_unstructured_idw_lsq_gradient() {
    let mesh = UnstructuredMesh3d::new(
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
    .expect("mesh");
    let eos = IdealGasEoS::AIR_STANDARD;
    let cell_center = mesh.cell_metric(crate::core::CellId(0)).center;
    let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
    let cell_prim = linear_primitive_at(cell_center, &eos);
    prim.density.values_mut()[0] = cell_prim.density;
    prim.pressure.values_mut()[0] = cell_prim.pressure;
    prim.velocity_x.values_mut()[0] = cell_prim.velocity[0];
    prim.velocity_y.values_mut()[0] = cell_prim.velocity[1];
    prim.velocity_z.values_mut()[0] = cell_prim.velocity[2];

    let faces = (0..mesh.num_faces())
        .map(|face| crate::core::FaceId(face as u32))
        .collect::<Vec<_>>();
    let mut ghosts = BoundaryGhostBuffer::new();
    for &face in &faces {
        let face_center = mesh.face_metric(face).center;
        let ghost_prim = linear_primitive_at(face_center, &eos);
        ghosts.insert_face(
            face,
            GhostCellState {
                conserved: ConservedState::from_primitive(&eos, &ghost_prim).expect("cons"),
            },
        );
    }
    let boundary = BoundarySet::new(vec![BoundaryPatch::new(
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
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");

    let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
    let mut exec = ExecutionContext::for_unit_test();
    compute_unstructured_gradients_idw_lsq(
        UnstructuredGradientLsqInput {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            primitives: &prim,
            eos: &eos,
            ghosts: &ghosts,
            min_pressure: 1.0e-8,
            viscous: None,
        },
        &mut grad,
        &mut exec,
    )
    .expect("grad");

    let g = grad.velocity_grad_at(0);
    assert!((g.du[0] - 2.0).abs() < 1.0e-12);
    assert!((g.du[1] + 3.0).abs() < 1.0e-12);
    assert!((g.du[2] - 0.5).abs() < 1.0e-12);
    assert!((g.dv[0] + 1.5).abs() < 1.0e-12);
    assert!((g.dv[1] - 0.25).abs() < 1.0e-12);
    assert!((g.dv[2] - 4.0).abs() < 1.0e-12);
    assert!((g.dw[0] - 0.75).abs() < 1.0e-12);
    assert!((g.dw[1] - 1.25).abs() < 1.0e-12);
    assert!((g.dw[2] + 2.5).abs() < 1.0e-12);
    assert!((grad.dt_dx.values()[0] - 10.0).abs() < 1.0e-12);
    assert!((grad.dt_dy.values()[0] + 5.0).abs() < 1.0e-12);
    assert!((grad.dt_dz.values()[0] - 2.0).abs() < 1.0e-12);
}

#[test]
fn boundary_lsq_sample_uses_face_center_and_inverse_squared_weight() {
    let mesh = UnstructuredMesh3d::new(
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
    .expect("mesh");
    let faces = (0..mesh.num_faces())
        .map(|face| crate::core::FaceId(face as u32))
        .collect::<Vec<_>>();
    let boundary = BoundarySet::new(vec![BoundaryPatch::new(
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
    let cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let owner = crate::core::CellId(0);
    let owner_center = mesh.cell_metric(owner).center;
    let bface = cache
        .face_topology
        .boundary
        .iter()
        .find(|f| f.owner == 0)
        .expect("boundary face");
    let face_center = mesh.face_metric(bface.face).center;
    let dr = bface.lsq_dr;
    assert!((dr.x - (face_center.x - owner_center.x)).abs() < 1.0e-12);
    assert!((dr.y - (face_center.y - owner_center.y)).abs() < 1.0e-12);
    assert!((dr.z - (face_center.z - owner_center.z)).abs() < 1.0e-12);
    let dist = dr.magnitude();
    assert!((bface.lsq_w - 1.0 / (dist * dist)).abs() < 1.0e-12);
    let mirror = Vector3::new(
        2.0 * face_center.x - owner_center.x,
        2.0 * face_center.y - owner_center.y,
        2.0 * face_center.z - owner_center.z,
    );
    let dr_mirror = Vector3::new(
        mirror.x - owner_center.x,
        mirror.y - owner_center.y,
        mirror.z - owner_center.z,
    );
    assert!((dr.magnitude() - dr_mirror.magnitude()).abs() > 1.0e-6);
}

fn linear_primitive_at(point: crate::core::Vector3, eos: &IdealGasEoS) -> PrimitiveState {
    let density = 1.0;
    let temperature = 300.0 + 10.0 * point.x - 5.0 * point.y + 2.0 * point.z;
    PrimitiveState {
        density,
        velocity: [
            2.0 * point.x - 3.0 * point.y + 0.5 * point.z,
            -1.5 * point.x + 0.25 * point.y + 4.0 * point.z,
            0.75 * point.x + 1.25 * point.y - 2.5 * point.z,
        ],
        pressure: density * eos.gas_constant * temperature,
        temperature,
    }
}

fn two_tet_mesh_and_boundary() -> (UnstructuredMesh3d, BoundarySet) {
    let mesh = UnstructuredMesh3d::new(
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
    .expect("mesh");
    let faces = (0..mesh.num_faces())
        .map(|face| crate::core::FaceId(face as u32))
        .collect::<Vec<_>>();
    let boundary = BoundarySet::new(vec![BoundaryPatch::new(
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
    (mesh, boundary)
}

fn assert_vector3_fields_match(a: &[Vector3], b: &[Vector3], tol: Real) {
    use crate::core::approx_eq;
    assert_eq!(a.len(), b.len());
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        assert!(approx_eq(lhs.x, rhs.x, tol));
        assert!(approx_eq(lhs.y, rhs.y, tol));
        assert!(approx_eq(lhs.z, rhs.z, tol));
    }
}

#[cfg(feature = "parallel-fvm")]
#[test]
fn parallel_idw_lsq_accumulate_matches_face_serial() {
    let (mesh, boundary) = two_tet_mesh_and_boundary();
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let eos = IdealGasEoS::AIR_STANDARD;
    let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
    for (cell, ux) in prim.velocity_x.values_mut().iter_mut().enumerate() {
        *ux = 10.0 + cell as f64 * 5.0;
        prim.velocity_y.values_mut()[cell] = 20.0 - cell as f64;
        prim.velocity_z.values_mut()[cell] = 3.0 * cell as f64;
        prim.density.values_mut()[cell] = 1.0 + 0.1 * cell as f64;
        prim.pressure.values_mut()[cell] = 101_325.0 + cell as f64 * 100.0;
    }
    let mut ghosts = BoundaryGhostBuffer::new();
    for face in &mesh_cache.face_topology.boundary {
        let owner = face.owner;
        let ghost_prim = PrimitiveState {
            density: prim.density.values()[owner],
            velocity: [
                prim.velocity_x.values()[owner],
                prim.velocity_y.values()[owner],
                prim.velocity_z.values()[owner],
            ],
            pressure: prim.pressure.values()[owner],
            temperature: 300.0,
        };
        ghosts.insert_face(
            face.face,
            GhostCellState {
                conserved: ConservedState::from_primitive(&eos, &ghost_prim).expect("ghost"),
            },
        );
    }
    let input = UnstructuredGradientLsqInput {
        mesh: &mesh,
        mesh_cache: &mesh_cache,
        primitives: &prim,
        eos: &eos,
        ghosts: &ghosts,
        min_pressure: 1.0e-8,
        viscous: None,
    };
    let n = mesh.num_cells();
    let metrics = MeshExecMetrics::new(n, mesh_cache.face_topology.interior.len(), n);
    let mut exec_serial = ExecutionContext::new(
        ExecConfig::for_test_backend(ExecBackend::CpuScalar),
        metrics,
    )
    .expect("serial exec");
    let mut exec_parallel =
        ExecutionContext::new(ExecConfig::default(), metrics).expect("parallel");
    let mut scratch_serial = UnstructuredGradientScratch::new(n);
    let mut scratch_parallel = UnstructuredGradientScratch::new(n);
    scratch_serial.prepare_temperatures(n);
    scratch_parallel.prepare_temperatures(n);
    cell_temperatures_into(&prim, &eos, None, &mut scratch_serial.temperatures).expect("t");
    scratch_parallel
        .temperatures
        .clone_from(&scratch_serial.temperatures);
    exec_serial.idwls_prepare_viscous(n);
    exec_parallel.idwls_prepare_viscous(n);

    accumulate_lsq_rhs_face_serial(&input, &scratch_serial, &mut exec_serial).expect("serial");
    accumulate_lsq_rhs_cell_parallel(&input, &scratch_parallel, &mut exec_parallel)
        .expect("parallel");

    let idwls_serial = exec_serial.idwls_rhs();
    let idwls_parallel = exec_parallel.idwls_rhs();
    assert_vector3_fields_match(idwls_serial.bu(), idwls_parallel.bu(), 1.0e-12);
    assert_vector3_fields_match(idwls_serial.bv(), idwls_parallel.bv(), 1.0e-12);
    assert_vector3_fields_match(idwls_serial.bw(), idwls_parallel.bw(), 1.0e-12);
    assert_vector3_fields_match(idwls_serial.bt(), idwls_parallel.bt(), 1.0e-12);
}

#[cfg(feature = "parallel-fvm")]
#[test]
fn parallel_inviscid_idw_lsq_accumulate_matches_face_serial() {
    let (mesh, boundary) = two_tet_mesh_and_boundary();
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let eos = IdealGasEoS::AIR_STANDARD;
    let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
    for (cell, ux) in prim.velocity_x.values_mut().iter_mut().enumerate() {
        *ux = 10.0 + cell as f64 * 5.0;
        prim.velocity_y.values_mut()[cell] = 20.0 - cell as f64;
        prim.velocity_z.values_mut()[cell] = 3.0 * cell as f64;
        prim.density.values_mut()[cell] = 1.0 + 0.1 * cell as f64;
        prim.pressure.values_mut()[cell] = 101_325.0 + cell as f64 * 100.0;
    }
    let mut ghosts = BoundaryGhostBuffer::new();
    for face in &mesh_cache.face_topology.boundary {
        let owner = face.owner;
        let ghost_prim = PrimitiveState {
            density: prim.density.values()[owner],
            velocity: [
                prim.velocity_x.values()[owner],
                prim.velocity_y.values()[owner],
                prim.velocity_z.values()[owner],
            ],
            pressure: prim.pressure.values()[owner],
            temperature: 300.0,
        };
        ghosts.insert_face(
            face.face,
            GhostCellState {
                conserved: ConservedState::from_primitive(&eos, &ghost_prim).expect("ghost"),
            },
        );
    }
    let input = UnstructuredGradientLsqInput {
        mesh: &mesh,
        mesh_cache: &mesh_cache,
        primitives: &prim,
        eos: &eos,
        ghosts: &ghosts,
        min_pressure: 1.0e-8,
        viscous: None,
    };
    let n = mesh.num_cells();
    let metrics = MeshExecMetrics::new(n, mesh_cache.face_topology.interior.len(), n);
    let mut exec_serial = ExecutionContext::new(
        ExecConfig::for_test_backend(ExecBackend::CpuScalar),
        metrics,
    )
    .expect("serial exec");
    let mut exec_parallel =
        ExecutionContext::new(ExecConfig::default(), metrics).expect("parallel");
    exec_serial.idwls_prepare_inviscid(n);
    exec_parallel.idwls_prepare_inviscid(n);

    super::inviscid_linear::accumulate_lsq_rhs_inviscid_face_serial(&input, &mut exec_serial)
        .expect("serial");
    super::inviscid_linear::accumulate_lsq_rhs_inviscid_cell_parallel(&input, &mut exec_parallel)
        .expect("parallel");

    let idwls_serial = exec_serial.idwls_rhs();
    let idwls_parallel = exec_parallel.idwls_rhs();
    assert_vector3_fields_match(idwls_serial.br(), idwls_parallel.br(), 1.0e-12);
    assert_vector3_fields_match(idwls_serial.bp(), idwls_parallel.bp(), 1.0e-12);
    assert_vector3_fields_match(idwls_serial.bu(), idwls_parallel.bu(), 1.0e-12);
    assert_vector3_fields_match(idwls_serial.bv(), idwls_parallel.bv(), 1.0e-12);
    assert_vector3_fields_match(idwls_serial.bw(), idwls_parallel.bw(), 1.0e-12);
}
