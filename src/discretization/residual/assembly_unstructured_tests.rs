use super::*;
use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::core::approx_eq;
use crate::discretization::{
    GradientFields, InviscidFluxConfig, UnstructuredGradientLimiter, UnstructuredGradientLsqInput,
    UnstructuredGradientScratch, UnstructuredSolverMeshCache,
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq,
};
use crate::exec::ExecutionContext;
use crate::field::ConservedFields;
use crate::mesh::{CellKind, UnstructuredCell};
use crate::physics::{FreestreamParams, IdealGasEoS};

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
        .map(|face| FaceId(face as u32))
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

fn perturbed_two_tet_primitives(mesh: &UnstructuredMesh3d) -> PrimitiveFields {
    let eos = IdealGasEoS::AIR_STANDARD;
    let fields = ConservedFields::from_freestream(
        mesh.num_cells(),
        &eos,
        &FreestreamParams {
            mach: 0.3,
            ..FreestreamParams::default()
        },
    )
    .expect("fields");
    let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
    primitives
        .fill_from_conserved(&fields, &eos, 1.0e-8)
        .expect("fill");
    for (cell, ux) in primitives.velocity_x.values_mut().iter_mut().enumerate() {
        *ux = 100.0 + cell as f64 * 50.0;
    }
    primitives
}

fn inviscid_interior_only_residual(
    params: &InviscidAssemblyUnstructuredParams<'_>,
    linear_order: bool,
) -> ConservedResidual {
    let mut residual = ConservedResidual::zeros(params.mesh.num_cells()).expect("rhs");
    let topology = params.face_topology.expect("topology");
    let coloring = &topology.interior_coloring;
    if linear_order {
        coloring.for_each_face_index_linear(topology.interior.len(), |face_idx| {
            accumulate_one_interior_inviscid_face(face_idx, &mut residual, params, topology)
                .expect("face");
        });
    } else {
        coloring.for_each_face_index(|face_idx| {
            accumulate_one_interior_inviscid_face(face_idx, &mut residual, params, topology)
                .expect("face");
        });
    }
    residual
}

fn assert_residuals_match(a: &ConservedResidual, b: &ConservedResidual) {
    for (va, vb) in a.density.values().iter().zip(b.density.values()) {
        assert!(approx_eq(*va, *vb, 1.0e-12));
    }
    for (va, vb) in a.momentum_x.values().iter().zip(b.momentum_x.values()) {
        assert!(approx_eq(*va, *vb, 1.0e-12));
    }
    for (va, vb) in a.total_energy.values().iter().zip(b.total_energy.values()) {
        assert!(approx_eq(*va, *vb, 1.0e-12));
    }
}

#[test]
fn uniform_field_on_closed_tet_has_near_zero_rhs() {
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
    let eos = IdealGasEoS::AIR_STANDARD;
    let fs = FreestreamParams {
        mach: 0.3,
        ..FreestreamParams::default()
    };
    let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
    let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("primitive");
    primitives
        .fill_from_conserved(&fields, &eos, 1.0e-8)
        .expect("fill");
    let mut ghosts = BoundaryGhostBuffer::new();
    let state = fields.cell_state(0).expect("state");
    let faces = (0..mesh.num_faces())
        .map(|face| FaceId(face as u32))
        .collect::<Vec<_>>();
    for &face in &faces {
        ghosts.insert_face(
            face,
            crate::discretization::GhostCellState { conserved: state },
        );
    }
    let boundary = BoundarySet::new(vec![BoundaryPatch::new(
        "farfield",
        faces,
        BoundaryKind::Farfield {
            mach: fs.mach,
            pressure: fs.pressure,
            temperature: fs.temperature,
            alpha: fs.alpha,
            beta: fs.beta,
        },
    )]);
    let mut residual = ConservedResidual::zeros(mesh.num_cells()).expect("residual");
    let exec = ExecutionContext::for_unit_test();
    let params = InviscidAssemblyUnstructuredParams {
        mesh: &mesh,
        eos: &eos,
        config: &InviscidFluxConfig::roe_first_order(),
        boundaries: &boundary,
        ghosts: &ghosts,
        primitives: &primitives,
        face_topology: None,
        mesh_cache: None,
        gradients: None,
        min_pressure: 1.0e-8,
        exec: &exec,
    };
    assemble_inviscid_residual_unstructured(&fields, &mut residual, &params).expect("rhs");
    assert!(residual.density_rms_norm() < 1.0e-10);
}

fn closed_tet_freestream_linear_reconstruction_rhs(limiter: UnstructuredGradientLimiter) -> Real {
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
    let eos = IdealGasEoS::AIR_STANDARD;
    let fs = FreestreamParams {
        mach: 0.3,
        ..FreestreamParams::default()
    };
    let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
    let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("primitive");
    primitives
        .fill_from_conserved(&fields, &eos, 1.0e-8)
        .expect("fill");
    let mut ghosts = BoundaryGhostBuffer::new();
    let state = fields.cell_state(0).expect("state");
    let faces = (0..mesh.num_faces())
        .map(|face| FaceId(face as u32))
        .collect::<Vec<_>>();
    for &face in &faces {
        ghosts.insert_face(
            face,
            crate::discretization::GhostCellState { conserved: state },
        );
    }
    let boundary = BoundarySet::new(vec![BoundaryPatch::new(
        "farfield",
        faces,
        BoundaryKind::Farfield {
            mach: fs.mach,
            pressure: fs.pressure,
            temperature: fs.temperature,
            alpha: fs.alpha,
            beta: fs.beta,
        },
    )]);
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let mut gradients = GradientFields::zeros(mesh.num_cells()).expect("grad");
    let mut scratch = UnstructuredGradientScratch::new(mesh.num_cells());
    let mut exec = ExecutionContext::for_unit_test();
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq(
        UnstructuredGradientLsqInput {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            primitives: &primitives,
            eos: &eos,
            ghosts: &ghosts,
            min_pressure: 1.0e-8,
            viscous: None,
        },
        &mut gradients,
        &mut scratch,
        &mut exec,
    )
    .expect("grad");
    let config = InviscidFluxConfig::muscl_roe().with_unstructured_gradient_limiter(limiter);
    let mut residual = ConservedResidual::zeros(mesh.num_cells()).expect("residual");
    let params = InviscidAssemblyUnstructuredParams {
        mesh: &mesh,
        eos: &eos,
        config: &config,
        boundaries: &boundary,
        ghosts: &ghosts,
        primitives: &primitives,
        face_topology: Some(&mesh_cache.face_topology),
        mesh_cache: Some(&mesh_cache),
        gradients: Some(&gradients),
        min_pressure: 1.0e-8,
        exec: &exec,
    };
    assemble_inviscid_residual_unstructured(&fields, &mut residual, &params).expect("rhs");
    residual.density_rms_norm()
}

#[test]
fn uniform_freestream_linear_reconstruction_bj_rhs_near_zero() {
    assert!(
        closed_tet_freestream_linear_reconstruction_rhs(
            UnstructuredGradientLimiter::BarthJespersen
        ) < 1.0e-9
    );
}

#[test]
fn uniform_freestream_linear_reconstruction_venk_rhs_near_zero() {
    assert!(
        closed_tet_freestream_linear_reconstruction_rhs(
            UnstructuredGradientLimiter::Venkatakrishnan
        ) < 1.0e-9
    );
}

#[test]
fn cached_interior_inviscid_matches_mesh_face_loop() {
    let (mesh, boundary) = two_tet_mesh_and_boundary();
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let eos = IdealGasEoS::AIR_STANDARD;
    let fs = FreestreamParams {
        mach: 0.3,
        ..FreestreamParams::default()
    };
    let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
    let primitives = perturbed_two_tet_primitives(&mesh);
    let mut ghosts = BoundaryGhostBuffer::new();
    let state = fields.cell_state(0).expect("state");
    let faces = (0..mesh.num_faces())
        .map(|face| FaceId(face as u32))
        .collect::<Vec<_>>();
    for &face in &faces {
        ghosts.insert_face(
            face,
            crate::discretization::GhostCellState { conserved: state },
        );
    }
    let config = InviscidFluxConfig::roe_first_order();
    let exec = ExecutionContext::for_unit_test();
    let params_mesh = InviscidAssemblyUnstructuredParams {
        mesh: &mesh,
        eos: &eos,
        config: &config,
        boundaries: &boundary,
        ghosts: &ghosts,
        primitives: &primitives,
        face_topology: None,
        mesh_cache: None,
        gradients: None,
        min_pressure: 1.0e-8,
        exec: &exec,
    };
    let params_cached = InviscidAssemblyUnstructuredParams {
        face_topology: Some(&mesh_cache.face_topology),
        mesh_cache: Some(&mesh_cache),
        ..params_mesh
    };
    let mut mesh_loop = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
    let mut cached = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
    assemble_inviscid_residual_unstructured(&fields, &mut mesh_loop, &params_mesh).expect("m");
    assemble_inviscid_residual_unstructured(&fields, &mut cached, &params_cached).expect("c");
    assert_residuals_match(&mesh_loop, &cached);
}

#[test]
fn colored_interior_inviscid_matches_linear_face_order() {
    let (mesh, boundary) = two_tet_mesh_and_boundary();
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let eos = IdealGasEoS::AIR_STANDARD;
    let primitives = perturbed_two_tet_primitives(&mesh);
    let config = InviscidFluxConfig::roe_first_order();
    let exec = ExecutionContext::for_unit_test();
    let params = InviscidAssemblyUnstructuredParams {
        mesh: &mesh,
        eos: &eos,
        config: &config,
        boundaries: &boundary,
        ghosts: &BoundaryGhostBuffer::new(),
        primitives: &primitives,
        face_topology: Some(&mesh_cache.face_topology),
        mesh_cache: Some(&mesh_cache),
        gradients: None,
        min_pressure: 1.0e-8,
        exec: &exec,
    };
    let linear = inviscid_interior_only_residual(&params, true);
    let colored = inviscid_interior_only_residual(&params, false);
    assert_residuals_match(&linear, &colored);
}

#[cfg(feature = "parallel-fvm")]
#[test]
fn parallel_interior_inviscid_matches_colored_serial() {
    let (mesh, boundary) = two_tet_mesh_and_boundary();
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let eos = IdealGasEoS::AIR_STANDARD;
    let fs = FreestreamParams {
        mach: 0.3,
        ..FreestreamParams::default()
    };
    let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
    let primitives = perturbed_two_tet_primitives(&mesh);
    let config = InviscidFluxConfig::roe_first_order();
    let exec = ExecutionContext::for_unit_test();
    let params = InviscidAssemblyUnstructuredParams {
        mesh: &mesh,
        eos: &eos,
        config: &config,
        boundaries: &boundary,
        ghosts: &BoundaryGhostBuffer::new(),
        primitives: &primitives,
        face_topology: Some(&mesh_cache.face_topology),
        mesh_cache: Some(&mesh_cache),
        gradients: None,
        min_pressure: 1.0e-8,
        exec: &exec,
    };
    let serial = inviscid_interior_only_residual(&params, false);
    let mut parallel = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
    assemble_interior_faces_cached(&mut parallel, &fields, &params, &mesh_cache.face_topology)
        .expect("par");
    assert_residuals_match(&serial, &parallel);
}

#[test]
fn exec_context_cpu_scalar_matches_legacy_path() {
    let (mesh, boundary) = two_tet_mesh_and_boundary();
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let eos = IdealGasEoS::AIR_STANDARD;
    let fs = FreestreamParams {
        mach: 0.3,
        ..FreestreamParams::default()
    };
    let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
    let primitives = perturbed_two_tet_primitives(&mesh);
    let mut ghosts = BoundaryGhostBuffer::new();
    let state = fields.cell_state(0).expect("state");
    let faces = (0..mesh.num_faces())
        .map(|face| FaceId(face as u32))
        .collect::<Vec<_>>();
    for &face in &faces {
        ghosts.insert_face(
            face,
            crate::discretization::GhostCellState { conserved: state },
        );
    }
    let config = InviscidFluxConfig::roe_first_order();
    let unit_exec = ExecutionContext::for_unit_test();
    let scalar_exec = ExecutionContext::new(
        crate::exec::ExecConfig {
            backend: crate::exec::ExecBackend::CpuScalar,
            ..crate::exec::ExecConfig::default()
        },
        crate::exec::MeshExecMetrics::new(mesh.num_cells(), mesh.num_faces(), 4),
    );
    let params_unit = InviscidAssemblyUnstructuredParams {
        mesh: &mesh,
        eos: &eos,
        config: &config,
        boundaries: &boundary,
        ghosts: &ghosts,
        primitives: &primitives,
        face_topology: Some(&mesh_cache.face_topology),
        mesh_cache: Some(&mesh_cache),
        gradients: None,
        min_pressure: 1.0e-8,
        exec: &unit_exec,
    };
    let mut unit = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
    let mut cpu_scalar = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
    assemble_inviscid_residual_unstructured(&fields, &mut unit, &params_unit).expect("unit");
    let params_scalar = InviscidAssemblyUnstructuredParams {
        exec: &scalar_exec,
        ..params_unit
    };
    assemble_inviscid_residual_unstructured(&fields, &mut cpu_scalar, &params_scalar)
        .expect("scalar");
    assert_residuals_match(&unit, &cpu_scalar);
}
