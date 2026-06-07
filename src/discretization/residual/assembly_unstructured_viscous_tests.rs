use super::*;
use crate::boundary::BoundaryPatch;
use crate::discretization::GhostCellState;
use crate::field::ConservedFields;
use crate::mesh::{CellKind, UnstructuredCell};
use crate::physics::{FreestreamParams, ViscousPhysicsConfig};

#[test]
fn uniform_closed_tet_has_near_zero_unstructured_viscous_rhs() {
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
        mach: 0.2,
        ..FreestreamParams::default()
    };
    let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
    let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
    primitives
        .fill_from_conserved(&fields, &eos, 1.0e-8)
        .expect("fill");
    let faces = (0..mesh.num_faces())
        .map(|face| crate::core::FaceId(face as u32))
        .collect::<Vec<_>>();
    let mut ghosts = BoundaryGhostBuffer::new();
    let state = fields.cell_state(0).expect("state");
    for &face in &faces {
        ghosts.insert_face(face, GhostCellState { conserved: state });
    }
    let boundary = BoundarySet::new(vec![BoundaryPatch::new(
        "farfield",
        faces,
        crate::boundary::BoundaryKind::Farfield {
            mach: fs.mach,
            pressure: fs.pressure,
            temperature: fs.temperature,
            alpha: fs.alpha,
            beta: fs.beta,
        },
    )]);
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let viscous = ViscousPhysicsConfig::default();
    let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
    let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
    let mut input = ViscousAssemblyUnstructuredInput {
        mesh: &mesh,
        mesh_cache: &mesh_cache,
        eos: &eos,
        viscous: &viscous,
        boundaries: &boundary,
        ghosts: &ghosts,
        primitives: &primitives,
        min_pressure: 1.0e-8,
        gradient_scratch: &mut grad,
    };
    compute_gradients_and_assemble_viscous_unstructured(&mut rhs, &mut input).expect("visc");
    assert!(rhs.density.values().iter().all(|v| v.abs() < 1.0e-12));
    assert!(rhs.momentum_x.values().iter().all(|v| v.abs() < 1.0e-8));
    assert!(rhs.total_energy.values().iter().all(|v| v.abs() < 1.0e-8));
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
        crate::boundary::BoundaryKind::Farfield {
            mach: 0.0,
            pressure: 101_325.0,
            temperature: 300.0,
            alpha: 0.0,
            beta: 0.0,
        },
    )]);
    (mesh, boundary)
}

fn accumulate_interior_viscous_test_state(
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    linear_order: bool,
) -> ConservedResidual {
    let mut residual = ConservedResidual::zeros(params.mesh.num_cells()).expect("rhs");
    let prim = params.primitives;
    let grad_slices = params.gradients.velocity_gradient_slices();
    let inputs = InteriorViscousFaceInputs {
        grad: &grad_slices,
        ux: prim.velocity_x.values(),
        uy: prim.velocity_y.values(),
        uz: prim.velocity_z.values(),
    };
    let mut residual_mut = InteriorViscousResidualMut {
        mx: residual.momentum_x.values_mut(),
        my: residual.momentum_y.values_mut(),
        mz: residual.momentum_z.values_mut(),
        energy: residual.total_energy.values_mut(),
    };
    let constant = scratch.constant_transport;
    let coloring = &params.face_topology.interior_coloring;
    if linear_order {
        coloring.for_each_face_index_linear(params.face_topology.interior.len(), |i| {
            accumulate_one_interior_face(i, &inputs, &mut residual_mut, params, scratch, constant);
        });
    } else {
        coloring.for_each_face_index(|i| {
            accumulate_one_interior_face(i, &inputs, &mut residual_mut, params, scratch, constant);
        });
    }
    residual
}

#[test]
fn colored_interior_viscous_accumulation_matches_linear_face_order() {
    use crate::core::approx_eq;
    use crate::physics::{IdealGasEoS, ViscosityModel};

    let (mesh, boundary) = two_tet_mesh_and_boundary();
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let eos = IdealGasEoS::AIR_STANDARD;
    let viscous = ViscousPhysicsConfig::new(ViscosityModel::constant(2.0e-5).expect("mu"), 0.72)
        .expect("visc");
    let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
    let fields = ConservedFields::from_freestream(
        mesh.num_cells(),
        &eos,
        &FreestreamParams {
            mach: 0.0,
            ..FreestreamParams::default()
        },
    )
    .expect("fields");
    primitives
        .fill_from_conserved(&fields, &eos, 1.0e-8)
        .expect("fill");
    for (cell, ux) in primitives.velocity_x.values_mut().iter_mut().enumerate() {
        *ux = 10.0 + cell as f64 * 5.0;
    }
    let mut gradients = GradientFields::zeros(mesh.num_cells()).expect("grad");
    for cell in 0..mesh.num_cells() {
        gradients.du_dx.values_mut()[cell] = 100.0;
    }
    let mut scratch = ViscousAssemblyUnstructuredScratch::new(mesh.num_cells());
    crate::discretization::gradient::cell_temperatures_into(
        &primitives,
        &eos,
        Some(&viscous),
        &mut scratch.gradient.temperatures,
    )
    .expect("t");
    scratch.constant_transport =
        Some(face_transport_coefficients(300.0, 300.0, &viscous, &eos).expect("tc"));
    let params = ViscousAssemblyUnstructuredParams {
        mesh: &mesh,
        face_topology: &mesh_cache.face_topology,
        eos: &eos,
        viscous: &viscous,
        ghosts: &BoundaryGhostBuffer::new(),
        primitives: &primitives,
        gradients: &gradients,
        min_pressure: 1.0e-8,
    };
    let linear = accumulate_interior_viscous_test_state(&params, &scratch, true);
    let colored = accumulate_interior_viscous_test_state(&params, &scratch, false);
    for (a, b) in linear
        .momentum_x
        .values()
        .iter()
        .zip(colored.momentum_x.values())
    {
        assert!(approx_eq(*a, *b, 1.0e-12));
    }
    for (a, b) in linear
        .total_energy
        .values()
        .iter()
        .zip(colored.total_energy.values())
    {
        assert!(approx_eq(*a, *b, 1.0e-12));
    }
}

#[cfg(feature = "parallel-fvm")]
#[test]
fn parallel_interior_viscous_accumulation_matches_colored_serial() {
    use crate::core::approx_eq;
    use crate::physics::{IdealGasEoS, ViscosityModel};

    let (mesh, boundary) = two_tet_mesh_and_boundary();
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let eos = IdealGasEoS::AIR_STANDARD;
    let viscous = ViscousPhysicsConfig::new(ViscosityModel::constant(2.0e-5).expect("mu"), 0.72)
        .expect("visc");
    let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
    let fields = ConservedFields::from_freestream(
        mesh.num_cells(),
        &eos,
        &FreestreamParams {
            mach: 0.0,
            ..FreestreamParams::default()
        },
    )
    .expect("fields");
    primitives
        .fill_from_conserved(&fields, &eos, 1.0e-8)
        .expect("fill");
    for (cell, ux) in primitives.velocity_x.values_mut().iter_mut().enumerate() {
        *ux = 10.0 + cell as f64 * 5.0;
    }
    let mut gradients = GradientFields::zeros(mesh.num_cells()).expect("grad");
    for cell in 0..mesh.num_cells() {
        gradients.du_dx.values_mut()[cell] = 100.0;
    }
    let mut scratch = ViscousAssemblyUnstructuredScratch::new(mesh.num_cells());
    crate::discretization::gradient::cell_temperatures_into(
        &primitives,
        &eos,
        Some(&viscous),
        &mut scratch.gradient.temperatures,
    )
    .expect("t");
    scratch.constant_transport =
        Some(face_transport_coefficients(300.0, 300.0, &viscous, &eos).expect("tc"));
    let params = ViscousAssemblyUnstructuredParams {
        mesh: &mesh,
        face_topology: &mesh_cache.face_topology,
        eos: &eos,
        viscous: &viscous,
        ghosts: &BoundaryGhostBuffer::new(),
        primitives: &primitives,
        gradients: &gradients,
        min_pressure: 1.0e-8,
    };
    let serial = accumulate_interior_viscous_test_state(&params, &scratch, false);
    let mut parallel = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
    accumulate_interior_faces_fused(&mut parallel, &params, &scratch).expect("par");
    for (a, b) in serial
        .momentum_x
        .values()
        .iter()
        .zip(parallel.momentum_x.values())
    {
        assert!(approx_eq(*a, *b, 1.0e-12));
    }
}
