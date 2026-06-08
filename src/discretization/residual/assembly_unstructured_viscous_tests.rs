use super::*;
use crate::boundary::BoundaryPatch;
use crate::discretization::GhostCellState;
use crate::exec::ExecutionContext;
use crate::field::ConservedFields;
use crate::mesh::{CellKind, UnstructuredCell};
use crate::physics::{FreestreamParams, ViscousPhysicsConfig};

fn test_exec() -> ExecutionContext {
    ExecutionContext::for_unit_test()
}

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
    let mut exec = test_exec();
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
        exec: &mut exec,
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
    scratch: &mut ViscousAssemblyUnstructuredScratch,
    linear_order: bool,
) -> ConservedResidual {
    #[cfg(not(feature = "simd-fvm"))]
    super::face_avg::fill_face_averaged_viscous_soa(params, scratch);
    let mut residual = ConservedResidual::zeros(params.mesh.num_cells()).expect("rhs");
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
            accumulate_one_interior_face(i, &mut residual_mut, params, scratch, constant);
        });
    } else {
        coloring.for_each_face_index(|i| {
            accumulate_one_interior_face(i, &mut residual_mut, params, scratch, constant);
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
    let _exec = test_exec();
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
    let linear = accumulate_interior_viscous_test_state(&params, &mut scratch, true);
    let colored = accumulate_interior_viscous_test_state(&params, &mut scratch, false);
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
    let mut exec = test_exec();
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
    let serial = accumulate_interior_viscous_test_state(&params, &mut scratch, false);
    let mut parallel = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
    assemble_interior_faces(&mut parallel, &params, &mut scratch, &mut exec).expect("par");
    for (a, b) in serial
        .momentum_x
        .values()
        .iter()
        .zip(parallel.momentum_x.values())
    {
        assert!(approx_eq(*a, *b, 1.0e-12));
    }
}

#[cfg(feature = "parallel-fvm")]
#[test]
fn viscous_interior_one_scatter_invocation_per_color_bucket() {
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
    let mut exec = test_exec();
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
    let colors = params.face_topology.interior_coloring.num_colors;
    let mut parallel = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
    exec.reset_scatter_invocation_count();
    assemble_interior_faces(&mut parallel, &params, &mut scratch, &mut exec).expect("par");
    assert_eq!(exec.scatter_invocation_count(), colors);
}

#[cfg(feature = "simd-fvm")]
#[test]
fn simd_batch4_cell_gather_matches_face_averaged_fill() {
    use crate::core::approx_eq;
    use crate::discretization::viscous::fused_interior_viscous_face_flux_averaged;
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
    scratch.constant_transport =
        Some(face_transport_coefficients(300.0, 300.0, &viscous, &eos).expect("tc"));
    let _exec = test_exec();
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
    let face = &params.face_topology.interior[0];
    let batch = crate::discretization::InteriorFaceBatchStatic4 {
        face_indices: [0, 0, 0, 0],
        owners: [face.owner; 4],
        neighbors: [face.neighbor; 4],
        nx: [face.normal.x; 4],
        ny: [face.normal.y; 4],
        nz: [face.normal.z; 4],
        owner_rhs_scale: [face.owner_rhs_scale; 4],
        neighbor_rhs_scale: [face.neighbor_rhs_scale; 4],
        area: [face.area; 4],
        owner_volume: [face.owner_volume; 4],
        neighbor_volume: [face.neighbor_volume; 4],
    };
    let mut geoms = [crate::exec::ColoredViscousFaceGeom::default(); 4];
    let mut fluxes = [crate::exec::ColoredViscousFaceFlux::default(); 4];
    let count = super::compute_viscous_batch4_into(
        &batch,
        &params,
        &scratch,
        scratch.constant_transport,
        &mut geoms,
        &mut fluxes,
    );
    assert_eq!(count, 4);
    for lane in 0..count as usize {
        let geom = geoms[lane];
        let flux_batch = fluxes[lane];
        let face_idx = batch.face_indices[lane];
        let lane_avg = super::face_avg::face_averaged_lane_at(face_idx, &params);
        let flux_ref = fused_interior_viscous_face_flux_averaged(
            lane_avg,
            geom.nx,
            geom.ny,
            geom.nz,
            geom.mu,
            geom.lambda,
        );
        assert!(approx_eq(flux_batch.mx, flux_ref.mx, 1.0e-12));
        assert!(approx_eq(flux_batch.my, flux_ref.my, 1.0e-12));
        assert!(approx_eq(flux_batch.mz, flux_ref.mz, 1.0e-12));
        assert!(approx_eq(flux_batch.energy, flux_ref.energy, 1.0e-12));
    }
}

#[cfg(feature = "simd-fvm")]
#[test]
fn viscous_batch4_scalar_fallback_compacts_skipped_faces() {
    use crate::core::Vector3;
    use crate::discretization::{
        InteriorFaceBatchStatic4, InteriorFaceColoring, UnstructuredFaceTopology,
        UnstructuredInteriorFace,
    };

    fn interior_face(owner_scale: f64) -> UnstructuredInteriorFace {
        UnstructuredInteriorFace {
            owner: 0,
            neighbor: 1,
            area: 1.0,
            normal: Vector3::new(1.0, 0.0, 0.0),
            owner_volume: 1.0,
            neighbor_volume: 1.0,
            inv_owner_volume: 1.0,
            inv_neighbor_volume: 1.0,
            owner_rhs_scale: owner_scale,
            neighbor_rhs_scale: if owner_scale == 0.0 {
                0.0
            } else {
                -owner_scale
            },
            lsq_dr: Vector3::new(1.0, 0.0, 0.0),
            lsq_w: 1.0,
            dr_owner_to_face: Vector3::new(0.5, 0.0, 0.0),
            dr_neighbor_to_face: Vector3::new(-0.5, 0.0, 0.0),
        }
    }

    let (mesh, _boundary) = two_tet_mesh_and_boundary();
    let eos = IdealGasEoS::AIR_STANDARD;
    let viscous = ViscousPhysicsConfig::default();
    let fields = ConservedFields::from_freestream(
        mesh.num_cells(),
        &eos,
        &FreestreamParams {
            mach: 0.0,
            ..FreestreamParams::default()
        },
    )
    .expect("fields");
    let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
    primitives
        .fill_from_conserved(&fields, &eos, 1.0e-8)
        .expect("fill");
    let gradients = GradientFields::zeros(mesh.num_cells()).expect("grad");
    let mut scratch = ViscousAssemblyUnstructuredScratch::new(mesh.num_cells());
    scratch.constant_transport =
        Some(face_transport_coefficients(300.0, 300.0, &viscous, &eos).expect("tc"));

    let interior = vec![
        interior_face(0.0),
        interior_face(0.5),
        interior_face(0.6),
        interior_face(0.7),
    ];
    let face_topology = UnstructuredFaceTopology {
        interior,
        boundary: Vec::new(),
        interior_coloring: InteriorFaceColoring {
            buckets: Vec::new(),
            num_colors: 0,
            bucket_batch_layouts: Vec::new(),
        },
    };
    let params = ViscousAssemblyUnstructuredParams {
        mesh: &mesh,
        face_topology: &face_topology,
        eos: &eos,
        viscous: &viscous,
        ghosts: &BoundaryGhostBuffer::new(),
        primitives: &primitives,
        gradients: &gradients,
        min_pressure: 1.0e-8,
    };

    let batch = InteriorFaceBatchStatic4 {
        face_indices: [0, 1, 2, 3],
        owners: [0; 4],
        neighbors: [1; 4],
        nx: [1.0; 4],
        ny: [0.0; 4],
        nz: [0.0; 4],
        owner_rhs_scale: [0.0, 0.5, 0.6, 0.7],
        neighbor_rhs_scale: [0.0, -0.5, -0.6, -0.7],
        area: [1.0; 4],
        owner_volume: [1.0; 4],
        neighbor_volume: [1.0; 4],
    };
    assert!(!batch.simd_eligible());

    let mut geoms = [crate::exec::ColoredViscousFaceGeom::default(); 4];
    let mut fluxes = [crate::exec::ColoredViscousFaceFlux::default(); 4];
    let count = super::compute_viscous_batch4_into(
        &batch,
        &params,
        &scratch,
        scratch.constant_transport,
        &mut geoms,
        &mut fluxes,
    );
    assert_eq!(count, 3);
    assert!((geoms[0].owner_scale - 0.5).abs() < 1.0e-12);
    assert!((geoms[1].owner_scale - 0.6).abs() < 1.0e-12);
    assert!((geoms[2].owner_scale - 0.7).abs() < 1.0e-12);
}
