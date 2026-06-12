use super::incompressible_momentum::{
    IncompressibleConvectionScheme, IncompressibleMomentumPredictorConfig,
    assemble_incompressible_momentum_predictor_with_boundary_3d,
};
use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::core::approx_eq;
use crate::field::IncompressibleFields;
use crate::mesh::{BoundaryMesh, StructuredMesh3d};

#[test]
fn central_convection_splits_internal_face_flux_between_owner_and_neighbor() {
    let mesh = StructuredMesh3d::uniform_box("box", 2, 1, 1, 2.0, 1.0, 1.0).expect("mesh");
    let fields =
        IncompressibleFields::uniform(mesh.num_cells(), 0.0, [1.0, 0.0, 0.0]).expect("fields");
    let config = IncompressibleMomentumPredictorConfig::new(0.0, 1.0)
        .expect("config")
        .with_convection_scheme(IncompressibleConvectionScheme::Central);

    let system = assemble_incompressible_momentum_predictor_with_boundary_3d(
        &mesh,
        &fields,
        &BoundarySet::default(),
        config,
    )
    .expect("system");

    let left = mesh.cell_index(0, 0, 0);
    let right = mesh.cell_index(1, 0, 0);
    let row = system.matrix.row_entries(left).collect::<Vec<_>>();
    assert!(
        row.iter()
            .any(|&(col, value)| col == left && approx_eq(value, 1.5, 1.0e-12))
    );
    assert!(
        row.iter()
            .any(|&(col, value)| col == right && approx_eq(value, 0.5, 1.0e-12))
    );
}

#[test]
fn moving_wall_adds_boundary_diffusion_source() {
    let mesh = StructuredMesh3d::uniform_box("box", 1, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
    let fields =
        IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
    let boundary = BoundarySet::new(vec![BoundaryPatch::new(
        "j_max",
        mesh.resolve_logical_boundary("j_max").expect("faces"),
        BoundaryKind::MovingWall {
            velocity: [2.0, 0.0, 0.0],
        },
    )]);
    let config = IncompressibleMomentumPredictorConfig::new(0.25, 1.0).expect("config");

    let system = assemble_incompressible_momentum_predictor_with_boundary_3d(
        &mesh, &fields, &boundary, config,
    )
    .expect("system");

    let row = system.matrix.row_entries(0).collect::<Vec<_>>();
    assert_eq!(row, vec![(0, 1.5)]);
    assert!(approx_eq(system.rhs_x[0], 1.0, 1.0e-12));
    assert!(approx_eq(system.rhs_y[0], 0.0, 1.0e-12));
    assert!(approx_eq(
        system.d_coefficient.values()[0],
        2.0 / 3.0,
        1.0e-12
    ));
}

#[test]
fn velocity_inlet_adds_upwind_boundary_convection_source() {
    let mesh = StructuredMesh3d::uniform_box("box", 1, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
    let fields =
        IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
    let boundary = BoundarySet::new(vec![BoundaryPatch::new(
        "i_min",
        mesh.resolve_logical_boundary("i_min").expect("faces"),
        BoundaryKind::IncompressibleVelocityInlet {
            velocity: [1.0, 0.0, 0.0],
        },
    )]);
    let config = IncompressibleMomentumPredictorConfig::new(0.0, 1.0).expect("config");

    let system = assemble_incompressible_momentum_predictor_with_boundary_3d(
        &mesh, &fields, &boundary, config,
    )
    .expect("system");

    let row = system.matrix.row_entries(0).collect::<Vec<_>>();
    assert_eq!(row, vec![(0, 1.0)]);
    assert!(approx_eq(system.rhs_x[0], 1.0, 1.0e-12));
    assert!(approx_eq(system.rhs_y[0], 0.0, 1.0e-12));
    assert!(approx_eq(system.rhs_z[0], 0.0, 1.0e-12));
}

#[test]
fn body_force_adds_component_rhs_source() {
    let mesh = StructuredMesh3d::uniform_box("box", 1, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
    let fields =
        IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
    let config = IncompressibleMomentumPredictorConfig::new(0.0, 1.0)
        .expect("config")
        .with_body_force([2.0, -3.0, 4.0])
        .expect("body force");

    let system = assemble_incompressible_momentum_predictor_with_boundary_3d(
        &mesh,
        &fields,
        &BoundarySet::default(),
        config,
    )
    .expect("system");

    assert!(approx_eq(system.rhs_x[0], 2.0, 1.0e-12));
    assert!(approx_eq(system.rhs_y[0], -3.0, 1.0e-12));
    assert!(approx_eq(system.rhs_z[0], 4.0, 1.0e-12));
}
