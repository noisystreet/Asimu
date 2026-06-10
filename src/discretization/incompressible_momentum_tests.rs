use super::incompressible_momentum::{
    IncompressibleConvectionScheme, IncompressibleMomentumPredictorConfig,
    assemble_incompressible_momentum_predictor_with_boundary_3d,
};
use crate::boundary::BoundarySet;
use crate::core::approx_eq;
use crate::field::IncompressibleFields;
use crate::mesh::StructuredMesh3d;

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
