use super::*;
use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::discretization::freestream_pair::FreestreamPairFixture;
use crate::discretization::{
    BoundaryGhostBuffer, UnstructuredGradientLimiter, UnstructuredGradientLsqInput,
    UnstructuredGradientLsqInputF32, UnstructuredGradientScratch,
    apply_compressible_boundary_conditions_typed,
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq,
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq_f32,
};
use crate::discretization::{GradientFields, GradientFieldsT, InviscidFluxConfig};
use crate::exec::ExecutionContext;
use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

fn single_tet_mesh_and_farfield_boundary(
    side: &crate::discretization::freestream_pair::UniformFarfieldSide<'_>,
) -> (UnstructuredMesh3d, BoundarySet) {
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
        .map(|face| crate::core::FaceId(face as u32))
        .collect::<Vec<_>>();
    let boundary = BoundarySet::new(vec![BoundaryPatch::new(
        "farfield",
        faces,
        BoundaryKind::Farfield {
            mach: side.fs.mach,
            pressure: side.fs.pressure,
            temperature: side.fs.temperature,
            alpha: 0.0,
            beta: 0.0,
        },
    )]);
    (mesh, boundary)
}

fn single_tet_fixture(
    side: &crate::discretization::freestream_pair::UniformFarfieldSide<'_>,
) -> (
    UnstructuredMesh3d,
    BoundarySet,
    ConservedFieldsT<f32>,
    BoundaryGhostBuffer,
    UnstructuredSolverMeshCache,
    PrimitiveFieldsT<f32>,
) {
    let (mesh, boundary) = single_tet_mesh_and_farfield_boundary(side);
    let fields = ConservedFieldsT::<f32>::from_real_fields(
        &crate::field::ConservedFields::from_freestream_context(
            mesh.num_cells(),
            &side.ctx,
            side.fs,
        )
        .expect("fields"),
    )
    .expect("typed");
    let mut ghosts = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
    apply_compressible_boundary_conditions_typed(
        &mesh,
        &boundary,
        &fields,
        &mut ghosts,
        &side.ctx,
        side.fs,
        None,
    )
    .expect("bc");
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let mut primitives = PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim");
    primitives
        .fill_from_conserved(&fields, side.eos, side.min_pressure)
        .expect("fill");
    (mesh, boundary, fields, ghosts, mesh_cache, primitives)
}

fn single_tet_fixture_f64(
    side: &crate::discretization::freestream_pair::UniformFarfieldSide<'_>,
) -> (
    UnstructuredMesh3d,
    BoundarySet,
    ConservedFieldsT<f64>,
    BoundaryGhostBuffer,
    UnstructuredSolverMeshCache,
    PrimitiveFieldsT<f64>,
) {
    let (mesh, boundary) = single_tet_mesh_and_farfield_boundary(side);
    let fields = ConservedFieldsT::<f64>::from_real_fields(
        &crate::field::ConservedFields::from_freestream_context(
            mesh.num_cells(),
            &side.ctx,
            side.fs,
        )
        .expect("fields"),
    )
    .expect("typed");
    let mut ghosts = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
    apply_compressible_boundary_conditions_typed(
        &mesh,
        &boundary,
        &fields,
        &mut ghosts,
        &side.ctx,
        side.fs,
        None,
    )
    .expect("bc");
    let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
    let mut primitives = PrimitiveFieldsT::<f64>::zeros(mesh.num_cells()).expect("prim");
    primitives
        .fill_from_conserved(&fields, side.eos, side.min_pressure)
        .expect("fill");
    (mesh, boundary, fields, ghosts, mesh_cache, primitives)
}

#[test]
fn f32_single_tet_uniform_freestream_has_near_zero_rhs() {
    let pair = FreestreamPairFixture::air_sutherland(0.2);
    let side = pair.inviscid_side();
    let (mesh, boundary, fields, ghosts, mesh_cache, primitives) = single_tet_fixture(&side);
    let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
    let config = InviscidFluxConfig::default();
    let mut exec = ExecutionContext::for_unit_test();
    let mut params = InviscidAssemblyUnstructuredTypedParams {
        mesh: &mesh,
        eos: side.eos,
        config: &config,
        boundaries: &boundary,
        ghosts: &ghosts,
        primitives: &primitives,
        mesh_cache: &mesh_cache,
        gradients: None,
        min_pressure: side.min_pressure,
        exec: &mut exec,
    };
    assemble_inviscid_residual_unstructured_typed(&fields, &mut rhs, &mut params)
        .expect("assemble");
    assert!(
        rhs.density
            .values()
            .iter()
            .all(|v| v.to_real().abs() < 1.0e-5),
        "f32 tet density rhs"
    );
}

#[test]
fn f32_single_tet_muscl_uniform_freestream_has_near_zero_rhs() {
    let pair = FreestreamPairFixture::air_sutherland(0.2);
    let side = pair.inviscid_side();
    let (mesh, boundary, fields, ghosts, mesh_cache, primitives) = single_tet_fixture(&side);
    let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
    let config = InviscidFluxConfig {
        unstructured_gradient_limiter: Some(UnstructuredGradientLimiter::BarthJespersen),
        ..InviscidFluxConfig::muscl_hllc()
    };
    let mut gradients = GradientFieldsT::<f32>::zeros(mesh.num_cells()).expect("grad");
    let mut exec = ExecutionContext::for_unit_test();
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq_f32(
        UnstructuredGradientLsqInputF32 {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            primitives: &primitives,
            eos: side.eos,
            ghosts: &ghosts,
            min_pressure: side.min_pressure,
            viscous: None,
        },
        &mut gradients,
        &mut exec,
    )
    .expect("gradients");
    let mut params = InviscidAssemblyUnstructuredTypedParams {
        mesh: &mesh,
        eos: side.eos,
        config: &config,
        boundaries: &boundary,
        ghosts: &ghosts,
        primitives: &primitives,
        mesh_cache: &mesh_cache,
        gradients: Some(&gradients),
        min_pressure: side.min_pressure,
        exec: &mut exec,
    };
    assemble_inviscid_residual_unstructured_typed(&fields, &mut rhs, &mut params)
        .expect("assemble");
    assert!(
        rhs.density
            .values()
            .iter()
            .all(|v| v.to_real().abs() < 1.0e-5),
        "f32 muscl tet density rhs"
    );
}

#[test]
fn f64_single_tet_muscl_uniform_freestream_has_near_zero_rhs() {
    let pair = FreestreamPairFixture::air_sutherland(0.2);
    let side = pair.inviscid_side();
    let (mesh, boundary, fields, ghosts, mesh_cache, primitives) = single_tet_fixture_f64(&side);
    let mut rhs = ConservedResidualT::<f64>::zeros(mesh.num_cells()).expect("rhs");
    let config = InviscidFluxConfig {
        unstructured_gradient_limiter: Some(UnstructuredGradientLimiter::BarthJespersen),
        ..InviscidFluxConfig::muscl_hllc()
    };
    let mut gradients = GradientFields::zeros(mesh.num_cells()).expect("grad");
    let mut scratch = UnstructuredGradientScratch::new(mesh.num_cells());
    let mut exec = ExecutionContext::for_unit_test();
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq(
        UnstructuredGradientLsqInput {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            primitives: &primitives,
            eos: side.eos,
            ghosts: &ghosts,
            min_pressure: side.min_pressure,
            viscous: None,
        },
        &mut gradients,
        &mut scratch,
        &mut exec,
    )
    .expect("gradients");
    let mut params = InviscidAssemblyUnstructuredTypedParams {
        mesh: &mesh,
        eos: side.eos,
        config: &config,
        boundaries: &boundary,
        ghosts: &ghosts,
        primitives: &primitives,
        mesh_cache: &mesh_cache,
        gradients: Some(&gradients),
        min_pressure: side.min_pressure,
        exec: &mut exec,
    };
    assemble_inviscid_residual_unstructured_typed(&fields, &mut rhs, &mut params)
        .expect("assemble");
    assert!(
        rhs.density
            .values()
            .iter()
            .all(|v| v.to_real().abs() < 1.0e-10),
        "f64 muscl tet density rhs"
    );
}

#[cfg(feature = "simd-fvm")]
#[test]
fn f32_single_tet_roe_simd_freestream_has_near_zero_rhs() {
    let pair = FreestreamPairFixture::air_sutherland(0.2);
    let side = pair.inviscid_side();
    let (mesh, boundary, fields, ghosts, mesh_cache, primitives) = single_tet_fixture(&side);
    let config = InviscidFluxConfig::roe_first_order();
    let mut exec = ExecutionContext::for_unit_test();
    let mut params = InviscidAssemblyUnstructuredTypedParams {
        mesh: &mesh,
        eos: side.eos,
        config: &config,
        boundaries: &boundary,
        ghosts: &ghosts,
        primitives: &primitives,
        mesh_cache: &mesh_cache,
        gradients: None,
        min_pressure: side.min_pressure,
        exec: &mut exec,
    };
    let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
    assemble_inviscid_residual_unstructured_typed(&fields, &mut rhs, &mut params).expect("simd");
    assert!(
        rhs.density
            .values()
            .iter()
            .all(|v| v.to_real().abs() < 1.0e-5),
        "f32 roe simd tet density rhs"
    );
}
