use super::*;
use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::discretization::InviscidFluxConfig;
use crate::discretization::freestream_pair::FreestreamPairFixture;
use crate::exec::ExecConfig;
use crate::field::ConservedFields;
use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
use crate::physics::{FreestreamParams, IdealGasEoS, ReferenceScales};
use crate::solver::time::DualTimeConfig;
use crate::solver::{
    CflSchedule, CompressibleEulerConfig, CompressibleEulerSolver, run_unstructured_with_observer,
};

fn single_tet_driver(
    side: &crate::discretization::freestream_pair::UniformFarfieldSide<'_>,
    reference: &ReferenceScales,
) -> (
    UnstructuredMesh3d,
    BoundarySet,
    IdealGasEoS,
    FreestreamParams,
    CompressibleEulerSolver,
    InviscidFluxConfig,
    ReferenceScales,
) {
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
        .collect();
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
    let inviscid = InviscidFluxConfig::default();
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig::default());
    (
        mesh,
        boundary,
        *side.eos,
        *side.fs,
        solver,
        inviscid,
        reference.clone(),
    )
}

#[test]
fn f32_unstructured_step_matches_f64_on_single_tet() {
    let pair = FreestreamPairFixture::air_sutherland(0.2);
    let side = pair.inviscid_side();
    let (mesh, boundary, eos, freestream, solver, inviscid, reference) =
        single_tet_driver(&side, &pair.reference);
    let driver = UnstructuredDriverConfig {
        solver: &solver,
        mesh: &mesh,
        eos: &eos,
        freestream: &freestream,
        inviscid: &inviscid,
        patches: &boundary,
        reference: Some(&reference),
        viscous: None,
        fixed_dt: Some(1.0e-4),
        local_time_step: true,
        time_scheme: TimeIntegrationScheme::Euler,
        lu_sgs: Default::default(),
        dual_time: None,
        cfl_schedule: CflSchedule::constant(0.4),
        max_steps: 1,
        residual_tolerance: None,
        exec_config: ExecConfig::default(),
        observer_field_sync_interval: None,
    };
    let base = ConservedFields::from_freestream_context(mesh.num_cells(), &side.ctx, side.fs)
        .expect("base fields");
    let mut fields_f32 = ConservedFieldsT::<f32>::from_real_fields(&base).expect("f32 fields");
    let mut fields_f64 = base;
    let (history_f32, out_f32) =
        run_unstructured_typed_with_observer::<f32>(&driver, &mut fields_f32, |_| Ok(()))
            .expect("f32 run");
    let history_f64 =
        run_unstructured_with_observer(&driver, &mut fields_f64, |_| Ok(())).expect("f64 run");
    assert_eq!(history_f32.len(), 1);
    assert_eq!(history_f64.len(), 1);
    assert!(history_f32[0].residual_rms.is_finite());
    assert!(history_f64[0].residual_rms.is_finite());
    let rel = (out_f32.density.values()[0] - fields_f64.density.values()[0]).abs()
        / fields_f64.density.values()[0].max(1.0e-12);
    assert!(rel < 1.0e-3, "rel={rel}");
}

#[test]
fn f32_unstructured_lusgs_sweep_matches_f64_on_single_tet() {
    let pair = FreestreamPairFixture::air_sutherland(0.2);
    let side = pair.inviscid_side();
    let (mesh, boundary, eos, freestream, solver, inviscid, reference) =
        single_tet_driver(&side, &pair.reference);
    let driver = UnstructuredDriverConfig {
        solver: &solver,
        mesh: &mesh,
        eos: &eos,
        freestream: &freestream,
        inviscid: &inviscid,
        patches: &boundary,
        reference: Some(&reference),
        viscous: None,
        fixed_dt: Some(1.0e-4),
        local_time_step: true,
        time_scheme: TimeIntegrationScheme::LuSgs,
        lu_sgs: crate::solver::LuSgsConfig {
            sweep: true,
            omega: 1.0,
            sweep_backward_damping: 0.5,
        },
        dual_time: None,
        cfl_schedule: CflSchedule::constant(0.4),
        max_steps: 1,
        residual_tolerance: None,
        exec_config: ExecConfig::default(),
        observer_field_sync_interval: None,
    };
    let base = ConservedFields::from_freestream_context(mesh.num_cells(), &side.ctx, side.fs)
        .expect("base fields");
    let mut fields_f32 = ConservedFieldsT::<f32>::from_real_fields(&base).expect("f32 fields");
    let mut fields_f64 = base;
    let (history_f32, out_f32) =
        run_unstructured_typed_with_observer::<f32>(&driver, &mut fields_f32, |_| Ok(()))
            .expect("f32 run");
    let history_f64 =
        run_unstructured_with_observer(&driver, &mut fields_f64, |_| Ok(())).expect("f64 run");
    assert_eq!(history_f32.len(), 1);
    assert_eq!(history_f64.len(), 1);
    assert!(history_f32[0].residual_rms.is_finite());
    assert!(history_f64[0].residual_rms.is_finite());
    let rel = (out_f32.density.values()[0] - fields_f64.density.values()[0]).abs()
        / fields_f64.density.values()[0].max(1.0e-12);
    assert!(rel < 1.0e-3, "rel={rel}");
}

#[test]
fn f32_unstructured_dual_time_matches_f64_on_single_tet() {
    let pair = FreestreamPairFixture::air_sutherland(0.2);
    let side = pair.inviscid_side();
    let (mesh, boundary, eos, freestream, solver, inviscid, reference) =
        single_tet_driver(&side, &pair.reference);
    let dual = DualTimeConfig {
        dt_phys: 1.0e-4,
        max_inner_steps: 10,
        inner_log10_tolerance: Some(-2.0),
    };
    let driver = UnstructuredDriverConfig {
        solver: &solver,
        mesh: &mesh,
        eos: &eos,
        freestream: &freestream,
        inviscid: &inviscid,
        patches: &boundary,
        reference: Some(&reference),
        viscous: None,
        fixed_dt: Some(dual.dt_phys),
        local_time_step: true,
        time_scheme: TimeIntegrationScheme::DualTime,
        lu_sgs: Default::default(),
        dual_time: Some(dual),
        cfl_schedule: CflSchedule::constant(0.4),
        max_steps: 1,
        residual_tolerance: None,
        exec_config: ExecConfig::default(),
        observer_field_sync_interval: None,
    };
    let base = ConservedFields::from_freestream_context(mesh.num_cells(), &side.ctx, side.fs)
        .expect("base fields");
    let mut fields_f32 = ConservedFieldsT::<f32>::from_real_fields(&base).expect("f32 fields");
    let mut fields_f64 = base;
    let (history_f32, out_f32) =
        run_unstructured_typed_with_observer::<f32>(&driver, &mut fields_f32, |_| Ok(()))
            .expect("f32 run");
    let history_f64 =
        run_unstructured_with_observer(&driver, &mut fields_f64, |_| Ok(())).expect("f64 run");
    assert_eq!(history_f32.len(), 1);
    assert_eq!(history_f64.len(), 1);
    assert!(history_f32[0].residual_rms.is_finite());
    assert!(history_f64[0].residual_rms.is_finite());
    let rel = (out_f32.density.values()[0] - fields_f64.density.values()[0]).abs()
        / fields_f64.density.values()[0].max(1.0e-12);
    assert!(rel < 1.0e-3, "rel={rel}");
}
