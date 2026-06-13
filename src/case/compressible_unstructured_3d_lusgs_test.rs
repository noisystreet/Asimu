use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::io::{CaseMesh, parse_case_str};
use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
use crate::solver::{UnstructuredDriverConfig, run_unstructured_with_observer};

/// 回归：非结构对角 LU-SGS 以推进时 `k1 = RHS(u0)` 的 RMS 为监控量；场演化时相邻步应可区分。
#[test]
fn diagonal_lusgs_pre_rhs_monitor_residual_differs_across_steps() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_lusgs_monitor"
[mesh]
kind = "structured_3d"
nx = 1
ny = 1
nz = 1

[physics]
gamma = 1.4
gas_constant = 287.0

[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15

[euler]
flux = "hllc"
reconstruction = "first_order"

[time]
scheme = "lu_sgs"
local_time_step = true
cfl = 0.001
max_steps = 2
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let mesh = case.mesh.as_unstructured_3d().expect("mesh");
    let eos = case.physics.eos().expect("eos");
    let freestream = case.freestream.expect("freestream");
    let inviscid = case.compressible_discretization().expect("disc").inviscid();
    let solver =
        crate::case::compressible_unstructured_3d::build_compressible_solver(&case, &inviscid)
            .expect("solver");
    let mut fields = case.build_conserved_fields().expect("fields");
    fields.density.values_mut()[0] *= 1.1;
    let driver = UnstructuredDriverConfig {
        solver: &solver,
        mesh,
        eos: &eos,
        freestream: &freestream,
        inviscid: &inviscid,
        patches: &case.boundary,
        reference: case.reference.as_ref(),
        viscous: case.physics.viscous.as_ref(),
        fixed_dt: case.time.dt,
        local_time_step: case.time.uses_local_time_step(),
        time_scheme: case.time.resolved_time_scheme(),
        lu_sgs: case.time.resolved_lusgs_config().expect("lu_sgs"),
        cfl_schedule: case.cfl_schedule().expect("cfl"),
        max_steps: case.resolved_max_steps(),
        residual_tolerance: None,
    };
    let history =
        run_unstructured_with_observer(&driver, &mut fields, |_| Ok(())).expect("history");
    assert_eq!(history.len(), 2);
    let step1 = &history[0];
    let step2 = &history[1];
    assert!(
        (step1.residual_rms - step2.residual_rms).abs() > 1.0e-12,
        "两步 ‖RHS(u0)‖ 监控残差应不同: r1={} r2={}",
        step1.residual_rms,
        step2.residual_rms,
    );
}

fn attach_single_tet_farfield(case: &mut crate::io::CaseSpec) {
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
    let fs = case.freestream.expect("freestream");
    case.mesh = CaseMesh::Unstructured3d(mesh);
    case.boundary = BoundarySet::new(vec![BoundaryPatch::new(
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
}
