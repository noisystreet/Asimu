use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::io::{CaseMesh, parse_case_str};
use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

#[test]
fn runs_single_tet_unstructured_smoke_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_smoke"
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
scheme = "euler"
local_time_step = true
max_steps = 1
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let result = super::compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 1);
    assert!(metrics.residual_rms.is_finite());
}

#[test]
fn runs_single_tet_unstructured_lusgs_sweep_step() {
    let mut case = parse_case_str(
        r#"
name = "unstructured_lusgs_sweep"
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
lusgs_sweep = true
lusgs_sweep_backward_damping = 0.5
max_steps = 1
"#,
    )
    .expect("parse");
    attach_single_tet_farfield(&mut case);
    let result = super::compressible_unstructured_3d::run(&case).expect("run");
    let metrics = result.compressible_3d.expect("metrics");
    assert_eq!(metrics.steps, 1);
    assert!(metrics.residual_rms.is_finite());
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
