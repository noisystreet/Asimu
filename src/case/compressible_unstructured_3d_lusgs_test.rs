use super::*;
use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::exec::{ExecConfig, ExecutionContext, MeshExecMetrics};
use crate::io::{CaseMesh, parse_case_str};
use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

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
    let mut env = UnstructuredRunEnv {
        case: &case,
        mesh,
        eos: &eos,
        freestream: &freestream,
        inviscid,
    };
    let mut fields = case.build_conserved_fields().expect("fields");
    fields.density.values_mut()[0] *= 1.1;
    let n = mesh.num_cells();
    let mesh_cache =
        crate::discretization::UnstructuredSolverMeshCache::from_mesh(mesh, &case.boundary)
            .expect("cache");
    let interior_faces = mesh_cache.face_topology.interior.len();
    let max_bucket_faces = mesh_cache
        .face_topology
        .interior_coloring
        .max_bucket_faces();
    let exec = ExecutionContext::new(
        ExecConfig::default(),
        MeshExecMetrics::new(n, interior_faces, max_bucket_faces),
    );
    let mut work = UnstructuredStepWork {
        storage: Rk4Storage::new(n).expect("storage"),
        state: SolverState::default(),
        integrator: RungeKutta4Integrator::new(RungeKutta4Config {
            dt: case.time.dt.unwrap_or(0.0),
            max_steps: case.resolved_max_steps(),
        }),
        ghosts: BoundaryGhostBuffer::with_face_capacity(mesh.num_faces()),
        primitives: PrimitiveFields::zeros(n).expect("prim"),
        gradients: GradientFields::zeros(n).expect("grad"),
        viscous_scratch: crate::discretization::ViscousAssemblyUnstructuredScratch::new(n),
        mesh_cache,
        exec,
        volumes: mesh.cell_volumes(),
        lusgs_couplings: LuSgsUnstructuredCouplings::from_mesh(mesh).expect("couplings"),
    };
    let step1 = advance_unstructured_step(&mut env, &mut fields, &mut work).expect("step1");
    let step2 = advance_unstructured_step(&mut env, &mut fields, &mut work).expect("step2");
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
