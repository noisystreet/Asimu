//! 可压缩 NS 边界与初始条件集成测试。

use asimu::boundary::BoundaryKind;
use asimu::discretization::{apply_compressible_boundary_conditions, BoundaryGhostBuffer};
use asimu::io::{parse_case_str, write_conserved_fields};
use asimu::physics::{FreestreamParams, IdealGasEoS};

#[test]
fn compressible_case_builds_and_applies_bc() {
    let content = r#"
name = "cns_box"
[mesh]
kind = "structured_3d"
nx = 3
ny = 3
nz = 3
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.2
[boundary.i_min]
kind = "wall"
[boundary.i_max]
kind = "farfield"
mach = 0.2
[boundary.j_min]
kind = "symmetry"
[boundary.j_max]
kind = "symmetry"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "outlet"
static_pressure = 100000.0
"#;
    let case = parse_case_str(content).expect("parse");
    let eos = case.physics.eos().expect("eos");
    let fields = case.build_conserved_fields().expect("fields");
    let mesh = case.mesh.as_3d().expect("3d");
    let fs = case.freestream.unwrap_or_default();
    let patches = &case.boundary;
    let mut ghosts = BoundaryGhostBuffer::new();
    apply_compressible_boundary_conditions(mesh, patches, &fields, &mut ghosts, &eos, &fs)
        .expect("bc");
}

#[test]
fn periodic_patch_kind_parses() {
    let content = r#"
name = "periodic"
[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.0
[boundary.i_min]
kind = "periodic"
partner = "i_max"
[boundary.i_max]
kind = "periodic"
partner = "i_min"
[boundary.j_min]
kind = "wall"
[boundary.j_max]
kind = "wall"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "wall"
"#;
    let case = parse_case_str(content).expect("parse");
    let left = case.boundary.find("i_min").expect("left");
    assert!(matches!(left.kind, BoundaryKind::Periodic { .. }));
}

#[test]
fn restart_roundtrip_via_case() {
    let eos = IdealGasEoS::AIR_STANDARD;
    let fields =
        asimu::field::ConservedFields::from_freestream(2, &eos, &FreestreamParams::default())
            .expect("fields");
    let path = std::env::temp_dir().join("asimu_case_restart.toml");
    write_conserved_fields(&path, &fields).expect("write");
    let content = format!(
        r#"
name = "restart_case"
[mesh]
kind = "structured_3d"
nx = 1
ny = 1
nz = 2
[physics]
gamma = 1.4
gas_constant = 287.0
[restart]
path = "{}"
[boundary.i_min]
kind = "wall"
[boundary.i_max]
kind = "wall"
[boundary.j_min]
kind = "wall"
[boundary.j_max]
kind = "wall"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "wall"
"#,
        path.display()
    );
    let case = parse_case_str(&content).expect("parse");
    let loaded = case.build_conserved_fields().expect("restart");
    assert_eq!(loaded.num_cells(), 2);
    let _ = std::fs::remove_file(path);
}
