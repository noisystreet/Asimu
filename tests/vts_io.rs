//! VTS 二进制读入集成测试（feature `io-vtk`）。

#![cfg(feature = "io-vtk")]

use std::path::PathBuf;

use asimu::io::load_vts;

fn mesh_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/mesh")
        .join(name)
}

#[test]
fn integration_loads_binary_vts() {
    let path = mesh_fixture("unit_square_2x2_binary.vts");
    let loaded = load_vts(&path).expect("load vts");
    assert_eq!(loaded.mesh.name(), "unit_square_2x2_binary");
    assert_eq!(loaded.mesh.num_cells(), 4);
}

#[test]
fn integration_rejects_ascii_vts() {
    let path = mesh_fixture("ascii_reject.vts");
    assert!(load_vts(&path).is_err());
}
