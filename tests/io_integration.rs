mod common;

use asimu::io::load_mesh_from_case;

#[test]
fn load_mesh_from_fixture() {
    let path = common::fixture_path("demo.case");
    let mesh = load_mesh_from_case(&path).expect("load mesh");
    assert_eq!(mesh.name, "demo-channel");
    assert_eq!(mesh.cell_count, 256);
}
