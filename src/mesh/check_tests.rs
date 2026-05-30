use super::*;
use crate::boundary::{BoundaryKind, BoundaryPatch};

#[test]
fn uniform_box_passes_checks() {
    let mesh = StructuredMesh3d::uniform_box("box", 4, 4, 2, 1.0, 2.0, 0.5).expect("mesh");
    let report = check_mesh3d(&mesh, None, "test").expect("check");
    assert!(report.passed());
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.code == "cell_volume" && f.severity == CheckSeverity::Info)
    );
}

#[test]
fn report_includes_spatial_bounds() {
    let mesh = StructuredMesh3d::uniform_box("box", 4, 4, 2, 1.0, 2.0, 0.5).expect("mesh");
    let report = check_mesh3d(&mesh, None, "test").expect("check");
    let text = format!("{report}");
    assert!(text.contains("x ∈ [0.000000, 1.000000]  (Lx ≈ 1.000000)"));
    assert!(text.contains("y ∈ [0.000000, 2.000000]  (Ly ≈ 2.000000)"));
    assert!(text.contains("z ∈ [0.000000, 0.500000]  (Lz ≈ 0.500000)"));
}

#[test]
fn report_lists_boundary_patches() {
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::mesh::BoundaryMesh;

    let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 1, 1.0, 1.0, 1.0).expect("mesh");
    let boundary = BoundarySet::new(vec![BoundaryPatch::new(
        "inflow",
        mesh.resolve_logical_boundary("i_min").expect("faces"),
        BoundaryKind::Inlet {
            total_pressure: 200_000.0,
            total_temperature: 300.0,
            velocity_direction: [1.0, 0.0, 0.0],
            mach: 2.0,
        },
    )]);
    let mut report = check_mesh3d(&mesh, Some(&boundary), "test").expect("check");
    report.boundary_note = Some("test".to_string());
    let text = format!("{report}");
    assert!(text.contains("boundary patches (1):"));
    assert!(text.contains("inflow"));
    assert!(text.contains("i_min×2"));
    assert!(text.contains("BC: M=2"));
}

#[test]
fn warns_on_empty_boundary_patch_list() {
    let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 1, 1.0, 1.0, 1.0).expect("mesh");
    let boundary = BoundarySet::new(vec![BoundaryPatch::new(
        "wall",
        vec![],
        BoundaryKind::Wall {
            no_slip: false,
            heat: crate::boundary::WallHeat::Adiabatic,
        },
    )]);
    let report = check_mesh3d(&mesh, Some(&boundary), "test").expect("check");
    assert!(!report.passed());
}

#[test]
fn cylinder_cgns_passes_when_present() {
    use std::path::PathBuf;

    use crate::io::{CaseMesh, load_case};

    let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
    if !case_path.is_file() {
        return;
    }
    let case = load_case(&case_path).expect("case");
    let CaseMesh::Structured3d(mesh) = &case.mesh else {
        panic!("3d");
    };
    let report = check_mesh3d(mesh, Some(&case.boundary), "cylinder").expect("check");
    assert!(
        report.passed(),
        "{}",
        report
            .findings
            .iter()
            .filter(|f| f.severity == CheckSeverity::Error)
            .map(|f| f.message.as_str())
            .collect::<Vec<_>>()
            .join("; ")
    );
}
