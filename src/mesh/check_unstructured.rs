//! 非结构 3D 网格预检。

use std::collections::HashSet;

use crate::boundary::{BoundaryRegistry, BoundarySet};
use crate::core::{CellId, FaceId};

use super::{
    BoundaryPatchReport, CellKind, CheckFinding, CheckSeverity, MeshCheckReport,
    UnstructuredMesh3d, unstructured_mesh3d_diagnostics,
};

/// 非结构 3D 网格预检（拓扑、体积、面度量）。
#[must_use]
pub fn check_unstructured_mesh3d(
    mesh: &UnstructuredMesh3d,
    boundary: Option<&BoundarySet>,
    source: impl Into<String>,
) -> MeshCheckReport {
    let mut findings = Vec::new();
    check_unstructured_cell_volumes(mesh, &mut findings);
    check_unstructured_face_metrics(mesh, &mut findings);
    summarize_unstructured_topology(mesh, &mut findings);
    let boundary_patches = summarize_boundary_patches(boundary);
    let boundary_note = check_unstructured_boundary(mesh, boundary, &mut findings);
    MeshCheckReport {
        source: source.into(),
        diagnostics: unstructured_mesh3d_diagnostics(mesh),
        boundary_patches,
        boundary_note,
        findings,
    }
}

fn check_unstructured_cell_volumes(mesh: &UnstructuredMesh3d, findings: &mut Vec<CheckFinding>) {
    let mut bad = 0usize;
    let mut min_vol = f64::INFINITY;
    let mut worst = 0usize;
    for cell in 0..mesh.num_cells() {
        let volume = mesh.cell_metric(CellId(cell as u32)).volume;
        if volume < min_vol {
            min_vol = volume;
            worst = cell;
        }
        if !volume.is_finite() || volume <= 1.0e-30 {
            bad += 1;
        }
    }
    if bad > 0 {
        findings.push(CheckFinding::error(
            "cell_volume",
            format!(
                "{bad}/{} 单元体积非正或非有限；V_min≈{min_vol:.6e} @ cell {worst}",
                mesh.num_cells()
            ),
        ));
    } else {
        findings.push(CheckFinding::info(
            "cell_volume",
            format!(
                "全部 {} 单元体积 > 0；V_min≈{min_vol:.6e} @ cell {worst}",
                mesh.num_cells()
            ),
        ));
    }
}

fn check_unstructured_face_metrics(mesh: &UnstructuredMesh3d, findings: &mut Vec<CheckFinding>) {
    let mut stats = FaceMetricStats::new();
    for face in 0..mesh.num_faces() {
        let metric = mesh.face_metric(FaceId(face as u32));
        stats.record(metric.area, metric.normal.magnitude());
    }
    push_face_metric_findings(findings, &stats);
}

fn summarize_unstructured_topology(mesh: &UnstructuredMesh3d, findings: &mut Vec<CheckFinding>) {
    let mut boundary = 0usize;
    let mut interior = 0usize;
    for face in 0..mesh.num_faces() {
        match mesh.face_neighbor(FaceId(face as u32)) {
            Ok(Some(_)) => interior += 1,
            Ok(None) => boundary += 1,
            Err(err) => findings.push(CheckFinding::error("face_neighbor", format!("{err}"))),
        }
    }
    findings.push(CheckFinding::info(
        "face_topology",
        format!(
            "faces={} interior={} boundary={boundary}",
            mesh.num_faces(),
            interior
        ),
    ));
    findings.push(CheckFinding::info("cell_kinds", summarize_cell_kinds(mesh)));
}

fn summarize_cell_kinds(mesh: &UnstructuredMesh3d) -> String {
    let mut tet = 0usize;
    let mut hex = 0usize;
    let mut pyramid = 0usize;
    let mut prism = 0usize;
    for cell in mesh.cells() {
        match cell.kind {
            CellKind::Tet => tet += 1,
            CellKind::Hex => hex += 1,
            CellKind::Pyramid => pyramid += 1,
            CellKind::Prism => prism += 1,
        }
    }
    format!("tet={tet} hex={hex} pyramid={pyramid} prism={prism}")
}

fn summarize_boundary_patches(boundary: Option<&BoundarySet>) -> Vec<BoundaryPatchReport> {
    boundary
        .map(|set| {
            set.patches()
                .iter()
                .map(|patch| BoundaryPatchReport {
                    name: patch.name.clone(),
                    faces: patch.face_ids.len(),
                    logical_faces: "unstructured".to_string(),
                    kind: patch.kind.summary_label().to_string(),
                    detail: patch.kind.detail_label(),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn check_unstructured_boundary(
    mesh: &UnstructuredMesh3d,
    boundary: Option<&BoundarySet>,
    findings: &mut Vec<CheckFinding>,
) -> Option<String> {
    let Some(boundary) = boundary else {
        return Some("非结构网格：未提供边界 patch，仅检查几何/拓扑".to_string());
    };
    if boundary.patches().is_empty() {
        findings.push(CheckFinding::warn(
            "boundary_patches",
            "非结构网格未读到任何边界 patch",
        ));
        return Some("未读到边界 patch".to_string());
    }
    if let Err(err) = BoundaryRegistry::validate_patches(boundary.patches()) {
        findings.push(CheckFinding::error("boundary_registry", format!("{err}")));
    }
    let boundary_face_count = count_unstructured_boundary_faces(mesh, findings);
    let mut covered = HashSet::new();
    let mut duplicate = 0usize;
    let mut invalid = 0usize;
    let mut interior = 0usize;
    for patch in boundary.patches() {
        for &face in &patch.face_ids {
            let index = face.index() as usize;
            if index >= mesh.num_faces() {
                invalid += 1;
                continue;
            }
            match mesh.face_neighbor(face) {
                Ok(Some(_)) => interior += 1,
                Ok(None) => {
                    if !covered.insert(face.index()) {
                        duplicate += 1;
                    }
                }
                Err(_) => invalid += 1,
            }
        }
    }
    push_boundary_findings(
        findings,
        boundary_face_count,
        covered.len(),
        duplicate,
        invalid,
        interior,
    );
    Some("来自非结构网格边界 patch".to_string())
}

fn count_unstructured_boundary_faces(
    mesh: &UnstructuredMesh3d,
    findings: &mut Vec<CheckFinding>,
) -> usize {
    let mut count = 0usize;
    for face in 0..mesh.num_faces() {
        match mesh.face_neighbor(FaceId(face as u32)) {
            Ok(Some(_)) => {}
            Ok(None) => count += 1,
            Err(err) => findings.push(CheckFinding::error("face_neighbor", format!("{err}"))),
        }
    }
    count
}

fn push_boundary_findings(
    findings: &mut Vec<CheckFinding>,
    boundary_face_count: usize,
    covered: usize,
    duplicate: usize,
    invalid: usize,
    interior: usize,
) {
    if invalid > 0 || interior > 0 || duplicate > 0 {
        findings.push(CheckFinding::error(
            "boundary_patch_faces",
            format!(
                "patch face 引用异常：invalid={invalid} interior={interior} duplicate={duplicate}"
            ),
        ));
    }
    if covered == boundary_face_count {
        findings.push(CheckFinding::info(
            "boundary_coverage",
            format!("边界 patch 覆盖全部 {boundary_face_count} 个非结构边界面"),
        ));
    } else {
        findings.push(CheckFinding::warn(
            "boundary_coverage",
            format!("边界 patch 覆盖 {covered}/{boundary_face_count} 个非结构边界面"),
        ));
    }
}

#[derive(Default)]
struct FaceMetricStats {
    bad_area: usize,
    bad_normal: usize,
    min_area: f64,
}

impl FaceMetricStats {
    fn new() -> Self {
        Self {
            bad_area: 0,
            bad_normal: 0,
            min_area: f64::INFINITY,
        }
    }

    fn record(&mut self, area: f64, normal_len: f64) {
        if area < self.min_area {
            self.min_area = area;
        }
        if !area.is_finite() || area <= 1.0e-30 {
            self.bad_area += 1;
        }
        if !normal_len.is_finite() || (normal_len - 1.0).abs() > 1.0e-6 {
            self.bad_normal += 1;
        }
    }
}

fn push_face_metric_findings(findings: &mut Vec<CheckFinding>, stats: &FaceMetricStats) {
    if stats.bad_area > 0 {
        findings.push(CheckFinding::error(
            "face_area",
            format!(
                "{} 个面面积非正或非有限；A_min ≈ {:.6e}",
                stats.bad_area, stats.min_area
            ),
        ));
    } else {
        findings.push(CheckFinding::info(
            "face_area",
            format!("全部非结构面面积均为正；A_min ≈ {:.6e}", stats.min_area),
        ));
    }

    if stats.bad_normal > 0 {
        findings.push(CheckFinding {
            code: "face_normal",
            severity: CheckSeverity::Error,
            message: format!("{} 个面法向未归一化（|n|-1 > 1e-6）", stats.bad_normal),
        });
    }
}
