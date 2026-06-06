//! 结构化网格预检（计算前几何 / 度量 / 边界完整性）。

use std::collections::HashSet;
use std::fmt;

use crate::boundary::{BoundaryRegistry, BoundarySet};
use crate::core::FaceId;
use crate::error::{AsimuError, Result};

use super::{
    LogicalFace3d, MeshDiagnostics, MultiBlockStructuredMesh3d, StructuredMesh1d, StructuredMesh2d,
    StructuredMesh3d, mesh1d_diagnostics, mesh2d_diagnostics, mesh3d_diagnostics,
    multiblock_mesh3d_diagnostics,
};

/// 检查项严重级别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckSeverity {
    Info,
    Warn,
    Error,
}

/// 单条检查结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckFinding {
    pub code: &'static str,
    pub severity: CheckSeverity,
    pub message: String,
}

impl CheckFinding {
    fn info(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            severity: CheckSeverity::Info,
            message: message.into(),
        }
    }

    fn warn(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            severity: CheckSeverity::Warn,
            message: message.into(),
        }
    }

    fn error(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            severity: CheckSeverity::Error,
            message: message.into(),
        }
    }
}

/// 网格预检选项。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MeshCheckOptions {
    /// 将警告视为错误（适合 CI）。
    pub strict: bool,
}

/// 单条边界 patch 摘要（名称、逻辑面分布、边界条件）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryPatchReport {
    pub name: String,
    pub kind: String,
    pub detail: String,
    pub faces: usize,
    pub logical_faces: String,
}

/// 网格预检报告。
#[derive(Debug, Clone, PartialEq)]
pub struct MeshCheckReport {
    pub source: String,
    pub diagnostics: MeshDiagnostics,
    pub boundary_patches: Vec<BoundaryPatchReport>,
    pub boundary_note: Option<String>,
    pub findings: Vec<CheckFinding>,
}

impl MeshCheckReport {
    #[must_use]
    pub fn passed(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|f| effective_severity(f, MeshCheckOptions::default()) == CheckSeverity::Error)
    }

    #[must_use]
    pub fn passed_with(&self, opts: MeshCheckOptions) -> bool {
        !self
            .findings
            .iter()
            .any(|f| effective_severity(f, opts) == CheckSeverity::Error)
    }

    #[must_use]
    pub fn count_errors(&self, opts: MeshCheckOptions) -> usize {
        self.findings
            .iter()
            .filter(|f| effective_severity(f, opts) == CheckSeverity::Error)
            .count()
    }

    #[must_use]
    pub fn count_warnings(&self, opts: MeshCheckOptions) -> usize {
        self.findings
            .iter()
            .filter(|f| effective_severity(f, opts) == CheckSeverity::Warn)
            .count()
    }
}

fn effective_severity(finding: &CheckFinding, opts: MeshCheckOptions) -> CheckSeverity {
    if opts.strict && finding.severity == CheckSeverity::Warn {
        CheckSeverity::Error
    } else {
        finding.severity
    }
}

impl fmt::Display for MeshCheckReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let opts = MeshCheckOptions::default();
        write_mesh_check_report(f, self, opts)
    }
}

/// 带选项的可读报告包装。
pub struct MeshCheckReportDisplay<'a> {
    pub report: &'a MeshCheckReport,
    pub opts: MeshCheckOptions,
}

impl fmt::Display for MeshCheckReportDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_mesh_check_report(f, self.report, self.opts)
    }
}

/// 格式化预检报告（人类可读）。
pub fn write_mesh_check_report(
    f: &mut fmt::Formatter<'_>,
    report: &MeshCheckReport,
    opts: MeshCheckOptions,
) -> fmt::Result {
    write_mesh_geometry_section(f, report)?;
    write_boundary_patch_section(f, report)?;
    write_findings_section(f, report, opts)
}

fn write_mesh_geometry_section(
    f: &mut fmt::Formatter<'_>,
    report: &MeshCheckReport,
) -> fmt::Result {
    let d = &report.diagnostics;
    writeln!(f, "source: {}", report.source)?;
    writeln!(
        f,
        "mesh: {}  dim={}  cells={} ({})  nodes={}",
        d.name,
        d.dimension,
        d.num_cells,
        d.cell_dims_label(),
        d.num_nodes
    )?;
    write_bounds_line(f, "x", d.bounds.x)?;
    if d.dimension >= 2 {
        write_bounds_line(f, "y", d.bounds.y)?;
    }
    if d.dimension >= 3 {
        write_bounds_line(f, "z", d.bounds.z)?;
    }
    write_spacing_and_diag_warnings(f, d)
}

fn write_spacing_and_diag_warnings(f: &mut fmt::Formatter<'_>, d: &MeshDiagnostics) -> fmt::Result {
    if let Some(spacing) = d.spacing {
        write_spacing_line(f, "Δx", spacing.dx)?;
        if d.dimension >= 2 {
            write_spacing_line(f, "Δy", spacing.dy)?;
        }
        if d.dimension >= 3 {
            write_spacing_line(f, "Δz", spacing.dz)?;
        }
    }
    for warning in &d.warnings {
        writeln!(f, "  diag: {warning}")?;
    }
    Ok(())
}

fn write_boundary_patch_section(
    f: &mut fmt::Formatter<'_>,
    report: &MeshCheckReport,
) -> fmt::Result {
    if let Some(note) = &report.boundary_note {
        writeln!(f, "boundary: {note}")?;
    }
    if report.boundary_patches.is_empty() {
        return Ok(());
    }
    writeln!(f, "boundary patches ({}):", report.boundary_patches.len())?;
    for patch in &report.boundary_patches {
        writeln!(
            f,
            "  {:<20} {:<18} faces={:<5}  {}",
            patch.name, patch.kind, patch.faces, patch.logical_faces
        )?;
        writeln!(f, "    BC: {}", patch.detail)?;
    }
    Ok(())
}

fn write_findings_section(
    f: &mut fmt::Formatter<'_>,
    report: &MeshCheckReport,
    opts: MeshCheckOptions,
) -> fmt::Result {
    let mut infos = 0usize;
    let mut warns = 0usize;
    let mut errors = 0usize;
    for finding in &report.findings {
        let sev = effective_severity(finding, opts);
        let tag = match sev {
            CheckSeverity::Info => {
                infos += 1;
                "info"
            }
            CheckSeverity::Warn => {
                warns += 1;
                "warn"
            }
            CheckSeverity::Error => {
                errors += 1;
                "ERROR"
            }
        };
        writeln!(f, "[{tag}] {}: {}", finding.code, finding.message)?;
    }

    writeln!(
        f,
        "summary: info={infos} warn={warns} error={errors}  status={}",
        if report.passed_with(opts) {
            "PASS"
        } else {
            "FAIL"
        }
    )
}

fn write_bounds_line(
    f: &mut fmt::Formatter<'_>,
    axis: &str,
    range: super::CoordRange,
) -> fmt::Result {
    let span = range.span();
    if span <= 0.0 {
        return Ok(());
    }
    if span < 1.0e-12 {
        writeln!(f, "  {axis} ≈ {:.6}", range.min)
    } else {
        writeln!(
            f,
            "  {axis} ∈ [{:.6}, {:.6}]  (L{axis} ≈ {span:.6})",
            range.min, range.max
        )
    }
}

fn write_spacing_line(
    f: &mut fmt::Formatter<'_>,
    label: &str,
    range: super::CoordRange,
) -> fmt::Result {
    if range.max <= 0.0 {
        return Ok(());
    }
    if (range.max - range.min).abs() < 1.0e-12 {
        writeln!(f, "  {label} ≈ {:.6}", range.min)
    } else {
        writeln!(f, "  {label} ∈ [{:.6}, {:.6}]", range.min, range.max)
    }
}

#[must_use]
pub fn check_mesh1d(mesh: &StructuredMesh1d, source: impl Into<String>) -> MeshCheckReport {
    let mut findings = Vec::new();
    check_coordinates_finite_1d(mesh, &mut findings);
    MeshCheckReport {
        source: source.into(),
        diagnostics: mesh1d_diagnostics(mesh),
        boundary_patches: Vec::new(),
        boundary_note: None,
        findings,
    }
}

#[must_use]
pub fn check_mesh2d(mesh: &StructuredMesh2d, source: impl Into<String>) -> MeshCheckReport {
    let mut findings = Vec::new();
    check_coordinates_finite_2d(mesh, &mut findings);
    promote_diagnostic_warnings(mesh2d_diagnostics(mesh).warnings, &mut findings);
    MeshCheckReport {
        source: source.into(),
        diagnostics: mesh2d_diagnostics(mesh),
        boundary_patches: Vec::new(),
        boundary_note: None,
        findings,
    }
}

/// 3D 结构化网格预检（含曲线度量与边界 patch）。
pub fn check_mesh3d(
    mesh: &StructuredMesh3d,
    boundary: Option<&BoundarySet>,
    source: impl Into<String>,
) -> Result<MeshCheckReport> {
    let mut findings = Vec::new();
    check_coordinates_finite_3d(mesh, &mut findings);
    promote_diagnostic_warnings(mesh3d_diagnostics(mesh).warnings.clone(), &mut findings);

    findings.push(CheckFinding::info(
        "metric_mode",
        format!(
            "度量模式: {}（cache={}）",
            if mesh.uses_curvilinear_metrics() {
                "curvilinear"
            } else {
                "cartesian"
            },
            if mesh.metric_cache().is_some() {
                "yes"
            } else {
                "no"
            }
        ),
    ));

    if mesh.uses_curvilinear_metrics() && mesh.metric_cache().is_none() {
        findings.push(CheckFinding::error(
            "metric_cache_missing",
            "曲线网格未构建 MetricCache",
        ));
    }

    check_cell_volumes_3d(mesh, &mut findings);
    check_face_metrics_3d(mesh, &mut findings);

    match mesh.min_positive_spacing() {
        Ok(h) => findings.push(CheckFinding::info(
            "min_cell_spacing",
            format!("最小正单元间距 h ≈ {h:.6e}"),
        )),
        Err(err) => findings.push(CheckFinding::error("min_cell_spacing", format!("{err}"))),
    }

    if mesh.uses_curvilinear_metrics() {
        match mesh.min_positive_face_spacing() {
            Ok(h) => findings.push(CheckFinding::info(
                "min_face_spacing",
                format!("最小正面间距 h_f ≈ {h:.6e}"),
            )),
            Err(err) => findings.push(CheckFinding::error("min_face_spacing", format!("{err}"))),
        }
    }

    let mut boundary_patches = Vec::new();
    if let Some(boundary) = boundary {
        check_boundary_patches(boundary, mesh, &mut findings, &mut boundary_patches);
    } else {
        findings.push(CheckFinding::warn(
            "boundary_missing",
            "未提供边界 patch（仅几何检查）",
        ));
    }

    Ok(MeshCheckReport {
        source: source.into(),
        diagnostics: mesh3d_diagnostics(mesh),
        boundary_patches,
        boundary_note: None,
        findings,
    })
}

/// 多块 3D 结构化网格预检（逐 block 几何诊断；接口连通校验待补充）。
pub fn check_multiblock_mesh3d(
    mesh: &MultiBlockStructuredMesh3d,
    source: impl Into<String>,
) -> Result<MeshCheckReport> {
    let mut findings = Vec::new();
    for block in mesh.blocks() {
        let block_report = check_mesh3d(&block.mesh, None, format!("block {}", block.name))?;
        for finding in block_report.findings {
            if finding.code == "boundary_missing" {
                continue;
            }
            findings.push(CheckFinding {
                code: finding.code,
                severity: finding.severity,
                message: format!("block {}: {}", block.name, finding.message),
            });
        }
    }
    if !mesh.interfaces().is_empty() {
        findings.push(CheckFinding::warn(
            "multiblock_interfaces",
            format!(
                "多块网格含 {} 条 1-to-1 接口：暂未校验接口几何连通；可压缩求解仅支持 LU-SGS 对角隐式",
                mesh.interfaces().len()
            ),
        ));
    } else if mesh.num_blocks() > 1 {
        findings.push(CheckFinding::warn(
            "multiblock_no_interfaces",
            "多块网格无 block 间接口：各 block 独立同步推进",
        ));
    }

    Ok(MeshCheckReport {
        source: source.into(),
        diagnostics: multiblock_mesh3d_diagnostics(mesh),
        boundary_patches: Vec::new(),
        boundary_note: Some(format!(
            "多块结构化 3D 网格：{} 个 block，{} 条接口",
            mesh.num_blocks(),
            mesh.interfaces().len()
        )),
        findings,
    })
}

fn promote_diagnostic_warnings(warnings: Vec<String>, findings: &mut Vec<CheckFinding>) {
    for warning in warnings {
        findings.push(CheckFinding::warn("diagnostics", warning));
    }
}

fn check_coordinates_finite_1d(mesh: &StructuredMesh1d, findings: &mut Vec<CheckFinding>) {
    if !mesh.origin.is_finite() || !mesh.length.is_finite() {
        findings.push(CheckFinding::error(
            "coords_finite",
            "1D 网格 origin/length 含非有限值",
        ));
    }
}

fn check_coordinates_finite_2d(mesh: &StructuredMesh2d, findings: &mut Vec<CheckFinding>) {
    let mut bad = 0usize;
    for coords in [&mesh.points_x, &mesh.points_y] {
        for &v in coords {
            if !v.is_finite() {
                bad += 1;
            }
        }
    }
    if bad > 0 {
        findings.push(CheckFinding::error(
            "coords_finite",
            format!("2D 节点坐标含 {bad} 个非有限值"),
        ));
    }
}

fn check_coordinates_finite_3d(mesh: &StructuredMesh3d, findings: &mut Vec<CheckFinding>) {
    let mut bad = 0usize;
    for coords in [&mesh.points_x, &mesh.points_y, &mesh.points_z] {
        for &v in coords {
            if !v.is_finite() {
                bad += 1;
            }
        }
    }
    if bad > 0 {
        findings.push(CheckFinding::error(
            "coords_finite",
            format!("3D 节点坐标含 {bad} 个非有限值"),
        ));
    }
}

fn check_cell_volumes_3d(mesh: &StructuredMesh3d, findings: &mut Vec<CheckFinding>) {
    let eps = 1.0e-30_f64;
    let mut zero = 0usize;
    let mut min_vol = f64::INFINITY;
    let mut worst = (0usize, 0usize, 0usize);
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let vol = mesh.cell_metric(i, j, k).volume;
                if !vol.is_finite() || vol <= eps {
                    zero += 1;
                }
                if vol < min_vol {
                    min_vol = vol;
                    worst = (i, j, k);
                }
            }
        }
    }
    if zero > 0 {
        findings.push(CheckFinding::error(
            "cell_volume",
            format!(
                "{zero}/{} 单元体积非正或非有限；最小体积≈{min_vol:.6e} @ ({},{},{})",
                mesh.num_cells(),
                worst.0,
                worst.1,
                worst.2
            ),
        ));
    } else {
        findings.push(CheckFinding::info(
            "cell_volume",
            format!(
                "全部 {} 单元体积 > 0；V_min ≈ {min_vol:.6e} @ ({},{},{})",
                mesh.num_cells(),
                worst.0,
                worst.1,
                worst.2
            ),
        ));
    }
}

fn check_face_metrics_3d(mesh: &StructuredMesh3d, findings: &mut Vec<CheckFinding>) {
    let mut stats = FaceMetricStats::new();
    scan_interior_face_metrics(mesh, &mut stats);
    scan_boundary_face_samples(mesh, &mut stats);
    push_face_metric_findings(findings, &stats);
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

fn scan_interior_face_metrics(mesh: &StructuredMesh3d, stats: &mut FaceMetricStats) {
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                let face = mesh.i_face_metric(i, j, k);
                stats.record(face.area, face.normal.magnitude());
            }
        }
    }
    for k in 0..mesh.nz {
        for j in 0..mesh.ny.saturating_sub(1) {
            for i in 0..mesh.nx {
                let face = mesh.j_face_metric(i, j, k);
                stats.record(face.area, face.normal.magnitude());
            }
        }
    }
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let face = mesh.k_face_metric(i, j, k);
                stats.record(face.area, face.normal.magnitude());
            }
        }
    }
}

fn scan_boundary_face_samples(mesh: &StructuredMesh3d, stats: &mut FaceMetricStats) {
    use super::LogicalFace3d;
    let sample_faces = [
        (LogicalFace3d::IMin, 0, 0, 0),
        (LogicalFace3d::IMax, mesh.nx.saturating_sub(1), 0, 0),
        (LogicalFace3d::JMin, 0, 0, 0),
        (LogicalFace3d::JMax, 0, mesh.ny.saturating_sub(1), 0),
        (LogicalFace3d::KMin, 0, 0, 0),
        (LogicalFace3d::KMax, 0, 0, mesh.nz.saturating_sub(1)),
    ];
    for (logical, i, j, k) in sample_faces {
        let face = mesh.boundary_face_metric(logical, i, j, k);
        stats.record(face.area, face.normal.magnitude());
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
            format!("抽样面面积均为正；A_min ≈ {:.6e}", stats.min_area),
        ));
    }

    if stats.bad_normal > 0 {
        findings.push(CheckFinding::error(
            "face_normal",
            format!("{} 个面法向未归一化（|n|-1 > 1e-6）", stats.bad_normal),
        ));
    }
}

fn check_boundary_patches(
    boundary: &BoundarySet,
    mesh: &StructuredMesh3d,
    findings: &mut Vec<CheckFinding>,
    boundary_patches: &mut Vec<BoundaryPatchReport>,
) {
    if let Err(err) = BoundaryRegistry::validate_patches(boundary.patches()) {
        findings.push(CheckFinding::error("boundary_patch", format!("{err}")));
        return;
    }

    let total_faces = boundary_face_count(mesh);
    let mut seen = HashSet::new();
    let mut duplicates = 0usize;
    let mut assigned = 0usize;
    for patch in boundary.patches() {
        for &face in &patch.face_ids {
            assigned += 1;
            if !seen.insert(face) {
                duplicates += 1;
            }
            if let Err(err) = validate_face_on_mesh(mesh, face) {
                findings.push(CheckFinding::error(
                    "boundary_face_id",
                    format!("patch \"{}\": {err}", patch.name),
                ));
            }
        }
    }

    if duplicates > 0 {
        findings.push(CheckFinding::warn(
            "boundary_overlap",
            format!("{duplicates} 个边界面被多个 patch 重复引用"),
        ));
    }

    let uncovered = total_faces.saturating_sub(seen.len());
    if uncovered > 0 {
        findings.push(CheckFinding::warn(
            "boundary_coverage",
            format!("{uncovered}/{total_faces} 个边界面未分配 patch"),
        ));
    } else if duplicates > 0 {
        findings.push(CheckFinding::info(
            "boundary_coverage",
            format!(
                "全部 {total_faces} 个边界面均有 patch 覆盖（{assigned} 条 face_id，{duplicates} 条重复）"
            ),
        ));
    } else {
        findings.push(CheckFinding::info(
            "boundary_coverage",
            format!("{total_faces} 个边界面均已分配 patch"),
        ));
    }

    findings.push(CheckFinding::info(
        "boundary_patches",
        format!("{} 个边界 patch", boundary.patches().len()),
    ));

    let mut patch_reports: Vec<BoundaryPatchReport> = boundary
        .patches()
        .iter()
        .map(|patch| BoundaryPatchReport {
            name: patch.name.clone(),
            kind: patch.kind.summary_label().to_string(),
            detail: patch.kind.detail_label(),
            faces: patch.face_ids.len(),
            logical_faces: summarize_patch_logical_faces(&patch.face_ids),
        })
        .collect();
    patch_reports.sort_by(|a, b| a.name.cmp(&b.name));
    boundary_patches.extend(patch_reports);
}

fn summarize_patch_logical_faces(face_ids: &[FaceId]) -> String {
    let mut counts = [0usize; LogicalFace3d::COUNT as usize];
    for &face in face_ids {
        if let Ok((logical, _)) = LogicalFace3d::decode(face) {
            counts[logical.tag() as usize] += 1;
        }
    }
    let mut parts = Vec::new();
    for logical in [
        LogicalFace3d::IMin,
        LogicalFace3d::IMax,
        LogicalFace3d::JMin,
        LogicalFace3d::JMax,
        LogicalFace3d::KMin,
        LogicalFace3d::KMax,
    ] {
        let count = counts[logical.tag() as usize];
        if count > 0 {
            parts.push(format!("{}×{count}", logical.label()));
        }
    }
    if parts.is_empty() {
        "(no faces)".to_string()
    } else {
        parts.join(", ")
    }
}

fn boundary_face_count(mesh: &StructuredMesh3d) -> usize {
    2 * mesh.ny * mesh.nz + 2 * mesh.nx * mesh.nz + 2 * mesh.nx * mesh.ny
}

fn validate_face_on_mesh(mesh: &StructuredMesh3d, face: FaceId) -> Result<()> {
    use super::LogicalFace3d;
    let (logical, local) = LogicalFace3d::decode(face)?;
    let max_local = match logical {
        LogicalFace3d::IMin | LogicalFace3d::IMax => mesh.ny * mesh.nz,
        LogicalFace3d::JMin | LogicalFace3d::JMax => mesh.nx * mesh.nz,
        LogicalFace3d::KMin | LogicalFace3d::KMax => mesh.nx * mesh.ny,
    };
    if local as usize >= max_local {
        return Err(AsimuError::Mesh(format!(
            "FaceId 局部索引 {local} 超出逻辑面容量 {max_local}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "check_tests.rs"]
mod tests;
