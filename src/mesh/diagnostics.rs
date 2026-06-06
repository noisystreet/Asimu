//! 结构化网格几何诊断（坐标范围、间距、简单一致性检查）。

use super::{
    MultiBlockStructuredMesh3d, StructuredMesh, StructuredMesh1d, StructuredMesh2d,
    StructuredMesh3d,
};

/// 单轴坐标范围。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordRange {
    pub min: f64,
    pub max: f64,
}

impl CoordRange {
    #[must_use]
    pub fn span(&self) -> f64 {
        self.max - self.min
    }
}

/// 节点坐标轴范围（2D/3D 网格未使用的轴为 `[0, 0]`）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshBounds {
    pub x: CoordRange,
    pub y: CoordRange,
    pub z: CoordRange,
}

/// 结构化网格沿 i/j/k 方向的边长统计。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpacingStats {
    pub dx: CoordRange,
    pub dy: CoordRange,
    pub dz: CoordRange,
}

/// 结构化网格诊断摘要。
#[derive(Debug, Clone, PartialEq)]
pub struct MeshDiagnostics {
    pub name: String,
    pub dimension: usize,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub num_cells: usize,
    pub num_nodes: usize,
    pub bounds: MeshBounds,
    pub spacing: Option<SpacingStats>,
    pub warnings: Vec<String>,
}

impl MeshDiagnostics {
    #[must_use]
    pub fn cell_dims_label(&self) -> String {
        match self.dimension {
            1 => format!("{}", self.nx),
            2 => format!("{}×{}", self.nx, self.ny),
            _ => format!("{}×{}×{}", self.nx, self.ny, self.nz),
        }
    }
}

#[must_use]
pub fn structured_mesh_diagnostics(mesh: &StructuredMesh) -> MeshDiagnostics {
    match mesh {
        StructuredMesh::D2(m) => mesh2d_diagnostics(m),
        StructuredMesh::D3(m) => mesh3d_diagnostics(m),
    }
}

#[must_use]
pub fn mesh1d_diagnostics(mesh: &StructuredMesh1d) -> MeshDiagnostics {
    let x = CoordRange {
        min: mesh.origin,
        max: mesh.origin + mesh.length,
    };
    let dx = mesh.dx();
    MeshDiagnostics {
        name: mesh.name.clone(),
        dimension: 1,
        nx: mesh.num_cells(),
        ny: 1,
        nz: 1,
        num_cells: mesh.num_cells(),
        num_nodes: mesh.num_cells() + 1,
        bounds: MeshBounds {
            x,
            y: CoordRange { min: 0.0, max: 0.0 },
            z: CoordRange { min: 0.0, max: 0.0 },
        },
        spacing: Some(SpacingStats {
            dx: CoordRange { min: dx, max: dx },
            dy: CoordRange { min: 0.0, max: 0.0 },
            dz: CoordRange { min: 0.0, max: 0.0 },
        }),
        warnings: Vec::new(),
    }
}

#[must_use]
pub fn mesh2d_diagnostics(mesh: &StructuredMesh2d) -> MeshDiagnostics {
    let bounds = bounds_from_points(&mesh.points_x, &mesh.points_y, &[]);
    let spacing = spacing_stats_2d(mesh);
    let warnings = collect_warnings_2d(mesh, spacing.as_ref());
    MeshDiagnostics {
        name: mesh.name.clone(),
        dimension: 2,
        nx: mesh.nx,
        ny: mesh.ny,
        nz: 1,
        num_cells: mesh.num_cells(),
        num_nodes: mesh.num_nodes(),
        bounds,
        spacing,
        warnings,
    }
}

#[must_use]
pub fn mesh3d_diagnostics(mesh: &StructuredMesh3d) -> MeshDiagnostics {
    let bounds = bounds_from_points(&mesh.points_x, &mesh.points_y, &mesh.points_z);
    let spacing = spacing_stats_3d(mesh);
    let mut warnings = collect_spacing_warnings(spacing.as_ref());
    if mesh.nz == 1 {
        warnings.push("nz=1：准 2D 挤出网格".to_string());
    }
    MeshDiagnostics {
        name: mesh.name.clone(),
        dimension: 3,
        nx: mesh.nx,
        ny: mesh.ny,
        nz: mesh.nz,
        num_cells: mesh.num_cells(),
        num_nodes: mesh.num_nodes(),
        bounds,
        spacing,
        warnings,
    }
}

#[must_use]
pub fn multiblock_mesh3d_diagnostics(mesh: &MultiBlockStructuredMesh3d) -> MeshDiagnostics {
    let mut bounds = MeshBounds {
        x: CoordRange {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        },
        y: CoordRange {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        },
        z: CoordRange {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        },
    };
    let mut spacing: Option<SpacingStats> = None;
    let mut warnings = Vec::new();

    for block in mesh.blocks() {
        let diag = mesh3d_diagnostics(&block.mesh);
        bounds.x = merge_range(bounds.x, diag.bounds.x);
        bounds.y = merge_range(bounds.y, diag.bounds.y);
        bounds.z = merge_range(bounds.z, diag.bounds.z);
        spacing = merge_spacing(spacing, diag.spacing);
        for warning in diag.warnings {
            warnings.push(format!("block {}: {warning}", block.name));
        }
    }

    warnings.push("多块网格首版仅支持读入/诊断，求解器尚不跨 block 装配".to_string());
    MeshDiagnostics {
        name: mesh.name.clone(),
        dimension: 3,
        nx: mesh.num_blocks(),
        ny: 1,
        nz: 1,
        num_cells: mesh.num_cells(),
        num_nodes: mesh.num_nodes(),
        bounds,
        spacing,
        warnings,
    }
}

fn merge_range(a: CoordRange, b: CoordRange) -> CoordRange {
    CoordRange {
        min: a.min.min(b.min),
        max: a.max.max(b.max),
    }
}

fn merge_spacing(a: Option<SpacingStats>, b: Option<SpacingStats>) -> Option<SpacingStats> {
    match (a, b) {
        (Some(a), Some(b)) => Some(SpacingStats {
            dx: merge_range(a.dx, b.dx),
            dy: merge_range(a.dy, b.dy),
            dz: merge_range(a.dz, b.dz),
        }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn bounds_from_points(xs: &[f64], ys: &[f64], zs: &[f64]) -> MeshBounds {
    MeshBounds {
        x: range(xs),
        y: range(ys),
        z: if zs.is_empty() {
            CoordRange { min: 0.0, max: 0.0 }
        } else {
            range(zs)
        },
    }
}

fn range(values: &[f64]) -> CoordRange {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        min = min.min(v);
        max = max.max(v);
    }
    if min.is_infinite() {
        min = 0.0;
        max = 0.0;
    }
    CoordRange { min, max }
}

fn spacing_stats_2d(mesh: &StructuredMesh2d) -> Option<SpacingStats> {
    let mut dx = CoordRange {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
    };
    let mut dy = CoordRange {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
    };
    for j in 0..=mesh.ny {
        for i in 0..mesh.nx {
            let step = (mesh.node_x(i + 1, j) - mesh.node_x(i, j)).abs();
            dx.min = dx.min.min(step);
            dx.max = dx.max.max(step);
        }
    }
    for j in 0..mesh.ny {
        for i in 0..=mesh.nx {
            let step = (mesh.node_y(i, j + 1) - mesh.node_y(i, j)).abs();
            dy.min = dy.min.min(step);
            dy.max = dy.max.max(step);
        }
    }
    Some(SpacingStats {
        dx,
        dy,
        dz: CoordRange { min: 0.0, max: 0.0 },
    })
}

fn spacing_stats_3d(mesh: &StructuredMesh3d) -> Option<SpacingStats> {
    let mut dx = CoordRange {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
    };
    let mut dy = CoordRange {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
    };
    let mut dz = CoordRange {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
    };
    for k in 0..=mesh.nz {
        for j in 0..=mesh.ny {
            for i in 0..mesh.nx {
                let step = (mesh.node_x(i + 1, j, k) - mesh.node_x(i, j, k)).abs();
                dx.min = dx.min.min(step);
                dx.max = dx.max.max(step);
            }
        }
    }
    for k in 0..=mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..=mesh.nx {
                let step = (mesh.node_y(i, j + 1, k) - mesh.node_y(i, j, k)).abs();
                dy.min = dy.min.min(step);
                dy.max = dy.max.max(step);
            }
        }
    }
    for k in 0..mesh.nz {
        for j in 0..=mesh.ny {
            for i in 0..=mesh.nx {
                let step = (mesh.node_z(i, j, k + 1) - mesh.node_z(i, j, k)).abs();
                dz.min = dz.min.min(step);
                dz.max = dz.max.max(step);
            }
        }
    }
    Some(SpacingStats { dx, dy, dz })
}

fn collect_warnings_2d(mesh: &StructuredMesh2d, spacing: Option<&SpacingStats>) -> Vec<String> {
    let _ = mesh;
    collect_spacing_warnings(spacing)
}

fn collect_spacing_warnings(spacing: Option<&SpacingStats>) -> Vec<String> {
    let Some(spacing) = spacing else {
        return Vec::new();
    };
    let mut warnings = Vec::new();
    if spacing.dx.min <= 0.0 {
        warnings.push("Δx 存在非正间距".to_string());
    }
    if spacing.dy.min <= 0.0 {
        warnings.push("Δy 存在非正间距".to_string());
    }
    if spacing.dz.min <= 0.0 && spacing.dz.max > 0.0 {
        warnings.push("Δz 存在非正间距".to_string());
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh1d_reports_x_range_and_dx() {
        let mesh = StructuredMesh1d::new("line", 4, 0.0, 1.0).expect("mesh");
        let diag = mesh1d_diagnostics(&mesh);
        assert_eq!(diag.bounds.x.min, 0.0);
        assert_eq!(diag.bounds.x.max, 1.0);
        assert_eq!(diag.spacing.expect("spacing").dx.min, 0.25);
    }

    #[test]
    fn mesh3d_uniform_box_has_expected_bounds() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 3, 4, 10.0, 20.0, 5.0).expect("mesh");
        let diag = mesh3d_diagnostics(&mesh);
        assert_eq!(diag.bounds.x.min, 0.0);
        assert_eq!(diag.bounds.x.max, 10.0);
        assert_eq!(diag.bounds.y.max, 20.0);
        assert_eq!(diag.bounds.z.max, 5.0);
        let spacing = diag.spacing.expect("spacing");
        assert!((spacing.dx.min - 5.0).abs() < 1.0e-12);
        assert!((spacing.dy.min - 20.0 / 3.0).abs() < 1.0e-12);
    }
}
