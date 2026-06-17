//! 非结构 LU-SGS 扫掠 device CSR 拓扑（由 `lusgs_couplings_f32` 预打包）。

use crate::discretization::unstructured_face_cache_f32::LuSgsUnstructuredCouplingsF32;
use crate::mesh::UnstructuredMesh3d;

/// 单元图着色：同色单元互不相邻，供 wavefront 并行 GS。
#[derive(Debug, Clone)]
pub struct LuSgsSweepCellColoring {
    pub num_colors: usize,
    /// CSR：`color_offsets[c]..color_offsets[c+1]` 为颜色 `c` 的单元下标。
    pub color_offsets: Vec<u32>,
    /// 按颜色分桶的 `CellId` 序单元下标（平坦存储）。
    pub color_cells: Vec<u32>,
}

impl LuSgsSweepCellColoring {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_colors == 0
    }

    #[must_use]
    pub fn max_bucket_cells(&self) -> usize {
        if self.num_colors == 0 {
            return 0;
        }
        (0..self.num_colors)
            .map(|c| (self.color_offsets[c + 1] - self.color_offsets[c]) as usize)
            .max()
            .unwrap_or(0)
    }
}

/// Host 侧 CSR：每单元出边耦合列表（与 `lu_sgs_sweep_unstructured_f32` 一致）。
#[derive(Debug, Clone)]
pub struct LuSgsSweepHostTopology {
    pub cell_offsets: Vec<u32>,
    pub neighbors: Vec<u32>,
    pub areas: Vec<f32>,
    /// 每耦合 3 分量法向（与 `areas` 同长度）。
    pub normals: Vec<f32>,
    pub volumes: Vec<f32>,
    pub cell_coloring: LuSgsSweepCellColoring,
}

impl LuSgsSweepHostTopology {
    #[must_use]
    pub fn from_mesh_and_couplings(
        mesh: &UnstructuredMesh3d,
        couplings: &LuSgsUnstructuredCouplingsF32,
    ) -> Self {
        let n = mesh.num_cells();
        let volumes: Vec<f32> = mesh.cell_volumes().iter().map(|v| *v as f32).collect();
        let mut cell_offsets = Vec::with_capacity(n + 1);
        cell_offsets.push(0);
        let mut neighbors = Vec::new();
        let mut areas = Vec::new();
        let mut normals = Vec::new();
        for cell_couplings in couplings.cells().iter().take(n) {
            for c in cell_couplings {
                neighbors.push(c.neighbor as u32);
                areas.push(c.area);
                normals.extend_from_slice(&c.normal);
            }
            cell_offsets.push(neighbors.len() as u32);
        }
        let cell_coloring = color_lusgs_sweep_cells(couplings, n);
        Self {
            cell_offsets,
            neighbors,
            areas,
            normals,
            volumes,
            cell_coloring,
        }
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.volumes.len()
    }
}

/// 贪心单元着色：相邻单元不同色（wavefront 并行 LU-SGS 前提）。
pub(crate) fn color_lusgs_sweep_cells(
    couplings: &LuSgsUnstructuredCouplingsF32,
    num_cells: usize,
) -> LuSgsSweepCellColoring {
    if num_cells == 0 {
        return LuSgsSweepCellColoring {
            num_colors: 0,
            color_offsets: vec![0],
            color_cells: Vec::new(),
        };
    }
    let mut adjacency = vec![Vec::<u32>::new(); num_cells];
    for (cell, cell_couplings) in couplings.cells().iter().enumerate().take(num_cells) {
        for c in cell_couplings {
            let nb = c.neighbor;
            if nb >= num_cells {
                continue;
            }
            push_unique_u32(&mut adjacency[cell], nb as u32);
            push_unique_u32(&mut adjacency[nb], cell as u32);
        }
    }

    let mut cell_colors = vec![0u8; num_cells];
    for cell in 0..num_cells {
        let mut used = Vec::new();
        for &nb in &adjacency[cell] {
            let c = cell_colors[nb as usize];
            push_unique_u8(&mut used, c);
        }
        used.sort_unstable();
        cell_colors[cell] = first_available_color(&used);
    }

    let num_colors = cell_colors
        .iter()
        .copied()
        .max()
        .map(|c| c as usize + 1)
        .unwrap_or(0);
    let mut buckets = vec![Vec::new(); num_colors];
    for (cell, &color) in cell_colors.iter().enumerate() {
        buckets[color as usize].push(cell as u32);
    }

    let mut color_offsets = Vec::with_capacity(num_colors + 1);
    color_offsets.push(0);
    let mut color_cells = Vec::new();
    for bucket in &buckets {
        color_cells.extend_from_slice(bucket);
        color_offsets.push(color_cells.len() as u32);
    }

    LuSgsSweepCellColoring {
        num_colors,
        color_offsets,
        color_cells,
    }
}

fn push_unique_u32(values: &mut Vec<u32>, value: u32) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn push_unique_u8(values: &mut Vec<u8>, value: u8) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn first_available_color(used_sorted: &[u8]) -> u8 {
    let mut candidate = 0u8;
    for &used in used_sorted {
        if used == candidate {
            candidate = candidate.saturating_add(1);
        } else if used > candidate {
            break;
        }
    }
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::FaceId;
    use crate::discretization::unstructured_face_cache_f32::LuSgsUnstructuredCouplingsF32;
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

    fn closed_tet_mesh() -> UnstructuredMesh3d {
        UnstructuredMesh3d::new(
            "tet",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell")],
        )
        .expect("mesh")
    }

    fn tet_couplings(mesh: &UnstructuredMesh3d) -> LuSgsUnstructuredCouplingsF32 {
        let faces = (0..mesh.num_faces())
            .map(|f| FaceId(f as u32))
            .collect::<Vec<_>>();
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "far",
            faces,
            BoundaryKind::Farfield {
                mach: 0.1,
                pressure: 1.0,
                temperature: 1.0,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        let cache = crate::discretization::UnstructuredSolverMeshCache::from_mesh(mesh, &boundary)
            .expect("cache");
        cache.lusgs_couplings_f32.clone()
    }

    #[test]
    fn cell_coloring_has_no_adjacent_conflicts() {
        let mesh = closed_tet_mesh();
        let couplings = tet_couplings(&mesh);
        let topo = LuSgsSweepHostTopology::from_mesh_and_couplings(&mesh, &couplings);
        let n = mesh.num_cells();
        assert!(topo.cell_coloring.num_colors >= 2);
        let mut cell_color = vec![0u8; n];
        for c in 0..topo.cell_coloring.num_colors {
            let begin = topo.cell_coloring.color_offsets[c] as usize;
            let end = topo.cell_coloring.color_offsets[c + 1] as usize;
            for &cell in &topo.cell_coloring.color_cells[begin..end] {
                cell_color[cell as usize] = c as u8;
            }
        }
        for (cell, cell_couplings) in couplings.cells().iter().enumerate().take(n) {
            for coupling in cell_couplings {
                assert_ne!(
                    cell_color[cell], cell_color[coupling.neighbor],
                    "adjacent cells {cell} and {} same color",
                    coupling.neighbor
                );
            }
        }
    }

    #[test]
    fn cell_coloring_covers_all_cells_once() {
        let mesh = closed_tet_mesh();
        let couplings = tet_couplings(&mesh);
        let coloring = color_lusgs_sweep_cells(&couplings, mesh.num_cells());
        let mut seen = vec![false; mesh.num_cells()];
        for &cell in &coloring.color_cells {
            let idx = cell as usize;
            assert!(!seen[idx], "duplicate cell {idx}");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}
