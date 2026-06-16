//! 非结构 LU-SGS 扫掠 device CSR 拓扑（由 `lusgs_couplings_f32` 预打包）。

use crate::discretization::unstructured_face_cache_f32::LuSgsUnstructuredCouplingsF32;
use crate::mesh::UnstructuredMesh3d;

/// Host 侧 CSR：每单元出边耦合列表（与 `lu_sgs_sweep_unstructured_f32` 一致）。
#[derive(Debug, Clone)]
pub struct LuSgsSweepHostTopology {
    pub cell_offsets: Vec<u32>,
    pub neighbors: Vec<u32>,
    pub areas: Vec<f32>,
    /// 每耦合 3 分量法向（与 `areas` 同长度）。
    pub normals: Vec<f32>,
    pub volumes: Vec<f32>,
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
        Self {
            cell_offsets,
            neighbors,
            areas,
            normals,
            volumes,
        }
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.volumes.len()
    }
}
