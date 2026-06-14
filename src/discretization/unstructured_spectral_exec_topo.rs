//! 非结构谱半径 CUDA 静态拓扑（host；与 `spectral_radius_unstructured_f32.cu` 布局一致）。

use crate::discretization::unstructured_face_cache::LsqRhsCellIncidence;
use crate::discretization::unstructured_face_cache_f32::UnstructuredFaceTopologyF32;

/// 与 CUDA kernel `SpectralInteriorFace` 布局一致。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SpectralInteriorFaceHost {
    pub owner: u32,
    pub neighbor: u32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub area: f32,
    pub inv_owner_volume: f32,
    pub inv_neighbor_volume: f32,
    pub owner_volume: f32,
    pub neighbor_volume: f32,
}

/// 与 CUDA kernel `SpectralBoundaryFace` 布局一致。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SpectralBoundaryFaceHost {
    pub owner: u32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub area: f32,
    pub inv_owner_volume: f32,
    pub owner_volume: f32,
}

/// 边界面 ghost 原变量（每步 H2D）。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SpectralGhostPrimHost {
    pub rho: f32,
    pub pressure: f32,
    pub u: f32,
    pub v: f32,
    pub w: f32,
}

/// 单元并行谱半径累加用静态拓扑（CSR 与 IDWLS 相同）。
#[derive(Debug, Clone)]
pub struct SpectralRadiusHostTopology {
    pub interior_faces: Vec<SpectralInteriorFaceHost>,
    pub boundary_faces: Vec<SpectralBoundaryFaceHost>,
    pub owner_offsets: Vec<u32>,
    pub owner_indices: Vec<u32>,
    pub neighbor_offsets: Vec<u32>,
    pub neighbor_indices: Vec<u32>,
    pub boundary_offsets: Vec<u32>,
    pub boundary_indices: Vec<u32>,
    pub num_cells: usize,
}

impl SpectralRadiusHostTopology {
    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }
}

fn flatten_csr(lists: &[Vec<usize>]) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = Vec::with_capacity(lists.len() + 1);
    offsets.push(0);
    let mut indices = Vec::new();
    for list in lists {
        for &idx in list {
            indices.push(idx as u32);
        }
        offsets.push(indices.len() as u32);
    }
    (offsets, indices)
}

/// 由 f32 面拓扑与单元–面 CSR 构建 host 拓扑。
#[must_use]
pub fn build_spectral_radius_host_topology(
    face_topology_f32: &UnstructuredFaceTopologyF32,
    incidence: &LsqRhsCellIncidence,
    num_cells: usize,
) -> SpectralRadiusHostTopology {
    let interior_faces = face_topology_f32
        .interior
        .iter()
        .map(|face| SpectralInteriorFaceHost {
            owner: face.owner as u32,
            neighbor: face.neighbor as u32,
            nx: face.normal[0],
            ny: face.normal[1],
            nz: face.normal[2],
            area: face.area,
            inv_owner_volume: face.inv_owner_volume,
            inv_neighbor_volume: face.inv_neighbor_volume,
            owner_volume: face.owner_volume,
            neighbor_volume: face.neighbor_volume,
        })
        .collect();
    let boundary_faces = face_topology_f32
        .boundary
        .iter()
        .map(|face| SpectralBoundaryFaceHost {
            owner: face.owner as u32,
            nx: face.normal[0],
            ny: face.normal[1],
            nz: face.normal[2],
            area: face.area,
            inv_owner_volume: if face.owner_volume > 1.0e-30_f32 {
                1.0 / face.owner_volume
            } else {
                0.0
            },
            owner_volume: face.owner_volume,
        })
        .collect();
    let (owner_offsets, owner_indices) = flatten_csr(&incidence.interior_as_owner);
    let (neighbor_offsets, neighbor_indices) = flatten_csr(&incidence.interior_as_neighbor);
    let (boundary_offsets, boundary_indices) = flatten_csr(&incidence.boundary_faces);
    SpectralRadiusHostTopology {
        interior_faces,
        boundary_faces,
        owner_offsets,
        owner_indices,
        neighbor_offsets,
        neighbor_indices,
        boundary_offsets,
        boundary_indices,
        num_cells,
    }
}
