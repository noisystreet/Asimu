//! IDWLS 粘性 RHS 静态拓扑（host；CUDA upload 与 `idwls_viscous_rhs_f32.cu` 布局一致）。

use crate::discretization::unstructured_face_cache::LsqRhsCellIncidence;
use crate::discretization::unstructured_face_cache_f32::UnstructuredFaceTopologyF32;

/// 与 CUDA kernel `IdwlsInteriorFace` 布局一致。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IdwlsInteriorFaceHost {
    pub owner: u32,
    pub neighbor: u32,
    pub lsq_dr: [f32; 3],
    pub lsq_w: f32,
}

/// 与 CUDA kernel `IdwlsBoundaryFace` 布局一致。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IdwlsBoundaryFaceHost {
    pub owner: u32,
    pub lsq_dr: [f32; 3],
    pub lsq_w: f32,
}

/// 边界面 ghost 样本（每步 H2D）。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IdwlsGhostSampleHost {
    pub u: f32,
    pub v: f32,
    pub w: f32,
    pub t: f32,
}

/// 单元并行 IDWLS 累加用静态拓扑。
#[derive(Debug, Clone)]
pub struct IdwlsViscousHostTopology {
    pub interior_faces: Vec<IdwlsInteriorFaceHost>,
    pub boundary_faces: Vec<IdwlsBoundaryFaceHost>,
    pub owner_offsets: Vec<u32>,
    pub owner_indices: Vec<u32>,
    pub neighbor_offsets: Vec<u32>,
    pub neighbor_indices: Vec<u32>,
    pub boundary_offsets: Vec<u32>,
    pub boundary_indices: Vec<u32>,
    pub num_cells: usize,
}

impl IdwlsViscousHostTopology {
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
pub fn build_idwls_viscous_host_topology(
    face_topology_f32: &UnstructuredFaceTopologyF32,
    incidence: &LsqRhsCellIncidence,
    num_cells: usize,
) -> IdwlsViscousHostTopology {
    let interior_faces = face_topology_f32
        .interior
        .iter()
        .map(|face| IdwlsInteriorFaceHost {
            owner: face.owner as u32,
            neighbor: face.neighbor as u32,
            lsq_dr: face.lsq_dr,
            lsq_w: face.lsq_w,
        })
        .collect();
    let boundary_faces = face_topology_f32
        .boundary
        .iter()
        .map(|face| IdwlsBoundaryFaceHost {
            owner: face.owner as u32,
            lsq_dr: face.lsq_dr,
            lsq_w: face.lsq_w,
        })
        .collect();
    let (owner_offsets, owner_indices) = flatten_csr(&incidence.interior_as_owner);
    let (neighbor_offsets, neighbor_indices) = flatten_csr(&incidence.interior_as_neighbor);
    let (boundary_offsets, boundary_indices) = flatten_csr(&incidence.boundary_faces);
    IdwlsViscousHostTopology {
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
