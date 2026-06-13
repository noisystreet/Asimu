//! 非结构内面静态几何（H2D；与 `kernels/cuda/inviscid_first_order_f32.cu` 布局一致）。

/// 单内面预计算几何 + RHS scale（法向在 upload 时已单位化）。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecInteriorFaceStatic {
    pub owner: u32,
    pub neighbor: u32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub owner_scale: f32,
    pub neighbor_scale: f32,
}

/// 着色桶：同色面不共享单元，device scatter 可无 atomic。
#[derive(Debug, Clone)]
pub struct ExecInteriorColorBucket {
    pub face_indices: Vec<u32>,
}

/// exec 侧内面拓扑快照（由 discretization 在 init/run 前转换）。
#[derive(Debug, Clone)]
pub struct ExecInteriorFaceTopology {
    pub faces: Vec<ExecInteriorFaceStatic>,
    pub color_buckets: Vec<ExecInteriorColorBucket>,
}

impl ExecInteriorFaceTopology {
    #[must_use]
    pub fn num_interior_faces(&self) -> usize {
        self.faces.len()
    }

    #[must_use]
    pub fn num_colors(&self) -> usize {
        self.color_buckets.len()
    }
}
