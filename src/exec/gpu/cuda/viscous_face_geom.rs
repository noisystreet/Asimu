//! 粘性内面静态几何（含输运系数；H2D）。

use cudarc::driver::DeviceRepr;

/// 单内面粘性几何 + RHS scale（法向 upload 前已单位化）。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceViscousFaceGeom {
    pub owner: u32,
    pub neighbor: u32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub mu: f32,
    pub lambda: f32,
    pub owner_scale: f32,
    pub neighbor_scale: f32,
}

unsafe impl DeviceRepr for DeviceViscousFaceGeom {}

/// exec 侧粘性内面拓扑（复用无粘着色桶索引）。
#[derive(Debug, Clone)]
pub struct ExecViscousInteriorTopology {
    pub faces: Vec<DeviceViscousFaceGeom>,
    pub color_buckets: Vec<super::face_geom::ExecInteriorColorBucket>,
}

impl ExecViscousInteriorTopology {
    #[must_use]
    pub fn num_interior_faces(&self) -> usize {
        self.faces.len()
    }

    #[must_use]
    pub fn num_colors(&self) -> usize {
        self.color_buckets.len()
    }
}
