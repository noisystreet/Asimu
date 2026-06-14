//! 非结构边界面静态几何（H2D；与 `kernels/cuda/*_boundary_f32` 布局一致）。

/// 无粘边界面（owner + 法向 + RHS scale）。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecInviscidBoundaryFaceStatic {
    pub owner: u32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub owner_scale: f32,
}

/// 粘性边界面静态几何 + BC 标志。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecViscousBoundaryFaceStatic {
    pub owner: u32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub owner_scale: f32,
    pub spacing: f32,
    pub flags: u32,
    pub wall_param: f32,
}

/// exec 侧无粘边界面拓扑（init 一次 H2D）。
#[derive(Debug, Clone)]
pub struct ExecInviscidBoundaryTopology {
    pub faces: Vec<ExecInviscidBoundaryFaceStatic>,
}

impl ExecInviscidBoundaryTopology {
    #[must_use]
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }
}

/// exec 侧粘性边界面拓扑（init 一次 H2D）。
#[derive(Debug, Clone)]
pub struct ExecViscousBoundaryTopology {
    pub faces: Vec<ExecViscousBoundaryFaceStatic>,
}

impl ExecViscousBoundaryTopology {
    #[must_use]
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }
}

/// 粘性边界面 ghost 原变量（每步 H2D；含静温）。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ViscousBoundaryGhostHost {
    pub rho: f32,
    pub pressure: f32,
    pub u: f32,
    pub v: f32,
    pub w: f32,
    pub temperature: f32,
}

unsafe impl cudarc::driver::DeviceRepr for ExecInviscidBoundaryFaceStatic {}
unsafe impl cudarc::driver::DeviceRepr for ExecViscousBoundaryFaceStatic {}
unsafe impl cudarc::driver::DeviceRepr for ViscousBoundaryGhostHost {}

use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

/// 粘性边界面 CUDA 装配输入（P2；减少跨层参数数量）。
#[derive(Debug, Clone, Copy)]
pub struct CudaViscousBoundaryInput<'a> {
    pub topo: &'a ExecViscousBoundaryTopology,
    pub topo_key: usize,
    pub boundary_ghosts: &'a [ViscousBoundaryGhostHost],
    pub temperatures: &'a [f32],
    pub viscous: &'a ViscousPhysicsConfig,
    pub eos: &'a IdealGasEoS,
}
