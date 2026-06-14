//! 谱半径 device 类型再导出（布局与 discretization host 拓扑一致）。

pub use crate::discretization::unstructured_spectral_exec_topo::{
    SpectralBoundaryFaceHost as DeviceSpectralBoundaryFace,
    SpectralGhostPrimHost as DeviceSpectralGhostPrim,
    SpectralInteriorFaceHost as DeviceSpectralInteriorFace,
    SpectralRadiusHostTopology as ExecSpectralRadiusTopology,
};

use crate::discretization::unstructured_spectral_exec_topo::{
    SpectralBoundaryFaceHost, SpectralGhostPrimHost, SpectralInteriorFaceHost,
};

unsafe impl cudarc::driver::DeviceRepr for SpectralInteriorFaceHost {}
unsafe impl cudarc::driver::DeviceRepr for SpectralBoundaryFaceHost {}
unsafe impl cudarc::driver::DeviceRepr for SpectralGhostPrimHost {}
