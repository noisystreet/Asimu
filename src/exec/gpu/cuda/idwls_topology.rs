//! IDWLS device 类型再导出（布局与 discretization host 拓扑一致）。

pub use crate::discretization::unstructured_idwls_exec_topo::{
    IdwlsBoundaryFaceHost as DeviceIdwlsBoundaryFace,
    IdwlsGhostSampleHost as DeviceIdwlsGhostSample,
    IdwlsInteriorFaceHost as DeviceIdwlsInteriorFace,
    IdwlsViscousHostTopology as ExecIdwlsViscousTopology,
};

use crate::discretization::unstructured_idwls_exec_topo::{
    IdwlsBoundaryFaceHost, IdwlsGhostSampleHost, IdwlsInteriorFaceHost,
};

unsafe impl cudarc::driver::DeviceRepr for IdwlsInteriorFaceHost {}
unsafe impl cudarc::driver::DeviceRepr for IdwlsBoundaryFaceHost {}
unsafe impl cudarc::driver::DeviceRepr for IdwlsGhostSampleHost {}
use crate::discretization::unstructured_face_cache_f32::LsqPrecomputedCellF32;

unsafe impl cudarc::driver::DeviceRepr for LsqPrecomputedCellF32 {}
