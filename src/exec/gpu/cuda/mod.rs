//! NVIDIA CUDA 后端（`cudarc`；ADR 0017 G0+）。

mod buffers;
mod face_geom;
mod inviscid;
mod mesh_cache;
mod module;

pub use face_geom::{ExecInteriorColorBucket, ExecInteriorFaceStatic, ExecInteriorFaceTopology};
pub use inviscid::{
    CUDA_FLUX_SCHEME_HVL, CUDA_FLUX_SCHEME_ROE, CudaBackendState, CudaFirstOrderInviscidParams,
};
