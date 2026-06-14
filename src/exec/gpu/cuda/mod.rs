//! NVIDIA CUDA 后端（`cudarc`；ADR 0017 G0+）。

mod boundary_face_geom;
mod boundary_mesh_cache;
mod buffers;
mod face_geom;
mod field;
mod gradient_buffers;
mod idwls;
mod idwls_mesh_cache;
mod idwls_topology;
mod inviscid;
mod lusgs_diagonal;
mod mesh_cache;
mod module;
mod pipeline;
mod spectral_radius;
mod spectral_radius_mesh_cache;
mod spectral_radius_topology;
mod spmv;
mod transfer;
mod viscous;
mod viscous_face_geom;
mod viscous_mesh_cache;
mod viscous_transport_params;

pub use boundary_face_geom::{
    BoundaryConservedGhostHost, CudaViscousBoundaryInput, ExecInviscidBoundaryFaceStatic,
    ExecInviscidBoundaryTopology, ExecViscousBoundaryFaceStatic, ExecViscousBoundaryTopology,
    ViscousBoundaryGhostHost,
};
pub use face_geom::{ExecInteriorColorBucket, ExecInteriorFaceStatic, ExecInteriorFaceTopology};
pub use idwls_mesh_cache::IdwlsViscousRhsHostOut;
pub use idwls_topology::{DeviceIdwlsGhostSample, ExecIdwlsViscousTopology};
pub use inviscid::{
    CUDA_FLUX_SCHEME_HVL, CUDA_FLUX_SCHEME_ROE, CudaBackendState, CudaFirstOrderInviscidParams,
    CudaPrepareRhsDeviceInput,
};
pub use spectral_radius_topology::ExecSpectralRadiusTopology;
pub use viscous_face_geom::{DeviceViscousFaceGeom, ExecViscousInteriorTopology};
pub use viscous_transport_params::{
    DeviceViscousTransportParams, build_device_viscous_transport_params,
};
