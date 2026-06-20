//! 可压缩 FVM 离散：Riemann 通量、MUSCL 重构、粘性通量、残差装配与 BC。
//!
//! 理论：[`docs/theory/inviscid_flux.md`](../../../docs/theory/inviscid_flux.md)、
//! [`interface_reconstruction.md`](../../../docs/theory/interface_reconstruction.md)。

pub mod bc_compressible;
pub mod face_flux;
pub mod face_flux_f32;
pub mod face_flux_jacobian;
pub mod face_flux_typed;
#[cfg(test)]
pub(crate) mod freestream_pair;
pub mod hllc;
pub mod hllc_f32;
pub mod inviscid;
pub mod inviscid_f32;
pub mod reconstruction;
pub mod reconstruction_unstructured;
pub mod reconstruction_unstructured_f32;
pub mod residual;
pub mod roe;
pub mod roe_f32;
pub mod slau2;
pub mod slau2_f32;
pub mod van_leer;
pub mod van_leer_f32;
pub mod van_leer_jacobian;
pub mod viscous;
pub mod viscous_assembly;
pub mod viscous_boundary_f32;
pub mod viscous_f32;
pub mod wall_thermal;

pub use bc_compressible::{
    BoundaryGhostBuffer, GhostCellState, apply_compressible_boundary_conditions,
    apply_compressible_boundary_conditions_typed, farfield_ghost, inlet_ghost, outlet_ghost,
    symmetry_ghost, wall_ghost,
};
pub use face_flux::{
    FaceFluxInput, face_inviscid_flux, face_inviscid_flux_first_order_boundary_soa,
    face_inviscid_flux_first_order_interior_soa, face_inviscid_flux_from_interface,
};
pub use face_flux_f32::{
    face_inviscid_flux_first_order_boundary_soa_f32,
    face_inviscid_flux_first_order_interior_soa_f32, face_inviscid_flux_from_interface_f32,
};
pub use face_flux_jacobian::{
    ConservedFluxJacobian, first_order_face_flux_jacobian_supported,
    first_order_interior_flux_jacobian, physical_inviscid_flux_jacobian_conserved,
};
pub use face_flux_typed::InviscidFaceFluxTyped;
pub use hllc::hllc_flux;
pub use inviscid::{InviscidFlux, physical_inviscid_flux};
pub use reconstruction::{
    InterfacePrimitiveStates, PrimitiveMusclStencil1d, interface_conserved_pair,
    reconstruct_face_primitives, reconstruct_first_order,
};
pub use reconstruction_unstructured::{
    UnstructuredLinearReconstructionCtx, reconstruct_unstructured_boundary_face,
    reconstruct_unstructured_interior_face,
};
pub use residual::{
    BoundaryGhosts1d, BoundaryInviscidFluxInput, InviscidAssemblyUnstructuredParams,
    InviscidBoundary1d, ViscousAssembly3dInput, ViscousAssembly3dParams,
    ViscousAssemblyUnstructuredF32Input, ViscousAssemblyUnstructuredInput,
    ViscousAssemblyUnstructuredParams, ViscousAssemblyUnstructuredScratch,
    ViscousAssemblyUnstructuredTypedInput, accumulate_boundary_face, accumulate_interior_face,
    assemble_inviscid_residual_1d, assemble_inviscid_residual_3d,
    assemble_inviscid_residual_3d_typed, assemble_inviscid_residual_unstructured,
    assemble_inviscid_residual_unstructured_typed, assemble_viscous_residual_3d,
    assemble_viscous_residual_unstructured, compute_gradients_and_assemble_viscous_3d,
    compute_gradients_and_assemble_viscous_unstructured_f32,
    compute_gradients_and_assemble_viscous_unstructured_typed,
    compute_gradients_and_assemble_viscous_unstructured_with_scratch, zero_gradient_ghosts_1d,
};
pub use roe::{RoeFluxConfig, roe_flux};
pub use slau2::slau2_flux;
pub use van_leer::{hanel_van_leer_flux, van_leer_flux};
pub use viscous::{ViscousFlux, face_transport_coefficients, viscous_face_flux};
pub use wall_thermal::{wall_face_conduction, wall_ghost_temperature};
