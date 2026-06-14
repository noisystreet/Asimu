//! CPU 热算子：标量实现 + 可选 `wide` SIMD（`simd-fvm`）。

mod hvl;
mod lsq;
mod lsq_f32;
mod lusgs;
mod roe;
mod viscous;

pub use hvl::face_inviscid_flux_first_order_hanel_batch4;
pub use lsq::{
    Symmetric3x3, accumulate_lsq_rhs_component, solve_symmetric_3x3, solve_symmetric_3x3_batch4,
};
pub use lsq_f32::{
    accumulate_lsq_rhs_component_f32, solve_lsq_precomputed_cell_f32, solve_symmetric_3x3_f32,
};
pub use lusgs::{ConservedSoA, ConservedSoAMut, LusgsDiagonalUpdate, assign_lusgs_diagonal_update};
pub use roe::{InviscidFlux5, face_inviscid_flux_first_order_roe_batch4};
pub use viscous::{
    VelocityGradientSoA, ViscousFaceBatchGeom, ViscousFaceGather4, ViscousFlux4,
    fused_interior_viscous_face_flux_batch4, fused_interior_viscous_face_flux_batch4_from_soa,
    gather_viscous_face_batch4,
};
