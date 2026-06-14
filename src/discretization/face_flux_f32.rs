//! f32 面通量兼容 re-export（实现见 [`face_flux_typed`]）。

pub use super::face_flux_typed::{
    InviscidFaceFluxTyped, face_inviscid_flux_first_order_boundary_soa_f32,
    face_inviscid_flux_first_order_interior_soa_f32, face_inviscid_flux_from_interface_f32,
};
