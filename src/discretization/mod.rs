//! 空间离散算子（v0.2 扩散 + v1.x 可压缩无粘）。
//!
//! 理论：[`docs/theory/fvm_diffusion.md`](../../docs/theory/fvm_diffusion.md)（扩散）、
//! [`interface_reconstruction.md`](../../docs/theory/interface_reconstruction.md)、
//! [`inviscid_flux.md`](../../docs/theory/inviscid_flux.md)（Euler FVM）。

pub mod bc;
pub mod bc_compressible;
pub mod diffusion_1d;
pub mod face_flux;
pub mod flux_common;
pub mod flux_config;
pub mod hllc;
pub mod inviscid;
pub mod reconstruction;
pub mod residual;
pub mod roe;
pub mod van_leer;

use crate::core::Real;
use crate::error::Result;
use crate::field::ScalarField;
use crate::linalg::LinearSystem;
use crate::mesh::Mesh;

pub use bc::{apply_boundary_conditions, apply_dirichlet, apply_dirichlet_face, apply_neumann};
pub use bc_compressible::{
    BoundaryGhostBuffer, GhostCellState, apply_compressible_boundary_conditions, farfield_ghost,
    inlet_ghost, outlet_ghost, symmetry_ghost, wall_ghost,
};
pub use diffusion_1d::assemble_diffusion_1d;
pub use face_flux::{FaceFluxInput, face_inviscid_flux};
pub use flux_config::{FluxScheme, InviscidFluxConfig, ReconstructionKind, SlopeLimiter};
pub use hllc::hllc_flux;
pub use inviscid::{InviscidFlux, physical_inviscid_flux};
pub use reconstruction::{
    InterfaceStates, MusclStencil1d, reconstruct_face_states, reconstruct_first_order,
};
pub use residual::{
    BoundaryGhosts1d, InviscidBoundary1d, accumulate_boundary_face, accumulate_interior_face,
    assemble_inviscid_residual_1d, assemble_inviscid_residual_3d, zero_gradient_ghosts_1d,
};
pub use roe::{RoeFluxConfig, roe_flux};
pub use van_leer::{hanel_van_leer_flux, van_leer_flux};

/// 占位装配入口：验证 field / mesh / system 尺寸一致。
///
/// v0.2 后续 PR 实现 1D FVM 扩散装配；当前仅清零 RHS。
pub fn assemble_diffusion_placeholder(
    mesh: &Mesh,
    field: &ScalarField,
    system: &mut LinearSystem,
    diffusivity: Real,
) -> Result<()> {
    let _ = diffusivity;
    debug_assert_eq!(mesh.cell_count, field.len());
    debug_assert_eq!(field.len(), system.len());
    for value in system.rhs_mut() {
        *value = 0.0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::ScalarField;

    #[test]
    fn placeholder_assemble_succeeds_on_matching_sizes() {
        let mesh = Mesh::new("line", 4).expect("mesh");
        let field = ScalarField::uniform(4, 0.0).expect("field");
        let mut system = LinearSystem::new(vec![1.0; 4]).expect("system");
        assemble_diffusion_placeholder(&mesh, &field, &mut system, 1.0).expect("assemble");
        assert!(system.rhs().iter().all(|&v| v == 0.0));
    }
}
