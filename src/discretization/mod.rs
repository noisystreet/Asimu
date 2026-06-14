//! 空间离散算子（v0.2 扩散 + v1.x 可压缩无粘）。
//!
//! 理论：[`docs/theory/fvm_diffusion.md`](../../docs/theory/fvm_diffusion.md)（扩散）、
//! [`interface_reconstruction.md`](../../docs/theory/interface_reconstruction.md)、
//! [`inviscid_flux.md`](../../docs/theory/inviscid_flux.md)（Euler FVM）。

pub mod bc;
pub mod bc_compressible;
pub mod diffusion_1d;
pub mod face_flux;
pub mod face_flux_f32;
pub mod face_flux_typed;
pub mod flux_common;
pub mod flux_config;
pub mod gradient;
#[cfg(test)]
mod gradient_tests;
pub mod gradient_typed;
pub mod gradient_unstructured;
pub mod gradient_unstructured_f32;
pub mod gradient_unstructured_inviscid_f32;
pub mod hllc;
pub mod hllc_f32;
pub mod incompressible;
pub mod incompressible_bc;
pub mod incompressible_boundary_flux;
pub mod incompressible_face_boundary;
pub mod incompressible_face_flux;
pub mod incompressible_momentum;
mod incompressible_momentum_geometry;
#[cfg(test)]
mod incompressible_momentum_tests;
pub mod incompressible_phi;
pub mod incompressible_pressure;
pub mod incompressible_rhie_chow;
pub mod incompressible_velocity_correction;
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
pub mod unstructured_face_cache;
pub mod unstructured_face_cache_f32;
pub mod unstructured_limiter;
pub mod van_leer;
pub mod van_leer_f32;
pub mod viscous;
pub mod viscous_assembly;
pub mod viscous_boundary_f32;
pub mod viscous_f32;
pub mod wall_thermal;

#[cfg(test)]
pub(crate) mod freestream_pair;

use crate::core::Real;
use crate::error::Result;
use crate::field::ScalarField;
use crate::linalg::LinearSystem;
use crate::mesh::Mesh;

pub use bc::{apply_boundary_conditions, apply_dirichlet, apply_dirichlet_face, apply_neumann};
pub use bc_compressible::{
    BoundaryGhostBuffer, GhostCellState, apply_compressible_boundary_conditions,
    apply_compressible_boundary_conditions_typed, farfield_ghost, inlet_ghost, outlet_ghost,
    symmetry_ghost, wall_ghost,
};
pub use diffusion_1d::assemble_diffusion_1d;
pub use face_flux::{
    FaceFluxInput, face_inviscid_flux, face_inviscid_flux_first_order_boundary_soa,
    face_inviscid_flux_first_order_interior_soa, face_inviscid_flux_from_interface,
};
pub use face_flux_f32::{
    face_inviscid_flux_first_order_boundary_soa_f32,
    face_inviscid_flux_first_order_interior_soa_f32, face_inviscid_flux_from_interface_f32,
};
pub use face_flux_typed::InviscidFaceFluxTyped;
pub use flux_config::{FluxScheme, InviscidFluxConfig, ReconstructionKind, SlopeLimiter};
pub use gradient::{
    GradientFields, GradientFieldsT, InviscidPrimitiveGradients, VelocityGradient,
    compute_structured_gradients_3d,
};
pub use gradient_unstructured::{
    UnstructuredGradientLsqInput, UnstructuredGradientScratch,
    compute_unstructured_gradients_idw_lsq, compute_unstructured_gradients_idw_lsq_with_scratch,
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq,
};
pub use gradient_unstructured_f32::{
    UnstructuredGradientLsqInputF32, UnstructuredGradientScratchF32,
    compute_unstructured_gradients_idw_lsq_f32,
};
pub use gradient_unstructured_inviscid_f32::compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq_f32;
pub use hllc::hllc_flux;
pub use incompressible::{
    IncompressiblePressureCorrectionConfig, IncompressiblePressureCorrectionSystem,
    IncompressibleVelocityLaplacian, assemble_incompressible_pressure_poisson_3d,
    compute_incompressible_divergence_3d, compute_incompressible_velocity_laplacian_3d,
};
pub use incompressible_bc::{
    IncompressibleBoundaryApplyStats, apply_incompressible_boundary_conditions_3d,
};
pub use incompressible_face_boundary::{
    IncompressibleBoundaryFaceState, IncompressibleMassFluxBoundaryKind,
    incompressible_boundary_face_state, incompressible_boundary_face_velocity,
    incompressible_pressure_correction_dirichlet,
};
pub use incompressible_face_flux::compute_incompressible_face_flux_divergence_3d;
pub use incompressible_momentum::{
    IncompressibleConvectionScheme, IncompressibleMomentumPredictorConfig,
    IncompressibleMomentumPredictorSystem, assemble_incompressible_momentum_predictor_3d,
    assemble_incompressible_momentum_predictor_with_boundary_3d,
    assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d,
};
pub use incompressible_phi::IncompressibleFaceFluxField;
pub use incompressible_pressure::assemble_incompressible_pressure_correction_3d;
pub use incompressible_rhie_chow::{
    PressureCorrectedRhieChowDivergenceConfig, compute_incompressible_rhie_chow_divergence_3d,
    compute_pressure_corrected_rhie_chow_divergence_3d,
};
pub use incompressible_velocity_correction::{
    RhieChowVelocityCorrectionConfig, corrected_incompressible_fields_rhie_chow_3d,
};
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
pub use unstructured_face_cache::{
    InteriorFaceBatchStatic4, InteriorFaceBucketBatchLayout, InteriorFaceColoring,
    LsqPrecomputedCell, UnstructuredFaceTopology, UnstructuredInteriorFace,
    UnstructuredSolverMeshCache,
};
pub use unstructured_face_cache_f32::{
    LsqPrecomputedCellF32, UnstructuredBoundaryFaceF32, UnstructuredFaceTopologyF32,
    UnstructuredInteriorFaceF32, neg_dr, vec3_from_f32, vec3_to_f32,
};
pub use unstructured_limiter::UnstructuredGradientLimiter;
pub use van_leer::{hanel_van_leer_flux, van_leer_flux};
pub use viscous::{ViscousFlux, face_transport_coefficients, viscous_face_flux};
pub use wall_thermal::{wall_face_conduction, wall_ghost_temperature};

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
