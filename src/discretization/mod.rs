//! 空间离散算子（v0.2 扩散 + v1.x 可压缩/不可压缩 FVM）。
//!
//! 共用算子（梯度、扩散、非结构拓扑）位于本层；regime 专用实现见
//! [`compressible`](compressible) 与 [`incompressible`](incompressible)。
//!
//! 理论：[`docs/theory/fvm_diffusion.md`](../../docs/theory/fvm_diffusion.md)（扩散）、
//! [`interface_reconstruction.md`](../../docs/theory/interface_reconstruction.md)、
//! [`inviscid_flux.md`](../../docs/theory/inviscid_flux.md)（Euler FVM）。

pub mod bc;
pub mod compressible;
pub mod diffusion_1d;
pub mod flux_common;
pub mod flux_config;
pub mod gradient;
#[cfg(test)]
mod gradient_tests;
pub mod gradient_typed;
pub mod gradient_unstructured;
pub mod gradient_unstructured_f32;
#[cfg(feature = "cuda")]
#[path = "gradient_unstructured_f32_cuda.rs"]
mod gradient_unstructured_f32_cuda;
pub mod gradient_unstructured_inviscid_f32;
pub mod incompressible;
pub mod periodic;
pub mod structured_face_cache_f32;
#[cfg(feature = "cuda")]
pub mod unstructured_boundary_exec_topo;
pub mod unstructured_face_cache;
pub mod unstructured_face_cache_f32;
pub mod unstructured_idwls_exec_topo;
#[cfg(feature = "cuda")]
pub mod unstructured_interior_exec_topo;
pub mod unstructured_limiter;
#[cfg(feature = "cuda")]
pub mod unstructured_lusgs_sweep_exec_topo;
pub mod unstructured_spectral_exec_topo;

// --- 可压 FVM：稳定库 API（类型与函数）---
pub use compressible::{
    BoundaryGhostBuffer, BoundaryGhosts1d, BoundaryInviscidFluxInput, FaceFluxInput,
    GhostCellState, InterfacePrimitiveStates, InviscidAssemblyUnstructuredParams,
    InviscidBoundary1d, InviscidFaceFluxTyped, InviscidFlux, PrimitiveMusclStencil1d,
    RoeFluxConfig, UnstructuredLinearReconstructionCtx, ViscousAssembly3dInput,
    ViscousAssembly3dParams, ViscousAssemblyUnstructuredF32Input, ViscousAssemblyUnstructuredInput,
    ViscousAssemblyUnstructuredParams, ViscousAssemblyUnstructuredScratch,
    ViscousAssemblyUnstructuredTypedInput, ViscousFlux, accumulate_boundary_face,
    accumulate_interior_face, apply_compressible_boundary_conditions,
    apply_compressible_boundary_conditions_typed, assemble_inviscid_residual_1d,
    assemble_inviscid_residual_3d, assemble_inviscid_residual_3d_typed,
    assemble_inviscid_residual_unstructured, assemble_inviscid_residual_unstructured_typed,
    assemble_viscous_residual_3d, assemble_viscous_residual_unstructured,
    compute_gradients_and_assemble_viscous_3d,
    compute_gradients_and_assemble_viscous_unstructured_f32,
    compute_gradients_and_assemble_viscous_unstructured_typed,
    compute_gradients_and_assemble_viscous_unstructured_with_scratch, face_inviscid_flux,
    face_inviscid_flux_first_order_boundary_soa, face_inviscid_flux_first_order_boundary_soa_f32,
    face_inviscid_flux_first_order_interior_soa, face_inviscid_flux_first_order_interior_soa_f32,
    face_inviscid_flux_from_interface, face_inviscid_flux_from_interface_f32,
    face_transport_coefficients, farfield_ghost, hanel_van_leer_flux, hllc_flux, inlet_ghost,
    interface_conserved_pair, outlet_ghost, physical_inviscid_flux, reconstruct_face_primitives,
    reconstruct_first_order, reconstruct_unstructured_boundary_face,
    reconstruct_unstructured_interior_face, roe_flux, slau2_flux, symmetry_ghost, van_leer_flux,
    viscous_face_flux, wall_face_conduction, wall_ghost, wall_ghost_temperature,
    zero_gradient_ghosts_1d,
};

/// 可压子模块路径别名（新代码请优先 `compressible::…`；`discretization::residual` 等保留兼容）。
pub use compressible::{
    bc_compressible, face_flux, face_flux_f32, face_flux_typed, hllc, hllc_f32, inviscid,
    inviscid_f32, reconstruction, reconstruction_unstructured, reconstruction_unstructured_f32,
    residual, roe, roe_f32, slau2, slau2_f32, van_leer, van_leer_f32, viscous, viscous_assembly,
    viscous_boundary_f32, viscous_f32, wall_thermal,
};

#[cfg(test)]
pub(crate) use compressible::freestream_pair;

// --- 不可压 FVM ---
pub use incompressible::{
    IncompressibleBoundaryApplyStats, IncompressibleBoundaryFaceState,
    IncompressibleBoundaryMassBalance, IncompressibleConvectionScheme, IncompressibleFaceFluxField,
    IncompressibleMassFluxBoundaryKind, IncompressibleMomentumPredictorConfig,
    IncompressibleMomentumPredictorSystem, IncompressiblePressureCorrectionConfig,
    IncompressiblePressureCorrectionSystem, IncompressibleVelocityLaplacian,
    PressureCorrectedRhieChowDivergenceConfig, RhieChowVelocityCorrectionConfig,
    apply_incompressible_boundary_conditions_3d, apply_pressure_correction_to_fields,
    apply_rhie_chow_pressure_projection_to_fields, assemble_incompressible_momentum_predictor_3d,
    assemble_incompressible_momentum_predictor_with_boundary_3d,
    assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d,
    assemble_incompressible_pressure_correction_3d, assemble_incompressible_pressure_poisson_3d,
    compute_incompressible_boundary_mass_balance_3d, compute_incompressible_divergence_3d,
    compute_incompressible_face_flux_divergence_3d, compute_incompressible_rhie_chow_divergence_3d,
    compute_incompressible_velocity_laplacian_3d,
    compute_pressure_corrected_rhie_chow_divergence_3d,
    corrected_incompressible_fields_rhie_chow_3d, incompressible_boundary_face_state,
    incompressible_boundary_face_velocity, incompressible_pressure_correction_dirichlet,
    subtract_d_pressure_gradient_from_velocity_3d,
};

use crate::core::Real;
use crate::error::Result;
use crate::field::ScalarField;
use crate::linalg::LinearSystem;
use crate::mesh::Mesh;

pub use bc::{apply_boundary_conditions, apply_dirichlet, apply_dirichlet_face, apply_neumann};
pub use diffusion_1d::assemble_diffusion_1d;
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
pub use structured_face_cache_f32::{StructuredFaceCacheF32, StructuredInteriorFaceF32};
pub use unstructured_face_cache::{
    InteriorFaceBatchStatic4, InteriorFaceBucketBatchLayout, InteriorFaceColoring,
    LsqPrecomputedCell, UnstructuredFaceTopology, UnstructuredInteriorFace,
    UnstructuredSolverMeshCache,
};
pub use unstructured_face_cache_f32::{
    GradientLimiterSampleF32, LsqPrecomputedCellF32, LuSgsCellCouplingF32,
    LuSgsUnstructuredCouplingsF32, UnstructuredBoundaryFaceF32, UnstructuredFaceTopologyF32,
    UnstructuredInteriorFaceF32, neg_dr, vec3_from_f32, vec3_to_f32,
};
pub use unstructured_limiter::UnstructuredGradientLimiter;

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
