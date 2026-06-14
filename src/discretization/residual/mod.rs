//! 可压缩 Euler 无粘残差装配（FVM 控制体积分）。
//!
//! 理论：[`docs/theory/inviscid_flux.md`](../../docs/theory/inviscid_flux.md) §3

mod assembly_1d;
mod assembly_3d;
mod assembly_3d_typed;
mod assembly_3d_viscous;
mod assembly_unstructured;
mod assembly_unstructured_typed;
mod assembly_unstructured_viscous;
mod assembly_unstructured_viscous_f32;
mod assembly_unstructured_viscous_typed;
mod face_flux_3d;
mod muscl_stencil_3d;

use crate::core::{ComputeFloat, Real};
use crate::discretization::InviscidFlux;
use crate::discretization::inviscid_f32::InviscidFluxF32;
use crate::error::Result;
use crate::field::{ConservedResidual, ConservedResidualT};

pub use assembly_1d::{
    BoundaryGhosts1d, InviscidBoundary1d, assemble_inviscid_residual_1d, zero_gradient_ghosts_1d,
};
pub use assembly_3d::{InviscidAssembly3dParams, assemble_inviscid_residual_3d};
pub use assembly_3d_typed::{InviscidAssembly3dTypedParams, assemble_inviscid_residual_3d_typed};
pub use assembly_3d_viscous::{
    ViscousAssembly3dInput, ViscousAssembly3dParams, assemble_viscous_residual_3d,
    compute_gradients_and_assemble_viscous_3d,
};
pub use assembly_unstructured::{
    InviscidAssemblyUnstructuredParams, assemble_inviscid_residual_unstructured,
};
pub use assembly_unstructured_typed::{
    InviscidAssemblyUnstructuredTypedParams, InviscidTypedScatterBackend,
    assemble_inviscid_residual_unstructured_typed,
};
pub use assembly_unstructured_viscous::{
    ViscousAssemblyUnstructuredInput, ViscousAssemblyUnstructuredParams,
    ViscousAssemblyUnstructuredScratch, assemble_viscous_residual_unstructured,
    compute_gradients_and_assemble_viscous_unstructured,
    compute_gradients_and_assemble_viscous_unstructured_with_scratch,
};
pub use assembly_unstructured_viscous_f32::{
    ViscousAssemblyUnstructuredF32Input, compute_gradients_and_assemble_viscous_unstructured_f32,
};
pub use assembly_unstructured_viscous_typed::{
    ViscousAssemblyUnstructuredTypedInput, ViscousTypedScatterBackend,
    compute_gradients_and_assemble_viscous_unstructured_typed,
};
pub use face_flux_3d::{
    BoundaryInviscidFluxInput, inviscid_boundary_face_flux,
    inviscid_boundary_face_flux_with_normal, inviscid_i_face_flux, inviscid_j_face_flux,
    inviscid_k_face_flux,
};

/// 忽略退化（零体积）控制体的体积下限。
const DEGENERATE_VOLUME: Real = 1.0e-30;
const DEGENERATE_VOLUME_F32: f32 = 1.0e-30;

pub(crate) fn is_degenerate_volume(volume: Real) -> bool {
    volume <= DEGENERATE_VOLUME
}

pub(crate) fn is_degenerate_volume_f32(volume: f32) -> bool {
    volume <= DEGENERATE_VOLUME_F32
}

fn add_inviscid_flux<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    cell: usize,
    flux: &InviscidFlux,
    scale: Real,
) -> Result<()> {
    residual.add_flux_to_cell(cell, flux.mass, flux.momentum, flux.energy, scale)
}

/// 将面通量写入 owner / neighbor 控制体右手项（f32 残差，无 Real 桥接）。
pub fn accumulate_interior_face_f32(
    residual: &mut ConservedResidualT<f32>,
    owner: usize,
    neighbor: usize,
    flux: &InviscidFluxF32,
    area: f32,
    owner_volume: f32,
    neighbor_volume: f32,
) -> Result<()> {
    let owner_scale = -area / owner_volume;
    let neighbor_scale = area / neighbor_volume;
    add_inviscid_flux_f32(residual, owner, flux, owner_scale)?;
    add_inviscid_flux_f32(residual, neighbor, flux, neighbor_scale)?;
    Ok(())
}

/// 边界面：仅 owner 单元贡献（f32 残差）。
pub fn accumulate_boundary_face_f32(
    residual: &mut ConservedResidualT<f32>,
    owner: usize,
    flux: &InviscidFluxF32,
    area: f32,
    owner_volume: f32,
) -> Result<()> {
    add_inviscid_flux_f32(residual, owner, flux, -area / owner_volume)
}

fn add_inviscid_flux_f32(
    residual: &mut ConservedResidualT<f32>,
    cell: usize,
    flux: &InviscidFluxF32,
    scale: f32,
) -> Result<()> {
    if cell >= residual.num_cells() {
        return Err(crate::error::AsimuError::Field(format!(
            "残差单元索引越界: {cell}"
        )));
    }
    residual.density.values_mut()[cell] += scale * flux.mass;
    residual.momentum_x.values_mut()[cell] += scale * flux.momentum[0];
    residual.momentum_y.values_mut()[cell] += scale * flux.momentum[1];
    residual.momentum_z.values_mut()[cell] += scale * flux.momentum[2];
    residual.total_energy.values_mut()[cell] += scale * flux.energy;
    Ok(())
}

/// 将面通量写入 owner / neighbor 控制体右手项（typed 残差）。
pub fn accumulate_interior_face_typed<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    owner: usize,
    neighbor: usize,
    flux: &InviscidFlux,
    area: Real,
    owner_volume: Real,
    neighbor_volume: Real,
) -> Result<()> {
    let owner_scale = -area / owner_volume;
    let neighbor_scale = area / neighbor_volume;
    add_inviscid_flux(residual, owner, flux, owner_scale)?;
    add_inviscid_flux(residual, neighbor, flux, neighbor_scale)?;
    Ok(())
}

/// 边界面：仅 owner 单元贡献（typed 残差）。
pub fn accumulate_boundary_face_typed<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    owner: usize,
    flux: &InviscidFlux,
    area: Real,
    owner_volume: Real,
) -> Result<()> {
    let owner_scale = -area / owner_volume;
    add_inviscid_flux(residual, owner, flux, owner_scale)
}

/// 将面通量写入 owner / neighbor 控制体右手项。
///
/// \(\mathrm{d}U_i/\mathrm{d}t = -\frac{1}{V_i}\sum_f \mathbf{F}_f \cdot \mathbf{S}_f\)；
/// 此处 `flux` 为沿 owner→neighbor 法向的数值通量，`area` 为面积。
pub fn accumulate_interior_face(
    residual: &mut ConservedResidual,
    owner: usize,
    neighbor: usize,
    flux: &InviscidFlux,
    area: Real,
    owner_volume: Real,
    neighbor_volume: Real,
) -> Result<()> {
    let owner_scale = -area / owner_volume;
    let neighbor_scale = area / neighbor_volume;
    add_inviscid_flux(residual, owner, flux, owner_scale)?;
    add_inviscid_flux(residual, neighbor, flux, neighbor_scale)?;
    Ok(())
}

/// 边界面：仅 owner 单元贡献（无内侧 neighbor）。
pub fn accumulate_boundary_face(
    residual: &mut ConservedResidual,
    owner: usize,
    flux: &InviscidFlux,
    area: Real,
    owner_volume: Real,
) -> Result<()> {
    add_inviscid_flux(residual, owner, flux, -area / owner_volume)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::InviscidFlux;

    #[test]
    fn interior_face_opposes_owner_and_neighbor() {
        let mut rhs = ConservedResidual::zeros(2).expect("rhs");
        let flux = InviscidFlux {
            mass: 2.0,
            momentum: [0.0, 0.0, 0.0],
            energy: 0.0,
        };
        accumulate_interior_face(&mut rhs, 0, 1, &flux, 1.0, 1.0, 1.0).expect("acc");
        assert!((rhs.density.values()[0] + 2.0).abs() < 1.0e-12);
        assert!((rhs.density.values()[1] - 2.0).abs() < 1.0e-12);
    }
}
