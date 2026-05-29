//! 可压缩 Euler 无粘残差装配（FVM 控制体积分）。

mod assembly_1d;
mod assembly_3d;

use crate::core::Real;
use crate::discretization::InviscidFlux;
use crate::error::Result;
use crate::field::ConservedResidual;

pub use assembly_1d::{BoundaryGhosts1d, assemble_inviscid_residual_1d};
pub use assembly_3d::assemble_inviscid_residual_3d;

fn add_inviscid_flux(
    residual: &mut ConservedResidual,
    cell: usize,
    flux: &InviscidFlux,
    scale: Real,
) -> Result<()> {
    residual.add_flux_to_cell(cell, flux.mass, flux.momentum, flux.energy, scale)
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
