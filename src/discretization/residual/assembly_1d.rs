//! 1D 结构化网格无粘残差装配。

use crate::core::Vector3;
use crate::discretization::{
    FaceFluxInput, InviscidFluxConfig, MusclStencil1d, face_inviscid_flux,
};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual};
use crate::mesh::StructuredMesh1d;
use crate::physics::IdealGasEoS;

use super::{accumulate_boundary_face, accumulate_interior_face};

struct InviscidFaceParams<'a> {
    eos: &'a IdealGasEoS,
    config: &'a InviscidFluxConfig,
    area: crate::core::Real,
    volume: crate::core::Real,
}

/// 1D 边界 ghost（`left` / `right`）；`None` 则跳过该边界面。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BoundaryGhosts1d {
    pub left: Option<crate::physics::ConservedState>,
    pub right: Option<crate::physics::ConservedState>,
}

/// 1D 无粘边界面处理方式。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum InviscidBoundary1d {
    /// 固定 ghost（每步不更新）。
    Fixed(BoundaryGhosts1d),
    /// 零梯度：每步以 owner 单元当前值作为 ghost。
    #[default]
    ZeroGradient,
}

impl InviscidBoundary1d {
    pub fn resolve(&self, fields: &ConservedFields) -> Result<BoundaryGhosts1d> {
        match self {
            Self::Fixed(ghosts) => Ok(*ghosts),
            Self::ZeroGradient => zero_gradient_ghosts_1d(fields),
        }
    }
}

/// 1D 零梯度边界 ghost（复制 owner 单元）。
pub fn zero_gradient_ghosts_1d(fields: &ConservedFields) -> Result<BoundaryGhosts1d> {
    let last = fields.num_cells() - 1;
    Ok(BoundaryGhosts1d {
        left: Some(fields.cell_state(0)?),
        right: Some(fields.cell_state(last)?),
    })
}

/// 装配 1D 无粘 Euler 残差：内部面 + 可选边界 ghost 面。
pub fn assemble_inviscid_residual_1d(
    mesh: &StructuredMesh1d,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
    boundaries: &BoundaryGhosts1d,
) -> Result<()> {
    let n = mesh.num_cells();
    if fields.num_cells() != n || residual.num_cells() != n {
        return Err(crate::error::AsimuError::Field(format!(
            "场/残差尺寸 {} 与网格单元数 {n} 不一致",
            fields.num_cells()
        )));
    }
    residual.clear();
    let params = InviscidFaceParams {
        eos,
        config,
        area: mesh.face_area(),
        volume: mesh.cell_volume(),
    };
    assemble_interior_faces_1d(mesh, fields, residual, &params)?;
    assemble_boundary_faces_1d(mesh, fields, residual, boundaries, &params)?;
    Ok(())
}

fn assemble_interior_faces_1d(
    mesh: &StructuredMesh1d,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    params: &InviscidFaceParams<'_>,
) -> Result<()> {
    let n = mesh.num_cells();
    let normal = Vector3::new(1.0, 0.0, 0.0);
    for i in 0..n.saturating_sub(1) {
        let left_of_owner = if i > 0 {
            Some(fields.cell_state(i - 1)?)
        } else {
            None
        };
        let owner = fields.cell_state(i)?;
        let neighbor = fields.cell_state(i + 1)?;
        let right_of_neighbor = if i + 2 < n {
            Some(fields.cell_state(i + 2)?)
        } else {
            None
        };
        let stencil = MusclStencil1d {
            left_of_owner: left_of_owner.as_ref(),
            owner: &owner,
            neighbor: &neighbor,
            right_of_neighbor: right_of_neighbor.as_ref(),
        };
        let flux = face_inviscid_flux(
            FaceFluxInput::from_stencil(stencil),
            normal,
            params.eos,
            params.config,
        )?;
        accumulate_interior_face(
            residual,
            i,
            i + 1,
            &flux,
            params.area,
            params.volume,
            params.volume,
        )?;
    }
    Ok(())
}

fn assemble_boundary_faces_1d(
    mesh: &StructuredMesh1d,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    boundaries: &BoundaryGhosts1d,
    params: &InviscidFaceParams<'_>,
) -> Result<()> {
    if let Some(ghost) = boundaries.left {
        let owner = fields.cell_state(0)?;
        let normal = Vector3::new(-1.0, 0.0, 0.0);
        let flux = face_inviscid_flux(
            FaceFluxInput::first_order(&owner, &ghost),
            normal,
            params.eos,
            params.config,
        )?;
        accumulate_boundary_face(residual, 0, &flux, params.area, params.volume)?;
    }
    if let Some(ghost) = boundaries.right {
        let owner = fields.cell_state(mesh.num_cells() - 1)?;
        let normal = Vector3::new(1.0, 0.0, 0.0);
        let flux = face_inviscid_flux(
            FaceFluxInput::first_order(&owner, &ghost),
            normal,
            params.eos,
            params.config,
        )?;
        let last = mesh.num_cells() - 1;
        accumulate_boundary_face(residual, last, &flux, params.area, params.volume)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::discretization::InviscidFluxConfig;
    use crate::physics::{ConservedState, PrimitiveState};

    use crate::physics::FreestreamParams;

    #[test]
    fn uniform_field_interior_only_has_zero_rhs() {
        let mesh = StructuredMesh1d::new("line", 4, 0.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fields = ConservedFields::from_freestream(4, &eos, &FreestreamParams::default())
            .expect("fields");
        let boundaries = zero_gradient_ghosts_1d(&fields).expect("bc");
        let mut rhs = ConservedResidual::zeros(4).expect("rhs");
        assemble_inviscid_residual_1d(
            &mesh,
            &fields,
            &mut rhs,
            &eos,
            &InviscidFluxConfig::default(),
            &boundaries,
        )
        .expect("assemble");
        assert!(rhs.density.values().iter().all(|&v| v.abs() < 1.0e-10));
        assert!(rhs.momentum_x.values().iter().all(|&v| v.abs() < 1.0e-10));
        assert!(rhs.total_energy.values().iter().all(|&v| v.abs() < 1.0e-10));
    }

    #[test]
    fn two_cell_discontinuity_has_opposing_mass_rhs() {
        let mesh = StructuredMesh1d::new("line", 2, 0.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let left = ConservedState::from_primitive(
            &eos,
            &PrimitiveState {
                density: 1.0,
                velocity: [0.0, 0.0, 0.0],
                pressure: 1.0,
                temperature: 1.0,
            },
        )
        .expect("left");
        let right = ConservedState::from_primitive(
            &eos,
            &PrimitiveState {
                density: 0.125,
                velocity: [0.0, 0.0, 0.0],
                pressure: 0.1,
                temperature: 1.0,
            },
        )
        .expect("right");
        let fields = ConservedFields::uniform(2, left).expect("f");
        let mut fields = fields;
        fields.density.values_mut()[1] = right.density;
        fields.total_energy.values_mut()[1] = right.total_energy;
        let mut rhs = ConservedResidual::zeros(2).expect("rhs");
        assemble_inviscid_residual_1d(
            &mesh,
            &fields,
            &mut rhs,
            &eos,
            &InviscidFluxConfig::default(),
            &BoundaryGhosts1d::default(),
        )
        .expect("assemble");
        let inv_dx = 1.0 / mesh.dx();
        assert!(approx_eq(
            rhs.density.values()[0],
            -0.390_660_485_785_962_96 * inv_dx,
            1.0e-8,
        ));
        assert!(approx_eq(
            rhs.density.values()[1],
            0.390_660_485_785_962_96 * inv_dx,
            1.0e-8,
        ));
    }
}
