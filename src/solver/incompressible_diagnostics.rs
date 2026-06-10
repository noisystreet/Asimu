use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::IncompressibleFields;
use crate::mesh::{BoundaryMesh, StructuredMesh3d};

const SIMPLEC_DIVERGENCE_LIMIT: Real = 1.0e50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompressiblePressureVelocityAlgorithm {
    Simplec,
    Piso,
}

impl IncompressiblePressureVelocityAlgorithm {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Simplec => "simplec",
            Self::Piso => "piso",
        }
    }
}

#[must_use]
pub(crate) fn pressure_velocity_algorithm(
    pressure_correctors: usize,
) -> IncompressiblePressureVelocityAlgorithm {
    if pressure_correctors.max(1) > 1 {
        IncompressiblePressureVelocityAlgorithm::Piso
    } else {
        IncompressiblePressureVelocityAlgorithm::Simplec
    }
}

#[must_use]
pub(crate) fn simplec_converged(
    tolerance: Option<Real>,
    min_iterations: usize,
    iterations: usize,
    residual: Real,
    momentum_residual: Real,
    velocity_delta: Real,
) -> bool {
    iterations >= min_iterations
        && tolerance
            .is_some_and(|tol| residual <= tol && momentum_residual <= tol && velocity_delta <= tol)
}

pub(crate) fn validate_simplec_step(
    residual: Real,
    momentum_residual: Real,
    velocity_delta: Real,
) -> Result<()> {
    if !residual.is_finite() || !momentum_residual.is_finite() || !velocity_delta.is_finite() {
        return Err(AsimuError::Solver("SIMPLEC 残差出现非有限值".to_string()));
    }
    if residual > SIMPLEC_DIVERGENCE_LIMIT
        || momentum_residual > SIMPLEC_DIVERGENCE_LIMIT
        || velocity_delta > SIMPLEC_DIVERGENCE_LIMIT
    {
        return Err(AsimuError::Solver(format!(
            "SIMPLEC 发散：continuity={residual:.4e}, momentum={momentum_residual:.4e}, velocity_delta={velocity_delta:.4e}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VelocityDeltaByRegion {
    pub(crate) all: Real,
    pub(crate) interior: Real,
    pub(crate) boundary: Real,
}

pub(crate) fn max_velocity_delta_by_region(
    mesh: &StructuredMesh3d,
    boundary: &BoundarySet,
    fields: &IncompressibleFields,
    u: &[Real],
    v: &[Real],
    w: &[Real],
) -> Result<VelocityDeltaByRegion> {
    let mut constrained_owner = vec![false; mesh.num_cells()];
    for patch in boundary.patches() {
        if !is_velocity_constrained_kind(&patch.kind) {
            continue;
        }
        for face_id in &patch.face_ids {
            let owner = mesh.face_owner(*face_id)?;
            constrained_owner[owner.index() as usize] = true;
        }
    }

    let mut delta = VelocityDeltaByRegion {
        all: 0.0,
        interior: 0.0,
        boundary: 0.0,
    };
    for idx in 0..fields.velocity_x.len() {
        let cell_delta = (u[idx] - fields.velocity_x.values()[idx])
            .abs()
            .max((v[idx] - fields.velocity_y.values()[idx]).abs())
            .max((w[idx] - fields.velocity_z.values()[idx]).abs());
        delta.all = delta.all.max(cell_delta);
        if constrained_owner[idx] {
            delta.boundary = delta.boundary.max(cell_delta);
        } else {
            delta.interior = delta.interior.max(cell_delta);
        }
    }
    Ok(delta)
}

fn is_velocity_constrained_kind(kind: &BoundaryKind) -> bool {
    matches!(
        kind,
        BoundaryKind::Wall { .. }
            | BoundaryKind::MovingWall { .. }
            | BoundaryKind::IncompressibleVelocityInlet { .. }
            | BoundaryKind::Inlet { .. }
    )
}
