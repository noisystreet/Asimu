use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::error::Result;
use crate::field::IncompressibleFields;
use crate::mesh::{BoundaryMesh, StructuredMesh3d};

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
