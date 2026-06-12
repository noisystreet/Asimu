//! 结构化不可压缩边界 ghost 速度，用于贴壁内面通量。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::discretization::incompressible_face_boundary::{
    cell_velocity, is_normal_component, tangential_velocity,
};
use crate::field::IncompressibleFields;
use crate::mesh::{BoundaryMesh, BoundaryMesh3d, StructuredMesh3d};

#[derive(Debug, Clone, Copy)]
pub(crate) enum BoundaryOwnerKind {
    NoSlipWall,
    MovingWall([Real; 3]),
    Symmetry,
    SlipWall,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BoundaryOwnerInfo {
    kind: BoundaryOwnerKind,
    outward_normal: [Real; 3],
}

/// 每个单元若为边界 owner，则记录其边界类型与外法向。
#[derive(Debug)]
pub(crate) struct IncompressibleBoundaryOwnerMap {
    owners: Vec<Option<BoundaryOwnerInfo>>,
}

impl IncompressibleBoundaryOwnerMap {
    pub(crate) fn build(mesh: &StructuredMesh3d, boundary: &BoundarySet) -> Self {
        let mut owners = vec![None; mesh.num_cells()];
        for patch in boundary.patches() {
            let Some(owner_kind) = classify_owner_kind(&patch.kind) else {
                continue;
            };
            for &face in &patch.face_ids {
                if let Ok(owner) = mesh.face_owner(face) {
                    let normal = mesh.face_normal_3d(face).expect("face normal");
                    owners[owner.index() as usize] = Some(BoundaryOwnerInfo {
                        kind: owner_kind,
                        outward_normal: [normal.x, normal.y, normal.z],
                    });
                }
            }
        }
        Self { owners }
    }

    pub(crate) fn get(&self, cell: usize) -> Option<BoundaryOwnerInfo> {
        self.owners.get(cell).copied().flatten()
    }
}

fn classify_owner_kind(kind: &BoundaryKind) -> Option<BoundaryOwnerKind> {
    match kind {
        BoundaryKind::Wall { no_slip: true, .. } => Some(BoundaryOwnerKind::NoSlipWall),
        BoundaryKind::Wall { no_slip: false, .. } => Some(BoundaryOwnerKind::SlipWall),
        BoundaryKind::MovingWall { velocity } => Some(BoundaryOwnerKind::MovingWall(*velocity)),
        BoundaryKind::Symmetry => Some(BoundaryOwnerKind::Symmetry),
        _ => None,
    }
}

/// 贴壁单元在面插值中使用的 ghost 速度（使面算术平均满足边界语义）。
#[must_use]
pub(crate) fn boundary_ghost_velocity(
    fields: &IncompressibleFields,
    interior: usize,
    kind: BoundaryOwnerKind,
    outward_normal: [Real; 3],
) -> [Real; 3] {
    let interior_v = cell_velocity(fields, interior);
    match kind {
        BoundaryOwnerKind::NoSlipWall => [-interior_v[0], -interior_v[1], -interior_v[2]],
        BoundaryOwnerKind::SlipWall => {
            let un = interior_v[0] * outward_normal[0]
                + interior_v[1] * outward_normal[1]
                + interior_v[2] * outward_normal[2];
            [
                interior_v[0] - 2.0 * un * outward_normal[0],
                interior_v[1] - 2.0 * un * outward_normal[1],
                interior_v[2] - 2.0 * un * outward_normal[2],
            ]
        }
        BoundaryOwnerKind::MovingWall(wall_velocity) => {
            let wall_t = tangential_velocity(wall_velocity, outward_normal);
            let mut ghost = [-interior_v[0], -interior_v[1], -interior_v[2]];
            for component in 0..3 {
                if is_normal_component(component, outward_normal) {
                    ghost[component] = -interior_v[component];
                } else {
                    ghost[component] = 2.0 * wall_t[component] - interior_v[component];
                }
            }
            ghost
        }
        BoundaryOwnerKind::Symmetry => {
            let mut ghost = interior_v;
            for component in 0..3 {
                if is_normal_component(component, outward_normal) {
                    ghost[component] = -interior_v[component];
                }
            }
            ghost
        }
    }
}

/// 内面速度分量：若一侧为边界 owner，则用 ghost 替代该侧 cell 速度做算术平均。
#[must_use]
pub(crate) fn interior_face_velocity(
    fields: &IncompressibleFields,
    left: usize,
    right: usize,
    component: usize,
    boundary: &IncompressibleBoundaryOwnerMap,
) -> Real {
    let left_v = side_velocity(fields, left, right, boundary)[component];
    let right_v = side_velocity(fields, right, left, boundary)[component];
    0.5 * (left_v + right_v)
}

fn side_velocity(
    fields: &IncompressibleFields,
    cell: usize,
    interior: usize,
    boundary: &IncompressibleBoundaryOwnerMap,
) -> [Real; 3] {
    match boundary.get(cell) {
        Some(info) => boundary_ghost_velocity(fields, interior, info.kind, info.outward_normal),
        None => cell_velocity(fields, cell),
    }
}
