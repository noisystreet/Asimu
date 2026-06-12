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

/// 动量对流用的结构化内面速度（与 Rhie-Chow 共用 ghost 语义）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct StructuredMomentumFaceQuery<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub fields: &'a IncompressibleFields,
    pub axis: usize,
    pub cell: (usize, usize, usize),
    pub upper: bool,
    pub periodic_x: bool,
    pub boundary: &'a IncompressibleBoundaryOwnerMap,
}

#[must_use]
pub(crate) fn structured_momentum_face_velocity(query: StructuredMomentumFaceQuery<'_>) -> Real {
    let StructuredMomentumFaceQuery {
        mesh,
        fields,
        axis,
        cell: (i, j, k),
        upper,
        periodic_x,
        boundary,
    } = query;
    let (left, right) = match axis {
        0 => {
            let left_i = if upper {
                i
            } else {
                west_with_periodic(i, mesh.nx, periodic_x)
            };
            let right_i = if upper {
                east_with_periodic(i, mesh.nx, periodic_x)
            } else {
                i
            };
            (
                mesh.cell_index(left_i, j, k),
                mesh.cell_index(right_i, j, k),
            )
        }
        1 => {
            let left_j = if upper { j } else { south(j) };
            let right_j = if upper { north(j, mesh.ny) } else { j };
            (
                mesh.cell_index(i, left_j, k),
                mesh.cell_index(i, right_j, k),
            )
        }
        _ => {
            let left_k = if upper { k } else { bottom(k) };
            let right_k = if upper { top(k, mesh.nz) } else { k };
            (
                mesh.cell_index(i, j, left_k),
                mesh.cell_index(i, j, right_k),
            )
        }
    };
    interior_face_velocity(fields, left, right, axis, boundary)
}

fn west_with_periodic(i: usize, nx: usize, periodic_x: bool) -> usize {
    if periodic_x && i == 0 {
        nx - 1
    } else {
        i.saturating_sub(1)
    }
}

fn east_with_periodic(i: usize, nx: usize, periodic_x: bool) -> usize {
    if periodic_x && i + 1 == nx {
        0
    } else {
        (i + 1).min(nx - 1)
    }
}

fn south(j: usize) -> usize {
    j.saturating_sub(1)
}

fn north(j: usize, ny: usize) -> usize {
    (j + 1).min(ny - 1)
}

fn bottom(k: usize) -> usize {
    k.saturating_sub(1)
}

fn top(k: usize, nz: usize) -> usize {
    (k + 1).min(nz - 1)
}
