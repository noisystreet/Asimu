//! 非结构 `block_lusgs` 预条件器静态拓扑（行 CSR + 面块 slot；仅依赖网格几何）。

use crate::core::Real;
use crate::discretization::unstructured_face_cache::UnstructuredInteriorFace;

/// 面块 off-diagonal slot：行/列单元与粘性抛物尺度（\(6 A_f^2/V_i^2\)）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockLusgsOffDiagonalSlot {
    pub row: usize,
    pub col: usize,
    pub face_idx: usize,
    /// 行单元侧抛物尺度 \(6 A_f^2 / V_{\mathrm{row}}^2\)。
    pub viscous_parabolic_scale: Real,
}

/// `block_lusgs` 预条件器 mesh 级缓存：行 CSR 与 off-diagonal 写入顺序。
#[derive(Debug, Clone, PartialEq)]
pub struct BlockLusgsPreconditionerTopology {
    pub row_offsets: Vec<usize>,
    pub off_diagonal: Vec<BlockLusgsOffDiagonalSlot>,
}

impl BlockLusgsPreconditionerTopology {
    /// 由内部面拓扑构建；写入顺序与 legacy `fill_off_diagonal_blocks` 一致（逐面 owner→neighbor）。
    #[must_use]
    pub fn from_interior_faces(num_cells: usize, interior: &[UnstructuredInteriorFace]) -> Self {
        let row_offsets = row_offsets_from_interior(num_cells, interior);
        let mut off_diagonal = Vec::with_capacity(interior.len() * 2);
        for (face_idx, face) in interior.iter().enumerate() {
            off_diagonal.push(BlockLusgsOffDiagonalSlot {
                row: face.owner,
                col: face.neighbor,
                face_idx,
                viscous_parabolic_scale: parabolic_scale(face.area, face.owner_volume),
            });
            off_diagonal.push(BlockLusgsOffDiagonalSlot {
                row: face.neighbor,
                col: face.owner,
                face_idx,
                viscous_parabolic_scale: parabolic_scale(face.area, face.neighbor_volume),
            });
        }
        Self {
            row_offsets,
            off_diagonal,
        }
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.row_offsets.len().saturating_sub(1)
    }

    #[must_use]
    pub fn num_off_diagonal_blocks(&self) -> usize {
        self.off_diagonal.len()
    }
}

fn row_offsets_from_interior(
    num_cells: usize,
    interior: &[UnstructuredInteriorFace],
) -> Vec<usize> {
    let mut row_counts = vec![0usize; num_cells];
    for face in interior {
        row_counts[face.owner] += 1;
        row_counts[face.neighbor] += 1;
    }
    let mut row_offsets = Vec::with_capacity(num_cells + 1);
    row_offsets.push(0);
    for count in row_counts {
        row_offsets.push(row_offsets.last().copied().unwrap_or(0) + count);
    }
    row_offsets
}

fn parabolic_scale(area: Real, volume: Real) -> Real {
    const PARABOLIC_SPECTRAL_FACTOR_3D: Real = 6.0;
    if area <= Real::EPSILON || volume <= 1.0e-30 {
        0.0
    } else {
        PARABOLIC_SPECTRAL_FACTOR_3D * area * area / (volume * volume)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Vector3;

    fn sample_interior_face(owner: usize, neighbor: usize) -> UnstructuredInteriorFace {
        UnstructuredInteriorFace {
            owner,
            neighbor,
            area: 0.5,
            normal: Vector3::new(1.0, 0.0, 0.0),
            owner_volume: 2.0,
            neighbor_volume: 3.0,
            inv_owner_volume: 0.5,
            inv_neighbor_volume: 1.0 / 3.0,
            owner_rhs_scale: -0.25,
            neighbor_rhs_scale: 1.0 / 6.0,
            lsq_dr: Vector3::new(1.0, 0.0, 0.0),
            lsq_w: 1.0,
            dr_owner_to_face: Vector3::new(0.5, 0.0, 0.0),
            dr_neighbor_to_face: Vector3::new(-0.5, 0.0, 0.0),
        }
    }

    #[test]
    fn topology_row_offsets_and_entry_order_match_two_face_chain() {
        let interior = vec![sample_interior_face(0, 1), sample_interior_face(1, 2)];
        let topo = BlockLusgsPreconditionerTopology::from_interior_faces(3, &interior);
        assert_eq!(topo.row_offsets, vec![0, 1, 3, 4]);
        assert_eq!(topo.off_diagonal.len(), 4);
        assert_eq!(topo.off_diagonal[0].row, 0);
        assert_eq!(topo.off_diagonal[0].col, 1);
        assert_eq!(topo.off_diagonal[1].row, 1);
        assert_eq!(topo.off_diagonal[1].col, 0);
    }
}
