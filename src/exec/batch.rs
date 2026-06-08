//! 四内面批静态几何 SoA（init-time；exec 自有，不依赖 discretization）。

use crate::core::{Real, Vector3};

const DEGENERATE_CELL_VOLUME: Real = 1.0e-30;

/// 四内面静态几何 SoA（不含 μ/λ 与时变场）。
///
/// ADR 0013 E2：由 mesh cache 构造期写入；SIMD batch 内核经本类型传入 `exec::cpu`。
#[derive(Debug, Clone, Copy)]
pub struct ExecFaceBatchStatic4 {
    pub face_indices: [usize; 4],
    pub owners: [usize; 4],
    pub neighbors: [usize; 4],
    pub nx: [Real; 4],
    pub ny: [Real; 4],
    pub nz: [Real; 4],
    pub owner_rhs_scale: [Real; 4],
    pub neighbor_rhs_scale: [Real; 4],
    pub area: [Real; 4],
    pub owner_volume: [Real; 4],
    pub neighbor_volume: [Real; 4],
}

impl ExecFaceBatchStatic4 {
    /// 四路均可走 SIMD 批内核（非退化体积且 RHS scale 非双零）。
    #[must_use]
    pub fn simd_eligible(&self) -> bool {
        for lane in 0..4 {
            if self.owner_rhs_scale[lane] == 0.0 && self.neighbor_rhs_scale[lane] == 0.0 {
                return false;
            }
            if is_degenerate_cell_volume(self.owner_volume[lane])
                || is_degenerate_cell_volume(self.neighbor_volume[lane])
            {
                return false;
            }
        }
        true
    }

    #[must_use]
    pub fn normal(&self, lane: usize) -> Vector3 {
        Vector3::new(self.nx[lane], self.ny[lane], self.nz[lane])
    }

    #[must_use]
    pub fn normals(&self) -> [Vector3; 4] {
        [
            self.normal(0),
            self.normal(1),
            self.normal(2),
            self.normal(3),
        ]
    }
}

fn is_degenerate_cell_volume(volume: Real) -> bool {
    volume <= DEGENERATE_CELL_VOLUME
}
