//! 着色桶粘性 compute/scatter 复用 flat buffer（ADR 0013 P8′ → E2）。

use crate::core::Real;

/// 单面几何与物性（exec scratch；与 discretization 粘性 scatter 布局一致）。
#[derive(Debug, Clone, Copy, Default)]
pub struct ColoredViscousFaceGeom {
    pub owner: usize,
    pub neighbor: usize,
    pub nx: Real,
    pub ny: Real,
    pub nz: Real,
    pub mu: Real,
    pub lambda: Real,
    pub owner_scale: Real,
    pub neighbor_scale: Real,
}

/// 单面粘性动量/能量通量（scatter 前）。
#[derive(Debug, Clone, Copy, Default)]
pub struct ColoredViscousFaceFlux {
    pub mx: Real,
    pub my: Real,
    pub mz: Real,
    pub energy: Real,
}

/// 单着色桶 batch×4 + remainder 槽；按 `max_bucket_faces` 预分配。
#[derive(Debug, Default)]
pub struct ColoredViscousFaceBuffer {
    geoms: Vec<ColoredViscousFaceGeom>,
    fluxes: Vec<ColoredViscousFaceFlux>,
    batch_counts: Vec<u8>,
    slot_valid: Vec<bool>,
}

/// 步内临时拆出的 flat buffer（compute 与 scratch 解耦借用）。
#[derive(Debug, Default)]
pub struct ColoredViscousWorkingSet {
    pub geoms: Vec<ColoredViscousFaceGeom>,
    pub fluxes: Vec<ColoredViscousFaceFlux>,
    pub batch_counts: Vec<u8>,
    pub slot_valid: Vec<bool>,
}

impl ColoredViscousFaceBuffer {
    #[must_use]
    pub fn with_capacity(max_bucket_faces: usize) -> Self {
        let max_batches = max_bucket_faces.div_ceil(4);
        Self {
            geoms: Vec::with_capacity(max_bucket_faces),
            fluxes: Vec::with_capacity(max_bucket_faces),
            batch_counts: Vec::with_capacity(max_batches),
            slot_valid: Vec::with_capacity(max_bucket_faces),
        }
    }

    /// 按桶内 batch 数与余面数扩容（热路径 trust：调用方已校验规模 ≤ init 容量）。
    pub fn ensure_bucket_layout(&mut self, num_batches: usize, num_remainder: usize) {
        self.batch_counts.resize(num_batches, 0);
        let total = num_batches * 4 + num_remainder;
        self.geoms
            .resize_with(total, ColoredViscousFaceGeom::default);
        self.fluxes
            .resize_with(total, ColoredViscousFaceFlux::default);
        self.slot_valid.resize(total, false);
    }

    /// 非 SIMD 并行桶：全部为 remainder 槽。
    pub fn ensure_face_slots(&mut self, num_faces: usize) {
        self.batch_counts.clear();
        self.geoms
            .resize_with(num_faces, ColoredViscousFaceGeom::default);
        self.fluxes
            .resize_with(num_faces, ColoredViscousFaceFlux::default);
        self.slot_valid.resize(num_faces, false);
    }

    #[must_use]
    pub fn geoms(&self) -> &[ColoredViscousFaceGeom] {
        &self.geoms
    }

    #[must_use]
    pub fn geoms_mut(&mut self) -> &mut [ColoredViscousFaceGeom] {
        &mut self.geoms
    }

    #[must_use]
    pub fn fluxes(&self) -> &[ColoredViscousFaceFlux] {
        &self.fluxes
    }

    #[must_use]
    pub fn fluxes_mut(&mut self) -> &mut [ColoredViscousFaceFlux] {
        &mut self.fluxes
    }

    #[must_use]
    pub fn batch_counts(&self) -> &[u8] {
        &self.batch_counts
    }

    #[must_use]
    pub fn batch_counts_mut(&mut self) -> &mut [u8] {
        &mut self.batch_counts
    }

    #[must_use]
    pub fn slot_valid(&self) -> &[bool] {
        &self.slot_valid
    }

    #[must_use]
    pub fn slot_valid_mut(&mut self) -> &mut [bool] {
        &mut self.slot_valid
    }

    /// 拆出工作集供并行 compute（避免与 `ExecutionContext` 重叠借用）。
    pub fn take_working_set(&mut self) -> ColoredViscousWorkingSet {
        ColoredViscousWorkingSet {
            geoms: std::mem::take(&mut self.geoms),
            fluxes: std::mem::take(&mut self.fluxes),
            batch_counts: std::mem::take(&mut self.batch_counts),
            slot_valid: std::mem::take(&mut self.slot_valid),
        }
    }

    /// 归还工作集（保留 capacity）。
    pub fn restore_working_set(&mut self, ws: ColoredViscousWorkingSet) {
        self.geoms = ws.geoms;
        self.fluxes = ws.fluxes;
        self.batch_counts = ws.batch_counts;
        self.slot_valid = ws.slot_valid;
    }
}

impl ColoredViscousWorkingSet {
    /// 按桶内 batch 数与余面数扩容（热路径 trust：调用方已校验规模 ≤ init 容量）。
    pub fn ensure_bucket_layout(&mut self, num_batches: usize, num_remainder: usize) {
        self.batch_counts.resize(num_batches, 0);
        let total = num_batches * 4 + num_remainder;
        self.geoms
            .resize_with(total, ColoredViscousFaceGeom::default);
        self.fluxes
            .resize_with(total, ColoredViscousFaceFlux::default);
        self.slot_valid.resize(total, false);
    }

    /// 由 batch4 compute 计数填充 scatter 有效掩码（`[0, num_batches×4)`；余面掩码由 compute 写入）。
    pub fn fill_batch_slot_valid(&mut self) {
        for (batch_idx, &count) in self.batch_counts.iter().enumerate() {
            let base = batch_idx * 4;
            let n = count as usize;
            for lane in 0..4 {
                self.slot_valid[base + lane] = lane < n;
            }
        }
    }

    /// 非 SIMD 并行桶：全部为 remainder 槽。
    pub fn ensure_face_slots(&mut self, num_faces: usize) {
        self.batch_counts.clear();
        self.geoms
            .resize_with(num_faces, ColoredViscousFaceGeom::default);
        self.fluxes
            .resize_with(num_faces, ColoredViscousFaceFlux::default);
        self.slot_valid.resize(num_faces, false);
    }
}

#[cfg(test)]
mod tests {
    use super::ColoredViscousWorkingSet;

    #[test]
    fn fill_batch_slot_valid_marks_lanes_from_counts() {
        let mut ws = ColoredViscousWorkingSet::default();
        ws.ensure_bucket_layout(2, 1);
        ws.batch_counts[0] = 3;
        ws.batch_counts[1] = 1;
        ws.slot_valid[8] = true;
        ws.fill_batch_slot_valid();
        assert!(ws.slot_valid[0] && ws.slot_valid[1] && ws.slot_valid[2] && !ws.slot_valid[3]);
        assert!(ws.slot_valid[4] && !ws.slot_valid[5] && !ws.slot_valid[6] && !ws.slot_valid[7]);
        assert!(ws.slot_valid[8]);
    }
}
