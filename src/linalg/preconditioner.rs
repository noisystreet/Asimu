use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::sparse::CsrMatrix;
use super::{Preconditioner, ensure_vector_len};

/// 恒等预条件器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdentityPreconditioner {
    n: usize,
}

impl IdentityPreconditioner {
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Preconditioner for IdentityPreconditioner {
    fn dimension(&self) -> usize {
        self.n
    }

    fn apply(&mut self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        ensure_vector_len(rhs, self.n, "identity preconditioner rhs")?;
        ensure_vector_len(out, self.n, "identity preconditioner out")?;
        out.copy_from_slice(rhs);
        Ok(())
    }
}

/// LU-SGS 对角预条件器：按分量缩放 \(M^{-1}r\)。
#[derive(Debug, Clone, PartialEq)]
pub struct LusgsDiagonalPreconditioner {
    scales: Vec<Real>,
}

impl LusgsDiagonalPreconditioner {
    pub fn from_scales(scales: Vec<Real>) -> Result<Self> {
        if scales.is_empty() {
            return Err(AsimuError::Linalg(
                "LU-SGS 对角预条件器尺寸必须大于 0".to_string(),
            ));
        }
        if scales.iter().any(|s| !s.is_finite() || *s < 0.0) {
            return Err(AsimuError::Linalg(
                "LU-SGS 对角预条件器 scale 必须为有限非负数".to_string(),
            ));
        }
        Ok(Self { scales })
    }

    /// 从单元 \(\Delta t_i,\sigma_i\) 构造，并按 `components_per_cell` 重复到守恒分量。
    pub fn from_lusgs_diagonal(
        dt: &[Real],
        sigma: &[Real],
        omega: Real,
        components_per_cell: usize,
    ) -> Result<Self> {
        if dt.len() != sigma.len() || dt.is_empty() || components_per_cell == 0 {
            return Err(AsimuError::Linalg(
                "LU-SGS 对角预条件器 dt/sigma/分量数不一致".to_string(),
            ));
        }
        if !omega.is_finite() || omega <= 0.0 || omega > 1.0 {
            return Err(AsimuError::Linalg(
                "LU-SGS 对角预条件器 omega 须在 (0,1] 内".to_string(),
            ));
        }
        let mut scales = Vec::with_capacity(dt.len() * components_per_cell);
        for (&dt_i, &sigma_i) in dt.iter().zip(sigma.iter()) {
            if !dt_i.is_finite() || dt_i <= 0.0 || !sigma_i.is_finite() || sigma_i < 0.0 {
                return Err(AsimuError::Linalg(
                    "LU-SGS 对角预条件器 dt 须为正且 sigma 非负".to_string(),
                ));
            }
            let scale = omega * dt_i / (1.0 + dt_i * sigma_i);
            for _ in 0..components_per_cell {
                scales.push(scale);
            }
        }
        Self::from_scales(scales)
    }
}

impl Preconditioner for LusgsDiagonalPreconditioner {
    fn dimension(&self) -> usize {
        self.scales.len()
    }

    fn apply(&mut self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        ensure_vector_len(rhs, self.dimension(), "lusgs diagonal rhs")?;
        ensure_vector_len(out, self.dimension(), "lusgs diagonal out")?;
        for ((dst, src), scale) in out.iter_mut().zip(rhs.iter()).zip(self.scales.iter()) {
            *dst = scale * src;
        }
        Ok(())
    }
}

/// 单元局部块对角预条件器：每个控制体一个固定大小的小块逆矩阵。
#[derive(Debug, Clone, PartialEq)]
pub struct CellBlockDiagonalPreconditioner {
    block_size: usize,
    inverse_blocks: Vec<Real>,
}

impl CellBlockDiagonalPreconditioner {
    pub fn from_blocks(block_size: usize, blocks: Vec<Real>) -> Result<Self> {
        let block_entries = block_size * block_size;
        if block_size == 0 || blocks.is_empty() || blocks.len() % block_entries != 0 {
            return Err(AsimuError::Linalg(
                "块对角预条件器 block_size/blocks 尺寸不一致".to_string(),
            ));
        }
        let num_blocks = blocks.len() / block_entries;
        let mut inverse_blocks = Vec::with_capacity(blocks.len());
        for block in 0..num_blocks {
            let start = block * block_size * block_size;
            inverse_blocks.extend(invert_dense_block(
                block_size,
                &blocks[start..start + block_size * block_size],
            )?);
        }
        Ok(Self {
            block_size,
            inverse_blocks,
        })
    }

    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.inverse_blocks.len() / (self.block_size * self.block_size)
    }
}

/// CSR 矩阵 Jacobi（对角）预条件器：\(z_i = r_i / A_{ii}\)。
#[derive(Debug, Clone, PartialEq)]
pub struct CsrJacobiPreconditioner {
    inverse_diagonal: Vec<Real>,
}

impl CsrJacobiPreconditioner {
    pub fn from_matrix(matrix: &CsrMatrix) -> Result<Self> {
        if matrix.nrows() != matrix.ncols() {
            return Err(AsimuError::Linalg("Jacobi 预条件器需要方阵".to_string()));
        }
        let n = matrix.nrows();
        let mut inverse_diagonal = vec![0.0; n];
        for (row, inv) in inverse_diagonal.iter_mut().enumerate() {
            let Some(diag) = matrix
                .row_entries(row)
                .find_map(|(col, value)| (col == row).then_some(value))
            else {
                return Err(AsimuError::Linalg(format!(
                    "Jacobi 预条件器缺少对角元: row={row}"
                )));
            };
            if diag.abs() <= Real::EPSILON {
                return Err(AsimuError::Linalg(format!(
                    "Jacobi 预条件器零对角元: row={row}"
                )));
            }
            *inv = 1.0 / diag;
        }
        Ok(Self { inverse_diagonal })
    }
}

impl Preconditioner for CsrJacobiPreconditioner {
    fn dimension(&self) -> usize {
        self.inverse_diagonal.len()
    }

    fn apply(&mut self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        ensure_vector_len(rhs, self.dimension(), "csr jacobi rhs")?;
        ensure_vector_len(out, self.dimension(), "csr jacobi out")?;
        for ((dst, src), inv) in out
            .iter_mut()
            .zip(rhs.iter())
            .zip(self.inverse_diagonal.iter())
        {
            *dst = inv * src;
        }
        Ok(())
    }
}

impl Preconditioner for CellBlockDiagonalPreconditioner {
    fn dimension(&self) -> usize {
        self.num_blocks() * self.block_size
    }

    fn apply(&mut self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        ensure_vector_len(rhs, self.dimension(), "cell block preconditioner rhs")?;
        ensure_vector_len(out, self.dimension(), "cell block preconditioner out")?;
        for block in 0..self.num_blocks() {
            let vec_offset = block * self.block_size;
            let mat_offset = block * self.block_size * self.block_size;
            for row in 0..self.block_size {
                let mut value = 0.0;
                for col in 0..self.block_size {
                    value += self.inverse_blocks[mat_offset + row * self.block_size + col]
                        * rhs[vec_offset + col];
                }
                out[vec_offset + row] = value;
            }
        }
        Ok(())
    }
}

fn invert_dense_block(block_size: usize, block: &[Real]) -> Result<Vec<Real>> {
    let width = block_size * 2;
    let mut aug = vec![0.0; block_size * width];
    for row in 0..block_size {
        for col in 0..block_size {
            aug[row * width + col] = block[row * block_size + col];
        }
        aug[row * width + block_size + row] = 1.0;
    }
    for pivot in 0..block_size {
        let (pivot_row, pivot_abs) = find_pivot_row(&aug, block_size, width, pivot);
        if pivot_abs <= Real::EPSILON {
            return Err(AsimuError::Linalg(
                "块对角预条件器遇到奇异局部块".to_string(),
            ));
        }
        swap_augmented_rows(&mut aug, width, pivot, pivot_row);
        normalize_pivot_row(&mut aug, width, pivot);
        eliminate_pivot_column(&mut aug, block_size, width, pivot);
    }
    let mut inverse = vec![0.0; block_size * block_size];
    for row in 0..block_size {
        for col in 0..block_size {
            inverse[row * block_size + col] = aug[row * width + block_size + col];
        }
    }
    Ok(inverse)
}

fn find_pivot_row(aug: &[Real], block_size: usize, width: usize, pivot: usize) -> (usize, Real) {
    let mut pivot_row = pivot;
    let mut pivot_abs = aug[pivot * width + pivot].abs();
    for row in pivot + 1..block_size {
        let candidate = aug[row * width + pivot].abs();
        if candidate > pivot_abs {
            pivot_abs = candidate;
            pivot_row = row;
        }
    }
    (pivot_row, pivot_abs)
}

fn swap_augmented_rows(aug: &mut [Real], width: usize, lhs: usize, rhs: usize) {
    if lhs == rhs {
        return;
    }
    for col in 0..width {
        aug.swap(lhs * width + col, rhs * width + col);
    }
}

fn normalize_pivot_row(aug: &mut [Real], width: usize, pivot: usize) {
    let denom = aug[pivot * width + pivot];
    for col in 0..width {
        aug[pivot * width + col] /= denom;
    }
}

fn eliminate_pivot_column(aug: &mut [Real], block_size: usize, width: usize, pivot: usize) {
    for row in 0..block_size {
        if row == pivot {
            continue;
        }
        let factor = aug[row * width + pivot];
        if factor.abs() <= Real::EPSILON {
            continue;
        }
        for col in 0..width {
            aug[row * width + col] -= factor * aug[pivot * width + col];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lusgs_diagonal_preconditioner_repeats_cell_scales() {
        let mut p = LusgsDiagonalPreconditioner::from_lusgs_diagonal(&[0.5], &[3.0], 0.5, 5)
            .expect("precond");
        let mut out = [0.0; 5];
        p.apply(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut out)
            .expect("apply");
        let scale = 0.5 * 0.5 / (1.0 + 0.5 * 3.0);
        assert!((out[4] - 5.0 * scale).abs() < 1.0e-12);
    }

    #[test]
    fn cell_block_preconditioner_solves_local_blocks() {
        let mut p = CellBlockDiagonalPreconditioner::from_blocks(
            2,
            vec![2.0, 0.0, 0.0, 4.0, 1.0, 1.0, 0.0, 2.0],
        )
        .expect("precond");
        let mut out = [0.0; 4];
        p.apply(&[2.0, 8.0, 3.0, 4.0], &mut out).expect("apply");
        assert!((out[0] - 1.0).abs() < 1.0e-12);
        assert!((out[1] - 2.0).abs() < 1.0e-12);
        assert!((out[2] - 1.0).abs() < 1.0e-12);
        assert!((out[3] - 2.0).abs() < 1.0e-12);
    }
}
