use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::{LinearOperator, Preconditioner, ensure_vector_len};

/// 压缩行存储矩阵。
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    nrows: usize,
    ncols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<Real>,
}

impl CsrMatrix {
    pub fn from_rows(nrows: usize, ncols: usize, rows: Vec<Vec<(usize, Real)>>) -> Result<Self> {
        if nrows == 0 || ncols == 0 || rows.len() != nrows {
            return Err(AsimuError::Linalg("CSR 矩阵尺寸无效".to_string()));
        }
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for mut row in rows {
            row.sort_by_key(|(col, _)| *col);
            let mut last_col = None;
            for (col, value) in row {
                if col >= ncols {
                    return Err(AsimuError::Linalg(format!("CSR 列索引越界: {col}")));
                }
                if last_col == Some(col) {
                    let last = values
                        .last_mut()
                        .expect("last value exists after duplicate column");
                    *last += value;
                } else if value.abs() > Real::EPSILON {
                    col_idx.push(col);
                    values.push(value);
                    last_col = Some(col);
                }
            }
            row_ptr.push(col_idx.len());
        }
        Ok(Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        })
    }

    #[must_use]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[must_use]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    pub fn values(&self) -> &[Real] {
        &self.values
    }

    /// CSR 只读视图（供 [`crate::exec::CsrSpmvView`]）。
    #[must_use]
    pub fn spmv_view(&self) -> crate::exec::CsrSpmvView<'_> {
        crate::exec::CsrSpmvView {
            nrows: self.nrows,
            ncols: self.ncols,
            row_ptr: &self.row_ptr,
            col_idx: &self.col_idx,
            values: &self.values,
        }
    }

    /// \(y \leftarrow A x\)，经 [`ExecutionContext`](crate::exec::ExecutionContext) 调度（并行/串行）。
    pub fn apply_with_context(
        &self,
        ctx: &mut crate::exec::ExecutionContext,
        x: &[Real],
        y: &mut [Real],
    ) -> Result<()> {
        ctx.csr_spmv(&self.spmv_view(), x, y)
    }

    pub(crate) fn row_entries(&self, row: usize) -> impl Iterator<Item = (usize, Real)> + '_ {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        self.col_idx[start..end]
            .iter()
            .copied()
            .zip(self.values[start..end].iter().copied())
    }
}

impl LinearOperator for CsrMatrix {
    fn dimension(&self) -> usize {
        self.ncols
    }

    fn apply(&mut self, x: &[Real], y: &mut [Real]) -> Result<()> {
        ensure_vector_len(x, self.ncols, "csr input")?;
        ensure_vector_len(y, self.nrows, "csr output")?;
        for (row, dst) in y.iter_mut().enumerate().take(self.nrows) {
            *dst = self
                .row_entries(row)
                .map(|(col, value)| value * x[col])
                .sum();
        }
        Ok(())
    }
}

/// CSR 矩阵只读视图，避免 Krylov 求解路径 clone 系数矩阵。
#[derive(Debug, Clone, Copy)]
pub struct CsrMatrixView<'a> {
    matrix: &'a CsrMatrix,
}

impl<'a> CsrMatrixView<'a> {
    #[must_use]
    pub fn new(matrix: &'a CsrMatrix) -> Self {
        Self { matrix }
    }
}

impl LinearOperator for CsrMatrixView<'_> {
    fn dimension(&self) -> usize {
        self.matrix.ncols()
    }

    fn apply(&mut self, x: &[Real], y: &mut [Real]) -> Result<()> {
        ensure_vector_len(x, self.matrix.ncols(), "csr view input")?;
        ensure_vector_len(y, self.matrix.nrows(), "csr view output")?;
        for (row, dst) in y.iter_mut().enumerate().take(self.matrix.nrows()) {
            *dst = self
                .matrix
                .row_entries(row)
                .map(|(col, value)| value * x[col])
                .sum();
        }
        Ok(())
    }
}

/// ILU(0) 预条件器（与 CSR 非零结构一致）。
#[derive(Debug, Clone, PartialEq)]
pub struct Ilu0Preconditioner {
    rows: Vec<Vec<(usize, Real)>>,
    diag: Vec<Real>,
}

impl Ilu0Preconditioner {
    pub fn factor(matrix: &CsrMatrix) -> Result<Self> {
        if matrix.nrows != matrix.ncols {
            return Err(AsimuError::Linalg("ILU(0) 需要方阵".to_string()));
        }
        let n = matrix.nrows;
        let mut rows = (0..n)
            .map(|i| matrix.row_entries(i).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut diag: Vec<Real> = vec![0.0; n];
        for i in 0..n {
            let lower_cols = rows[i]
                .iter()
                .map(|(col, _)| *col)
                .filter(|col| *col < i)
                .collect::<Vec<_>>();
            for k in lower_cols {
                if diag[k].abs() <= Real::EPSILON {
                    return Err(AsimuError::Linalg(format!("ILU(0) 零主元: row={k}")));
                }
                let factor = row_value(&rows[i], k).unwrap_or(0.0) / diag[k];
                set_row_value(&mut rows[i], k, factor)?;
                for (j, ukj) in rows[k].clone() {
                    if j > k
                        && let Some(aij) = row_value_mut(&mut rows[i], j)
                    {
                        *aij -= factor * ukj;
                    }
                }
            }
            diag[i] = row_value(&rows[i], i)
                .ok_or_else(|| AsimuError::Linalg(format!("ILU(0) 缺少对角元: row={i}")))?;
            if diag[i].abs() <= Real::EPSILON {
                return Err(AsimuError::Linalg(format!("ILU(0) 零主元: row={i}")));
            }
        }
        Ok(Self { rows, diag })
    }
}

impl Preconditioner for Ilu0Preconditioner {
    fn dimension(&self) -> usize {
        self.rows.len()
    }

    fn apply(&mut self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        let n = self.dimension();
        ensure_vector_len(rhs, n, "ilu rhs")?;
        ensure_vector_len(out, n, "ilu out")?;
        let mut y = rhs.to_vec();
        for i in 0..n {
            for &(col, value) in &self.rows[i] {
                if col < i {
                    y[i] -= value * y[col];
                }
            }
        }
        out.copy_from_slice(&y);
        for i in (0..n).rev() {
            for &(col, value) in &self.rows[i] {
                if col > i {
                    out[i] -= value * out[col];
                }
            }
            out[i] /= self.diag[i];
        }
        Ok(())
    }
}

fn row_value(row: &[(usize, Real)], col: usize) -> Option<Real> {
    row.iter()
        .find_map(|(c, value)| (*c == col).then_some(*value))
}

fn row_value_mut(row: &mut [(usize, Real)], col: usize) -> Option<&mut Real> {
    row.iter_mut()
        .find_map(|(c, value)| (*c == col).then_some(value))
}

fn set_row_value(row: &mut [(usize, Real)], col: usize, new_value: Real) -> Result<()> {
    let value = row_value_mut(row, col)
        .ok_or_else(|| AsimuError::Linalg(format!("ILU(0) 内部结构缺少列 {col}")))?;
    *value = new_value;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ilu0_preconditioner_solves_triangular_factor_system() {
        let matrix = CsrMatrix::from_rows(
            3,
            3,
            vec![
                vec![(0, 2.0), (1, -1.0)],
                vec![(0, -1.0), (1, 2.0), (2, -1.0)],
                vec![(1, -1.0), (2, 2.0)],
            ],
        )
        .expect("csr");
        let mut ilu = Ilu0Preconditioner::factor(&matrix).expect("ilu");
        let mut z = [0.0; 3];
        ilu.apply(&[1.0, 0.0, 1.0], &mut z).expect("apply");
        assert!((z[0] - 1.0).abs() < 1.0e-10);
        assert!((z[1] - 1.0).abs() < 1.0e-10);
        assert!((z[2] - 1.0).abs() < 1.0e-10);
    }
}
