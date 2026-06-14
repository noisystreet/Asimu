//! CSR SpMV（经 [`ExecutionContext`](super::context::ExecutionContext) 调度）。

use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::context::ExecutionContext;
#[cfg(feature = "cuda")]
use super::device::ExecDevice;

/// CSR 矩阵只读视图（exec 不依赖 `linalg::CsrMatrix`）。
pub struct CsrSpmvView<'a> {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: &'a [usize],
    pub col_idx: &'a [usize],
    pub values: &'a [Real],
}

impl ExecutionContext {
    /// \(y \leftarrow A x\)（CUDA：`cusparse`；CPU：`CpuParallel` 行并行或串行）。
    pub fn csr_spmv(&mut self, matrix: &CsrSpmvView<'_>, x: &[Real], y: &mut [Real]) -> Result<()> {
        validate_csr_dims(matrix, x, y)?;
        #[cfg(feature = "cuda")]
        {
            if self.device() == ExecDevice::GpuCuda {
                return self.dispatch_cuda_csr_spmv(matrix, x, y);
            }
        }
        if self.uses_parallel_cell_loops() {
            csr_spmv_parallel(matrix, x, y);
        } else {
            csr_spmv_serial(matrix, x, y);
        }
        Ok(())
    }
}

fn validate_csr_dims(matrix: &CsrSpmvView<'_>, x: &[Real], y: &mut [Real]) -> Result<()> {
    if x.len() != matrix.ncols {
        return Err(AsimuError::Linalg(format!(
            "csr input 长度 {} 与列数 {} 不一致",
            x.len(),
            matrix.ncols
        )));
    }
    if y.len() != matrix.nrows {
        return Err(AsimuError::Linalg(format!(
            "csr output 长度 {} 与行数 {} 不一致",
            y.len(),
            matrix.nrows
        )));
    }
    Ok(())
}

fn csr_spmv_serial(matrix: &CsrSpmvView<'_>, x: &[Real], y: &mut [Real]) {
    for (row, dst) in y.iter_mut().enumerate().take(matrix.nrows) {
        *dst = csr_row_dot(matrix, row, x);
    }
}

#[cfg(feature = "parallel-fvm")]
fn csr_spmv_parallel(matrix: &CsrSpmvView<'_>, x: &[Real], y: &mut [Real]) {
    use rayon::prelude::*;

    y.par_iter_mut()
        .enumerate()
        .take(matrix.nrows)
        .for_each(|(row, dst)| {
            *dst = csr_row_dot(matrix, row, x);
        });
}

#[cfg(not(feature = "parallel-fvm"))]
fn csr_spmv_parallel(matrix: &CsrSpmvView<'_>, x: &[Real], y: &mut [Real]) {
    csr_spmv_serial(matrix, x, y);
}

fn csr_row_dot(matrix: &CsrSpmvView<'_>, row: usize, x: &[Real]) -> Real {
    let start = matrix.row_ptr[row];
    let end = matrix.row_ptr[row + 1];
    matrix.col_idx[start..end]
        .iter()
        .zip(matrix.values[start..end].iter())
        .map(|(&col, &value)| value * x[col])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{ExecConfig, ExecutionContext, MeshExecMetrics};

    fn sample_matrix() -> (Vec<usize>, Vec<usize>, Vec<Real>) {
        let row_ptr = vec![0, 2, 4, 5];
        let col_idx = vec![0, 1, 0, 2, 2];
        let values = vec![2.0, -1.0, -1.0, 2.0, 2.0];
        (row_ptr, col_idx, values)
    }

    #[test]
    fn csr_spmv_matches_dense_reference() {
        let (row_ptr, col_idx, values) = sample_matrix();
        let matrix = CsrSpmvView {
            nrows: 3,
            ncols: 3,
            row_ptr: &row_ptr,
            col_idx: &col_idx,
            values: &values,
        };
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0; 3];
        let mut ctx = ExecutionContext::new(ExecConfig::default(), MeshExecMetrics::new(3, 0, 0))
            .expect("ctx");
        ctx.csr_spmv(&matrix, &x, &mut y).expect("spmv");
        assert!((y[0] - 0.0).abs() < 1.0e-12);
        assert!((y[1] - 5.0).abs() < 1.0e-12);
        assert!((y[2] - 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn serial_and_parallel_spmv_match() {
        if !cfg!(feature = "parallel-fvm") {
            return;
        }
        let (row_ptr, col_idx, values) = sample_matrix();
        let matrix = CsrSpmvView {
            nrows: 3,
            ncols: 3,
            row_ptr: &row_ptr,
            col_idx: &col_idx,
            values: &values,
        };
        let x = [1.0, 0.5, 2.0];
        let mut y_serial = [0.0; 3];
        let mut y_parallel = [0.0; 3];
        let mut serial = ExecutionContext::new(
            ExecConfig::for_test_backend(crate::exec::ExecBackend::CpuScalar),
            MeshExecMetrics::new(3, 0, 0),
        )
        .expect("serial ctx");
        let mut parallel =
            ExecutionContext::new(ExecConfig::default(), MeshExecMetrics::new(3, 0, 0))
                .expect("parallel ctx");
        serial.csr_spmv(&matrix, &x, &mut y_serial).expect("serial");
        parallel
            .csr_spmv(&matrix, &x, &mut y_parallel)
            .expect("parallel");
        for (a, b) in y_serial.iter().zip(y_parallel.iter()) {
            assert!((a - b).abs() < 1.0e-12);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "gpu"]
    fn cpu_csr_spmv_matches_cuda_csr_spmv() {
        use crate::core::ExecDevice;
        use crate::core::approx_eq;

        let (row_ptr, col_idx, values) = sample_matrix();
        let matrix = CsrSpmvView {
            nrows: 3,
            ncols: 3,
            row_ptr: &row_ptr,
            col_idx: &col_idx,
            values: &values,
        };
        let x = [1.0, 0.5, 2.0];
        let mut y_cpu = [0.0; 3];
        let mut y_cuda = [0.0; 3];
        let mut cpu_ctx =
            ExecutionContext::new(ExecConfig::default(), MeshExecMetrics::new(3, 0, 0))
                .expect("cpu ctx");
        let cuda_config = ExecConfig {
            device: ExecDevice::GpuCuda,
            ..Default::default()
        };
        let mut cuda_ctx =
            ExecutionContext::new(cuda_config, MeshExecMetrics::new(3, 0, 0)).expect("cuda ctx");
        cpu_ctx.csr_spmv(&matrix, &x, &mut y_cpu).expect("cpu");
        cuda_ctx.csr_spmv(&matrix, &x, &mut y_cuda).expect("cuda");
        for i in 0..3 {
            assert!(
                approx_eq(y_cpu[i], y_cuda[i], 1.0e-10),
                "spmv mismatch at {i}: cpu={} cuda={}",
                y_cpu[i],
                y_cuda[i]
            );
        }
    }
}
