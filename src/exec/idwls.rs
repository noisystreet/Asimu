//! IDWLS RHS 并行调度（经 [`ExecutionContext`](super::context::ExecutionContext)）。

use crate::core::Vector3;
use crate::error::Result;

use super::context::ExecutionContext;
use super::device::{ExecCpuPolicy, ExecDevice};
use super::scratch::IdwlsRhsBuffer;

impl ExecutionContext {
    /// 清零粘性 IDWLS RHS 槽。
    pub fn idwls_prepare_viscous(&mut self, num_cells: usize) {
        self.scratch_mut().idwls_mut().prepare_viscous(num_cells);
    }

    /// 清零无粘二阶 IDWLS RHS 槽。
    pub fn idwls_prepare_inviscid(&mut self, num_cells: usize) {
        self.scratch_mut().idwls_mut().prepare_inviscid(num_cells);
    }

    /// 清零粘性 IDWLS RHS 槽（f32）。
    pub fn idwls_prepare_viscous_f32(&mut self, num_cells: usize) {
        self.scratch_mut()
            .idwls_mut()
            .prepare_viscous_f32(num_cells);
    }

    /// 清零无粘二阶 IDWLS RHS 槽（f32）。
    pub fn idwls_prepare_inviscid_f32(&mut self, num_cells: usize) {
        self.scratch_mut()
            .idwls_mut()
            .prepare_inviscid_f32(num_cells);
    }

    #[must_use]
    pub fn idwls_rhs(&self) -> &IdwlsRhsBuffer {
        self.scratch().idwls()
    }

    #[must_use]
    pub fn idwls_rhs_f32(&self) -> &IdwlsRhsBuffer {
        self.scratch().idwls()
    }

    /// 单元并行累加粘性 IDWLS RHS（`CpuParallel` → rayon；否则调用方走面串行路径）。
    pub fn idwls_accumulate_viscous_cells<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(usize, &mut Vector3, &mut Vector3, &mut Vector3, &mut Vector3) -> Result<()> + Sync,
    {
        if !self.uses_parallel_cell_loops() {
            return Ok(());
        }
        let idwls = self.scratch_mut().idwls_mut();
        let (bu, bv, bw, bt) = idwls.viscous_arrays_mut();
        crate::exec::parallel::par_try_for_each_cell_rhs4(bu, bv, bw, bt, f)
    }

    /// 单元并行累加无粘二阶 IDWLS RHS。
    pub fn idwls_accumulate_inviscid_cells<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(
                usize,
                &mut Vector3,
                &mut Vector3,
                &mut Vector3,
                &mut Vector3,
                &mut Vector3,
            ) -> Result<()>
            + Sync,
    {
        if !self.uses_parallel_cell_loops() {
            return Ok(());
        }
        let idwls = self.scratch_mut().idwls_mut();
        let (br, bp, bu, bv, bw) = idwls.inviscid_arrays_mut();
        crate::exec::parallel::par_try_for_each_cell_rhs5(br, bp, bu, bv, bw, f)
    }

    #[must_use]
    pub fn uses_parallel_cell_loops(&self) -> bool {
        cfg!(feature = "parallel-fvm")
            && self.device() == ExecDevice::Cpu
            && self.cpu_policy() == ExecCpuPolicy::Parallel
    }
}
