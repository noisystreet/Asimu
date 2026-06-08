//! IDWLS 正规方程 RHS 缓冲（ADR 0013 E3；与 flux scratch 共用 `ExecScratch`）。

use crate::core::Vector3;

/// 粘性 IDWLS 四分量 RHS slice 元组。
pub type ViscousIdwlsArraysMut<'a> = (
    &'a mut [Vector3],
    &'a mut [Vector3],
    &'a mut [Vector3],
    &'a mut [Vector3],
);

/// 无粘二阶 IDWLS 五分量 RHS slice 元组。
pub type InviscidIdwlsArraysMut<'a> = (
    &'a mut [Vector3],
    &'a mut [Vector3],
    &'a mut [Vector3],
    &'a mut [Vector3],
    &'a mut [Vector3],
);

/// 每单元 IDWLS RHS 向量 \(b_u,b_v,b_w,b_T,b_\rho,b_p\)（步间复用）。
#[derive(Debug, Default)]
pub struct IdwlsRhsBuffer {
    bu: Vec<Vector3>,
    bv: Vec<Vector3>,
    bw: Vec<Vector3>,
    bt: Vec<Vector3>,
    br: Vec<Vector3>,
    bp: Vec<Vector3>,
}

impl IdwlsRhsBuffer {
    #[must_use]
    pub fn with_capacity(num_cells: usize) -> Self {
        Self {
            bu: Vec::with_capacity(num_cells),
            bv: Vec::with_capacity(num_cells),
            bw: Vec::with_capacity(num_cells),
            bt: Vec::with_capacity(num_cells),
            br: Vec::with_capacity(num_cells),
            bp: Vec::with_capacity(num_cells),
        }
    }

    /// 粘性路径：清零 \(b_u,b_v,b_w,b_T\)。
    pub fn prepare_viscous(&mut self, num_cells: usize) {
        let zero = Vector3::new(0.0, 0.0, 0.0);
        self.bu.resize(num_cells, zero);
        self.bv.resize(num_cells, zero);
        self.bw.resize(num_cells, zero);
        self.bt.resize(num_cells, zero);
        for i in 0..num_cells {
            self.bu[i] = zero;
            self.bv[i] = zero;
            self.bw[i] = zero;
            self.bt[i] = zero;
        }
    }

    /// 无粘二阶线性重构：清零 \(b_\rho,b_p,b_u,b_v,b_w\)。
    pub fn prepare_inviscid(&mut self, num_cells: usize) {
        let zero = Vector3::new(0.0, 0.0, 0.0);
        self.bu.resize(num_cells, zero);
        self.bv.resize(num_cells, zero);
        self.bw.resize(num_cells, zero);
        self.br.resize(num_cells, zero);
        self.bp.resize(num_cells, zero);
        for i in 0..num_cells {
            self.bu[i] = zero;
            self.bv[i] = zero;
            self.bw[i] = zero;
            self.br[i] = zero;
            self.bp[i] = zero;
        }
    }

    #[must_use]
    pub fn bu(&self) -> &[Vector3] {
        &self.bu
    }

    #[must_use]
    pub fn bu_mut(&mut self) -> &mut [Vector3] {
        &mut self.bu
    }

    #[must_use]
    pub fn bv(&self) -> &[Vector3] {
        &self.bv
    }

    #[must_use]
    pub fn bv_mut(&mut self) -> &mut [Vector3] {
        &mut self.bv
    }

    #[must_use]
    pub fn bw(&self) -> &[Vector3] {
        &self.bw
    }

    #[must_use]
    pub fn bw_mut(&mut self) -> &mut [Vector3] {
        &mut self.bw
    }

    #[must_use]
    pub fn bt(&self) -> &[Vector3] {
        &self.bt
    }

    #[must_use]
    pub fn bt_mut(&mut self) -> &mut [Vector3] {
        &mut self.bt
    }

    #[must_use]
    pub fn br(&self) -> &[Vector3] {
        &self.br
    }

    #[must_use]
    pub fn br_mut(&mut self) -> &mut [Vector3] {
        &mut self.br
    }

    #[must_use]
    pub fn bp(&self) -> &[Vector3] {
        &self.bp
    }

    #[must_use]
    pub fn bp_mut(&mut self) -> &mut [Vector3] {
        &mut self.bp
    }

    /// 粘性路径可变 slice 元组（避免多字段重叠借用）。
    pub fn viscous_arrays_mut(&mut self) -> ViscousIdwlsArraysMut<'_> {
        (&mut self.bu, &mut self.bv, &mut self.bw, &mut self.bt)
    }

    /// 无粘二阶路径可变 slice 元组。
    pub fn inviscid_arrays_mut(&mut self) -> InviscidIdwlsArraysMut<'_> {
        (
            &mut self.br,
            &mut self.bp,
            &mut self.bu,
            &mut self.bv,
            &mut self.bw,
        )
    }
}
