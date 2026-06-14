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

/// 无粘二阶 IDWLS 五分量 RHS slice 元组（f32）。
pub type InviscidIdwlsArraysMutF32<'a> = (
    &'a mut [[f32; 3]],
    &'a mut [[f32; 3]],
    &'a mut [[f32; 3]],
    &'a mut [[f32; 3]],
    &'a mut [[f32; 3]],
);

/// 粘性 IDWLS 四分量 RHS slice 元组（f32）。
pub type ViscousIdwlsArraysMutF32<'a> = (
    &'a mut [[f32; 3]],
    &'a mut [[f32; 3]],
    &'a mut [[f32; 3]],
    &'a mut [[f32; 3]],
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
    bu_f32: Vec<[f32; 3]>,
    bv_f32: Vec<[f32; 3]>,
    bw_f32: Vec<[f32; 3]>,
    bt_f32: Vec<[f32; 3]>,
    br_f32: Vec<[f32; 3]>,
    bp_f32: Vec<[f32; 3]>,
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
            bu_f32: Vec::with_capacity(num_cells),
            bv_f32: Vec::with_capacity(num_cells),
            bw_f32: Vec::with_capacity(num_cells),
            bt_f32: Vec::with_capacity(num_cells),
            br_f32: Vec::with_capacity(num_cells),
            bp_f32: Vec::with_capacity(num_cells),
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

    /// 粘性路径 f32：清零 \(b_u,b_v,b_w,b_T\)。
    pub fn prepare_viscous_f32(&mut self, num_cells: usize) {
        let zero = [0.0f32; 3];
        self.bu_f32.resize(num_cells, zero);
        self.bv_f32.resize(num_cells, zero);
        self.bw_f32.resize(num_cells, zero);
        self.bt_f32.resize(num_cells, zero);
        for i in 0..num_cells {
            self.bu_f32[i] = zero;
            self.bv_f32[i] = zero;
            self.bw_f32[i] = zero;
            self.bt_f32[i] = zero;
        }
    }

    /// 无粘二阶 f32：清零 \(b_\rho,b_p,b_u,b_v,b_w\)。
    pub fn prepare_inviscid_f32(&mut self, num_cells: usize) {
        let zero = [0.0f32; 3];
        self.bu_f32.resize(num_cells, zero);
        self.bv_f32.resize(num_cells, zero);
        self.bw_f32.resize(num_cells, zero);
        self.br_f32.resize(num_cells, zero);
        self.bp_f32.resize(num_cells, zero);
        for i in 0..num_cells {
            self.bu_f32[i] = zero;
            self.bv_f32[i] = zero;
            self.bw_f32[i] = zero;
            self.br_f32[i] = zero;
            self.bp_f32[i] = zero;
        }
    }

    #[must_use]
    pub fn bu_f32(&self) -> &[[f32; 3]] {
        &self.bu_f32
    }

    #[must_use]
    pub fn bv_f32(&self) -> &[[f32; 3]] {
        &self.bv_f32
    }

    #[must_use]
    pub fn bw_f32(&self) -> &[[f32; 3]] {
        &self.bw_f32
    }

    #[must_use]
    pub fn bt_f32(&self) -> &[[f32; 3]] {
        &self.bt_f32
    }

    #[must_use]
    pub fn br_f32(&self) -> &[[f32; 3]] {
        &self.br_f32
    }

    #[must_use]
    pub fn bp_f32(&self) -> &[[f32; 3]] {
        &self.bp_f32
    }

    pub fn viscous_arrays_mut_f32(&mut self) -> ViscousIdwlsArraysMutF32<'_> {
        (
            &mut self.bu_f32,
            &mut self.bv_f32,
            &mut self.bw_f32,
            &mut self.bt_f32,
        )
    }

    pub fn inviscid_arrays_mut_f32(&mut self) -> InviscidIdwlsArraysMutF32<'_> {
        (
            &mut self.br_f32,
            &mut self.bp_f32,
            &mut self.bu_f32,
            &mut self.bv_f32,
            &mut self.bw_f32,
        )
    }
}
