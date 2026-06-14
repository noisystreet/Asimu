//! 单元原始变量场（SoA），供面通量装配复用，避免每面重复守恒→原始恢复。

use tracing::info_span;

use crate::core::{ComputeFloat, Real};
use crate::error::{AsimuError, Result};
use crate::physics::{IdealGasEoS, PrimitiveState};

use super::{ConservedFieldsT, ScalarFieldT, primitive_from_conserved_relaxed};

/// 与 [`ConservedFieldsT`] 同长度的原始变量 SoA。
#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveFieldsT<T: ComputeFloat> {
    pub density: ScalarFieldT<T>,
    pub pressure: ScalarFieldT<T>,
    pub velocity_x: ScalarFieldT<T>,
    pub velocity_y: ScalarFieldT<T>,
    pub velocity_z: ScalarFieldT<T>,
}

/// 默认工程标量原始变量场（`f64`）。
pub type PrimitiveFields = PrimitiveFieldsT<Real>;

impl<T: ComputeFloat> PrimitiveFieldsT<T> {
    pub fn zeros(num_cells: usize) -> Result<Self> {
        Ok(Self {
            density: ScalarFieldT::uniform(num_cells, T::zero())?,
            pressure: ScalarFieldT::uniform(num_cells, T::zero())?,
            velocity_x: ScalarFieldT::uniform(num_cells, T::zero())?,
            velocity_y: ScalarFieldT::uniform(num_cells, T::zero())?,
            velocity_z: ScalarFieldT::uniform(num_cells, T::zero())?,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.density.len()
    }

    /// 从守恒场批量恢复原始变量（RK 中间态用宽松压力下限）。
    pub fn fill_from_conserved(
        &mut self,
        fields: &ConservedFieldsT<T>,
        eos: &IdealGasEoS,
        min_pressure: Real,
    ) -> Result<()> {
        let n = fields.num_cells();
        if self.num_cells() != n {
            return Err(AsimuError::Field(format!(
                "PrimitiveFields 长度 {} 与守恒场 {n} 不一致",
                self.num_cells()
            )));
        }
        let _span = info_span!("fill_primitives", cells = n).entered();
        for i in 0..n {
            let cons = fields.cell_state(i)?;
            let prim = primitive_from_conserved_relaxed(eos, &cons, min_pressure)?;
            self.density.values_mut()[i] = T::from_real(prim.density);
            self.pressure.values_mut()[i] = T::from_real(prim.pressure);
            self.velocity_x.values_mut()[i] = T::from_real(prim.velocity[0]);
            self.velocity_y.values_mut()[i] = T::from_real(prim.velocity[1]);
            self.velocity_z.values_mut()[i] = T::from_real(prim.velocity[2]);
        }
        Ok(())
    }

    /// 转为 `Real` 原始变量场（谱半径等仍走 f64 路径时用）。
    pub fn cast_real(&self) -> Result<PrimitiveFields> {
        Ok(PrimitiveFields {
            density: ScalarFieldT::from_real_values(self.density.to_real_values())?,
            pressure: ScalarFieldT::from_real_values(self.pressure.to_real_values())?,
            velocity_x: ScalarFieldT::from_real_values(self.velocity_x.to_real_values())?,
            velocity_y: ScalarFieldT::from_real_values(self.velocity_y.to_real_values())?,
            velocity_z: ScalarFieldT::from_real_values(self.velocity_z.to_real_values())?,
        })
    }

    #[must_use]
    pub fn cell_primitive(&self, index: usize) -> PrimitiveState {
        let rho = self.density.values()[index].to_real();
        let pressure = self.pressure.values()[index].to_real();
        PrimitiveState {
            density: rho,
            velocity: [
                self.velocity_x.values()[index].to_real(),
                self.velocity_y.values()[index].to_real(),
                self.velocity_z.values()[index].to_real(),
            ],
            pressure,
            temperature: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::ConservedState;

    #[test]
    fn f32_primitive_cache_matches_conserved_state() {
        let state = ConservedState {
            density: 1.0,
            momentum: [0.2, 0.0, 0.0],
            total_energy: 2.5,
        };
        let fields = ConservedFieldsT::<f32>::uniform(1, state).expect("fields");
        let mut prim = PrimitiveFieldsT::<f32>::zeros(1).expect("prim");
        let eos = IdealGasEoS::AIR_STANDARD;
        prim.fill_from_conserved(&fields, &eos, 0.0).expect("fill");
        assert!((prim.density.values()[0].to_real() - 1.0).abs() < 1.0e-5);
    }
}
