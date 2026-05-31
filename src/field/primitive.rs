//! 单元原始变量场（SoA），供面通量装配复用，避免每面重复守恒→原始恢复。

use tracing::info_span;

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::physics::{IdealGasEoS, PrimitiveState};

use super::{ConservedFields, ScalarField, primitive_from_conserved_relaxed};

/// 与 [`ConservedFields`] 同长度的原始变量 SoA。
#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveFields {
    pub density: ScalarField,
    pub pressure: ScalarField,
    pub velocity_x: ScalarField,
    pub velocity_y: ScalarField,
    pub velocity_z: ScalarField,
}

impl PrimitiveFields {
    pub fn zeros(num_cells: usize) -> Result<Self> {
        Ok(Self {
            density: ScalarField::uniform(num_cells, 0.0)?,
            pressure: ScalarField::uniform(num_cells, 0.0)?,
            velocity_x: ScalarField::uniform(num_cells, 0.0)?,
            velocity_y: ScalarField::uniform(num_cells, 0.0)?,
            velocity_z: ScalarField::uniform(num_cells, 0.0)?,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.density.len()
    }

    /// 从守恒场批量恢复原始变量（RK 中间态用宽松压力下限）。
    pub fn fill_from_conserved(
        &mut self,
        fields: &ConservedFields,
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
            self.density.values_mut()[i] = prim.density;
            self.pressure.values_mut()[i] = prim.pressure;
            self.velocity_x.values_mut()[i] = prim.velocity[0];
            self.velocity_y.values_mut()[i] = prim.velocity[1];
            self.velocity_z.values_mut()[i] = prim.velocity[2];
        }
        Ok(())
    }

    #[must_use]
    pub fn cell_primitive(&self, index: usize) -> PrimitiveState {
        let rho = self.density.values()[index];
        let pressure = self.pressure.values()[index];
        PrimitiveState {
            density: rho,
            velocity: [
                self.velocity_x.values()[index],
                self.velocity_y.values()[index],
                self.velocity_z.values()[index],
            ],
            pressure,
            temperature: 0.0,
        }
    }
}
