//! 守恒变量场（可压缩 NS）。

use crate::error::{AsimuError, Result};
use crate::physics::{ConservedState, FreestreamParams, IdealGasEoS, PrimitiveState};

use super::ScalarField;

/// 单元守恒变量集合（SoA）。
#[derive(Debug, Clone, PartialEq)]
pub struct ConservedFields {
    pub density: ScalarField,
    pub momentum_x: ScalarField,
    pub momentum_y: ScalarField,
    pub momentum_z: ScalarField,
    pub total_energy: ScalarField,
}

impl ConservedFields {
    pub fn uniform(num_cells: usize, state: ConservedState) -> Result<Self> {
        Ok(Self {
            density: ScalarField::uniform(num_cells, state.density)?,
            momentum_x: ScalarField::uniform(num_cells, state.momentum[0])?,
            momentum_y: ScalarField::uniform(num_cells, state.momentum[1])?,
            momentum_z: ScalarField::uniform(num_cells, state.momentum[2])?,
            total_energy: ScalarField::uniform(num_cells, state.total_energy)?,
        })
    }

    pub fn from_freestream(
        num_cells: usize,
        eos: &IdealGasEoS,
        params: &FreestreamParams,
    ) -> Result<Self> {
        let prim = eos.freestream_primitive(
            params.mach,
            params.pressure,
            params.temperature,
            params.effective_direction(),
        )?;
        let state = ConservedState::from_primitive(eos, &prim)?;
        Self::uniform(num_cells, state)
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.density.len()
    }

    pub fn cell_state(&self, index: usize) -> Result<ConservedState> {
        Ok(ConservedState {
            density: self.density.values()[index],
            momentum: [
                self.momentum_x.values()[index],
                self.momentum_y.values()[index],
                self.momentum_z.values()[index],
            ],
            total_energy: self.total_energy.values()[index],
        })
    }

    pub fn primitive_at(&self, index: usize, eos: &IdealGasEoS) -> Result<PrimitiveState> {
        primitive_from_conserved(eos, &self.cell_state(index)?)
    }
}

/// 守恒变量 → 原始变量（理想气体）。
pub fn primitive_from_conserved(eos: &IdealGasEoS, cons: &ConservedState) -> Result<PrimitiveState> {
    let rho = cons.density;
    if rho <= 0.0 {
        return Err(AsimuError::Field("密度必须大于 0".to_string()));
    }
    let velocity = [
        cons.momentum[0] / rho,
        cons.momentum[1] / rho,
        cons.momentum[2] / rho,
    ];
    let ke = 0.5 * rho * (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]);
    let e = cons.total_energy / rho - ke / rho;
    if e <= 0.0 {
        return Err(AsimuError::Field("内能必须大于 0".to_string()));
    }
    let pressure = (eos.gamma - 1.0) * rho * e;
    let temperature = pressure / (rho * eos.gas_constant);
    Ok(PrimitiveState {
        density: rho,
        velocity,
        pressure,
        temperature,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freestream_uniform_field_has_correct_density() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let params = FreestreamParams {
            mach: 0.3,
            pressure: 101_325.0,
            temperature: 288.15,
            ..FreestreamParams::default()
        };
        let fields = ConservedFields::from_freestream(16, &eos, &params).expect("fields");
        assert_eq!(fields.num_cells(), 16);
        let prim = fields.primitive_at(0, &eos).expect("prim");
        assert!((prim.density - fields.density.values()[0]).abs() < 1.0e-10);
    }
}
