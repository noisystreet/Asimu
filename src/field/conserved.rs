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

    /// 保证 \(\rho>0\) 且 \(E>\mathrm{KE}+p_\mathrm{floor}/(\gamma-1)\)（显式 RK 步后调用）。
    pub fn enforce_positivity(&mut self, eos: &IdealGasEoS, min_pressure: crate::core::Real) {
        let gamma = eos.gamma;
        let min_internal = min_pressure / (gamma - 1.0);
        let rho_min = 1.0e-12;
        for i in 0..self.num_cells() {
            let rho_old = self.density.values()[i];
            let rho = if rho_old.is_finite() && rho_old > 0.0 {
                rho_old.max(rho_min)
            } else {
                rho_min
            };
            if rho_old.is_finite() && rho_old > 0.0 && rho_old < rho_min {
                let scale = rho / rho_old;
                self.momentum_x.values_mut()[i] *= scale;
                self.momentum_y.values_mut()[i] *= scale;
                self.momentum_z.values_mut()[i] *= scale;
            } else if !(rho_old.is_finite() && rho_old > 0.0) {
                self.momentum_x.values_mut()[i] = 0.0;
                self.momentum_y.values_mut()[i] = 0.0;
                self.momentum_z.values_mut()[i] = 0.0;
            }
            self.density.values_mut()[i] = rho;
            let mx = self.momentum_x.values()[i];
            let my = self.momentum_y.values()[i];
            let mz = self.momentum_z.values()[i];
            let ke = 0.5 * (mx * mx + my * my + mz * mz) / rho;
            let e_min = ke + min_internal;
            let energy = self.total_energy.values_mut();
            if !energy[i].is_finite() || energy[i] < e_min {
                energy[i] = e_min;
            }
        }
    }
}

/// 守恒变量 → 原始变量（理想气体）。
pub fn primitive_from_conserved(
    eos: &IdealGasEoS,
    cons: &ConservedState,
) -> Result<PrimitiveState> {
    let rho = cons.density;
    if rho <= 0.0 {
        return Err(AsimuError::Field("密度必须大于 0".to_string()));
    }
    let velocity = [
        cons.momentum[0] / rho,
        cons.momentum[1] / rho,
        cons.momentum[2] / rho,
    ];
    let ke = 0.5
        * rho
        * (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]);
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

/// 通量/边界装配用的宽松 primitive 恢复（RK 中间态对压力做下限）。
pub fn primitive_from_conserved_relaxed(
    eos: &IdealGasEoS,
    cons: &ConservedState,
    min_pressure: crate::core::Real,
) -> Result<PrimitiveState> {
    let rho = cons.density;
    if rho <= 0.0 {
        return Err(AsimuError::Field("密度必须大于 0".to_string()));
    }
    let velocity = [
        cons.momentum[0] / rho,
        cons.momentum[1] / rho,
        cons.momentum[2] / rho,
    ];
    let ke = 0.5
        * rho
        * (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]);
    let pressure = ((eos.gamma - 1.0) * (cons.total_energy - ke)).max(min_pressure);
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
