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

    pub fn primitive_at(
        &self,
        index: usize,
        eos: &IdealGasEoS,
        min_pressure: crate::core::Real,
    ) -> Result<PrimitiveState> {
        primitive_from_conserved_relaxed(eos, &self.cell_state(index)?, min_pressure)
    }

    /// 保证 \(\rho>0\) 且 \(E>\mathrm{KE}+p_\mathrm{floor}/(\gamma-1)\)（显式 RK 步后调用）。
    pub fn enforce_positivity(&mut self, eos: &IdealGasEoS, min_pressure: crate::core::Real) {
        for i in 0..self.num_cells() {
            let mut state = ConservedState {
                density: self.density.values()[i],
                momentum: [
                    self.momentum_x.values()[i],
                    self.momentum_y.values()[i],
                    self.momentum_z.values()[i],
                ],
                total_energy: self.total_energy.values()[i],
            };
            clamp_conserved_positivity(&mut state, eos.gamma, min_pressure);
            self.write_cell_state(i, &state);
        }
    }

    fn write_cell_state(&mut self, index: usize, state: &ConservedState) {
        self.density.values_mut()[index] = state.density;
        self.momentum_x.values_mut()[index] = state.momentum[0];
        self.momentum_y.values_mut()[index] = state.momentum[1];
        self.momentum_z.values_mut()[index] = state.momentum[2];
        self.total_energy.values_mut()[index] = state.total_energy;
    }
}

/// 来流静压的 1%（下限 1e-6 Pa），与求解器 CFL/正性保持一致。
#[must_use]
pub fn positivity_pressure_floor(freestream_pressure: crate::core::Real) -> crate::core::Real {
    (freestream_pressure * 1.0e-2).max(1.0e-6)
}

/// 单单元守恒量正性钳制（RK 阶段态与边界 owner 共用）。
pub fn clamp_conserved_positivity(
    state: &mut ConservedState,
    gamma: crate::core::Real,
    min_pressure: crate::core::Real,
) {
    let min_internal = min_pressure / (gamma - 1.0);
    let rho_min = 1.0e-12;
    let rho_old = state.density;
    let rho = if rho_old.is_finite() && rho_old > 0.0 {
        rho_old.max(rho_min)
    } else {
        rho_min
    };
    if rho_old.is_finite() && rho_old > 0.0 && rho_old < rho_min {
        let scale = rho / rho_old;
        state.momentum[0] *= scale;
        state.momentum[1] *= scale;
        state.momentum[2] *= scale;
    } else if !(rho_old.is_finite() && rho_old > 0.0) {
        state.momentum = [0.0, 0.0, 0.0];
    }
    state.density = rho;
    let ke = 0.5
        * (state.momentum[0] * state.momentum[0]
            + state.momentum[1] * state.momentum[1]
            + state.momentum[2] * state.momentum[2])
        / rho;
    let e_min = ke + min_internal;
    if !state.total_energy.is_finite() || state.total_energy < e_min {
        state.total_energy = e_min;
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
    let internal = cons.total_energy - ke;
    if internal <= 0.0 {
        // RK 中间态或输出阶段：回退到压力下限，避免中断时间步进。
        return primitive_from_conserved_relaxed(eos, cons, 1.0e-6);
    }
    let pressure = (eos.gamma - 1.0) * internal;
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
    fn clamp_allows_strict_primitive_recovery() {
        let eos = IdealGasEoS::new(1.4, 287.0).expect("eos");
        let mut state = ConservedState {
            density: 1.0e-11,
            momentum: [1.0, 0.0, 0.0],
            total_energy: 0.5,
        };
        clamp_conserved_positivity(&mut state, eos.gamma, 10.0);
        primitive_from_conserved(&eos, &state).expect("strict primitive after clamp");
    }

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
        let prim = fields
            .primitive_at(0, &eos, positivity_pressure_floor(params.pressure))
            .expect("prim");
        assert!((prim.density - fields.density.values()[0]).abs() < 1.0e-10);
    }
}
