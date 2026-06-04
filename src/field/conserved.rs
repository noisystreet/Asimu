//! 守恒变量场（可压缩 NS）。

use crate::error::{AsimuError, Result};
use crate::physics::{
    ConservedState, FreestreamContext, FreestreamParams, IdealGasEoS, PrimitiveState,
};

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
        Self::from_freestream_context(num_cells, &FreestreamContext::dimensional(eos), params)
    }

    /// 经 [`FreestreamContext`](crate::physics::FreestreamContext) 构造均匀来流场（有量纲 / 无量纲统一入口）。
    ///
    /// 理论：[`docs/theory/nondimensional.md`](../../docs/theory/nondimensional.md) §2、§6。
    pub fn from_freestream_context(
        num_cells: usize,
        ctx: &FreestreamContext<'_>,
        params: &FreestreamParams,
    ) -> Result<Self> {
        Self::uniform(num_cells, ctx.conserved(params)?)
    }

    /// 无量纲来流（等价于 `from_freestream_context` + 无量纲 `FreestreamContext`）。
    pub fn from_nondimensional_freestream(
        num_cells: usize,
        eos: &IdealGasEoS,
        params: &FreestreamParams,
    ) -> Result<Self> {
        Self::from_freestream_context(num_cells, &FreestreamContext::nondimensional(eos), params)
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
    pub fn enforce_positivity(&mut self, _eos: &IdealGasEoS, _min_pressure: crate::core::Real) {
        // 已禁用正性钳制——不做任何操作。
    }

    /// 将无量纲守恒量还原为有量纲 SI（输出 VTK/CGNS 用）。
    pub fn to_dimensional(&self, reference: &crate::physics::ReferenceScales) -> Result<Self> {
        let mut out = self.clone();
        let mom_scale = reference.density * reference.velocity;
        let energy_scale = reference.density * reference.velocity * reference.velocity;
        for v in out.density.values_mut() {
            *v *= reference.density;
        }
        for v in out.momentum_x.values_mut() {
            *v *= mom_scale;
        }
        for v in out.momentum_y.values_mut() {
            *v *= mom_scale;
        }
        for v in out.momentum_z.values_mut() {
            *v *= mom_scale;
        }
        for v in out.total_energy.values_mut() {
            *v *= energy_scale;
        }
        Ok(out)
    }

    #[allow(dead_code)]
    fn write_cell_state(&mut self, index: usize, state: &ConservedState) {
        self.density.values_mut()[index] = state.density;
        self.momentum_x.values_mut()[index] = state.momentum[0];
        self.momentum_y.values_mut()[index] = state.momentum[1];
        self.momentum_z.values_mut()[index] = state.momentum[2];
        self.total_energy.values_mut()[index] = state.total_energy;
    }
}

/// 来流静压的 1%（下限 1e-6 Pa 或 1e-12 无量纲），与求解器正性限制一致。
#[must_use]
pub fn positivity_pressure_floor(freestream_pressure: crate::core::Real) -> crate::core::Real {
    const ABSOLUTE_MIN: crate::core::Real = 1.0e-6;
    const RELATIVE_FRACTION: crate::core::Real = 0.01;
    if freestream_pressure > 0.0 {
        (RELATIVE_FRACTION * freestream_pressure).max(ABSOLUTE_MIN)
    } else {
        ABSOLUTE_MIN
    }
}

/// 单单元守恒量正性钳制（调试模式下已禁用——不做任何钳制）。
pub fn clamp_conserved_positivity(
    _state: &mut ConservedState,
    _gamma: crate::core::Real,
    _min_pressure: crate::core::Real,
) {
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
        // 已禁用夹紧——直接报错暴露数值问题根源。
        return Err(AsimuError::Field(format!(
            "内能非正: rho={rho}, KE={ke}, total_energy={}",
            cons.total_energy
        )));
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

/// 通量/边界装配用的宽松 primitive 恢复（压力不低于 `min_pressure`）。
pub fn primitive_from_conserved_relaxed(
    eos: &IdealGasEoS,
    cons: &ConservedState,
    min_pressure: crate::core::Real,
) -> Result<PrimitiveState> {
    let mut prim = primitive_from_conserved(eos, cons)?;
    if prim.pressure < min_pressure {
        prim.pressure = min_pressure;
        prim.temperature = prim.pressure / (prim.density.max(1.0e-30) * eos.gas_constant);
    }
    Ok(prim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positivity_pressure_floor_is_one_percent_of_freestream() {
        assert!((positivity_pressure_floor(1000.0) - 10.0).abs() < 1.0e-12);
        assert!((positivity_pressure_floor(1.0 / 1.4) - 0.01 / 1.4).abs() < 1.0e-12);
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

    #[test]
    fn dimensionalize_reverses_reference_scaling() {
        use crate::physics::{
            FreestreamParams, ReferenceScales, ViscosityModel, ViscousPhysicsConfig,
        };

        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            pressure: 1000.0,
            temperature: 300.0,
            mach: 2.0,
            ..FreestreamParams::default()
        };
        let viscous = ViscousPhysicsConfig::new(ViscosityModel::AIR_SUTHERLAND, 0.72).expect("v");
        let reference = ReferenceScales::from_freestream(&eos, &fs, Some(&viscous)).expect("ref");
        let dim = ConservedFields::from_freestream(4, &eos, &fs).expect("dim");
        let mut nd_eos = eos;
        nd_eos.gas_constant = reference.nondimensional_gas_constant();
        let nd_fs = FreestreamParams {
            mach: fs.mach,
            pressure: 1.0 / eos.gamma,
            temperature: 1.0,
            ..FreestreamParams::default()
        };
        let nd = ConservedFields::from_nondimensional_freestream(4, &nd_eos, &nd_fs).expect("nd");
        let back = nd.to_dimensional(&reference).expect("back");
        assert!(
            (back.density.values()[0] - dim.density.values()[0]).abs() / dim.density.values()[0]
                < 1.0e-10
        );
    }
}
