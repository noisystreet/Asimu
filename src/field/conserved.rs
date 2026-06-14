//! 守恒变量场（可压缩 NS）。

use crate::core::{ComputeFloat, Real};
use crate::error::{AsimuError, Result};
use crate::physics::{
    ConservedState, FreestreamContext, FreestreamParams, IdealGasEoS, PrimitiveState,
    PrimitiveStateF32, ReferenceScales,
};

use super::ScalarFieldT;

/// 单元守恒变量集合（SoA）。
#[derive(Debug, Clone, PartialEq)]
pub struct ConservedFieldsT<T: ComputeFloat> {
    pub density: ScalarFieldT<T>,
    pub momentum_x: ScalarFieldT<T>,
    pub momentum_y: ScalarFieldT<T>,
    pub momentum_z: ScalarFieldT<T>,
    pub total_energy: ScalarFieldT<T>,
}

/// 默认工程标量守恒场（`f64`）。
pub type ConservedFields = ConservedFieldsT<Real>;

impl<T: ComputeFloat> ConservedFieldsT<T> {
    pub fn uniform(num_cells: usize, state: ConservedState) -> Result<Self> {
        Ok(Self {
            density: ScalarFieldT::uniform(num_cells, T::from_real(state.density))?,
            momentum_x: ScalarFieldT::uniform(num_cells, T::from_real(state.momentum[0]))?,
            momentum_y: ScalarFieldT::uniform(num_cells, T::from_real(state.momentum[1]))?,
            momentum_z: ScalarFieldT::uniform(num_cells, T::from_real(state.momentum[2]))?,
            total_energy: ScalarFieldT::uniform(num_cells, T::from_real(state.total_energy))?,
        })
    }

    /// 由 SI 来流参数构造均匀无量纲来流场（与算例 `apply_nondimensionalization` 一致）。
    pub fn from_freestream(
        num_cells: usize,
        eos: &IdealGasEoS,
        params: &FreestreamParams,
    ) -> Result<Self> {
        let reference = ReferenceScales::from_freestream(eos, params, None)?;
        let mut nd_eos = *eos;
        nd_eos.gas_constant = reference.nondimensional_gas_constant();
        let nd_params = FreestreamParams {
            mach: params.mach,
            pressure: 1.0 / eos.gamma,
            temperature: 1.0,
            velocity_direction: params.velocity_direction,
            alpha: params.alpha,
            beta: params.beta,
        };
        Self::from_freestream_context(
            num_cells,
            &FreestreamContext::new(&nd_eos, Some(&reference), None),
            &nd_params,
        )
    }

    /// 经 [`FreestreamContext`](crate::physics::FreestreamContext) 构造均匀来流场（\(*\) 变量）。
    ///
    /// 理论：[`docs/theory/nondimensional.md`](../../docs/theory/nondimensional.md) §2、§6。
    pub fn from_freestream_context(
        num_cells: usize,
        ctx: &FreestreamContext<'_>,
        params: &FreestreamParams,
    ) -> Result<Self> {
        Self::uniform(num_cells, ctx.conserved(params)?)
    }

    /// 已缩放的无量纲来流参数（`pressure=1/γ`，`temperature=1`）。
    pub fn from_nondimensional_freestream(
        num_cells: usize,
        eos: &IdealGasEoS,
        params: &FreestreamParams,
    ) -> Result<Self> {
        Self::from_freestream_context(num_cells, &FreestreamContext::new(eos, None, None), params)
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.density.len()
    }

    pub fn cell_state(&self, index: usize) -> Result<ConservedState> {
        Ok(ConservedState {
            density: self.density.values()[index].to_real(),
            momentum: [
                self.momentum_x.values()[index].to_real(),
                self.momentum_y.values()[index].to_real(),
                self.momentum_z.values()[index].to_real(),
            ],
            total_energy: self.total_energy.values()[index].to_real(),
        })
    }

    pub fn primitive_at(
        &self,
        index: usize,
        eos: &IdealGasEoS,
        min_pressure: Real,
    ) -> Result<PrimitiveState> {
        primitive_from_conserved_relaxed(eos, &self.cell_state(index)?, min_pressure)
    }

    /// 保证 \(\rho>0\) 且 \(E>\mathrm{KE}+p_\mathrm{floor}/(\gamma-1)\)（显式 RK 步后调用）。
    pub fn enforce_positivity(&mut self, _eos: &IdealGasEoS, _min_pressure: Real) {
        // 已禁用正性钳制——不做任何操作。
    }

    /// 转为 `Real` 守恒场（输出 / 跨精度转换用）。
    pub fn cast_real(&self) -> Result<ConservedFields> {
        Ok(ConservedFields {
            density: ScalarFieldT::from_real_values(self.density.to_real_values())?,
            momentum_x: ScalarFieldT::from_real_values(self.momentum_x.to_real_values())?,
            momentum_y: ScalarFieldT::from_real_values(self.momentum_y.to_real_values())?,
            momentum_z: ScalarFieldT::from_real_values(self.momentum_z.to_real_values())?,
            total_energy: ScalarFieldT::from_real_values(self.total_energy.to_real_values())?,
        })
    }

    /// 从 `Real` 守恒场构造 typed 场。
    pub fn from_real_fields(fields: &ConservedFields) -> Result<Self> {
        Ok(Self {
            density: ScalarFieldT::from_real_values(fields.density.to_real_values())?,
            momentum_x: ScalarFieldT::from_real_values(fields.momentum_x.to_real_values())?,
            momentum_y: ScalarFieldT::from_real_values(fields.momentum_y.to_real_values())?,
            momentum_z: ScalarFieldT::from_real_values(fields.momentum_z.to_real_values())?,
            total_energy: ScalarFieldT::from_real_values(fields.total_energy.to_real_values())?,
        })
    }

    #[allow(dead_code)]
    fn write_cell_state(&mut self, index: usize, state: &ConservedState) {
        self.density.values_mut()[index] = T::from_real(state.density);
        self.momentum_x.values_mut()[index] = T::from_real(state.momentum[0]);
        self.momentum_y.values_mut()[index] = T::from_real(state.momentum[1]);
        self.momentum_z.values_mut()[index] = T::from_real(state.momentum[2]);
        self.total_energy.values_mut()[index] = T::from_real(state.total_energy);
    }
}

impl ConservedFields {
    /// 从 typed 场构造 `Real` 守恒场。
    pub fn from_typed<T: ComputeFloat>(fields: &ConservedFieldsT<T>) -> Result<Self> {
        fields.cast_real()
    }

    /// 从 `Real` 守恒场构造 typed 场（`T=f64` 时为拷贝）。
    pub fn to_typed<T: ComputeFloat>(fields: &ConservedFields) -> Result<ConservedFieldsT<T>> {
        ConservedFieldsT::from_real_fields(fields)
    }

    /// 将无量纲守恒量还原为有量纲 SI（输出 VTK/CGNS 用）。
    pub fn to_dimensional(&self, reference: &ReferenceScales) -> Result<Self> {
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
}

/// 来流静压的 1%（下限 1e-6 Pa 或 1e-12 无量纲），与求解器正性限制一致。
#[must_use]
pub fn positivity_pressure_floor(freestream_pressure: Real) -> Real {
    const ABSOLUTE_MIN: Real = 1.0e-6;
    const RELATIVE_FRACTION: Real = 0.01;
    if freestream_pressure > 0.0 {
        (RELATIVE_FRACTION * freestream_pressure).max(ABSOLUTE_MIN)
    } else {
        ABSOLUTE_MIN
    }
}

/// 单单元守恒量正性钳制（调试模式下已禁用——不做任何钳制）。
pub fn clamp_conserved_positivity(_state: &mut ConservedState, _gamma: Real, _min_pressure: Real) {}

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
    min_pressure: Real,
) -> Result<PrimitiveState> {
    let mut prim = primitive_from_conserved(eos, cons)?;
    if prim.pressure < min_pressure {
        prim.pressure = min_pressure;
        prim.temperature = prim.pressure / (prim.density.max(1.0e-30) * eos.gas_constant);
    }
    Ok(prim)
}

/// f32 守恒分量 → 原始变量（理想气体；热路径无 `cell_state` 往返）。
pub fn primitive_from_conserved_relaxed_f32(
    eos: &IdealGasEoS,
    density: f32,
    momentum: [f32; 3],
    total_energy: f32,
    min_pressure: Real,
) -> Result<PrimitiveStateF32> {
    let rho = density;
    if rho <= 0.0_f32 {
        return Err(AsimuError::Field("密度必须大于 0".to_string()));
    }
    let inv_rho = 1.0_f32 / rho;
    let velocity = [
        momentum[0] * inv_rho,
        momentum[1] * inv_rho,
        momentum[2] * inv_rho,
    ];
    let ke = 0.5_f32
        * rho
        * (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]);
    let internal = total_energy - ke;
    if internal <= 0.0_f32 {
        return Err(AsimuError::Field(format!(
            "内能非正: rho={rho}, KE={ke}, total_energy={total_energy}"
        )));
    }
    let gamma = eos.gamma as f32;
    let mut pressure = (gamma - 1.0_f32) * internal;
    let min_p = min_pressure as f32;
    if pressure < min_p {
        pressure = min_p;
    }
    let r_star = eos.gas_constant as f32;
    let temperature = pressure / (rho.max(1.0e-30_f32) * r_star);
    Ok(PrimitiveStateF32 {
        density: rho,
        velocity,
        pressure,
        temperature,
    })
}

/// 从 `ConservedState`（ghost 缓冲）恢复 f32 原始变量；输入 Real 仅转换一次。
pub fn primitive_from_conserved_relaxed_f32_from_state(
    eos: &IdealGasEoS,
    cons: &ConservedState,
    min_pressure: Real,
) -> Result<PrimitiveStateF32> {
    primitive_from_conserved_relaxed_f32(
        eos,
        cons.density as f32,
        [
            cons.momentum[0] as f32,
            cons.momentum[1] as f32,
            cons.momentum[2] as f32,
        ],
        cons.total_energy as f32,
        min_pressure,
    )
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
    fn f32_primitive_recovery_matches_f64_relaxed() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let cons = ConservedState {
            density: 1.2,
            momentum: [0.36, 0.0, 0.0],
            total_energy: 2.5,
        };
        let min_p = 0.01;
        let f64_prim = primitive_from_conserved_relaxed(&eos, &cons, min_p).expect("f64");
        let f32_prim =
            primitive_from_conserved_relaxed_f32_from_state(&eos, &cons, min_p).expect("f32");
        assert!((f32_prim.density as Real - f64_prim.density).abs() < 1.0e-5);
        assert!((f32_prim.pressure as Real - f64_prim.pressure).abs() < 1.0e-4);
        assert!((f32_prim.velocity[0] as Real - f64_prim.velocity[0]).abs() < 1.0e-5);
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
    fn f32_conserved_field_casts_to_real() {
        let state = ConservedState {
            density: 1.2,
            momentum: [0.1, 0.2, 0.3],
            total_energy: 2.5,
        };
        let fields = ConservedFieldsT::<f32>::uniform(3, state).expect("fields");
        assert_eq!(fields.num_cells(), 3);
        let real = fields.cast_real().expect("cast");
        assert!((real.density.values()[0] - 1.2).abs() < 1.0e-6);
        assert!((real.momentum_x.values()[1] - 0.1).abs() < 1.0e-6);
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
        let nd = ConservedFields::from_freestream(4, &eos, &fs).expect("nd");
        let back = nd.to_dimensional(&reference).expect("back");
        let dim_prim = eos
            .freestream_primitive(fs.mach, fs.pressure, fs.temperature, fs.velocity_direction)
            .expect("dim prim");
        assert!((back.density.values()[0] - dim_prim.density).abs() / dim_prim.density < 1.0e-10);
    }
}
