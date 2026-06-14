//! 粘性面输运系数 CUDA kernel 参数（与 `viscous_face_transport_f32.cu` 布局一致）。

use cudarc::driver::DeviceRepr;

use crate::error::Result;
use crate::physics::{IdealGasEoS, ViscosityModel, ViscousPhysicsConfig};

pub const CUDA_VISCOSITY_MODEL_CONSTANT: u32 = 0;
pub const CUDA_VISCOSITY_MODEL_SUTHERLAND: u32 = 1;

/// device kernel 输运参数（按值传入 launch）。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceViscousTransportParams {
    pub model_kind: u32,
    pub mu_const: f32,
    pub lambda_const: f32,
    pub mu_ref: f32,
    pub t_ref: f32,
    pub sutherland_s: f32,
    pub prandtl: f32,
    /// `viscosity_ref` 时 \(1/\mathrm{Re}/\mu_{\mathrm{ref}}\)；否则 0（kernel 跳过）。
    pub viscosity_ref_scale: f32,
    /// 无量纲静温还原 \(T_{\mathrm{ref}}\)；有量纲时为 0。
    pub temperature_ref: f32,
    pub cp: f32,
}

unsafe impl DeviceRepr for DeviceViscousTransportParams {}

/// 由 `ViscousPhysicsConfig` 构建 device 参数（与 CPU `face_transport_coefficients_f32` 一致）。
pub fn build_device_viscous_transport_params(
    viscous: &ViscousPhysicsConfig,
    eos: &IdealGasEoS,
) -> Result<DeviceViscousTransportParams> {
    let cp = viscous.specific_heat_capacity_f32(eos);
    let prandtl = viscous.prandtl as f32;
    let viscosity_ref_scale = viscous
        .viscosity_ref
        .map(|mu_ref| (viscous.inv_reynolds / mu_ref) as f32)
        .unwrap_or(0.0);
    let temperature_ref = viscous.temperature_ref.map(|t| t as f32).unwrap_or(0.0);

    match viscous.model {
        ViscosityModel::Constant { mu } => {
            let (mu_f, lambda_f) = viscous.face_transport_coefficients_f32(1.0, 1.0, eos)?;
            let _ = mu;
            Ok(DeviceViscousTransportParams {
                model_kind: CUDA_VISCOSITY_MODEL_CONSTANT,
                mu_const: mu_f,
                lambda_const: lambda_f,
                mu_ref: 0.0,
                t_ref: 0.0,
                sutherland_s: 0.0,
                prandtl,
                viscosity_ref_scale,
                temperature_ref,
                cp,
            })
        }
        ViscosityModel::Sutherland {
            mu_ref,
            t_ref,
            sutherland_constant,
        } => Ok(DeviceViscousTransportParams {
            model_kind: CUDA_VISCOSITY_MODEL_SUTHERLAND,
            mu_const: 0.0,
            lambda_const: 0.0,
            mu_ref: mu_ref as f32,
            t_ref: t_ref as f32,
            sutherland_s: sutherland_constant as f32,
            prandtl,
            viscosity_ref_scale,
            temperature_ref,
            cp,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::physics::{IdealGasEoS, ViscosityModel, ViscousPhysicsConfig};

    #[test]
    fn constant_model_matches_face_transport_f32() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous =
            ViscousPhysicsConfig::new(ViscosityModel::constant(1.0e-5).expect("mu"), 0.72)
                .expect("viscous");
        let (mu, lambda) = viscous
            .face_transport_coefficients_f32(1.0, 1.0, &eos)
            .expect("tc");
        let params = build_device_viscous_transport_params(&viscous, &eos).expect("params");
        assert_eq!(params.model_kind, CUDA_VISCOSITY_MODEL_CONSTANT);
        assert!(approx_eq(params.mu_const as f64, mu as f64, 1.0e-6));
        assert!(approx_eq(params.lambda_const as f64, lambda as f64, 1.0e-6));
    }
}
