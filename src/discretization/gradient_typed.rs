//! 单元中心梯度 typed 场（ADR 0016 P3）。

use crate::core::{ComputeFloat, Real};
use crate::error::Result;
use crate::field::ScalarFieldT;

/// 速度分量与温度的单元中心梯度（SoA）。
#[derive(Debug, Clone, PartialEq)]
pub struct GradientFieldsT<T: ComputeFloat> {
    pub du_dx: ScalarFieldT<T>,
    pub du_dy: ScalarFieldT<T>,
    pub du_dz: ScalarFieldT<T>,
    pub dv_dx: ScalarFieldT<T>,
    pub dv_dy: ScalarFieldT<T>,
    pub dv_dz: ScalarFieldT<T>,
    pub dw_dx: ScalarFieldT<T>,
    pub dw_dy: ScalarFieldT<T>,
    pub dw_dz: ScalarFieldT<T>,
    pub dt_dx: ScalarFieldT<T>,
    pub dt_dy: ScalarFieldT<T>,
    pub dt_dz: ScalarFieldT<T>,
    pub drho_dx: ScalarFieldT<T>,
    pub drho_dy: ScalarFieldT<T>,
    pub drho_dz: ScalarFieldT<T>,
    pub dp_dx: ScalarFieldT<T>,
    pub dp_dy: ScalarFieldT<T>,
    pub dp_dz: ScalarFieldT<T>,
}

/// f64 梯度场别名（兼容旧 API）。
pub type GradientFields = GradientFieldsT<Real>;

/// 非结构无粘重构用原始变量梯度分量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InviscidPrimitiveGradientsT<T: ComputeFloat> {
    pub drho: [T; 3],
    pub du: [T; 3],
    pub dv: [T; 3],
    pub dw: [T; 3],
    pub dp: [T; 3],
}

pub type InviscidPrimitiveGradients = InviscidPrimitiveGradientsT<Real>;

/// 单元 \((u,v,w,T)\) 梯度张量分量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityGradientT<T: ComputeFloat> {
    pub du: [T; 3],
    pub dv: [T; 3],
    pub dw: [T; 3],
    pub dt: [T; 3],
}

pub type VelocityGradient = VelocityGradientT<Real>;

/// 单元梯度 SoA 切片视图。
pub struct VelocityGradientSlicesT<'a, T: ComputeFloat> {
    pub du_dx: &'a [T],
    pub du_dy: &'a [T],
    pub du_dz: &'a [T],
    pub dv_dx: &'a [T],
    pub dv_dy: &'a [T],
    pub dv_dz: &'a [T],
    pub dw_dx: &'a [T],
    pub dw_dy: &'a [T],
    pub dw_dz: &'a [T],
    pub dt_dx: &'a [T],
    pub dt_dy: &'a [T],
    pub dt_dz: &'a [T],
}

pub type VelocityGradientSlices<'a> = VelocityGradientSlicesT<'a, Real>;

impl<T: ComputeFloat> GradientFieldsT<T> {
    pub fn zeros(num_cells: usize) -> Result<Self> {
        let zero = ScalarFieldT::uniform(num_cells, T::zero())?;
        Ok(Self {
            du_dx: zero.clone(),
            du_dy: zero.clone(),
            du_dz: zero.clone(),
            dv_dx: zero.clone(),
            dv_dy: zero.clone(),
            dv_dz: zero.clone(),
            dw_dx: zero.clone(),
            dw_dy: zero.clone(),
            dw_dz: zero.clone(),
            dt_dx: zero.clone(),
            dt_dy: zero.clone(),
            dt_dz: zero.clone(),
            drho_dx: zero.clone(),
            drho_dy: zero.clone(),
            drho_dz: zero.clone(),
            dp_dx: zero.clone(),
            dp_dy: zero.clone(),
            dp_dz: zero,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.du_dx.len()
    }

    pub(crate) fn clear(&mut self) {
        for f in [
            &mut self.du_dx,
            &mut self.du_dy,
            &mut self.du_dz,
            &mut self.dv_dx,
            &mut self.dv_dy,
            &mut self.dv_dz,
            &mut self.dw_dx,
            &mut self.dw_dy,
            &mut self.dw_dz,
            &mut self.dt_dx,
            &mut self.dt_dy,
            &mut self.dt_dz,
            &mut self.drho_dx,
            &mut self.drho_dy,
            &mut self.drho_dz,
            &mut self.dp_dx,
            &mut self.dp_dy,
            &mut self.dp_dz,
        ] {
            for v in f.values_mut() {
                *v = T::zero();
            }
        }
    }

    #[must_use]
    pub fn velocity_grad_at(&self, cell: usize) -> VelocityGradientT<T> {
        VelocityGradientT {
            du: [
                self.du_dx.values()[cell],
                self.du_dy.values()[cell],
                self.du_dz.values()[cell],
            ],
            dv: [
                self.dv_dx.values()[cell],
                self.dv_dy.values()[cell],
                self.dv_dz.values()[cell],
            ],
            dw: [
                self.dw_dx.values()[cell],
                self.dw_dy.values()[cell],
                self.dw_dz.values()[cell],
            ],
            dt: [
                self.dt_dx.values()[cell],
                self.dt_dy.values()[cell],
                self.dt_dz.values()[cell],
            ],
        }
    }

    #[must_use]
    pub fn inviscid_primitive_grad_at(&self, cell: usize) -> InviscidPrimitiveGradientsT<T> {
        InviscidPrimitiveGradientsT {
            drho: [
                self.drho_dx.values()[cell],
                self.drho_dy.values()[cell],
                self.drho_dz.values()[cell],
            ],
            du: [
                self.du_dx.values()[cell],
                self.du_dy.values()[cell],
                self.du_dz.values()[cell],
            ],
            dv: [
                self.dv_dx.values()[cell],
                self.dv_dy.values()[cell],
                self.dv_dz.values()[cell],
            ],
            dw: [
                self.dw_dx.values()[cell],
                self.dw_dy.values()[cell],
                self.dw_dz.values()[cell],
            ],
            dp: [
                self.dp_dx.values()[cell],
                self.dp_dy.values()[cell],
                self.dp_dz.values()[cell],
            ],
        }
    }

    #[must_use]
    pub fn velocity_gradient_slices(&self) -> VelocityGradientSlicesT<'_, T> {
        VelocityGradientSlicesT {
            du_dx: self.du_dx.values(),
            du_dy: self.du_dy.values(),
            du_dz: self.du_dz.values(),
            dv_dx: self.dv_dx.values(),
            dv_dy: self.dv_dy.values(),
            dv_dz: self.dv_dz.values(),
            dw_dx: self.dw_dx.values(),
            dw_dy: self.dw_dy.values(),
            dw_dz: self.dw_dz.values(),
            dt_dx: self.dt_dx.values(),
            dt_dy: self.dt_dy.values(),
            dt_dz: self.dt_dz.values(),
        }
    }

    /// 将 typed 梯度转为 f64（MUSCL f64 重构桥接）。
    pub fn to_real_fields(&self) -> Result<GradientFields> {
        let velocity = cast_velocity_gradient_scalars(self)?;
        let temperature = cast_temperature_gradient_scalars(self)?;
        let primitive = cast_primitive_gradient_scalars(self)?;
        Ok(GradientFields {
            du_dx: velocity.du_dx,
            du_dy: velocity.du_dy,
            du_dz: velocity.du_dz,
            dv_dx: velocity.dv_dx,
            dv_dy: velocity.dv_dy,
            dv_dz: velocity.dv_dz,
            dw_dx: velocity.dw_dx,
            dw_dy: velocity.dw_dy,
            dw_dz: velocity.dw_dz,
            dt_dx: temperature.dt_dx,
            dt_dy: temperature.dt_dy,
            dt_dz: temperature.dt_dz,
            drho_dx: primitive.drho_dx,
            drho_dy: primitive.drho_dy,
            drho_dz: primitive.drho_dz,
            dp_dx: primitive.dp_dx,
            dp_dy: primitive.dp_dy,
            dp_dz: primitive.dp_dz,
        })
    }
}

struct VelocityGradientScalars {
    du_dx: ScalarFieldT<Real>,
    du_dy: ScalarFieldT<Real>,
    du_dz: ScalarFieldT<Real>,
    dv_dx: ScalarFieldT<Real>,
    dv_dy: ScalarFieldT<Real>,
    dv_dz: ScalarFieldT<Real>,
    dw_dx: ScalarFieldT<Real>,
    dw_dy: ScalarFieldT<Real>,
    dw_dz: ScalarFieldT<Real>,
}

struct TemperatureGradientScalars {
    dt_dx: ScalarFieldT<Real>,
    dt_dy: ScalarFieldT<Real>,
    dt_dz: ScalarFieldT<Real>,
}

struct PrimitiveGradientScalars {
    drho_dx: ScalarFieldT<Real>,
    drho_dy: ScalarFieldT<Real>,
    drho_dz: ScalarFieldT<Real>,
    dp_dx: ScalarFieldT<Real>,
    dp_dy: ScalarFieldT<Real>,
    dp_dz: ScalarFieldT<Real>,
}

fn cast_scalar_field<T: ComputeFloat>(field: &ScalarFieldT<T>) -> Result<ScalarFieldT<Real>> {
    ScalarFieldT::from_real_values(field.to_real_values())
}

fn cast_velocity_gradient_scalars<T: ComputeFloat>(
    gradients: &GradientFieldsT<T>,
) -> Result<VelocityGradientScalars> {
    Ok(VelocityGradientScalars {
        du_dx: cast_scalar_field(&gradients.du_dx)?,
        du_dy: cast_scalar_field(&gradients.du_dy)?,
        du_dz: cast_scalar_field(&gradients.du_dz)?,
        dv_dx: cast_scalar_field(&gradients.dv_dx)?,
        dv_dy: cast_scalar_field(&gradients.dv_dy)?,
        dv_dz: cast_scalar_field(&gradients.dv_dz)?,
        dw_dx: cast_scalar_field(&gradients.dw_dx)?,
        dw_dy: cast_scalar_field(&gradients.dw_dy)?,
        dw_dz: cast_scalar_field(&gradients.dw_dz)?,
    })
}

fn cast_temperature_gradient_scalars<T: ComputeFloat>(
    gradients: &GradientFieldsT<T>,
) -> Result<TemperatureGradientScalars> {
    Ok(TemperatureGradientScalars {
        dt_dx: cast_scalar_field(&gradients.dt_dx)?,
        dt_dy: cast_scalar_field(&gradients.dt_dy)?,
        dt_dz: cast_scalar_field(&gradients.dt_dz)?,
    })
}

fn cast_primitive_gradient_scalars<T: ComputeFloat>(
    gradients: &GradientFieldsT<T>,
) -> Result<PrimitiveGradientScalars> {
    Ok(PrimitiveGradientScalars {
        drho_dx: cast_scalar_field(&gradients.drho_dx)?,
        drho_dy: cast_scalar_field(&gradients.drho_dy)?,
        drho_dz: cast_scalar_field(&gradients.drho_dz)?,
        dp_dx: cast_scalar_field(&gradients.dp_dx)?,
        dp_dy: cast_scalar_field(&gradients.dp_dy)?,
        dp_dz: cast_scalar_field(&gradients.dp_dz)?,
    })
}

impl GradientFieldsT<f32> {
    pub fn cast_to_real(&self) -> Result<GradientFields> {
        self.to_real_fields()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    #[test]
    fn f32_gradient_roundtrip_to_real() {
        let mut g = GradientFieldsT::<f32>::zeros(2).expect("g");
        g.du_dx.values_mut()[0] = 1.25f32;
        g.dt_dz.values_mut()[1] = -0.5f32;
        let real = g.to_real_fields().expect("cast");
        assert!(approx_eq(real.du_dx.values()[0], 1.25, 1.0e-6));
        assert!(approx_eq(real.dt_dz.values()[1], -0.5, 1.0e-6));
    }
}
