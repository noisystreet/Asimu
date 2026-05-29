//! 理想气体状态方程（可压缩 NS）。

use crate::core::Real;
use crate::error::{AsimuError, Result};

/// 理想气体：\(p = \rho R T\)，\(e = c_v T\)。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdealGasEoS {
    pub gamma: Real,
    pub gas_constant: Real,
}

impl IdealGasEoS {
    pub const AIR_STANDARD: Self = Self {
        gamma: 1.4,
        gas_constant: 287.052_871_936_417,
    };

    pub fn new(gamma: Real, gas_constant: Real) -> Result<Self> {
        if gamma <= 1.0 {
            return Err(AsimuError::Config("gamma 必须大于 1".to_string()));
        }
        if gas_constant <= 0.0 {
            return Err(AsimuError::Config("gas_constant 必须大于 0".to_string()));
        }
        Ok(Self { gamma, gas_constant })
    }

    #[must_use]
    pub fn cv(&self) -> Real {
        self.gas_constant / (self.gamma - 1.0)
    }

    /// 声速 \(a = \sqrt{\gamma R T}\)。
    pub fn sound_speed(&self, temperature: Real) -> Result<Real> {
        if temperature <= 0.0 {
            return Err(AsimuError::Config("温度必须大于 0".to_string()));
        }
        Ok((self.gamma * self.gas_constant * temperature).sqrt())
    }

    /// \(\rho = p / (R T)\)。
    pub fn density(&self, pressure: Real, temperature: Real) -> Result<Real> {
        if pressure <= 0.0 || temperature <= 0.0 {
            return Err(AsimuError::Config("压力与温度必须大于 0".to_string()));
        }
        Ok(pressure / (self.gas_constant * temperature))
    }

    /// 比内能 \(e = p / ((\gamma-1)\rho)\)。
    pub fn specific_internal_energy(&self, pressure: Real, density: Real) -> Result<Real> {
        if density <= 0.0 || pressure <= 0.0 {
            return Err(AsimuError::Config("压力与密度必须大于 0".to_string()));
        }
        Ok(pressure / ((self.gamma - 1.0) * density))
    }

    /// \(\rho E = \rho e + \frac{1}{2}\rho|\mathbf{u}|^2\)。
    pub fn total_energy_density(
        &self,
        density: Real,
        pressure: Real,
        velocity: [Real; 3],
    ) -> Result<Real> {
        let e = self.specific_internal_energy(pressure, density)?;
        let ke = 0.5 * density * (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]);
        Ok(density * e + ke)
    }

    /// 由 Mach 数与静参数构造来流原始变量。
    pub fn freestream_primitive(
        &self,
        mach: Real,
        pressure: Real,
        temperature: Real,
        velocity_direction: [Real; 3],
    ) -> Result<PrimitiveState> {
        if mach < 0.0 {
            return Err(AsimuError::Config("Mach 数不能为负".to_string()));
        }
        let a = self.sound_speed(temperature)?;
        let speed = mach * a;
        let dir = normalize_direction(velocity_direction)?;
        let velocity = [dir[0] * speed, dir[1] * speed, dir[2] * speed];
        let density = self.density(pressure, temperature)?;
        Ok(PrimitiveState {
            density,
            velocity,
            pressure,
            temperature,
        })
    }
}

/// 原始变量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrimitiveState {
    pub density: Real,
    pub velocity: [Real; 3],
    pub pressure: Real,
    pub temperature: Real,
}

/// 守恒变量 \([\rho, \rho u, \rho v, \rho w, \rho E]\)。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConservedState {
    pub density: Real,
    pub momentum: [Real; 3],
    pub total_energy: Real,
}

impl ConservedState {
    pub fn from_primitive(eos: &IdealGasEoS, prim: &PrimitiveState) -> Result<Self> {
        let rho = prim.density;
        let momentum = [
            rho * prim.velocity[0],
            rho * prim.velocity[1],
            rho * prim.velocity[2],
        ];
        let total_energy = eos.total_energy_density(rho, prim.pressure, prim.velocity)?;
        Ok(Self {
            density: rho,
            momentum,
            total_energy,
        })
    }
}

/// 来流参数（case `[freestream]`）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreestreamParams {
    pub mach: Real,
    pub pressure: Real,
    pub temperature: Real,
    pub velocity_direction: [Real; 3],
    pub alpha: Real,
    pub beta: Real,
}

impl Default for FreestreamParams {
    fn default() -> Self {
        Self {
            mach: 0.0,
            pressure: 101_325.0,
            temperature: 288.15,
            velocity_direction: [1.0, 0.0, 0.0],
            alpha: 0.0,
            beta: 0.0,
        }
    }
}

impl FreestreamParams {
    /// 按攻角/侧滑角构造速度方向（度）。
    pub fn velocity_direction_from_angles(&self) -> [Real; 3] {
        let alpha = self.alpha.to_radians();
        let beta = self.beta.to_radians();
        let ca = alpha.cos();
        let sa = alpha.sin();
        let cb = beta.cos();
        let sb = beta.sin();
        normalize_direction([ca * cb, ca * sb, sa]).unwrap_or([1.0, 0.0, 0.0])
    }

    pub fn effective_direction(&self) -> [Real; 3] {
        if self.alpha.abs() > Real::EPSILON || self.beta.abs() > Real::EPSILON {
            self.velocity_direction_from_angles()
        } else {
            self.velocity_direction
        }
    }
}

fn normalize_direction(v: [Real; 3]) -> Result<[Real; 3]> {
    let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if mag < Real::EPSILON {
        return Err(AsimuError::Config("速度方向不能为零向量".to_string()));
    }
    Ok([v[0] / mag, v[1] / mag, v[2] / mag])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freestream_mach_zero_is_isothermal_rest() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let prim = eos
            .freestream_primitive(0.0, 101_325.0, 300.0, [1.0, 0.0, 0.0])
            .expect("fs");
        assert!(prim.velocity.iter().all(|&v| v.abs() < 1.0e-12));
        assert!((prim.density - 101_325.0 / (287.052_871_936_417 * 300.0)).abs() < 1.0e-6);
    }

    #[test]
    fn conserved_from_primitive_roundtrip_energy() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let prim = eos
            .freestream_primitive(0.5, 101_325.0, 288.15, [1.0, 0.0, 0.0])
            .expect("fs");
        let cons = ConservedState::from_primitive(&eos, &prim).expect("cons");
        assert!((cons.density - prim.density).abs() < 1.0e-10);
        assert!(cons.total_energy > 0.0);
    }
}
