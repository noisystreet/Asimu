//! 不可压缩求解主变量场（cell-centered SoA）。

use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::ScalarField;

/// 不可压缩 Navier-Stokes 主状态：压力与三维速度。
#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleFields {
    pub pressure: ScalarField,
    pub velocity_x: ScalarField,
    pub velocity_y: ScalarField,
    pub velocity_z: ScalarField,
}

impl IncompressibleFields {
    /// 构造均匀不可压缩初场。
    pub fn uniform(num_cells: usize, pressure: Real, velocity: [Real; 3]) -> Result<Self> {
        Ok(Self {
            pressure: ScalarField::uniform(num_cells, pressure)?,
            velocity_x: ScalarField::uniform(num_cells, velocity[0])?,
            velocity_y: ScalarField::uniform(num_cells, velocity[1])?,
            velocity_z: ScalarField::uniform(num_cells, velocity[2])?,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.pressure.len()
    }

    pub fn validate_len(&self, expected: usize) -> Result<()> {
        for (name, len) in [
            ("Pressure", self.pressure.len()),
            ("VelocityX", self.velocity_x.len()),
            ("VelocityY", self.velocity_y.len()),
            ("VelocityZ", self.velocity_z.len()),
        ] {
            if len != expected {
                return Err(AsimuError::Field(format!(
                    "不可压字段 {name} 长度 {len} 与单元数 {expected} 不一致"
                )));
            }
        }
        Ok(())
    }
}
