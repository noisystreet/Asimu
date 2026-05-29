//! 守恒变量时间导数 / 右手项（FVM 残差）。

use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::ScalarField;

/// 控制体残差 SoA：\(\mathrm{d}U/\mathrm{d}t\) 或伪时间右手项。
#[derive(Debug, Clone, PartialEq)]
pub struct ConservedResidual {
    pub density: ScalarField,
    pub momentum_x: ScalarField,
    pub momentum_y: ScalarField,
    pub momentum_z: ScalarField,
    pub total_energy: ScalarField,
}

impl ConservedResidual {
    pub fn zeros(num_cells: usize) -> Result<Self> {
        Ok(Self {
            density: ScalarField::uniform(num_cells, 0.0)?,
            momentum_x: ScalarField::uniform(num_cells, 0.0)?,
            momentum_y: ScalarField::uniform(num_cells, 0.0)?,
            momentum_z: ScalarField::uniform(num_cells, 0.0)?,
            total_energy: ScalarField::uniform(num_cells, 0.0)?,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.density.len()
    }

    pub fn clear(&mut self) {
        for v in self.density.values_mut() {
            *v = 0.0;
        }
        for v in self.momentum_x.values_mut() {
            *v = 0.0;
        }
        for v in self.momentum_y.values_mut() {
            *v = 0.0;
        }
        for v in self.momentum_z.values_mut() {
            *v = 0.0;
        }
        for v in self.total_energy.values_mut() {
            *v = 0.0;
        }
    }

    pub fn add_flux_to_cell(
        &mut self,
        cell: usize,
        mass: Real,
        momentum: [Real; 3],
        energy: Real,
        scale: Real,
    ) -> Result<()> {
        if cell >= self.num_cells() {
            return Err(AsimuError::Field(format!("残差单元索引越界: {cell}")));
        }
        self.density.values_mut()[cell] += scale * mass;
        self.momentum_x.values_mut()[cell] += scale * momentum[0];
        self.momentum_y.values_mut()[cell] += scale * momentum[1];
        self.momentum_z.values_mut()[cell] += scale * momentum[2];
        self.total_energy.values_mut()[cell] += scale * energy;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_accumulate() {
        let mut rhs = ConservedResidual::zeros(2).expect("rhs");
        rhs.add_flux_to_cell(0, 1.0, [2.0, 0.0, 0.0], 3.0, -0.5)
            .expect("add");
        assert!((rhs.density.values()[0] + 0.5).abs() < 1.0e-12);
        assert!((rhs.momentum_x.values()[0] + 1.0).abs() < 1.0e-12);
    }
}
