//! 守恒变量时间导数 / 右手项（FVM 残差）。

use crate::core::{ComputeFloat, Real};
use crate::error::{AsimuError, Result};

use super::ScalarFieldT;

/// 控制体残差 SoA：\(\mathrm{d}U/\mathrm{d}t\) 或伪时间右手项。
#[derive(Debug, Clone, PartialEq)]
pub struct ConservedResidualT<T: ComputeFloat> {
    pub density: ScalarFieldT<T>,
    pub momentum_x: ScalarFieldT<T>,
    pub momentum_y: ScalarFieldT<T>,
    pub momentum_z: ScalarFieldT<T>,
    pub total_energy: ScalarFieldT<T>,
}

/// 默认工程标量残差（`f64`）。
pub type ConservedResidual = ConservedResidualT<Real>;

impl<T: ComputeFloat> ConservedResidualT<T> {
    pub fn zeros(num_cells: usize) -> Result<Self> {
        Ok(Self {
            density: ScalarFieldT::uniform(num_cells, T::zero())?,
            momentum_x: ScalarFieldT::uniform(num_cells, T::zero())?,
            momentum_y: ScalarFieldT::uniform(num_cells, T::zero())?,
            momentum_z: ScalarFieldT::uniform(num_cells, T::zero())?,
            total_energy: ScalarFieldT::uniform(num_cells, T::zero())?,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.density.len()
    }

    pub fn clear(&mut self) {
        for v in self.density.values_mut() {
            *v = T::zero();
        }
        for v in self.momentum_x.values_mut() {
            *v = T::zero();
        }
        for v in self.momentum_y.values_mut() {
            *v = T::zero();
        }
        for v in self.momentum_z.values_mut() {
            *v = T::zero();
        }
        for v in self.total_energy.values_mut() {
            *v = T::zero();
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
        self.density.values_mut()[cell] =
            self.density.values()[cell].add_mul_real(T::from_real(mass), scale);
        self.momentum_x.values_mut()[cell] =
            self.momentum_x.values()[cell].add_mul_real(T::from_real(momentum[0]), scale);
        self.momentum_y.values_mut()[cell] =
            self.momentum_y.values()[cell].add_mul_real(T::from_real(momentum[1]), scale);
        self.momentum_z.values_mut()[cell] =
            self.momentum_z.values()[cell].add_mul_real(T::from_real(momentum[2]), scale);
        self.total_energy.values_mut()[cell] =
            self.total_energy.values()[cell].add_mul_real(T::from_real(energy), scale);
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
        assert!((rhs.density.values()[0].to_real() + 0.5).abs() < 1.0e-6);
        assert!((rhs.momentum_x.values()[0].to_real() + 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn f32_residual_accumulates_flux() {
        let mut rhs = ConservedResidualT::<f32>::zeros(1).expect("rhs");
        rhs.add_flux_to_cell(0, 2.0, [1.0, 0.0, 0.0], 0.5, 1.0)
            .expect("add");
        assert!((rhs.density.values()[0].to_real() - 2.0).abs() < 1.0e-6);
    }
}
