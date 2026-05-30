//! 守恒场与残差的线性组合（RK 阶段更新）。

use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::{ConservedFields, ConservedResidual};

impl ConservedFields {
    /// `self ← src`。
    pub fn copy_from(&mut self, src: &Self) -> Result<()> {
        ensure_same_size(self.num_cells(), src.num_cells())?;
        self.density
            .values_mut()
            .copy_from_slice(src.density.values());
        self.momentum_x
            .values_mut()
            .copy_from_slice(src.momentum_x.values());
        self.momentum_y
            .values_mut()
            .copy_from_slice(src.momentum_y.values());
        self.momentum_z
            .values_mut()
            .copy_from_slice(src.momentum_z.values());
        self.total_energy
            .values_mut()
            .copy_from_slice(src.total_energy.values());
        Ok(())
    }

    /// `self ← base + factor * dt[i] * residual[i]`。
    pub fn assign_axpy_dt(
        &mut self,
        base: &Self,
        residual: &ConservedResidual,
        dt: &[Real],
        factor: Real,
    ) -> Result<()> {
        ensure_same_size(self.num_cells(), base.num_cells())?;
        ensure_residual_size(self.num_cells(), residual.num_cells())?;
        ensure_dt_size(self.num_cells(), dt.len())?;
        let rho_min = 1.0e-12;
        for (i, &dt_i) in dt.iter().enumerate() {
            let mut rho = base.density.values()[i] + factor * dt_i * residual.density.values()[i];
            let mut mx =
                base.momentum_x.values()[i] + factor * dt_i * residual.momentum_x.values()[i];
            let mut my =
                base.momentum_y.values()[i] + factor * dt_i * residual.momentum_y.values()[i];
            let mut mz =
                base.momentum_z.values()[i] + factor * dt_i * residual.momentum_z.values()[i];
            let energy =
                base.total_energy.values()[i] + factor * dt_i * residual.total_energy.values()[i];
            let rho_old = rho;
            let rho_clamped = if rho_old.is_finite() && rho_old > 0.0 {
                rho_old.max(rho_min)
            } else {
                rho_min
            };
            if rho_old.is_finite() && rho_old > 0.0 && rho_old < rho_min {
                let scale = rho_clamped / rho_old;
                mx *= scale;
                my *= scale;
                mz *= scale;
            } else if !(rho_old.is_finite() && rho_old > 0.0) {
                mx = 0.0;
                my = 0.0;
                mz = 0.0;
            }
            rho = rho_clamped;
            self.density.values_mut()[i] = rho;
            self.momentum_x.values_mut()[i] = mx;
            self.momentum_y.values_mut()[i] = my;
            self.momentum_z.values_mut()[i] = mz;
            self.total_energy.values_mut()[i] = energy;
        }
        Ok(())
    }

    /// `self ← base + scale * residual`。
    pub fn assign_axpy(
        &mut self,
        base: &Self,
        residual: &ConservedResidual,
        scale: Real,
    ) -> Result<()> {
        ensure_same_size(self.num_cells(), base.num_cells())?;
        ensure_residual_size(self.num_cells(), residual.num_cells())?;
        axpy_component(
            self.density.values_mut(),
            base.density.values(),
            residual.density.values(),
            scale,
        );
        axpy_component(
            self.momentum_x.values_mut(),
            base.momentum_x.values(),
            residual.momentum_x.values(),
            scale,
        );
        axpy_component(
            self.momentum_y.values_mut(),
            base.momentum_y.values(),
            residual.momentum_y.values(),
            scale,
        );
        axpy_component(
            self.momentum_z.values_mut(),
            base.momentum_z.values(),
            residual.momentum_z.values(),
            scale,
        );
        axpy_component(
            self.total_energy.values_mut(),
            base.total_energy.values(),
            residual.total_energy.values(),
            scale,
        );
        Ok(())
    }

    /// `self ← self + scale * residual`。
    pub fn add_axpy(&mut self, residual: &ConservedResidual, scale: Real) -> Result<()> {
        ensure_residual_size(self.num_cells(), residual.num_cells())?;
        add_scaled_slice(self.density.values_mut(), residual.density.values(), scale);
        add_scaled_slice(
            self.momentum_x.values_mut(),
            residual.momentum_x.values(),
            scale,
        );
        add_scaled_slice(
            self.momentum_y.values_mut(),
            residual.momentum_y.values(),
            scale,
        );
        add_scaled_slice(
            self.momentum_z.values_mut(),
            residual.momentum_z.values(),
            scale,
        );
        add_scaled_slice(
            self.total_energy.values_mut(),
            residual.total_energy.values(),
            scale,
        );
        Ok(())
    }
}

impl ConservedResidual {
    /// `self ← scale * src`。
    pub fn assign_scaled(&mut self, src: &Self, scale: Real) -> Result<()> {
        ensure_residual_size(self.num_cells(), src.num_cells())?;
        scale_component(self.density.values_mut(), src.density.values(), scale);
        scale_component(self.momentum_x.values_mut(), src.momentum_x.values(), scale);
        scale_component(self.momentum_y.values_mut(), src.momentum_y.values(), scale);
        scale_component(self.momentum_z.values_mut(), src.momentum_z.values(), scale);
        scale_component(
            self.total_energy.values_mut(),
            src.total_energy.values(),
            scale,
        );
        Ok(())
    }

    /// `self ← self + scale * src`。
    pub fn add_scaled(&mut self, src: &Self, scale: Real) -> Result<()> {
        ensure_residual_size(self.num_cells(), src.num_cells())?;
        add_scaled_slice(self.density.values_mut(), src.density.values(), scale);
        add_scaled_slice(self.momentum_x.values_mut(), src.momentum_x.values(), scale);
        add_scaled_slice(self.momentum_y.values_mut(), src.momentum_y.values(), scale);
        add_scaled_slice(self.momentum_z.values_mut(), src.momentum_z.values(), scale);
        add_scaled_slice(
            self.total_energy.values_mut(),
            src.total_energy.values(),
            scale,
        );
        Ok(())
    }

    /// RK4 组合：\(\frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)\)。
    pub fn assign_rk4_increment(
        &mut self,
        k1: &Self,
        k2: &Self,
        k3: &Self,
        k4: &Self,
    ) -> Result<()> {
        let n = k1.num_cells();
        ensure_residual_size(n, k2.num_cells())?;
        ensure_residual_size(n, k3.num_cells())?;
        ensure_residual_size(n, k4.num_cells())?;
        combine_rk4_component(
            self.density.values_mut(),
            k1.density.values(),
            k2.density.values(),
            k3.density.values(),
            k4.density.values(),
        );
        combine_rk4_component(
            self.momentum_x.values_mut(),
            k1.momentum_x.values(),
            k2.momentum_x.values(),
            k3.momentum_x.values(),
            k4.momentum_x.values(),
        );
        combine_rk4_component(
            self.momentum_y.values_mut(),
            k1.momentum_y.values(),
            k2.momentum_y.values(),
            k3.momentum_y.values(),
            k4.momentum_y.values(),
        );
        combine_rk4_component(
            self.momentum_z.values_mut(),
            k1.momentum_z.values(),
            k2.momentum_z.values(),
            k3.momentum_z.values(),
            k4.momentum_z.values(),
        );
        combine_rk4_component(
            self.total_energy.values_mut(),
            k1.total_energy.values(),
            k2.total_energy.values(),
            k3.total_energy.values(),
            k4.total_energy.values(),
        );
        Ok(())
    }

    /// 全场密度残差 L2 范数：\(\|\dot\rho\|_2 = \sqrt{\sum_i \dot\rho_i^2}\)（随网格单元数增大）。
    #[must_use]
    pub fn density_l2_norm(&self) -> Real {
        l2_norm(self.density.values())
    }

    /// 全场密度残差 RMS：\(\mathrm{RMS}(\dot\rho)=\|\dot\rho\|_2/\sqrt{N}\)（可与不同规模网格对比）。
    #[must_use]
    pub fn density_rms_norm(&self) -> Real {
        rms_norm(self.density.values())
    }

    /// 五方程守恒残差 RMS（所有单元、所有分量）：\(\sqrt{\sum|\dot U|^2 / (5N)}\)。
    #[must_use]
    pub fn conserved_rms_norm(&self) -> Real {
        let n = self.num_cells();
        if n == 0 {
            return 0.0;
        }
        let mut sum_sq = 0.0;
        for values in [
            self.density.values(),
            self.momentum_x.values(),
            self.momentum_y.values(),
            self.momentum_z.values(),
            self.total_energy.values(),
        ] {
            sum_sq += values.iter().map(|v| v * v).sum::<Real>();
        }
        (sum_sq / (5.0 * n as Real)).sqrt()
    }
}

#[must_use]
fn l2_norm(values: &[Real]) -> Real {
    values.iter().map(|v| v * v).sum::<Real>().sqrt()
}

#[must_use]
fn rms_norm(values: &[Real]) -> Real {
    if values.is_empty() {
        return 0.0;
    }
    l2_norm(values) / (values.len() as Real).sqrt()
}

fn ensure_same_size(left: usize, right: usize) -> Result<()> {
    if left != right {
        return Err(AsimuError::Field(format!(
            "守恒场尺寸不一致: {left} vs {right}"
        )));
    }
    Ok(())
}

fn ensure_residual_size(fields: usize, residual: usize) -> Result<()> {
    if fields != residual {
        return Err(AsimuError::Field(format!(
            "场/残差尺寸不一致: {fields} vs {residual}"
        )));
    }
    Ok(())
}

fn ensure_dt_size(fields: usize, dt_len: usize) -> Result<()> {
    if fields != dt_len {
        return Err(AsimuError::Field(format!(
            "逐单元 dt 长度 {dt_len} 与场单元数 {fields} 不一致"
        )));
    }
    Ok(())
}

fn axpy_component(dst: &mut [Real], base: &[Real], inc: &[Real], scale: Real) {
    for (d, (&b, &r)) in dst.iter_mut().zip(base.iter().zip(inc.iter())) {
        *d = b + scale * r;
    }
}

fn add_scaled_slice(dst: &mut [Real], src: &[Real], scale: Real) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += scale * s;
    }
}

fn scale_component(dst: &mut [Real], src: &[Real], scale: Real) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = scale * s;
    }
}

fn combine_rk4_component(dst: &mut [Real], k1: &[Real], k2: &[Real], k3: &[Real], k4: &[Real]) {
    let sixth = 1.0 / 6.0;
    for (d, (&a, (&b, (&c, &e)))) in dst
        .iter_mut()
        .zip(k1.iter().zip(k2.iter().zip(k3.iter().zip(k4.iter()))))
    {
        *d = sixth * (a + 2.0 * b + 2.0 * c + e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    #[test]
    fn rms_norm_scales_with_cell_count() {
        let mut rhs = ConservedResidual::zeros(4).expect("rhs");
        for v in rhs.density.values_mut() {
            *v = 3.0;
        }
        assert!((rhs.density_l2_norm() - 6.0).abs() < 1.0e-12);
        assert!((rhs.density_rms_norm() - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn assign_axpy_updates_all_components() {
        let base = ConservedFields::uniform(
            2,
            crate::physics::ConservedState {
                density: 1.0,
                momentum: [0.0, 0.0, 0.0],
                total_energy: 2.0,
            },
        )
        .expect("base");
        let mut rhs = ConservedResidual::zeros(2).expect("rhs");
        rhs.density.values_mut()[0] = 4.0;
        let mut out = ConservedFields::uniform(
            2,
            crate::physics::ConservedState {
                density: 0.0,
                momentum: [0.0, 0.0, 0.0],
                total_energy: 0.0,
            },
        )
        .expect("out");
        out.assign_axpy(&base, &rhs, 0.5).expect("axpy");
        assert!(approx_eq(out.density.values()[0], 3.0, 1.0e-12));
        assert!(approx_eq(out.density.values()[1], 1.0, 1.0e-12));
    }
}
