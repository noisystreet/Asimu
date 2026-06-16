//! 守恒场与残差的线性组合（RK 阶段更新）。

use crate::core::{ComputeFloat, Real};
use crate::error::{AsimuError, Result};

use super::{ConservedFieldsT, ConservedResidualT};

impl<T: ComputeFloat> ConservedFieldsT<T> {
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
        residual: &ConservedResidualT<T>,
        dt: &[Real],
        factor: Real,
        _gamma: Real,
        _min_pressure: Real,
    ) -> Result<()> {
        ensure_same_size(self.num_cells(), base.num_cells())?;
        ensure_residual_size(self.num_cells(), residual.num_cells())?;
        ensure_dt_size(self.num_cells(), dt.len())?;
        for (i, &dt_i) in dt.iter().enumerate() {
            let scale = factor * dt_i;
            self.density.values_mut()[i] =
                base.density.values()[i].add_mul_real(residual.density.values()[i], scale);
            self.momentum_x.values_mut()[i] =
                base.momentum_x.values()[i].add_mul_real(residual.momentum_x.values()[i], scale);
            self.momentum_y.values_mut()[i] =
                base.momentum_y.values()[i].add_mul_real(residual.momentum_y.values()[i], scale);
            self.momentum_z.values_mut()[i] =
                base.momentum_z.values()[i].add_mul_real(residual.momentum_z.values()[i], scale);
            self.total_energy.values_mut()[i] = base.total_energy.values()[i]
                .add_mul_real(residual.total_energy.values()[i], scale);
        }
        Ok(())
    }
}

impl ConservedFieldsT<f32> {
    /// `self ← base + factor * dt[i] * residual[i]`（f32 逐单元 dt）。
    pub fn assign_axpy_dt_f32(
        &mut self,
        base: &Self,
        residual: &ConservedResidualT<f32>,
        dt: &[f32],
        factor: f32,
        _gamma: f32,
        _min_pressure: f32,
    ) -> Result<()> {
        ensure_same_size(self.num_cells(), base.num_cells())?;
        ensure_residual_size(self.num_cells(), residual.num_cells())?;
        ensure_dt_size_f32(self.num_cells(), dt.len())?;
        for (i, &dt_i) in dt.iter().enumerate() {
            let scale = factor * dt_i;
            self.density.values_mut()[i] =
                base.density.values()[i] + residual.density.values()[i] * scale;
            self.momentum_x.values_mut()[i] =
                base.momentum_x.values()[i] + residual.momentum_x.values()[i] * scale;
            self.momentum_y.values_mut()[i] =
                base.momentum_y.values()[i] + residual.momentum_y.values()[i] * scale;
            self.momentum_z.values_mut()[i] =
                base.momentum_z.values()[i] + residual.momentum_z.values()[i] * scale;
            self.total_energy.values_mut()[i] =
                base.total_energy.values()[i] + residual.total_energy.values()[i] * scale;
        }
        Ok(())
    }
}

impl<T: ComputeFloat> ConservedFieldsT<T> {
    /// `self[cell] += scale * increment`（守恒分量）。
    pub fn add_conserved_increment(
        &mut self,
        cell: usize,
        scale: Real,
        increment: [Real; 5],
        _gamma: Real,
        _min_pressure: Real,
    ) -> Result<()> {
        if cell >= self.num_cells() {
            return Err(AsimuError::Field(format!("单元索引越界: {cell}")));
        }
        self.density.values_mut()[cell] =
            self.density.values()[cell].add_mul_real(T::from_real(increment[0]), scale);
        self.momentum_x.values_mut()[cell] =
            self.momentum_x.values()[cell].add_mul_real(T::from_real(increment[1]), scale);
        self.momentum_y.values_mut()[cell] =
            self.momentum_y.values()[cell].add_mul_real(T::from_real(increment[2]), scale);
        self.momentum_z.values_mut()[cell] =
            self.momentum_z.values()[cell].add_mul_real(T::from_real(increment[3]), scale);
        self.total_energy.values_mut()[cell] =
            self.total_energy.values()[cell].add_mul_real(T::from_real(increment[4]), scale);
        Ok(())
    }

    /// `self ← base + scale * residual`。
    pub fn assign_axpy(
        &mut self,
        base: &Self,
        residual: &ConservedResidualT<T>,
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
    pub fn add_axpy(&mut self, residual: &ConservedResidualT<T>, scale: Real) -> Result<()> {
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

    /// 对角 LU-SGS：`self ← self + ω·Δt·R / (1 + Δt·σ)`（scale 在 `f64` 中计算）。
    #[allow(clippy::too_many_arguments)]
    pub fn assign_lusgs_diagonal_increment(
        &mut self,
        residual: &ConservedResidualT<T>,
        sigma: &[Real],
        volumes: &[Real],
        dt: Real,
        omega: Real,
        gamma: Real,
        min_pressure: Real,
    ) -> Result<()> {
        let n = self.num_cells();
        ensure_residual_size(n, residual.num_cells())?;
        if sigma.len() != n || volumes.len() != n {
            return Err(AsimuError::Field(format!(
                "lu_sgs: sigma/volume 长度 {}/{} 与场单元数 {n} 不一致",
                sigma.len(),
                volumes.len()
            )));
        }
        if !(dt > 0.0 && omega > 0.0) {
            return Err(AsimuError::Field("lu_sgs: dt 与 omega 须为正".to_string()));
        }
        let _ = volumes;
        for (i, &sig) in sigma.iter().enumerate().take(n) {
            let denom = 1.0 + dt * sig;
            let scale = omega * dt / denom;
            apply_lusgs_component_update_typed(self, i, residual, scale, gamma, min_pressure);
        }
        Ok(())
    }
}

impl<T: ComputeFloat> ConservedResidualT<T> {
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

    /// 全场密度残差 L2 范数（`f64` 累加，ADR 0016 §4）。
    #[must_use]
    pub fn density_l2_norm(&self) -> Real {
        l2_norm_real(self.density.values())
    }

    /// 全场密度残差 RMS（`f64` 累加）。
    #[must_use]
    pub fn density_rms_norm(&self) -> Real {
        rms_norm_real(self.density.values())
    }

    /// 五方程守恒残差 RMS（`f64` 累加）。
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
            sum_sq += values
                .iter()
                .map(|v| {
                    let r = v.to_real();
                    r * r
                })
                .sum::<Real>();
        }
        (sum_sq / (5.0 * n as Real)).sqrt()
    }
}

#[must_use]
fn l2_norm_real<T: ComputeFloat>(values: &[T]) -> Real {
    values
        .iter()
        .map(|v| {
            let r = v.to_real();
            r * r
        })
        .sum::<Real>()
        .sqrt()
}

#[must_use]
fn rms_norm_real<T: ComputeFloat>(values: &[T]) -> Real {
    if values.is_empty() {
        return 0.0;
    }
    l2_norm_real(values) / (values.len() as Real).sqrt()
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

fn ensure_dt_size_f32(fields: usize, dt_len: usize) -> Result<()> {
    ensure_dt_size(fields, dt_len)
}

fn axpy_component<T: ComputeFloat>(dst: &mut [T], base: &[T], inc: &[T], scale: Real) {
    for (d, (&b, &r)) in dst.iter_mut().zip(base.iter().zip(inc.iter())) {
        *d = b.add_mul_real(r, scale);
    }
}

fn add_scaled_slice<T: ComputeFloat>(dst: &mut [T], src: &[T], scale: Real) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = d.add_mul_real(s, scale);
    }
}

fn scale_component<T: ComputeFloat>(dst: &mut [T], src: &[T], scale: Real) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = T::from_real(s.to_real() * scale);
    }
}

fn apply_lusgs_component_update_typed<T: ComputeFloat>(
    fields: &mut ConservedFieldsT<T>,
    i: usize,
    residual: &ConservedResidualT<T>,
    scale: Real,
    _gamma: Real,
    _min_pressure: Real,
) {
    fields.density.values_mut()[i] =
        fields.density.values()[i].add_mul_real(residual.density.values()[i], scale);
    fields.momentum_x.values_mut()[i] =
        fields.momentum_x.values()[i].add_mul_real(residual.momentum_x.values()[i], scale);
    fields.momentum_y.values_mut()[i] =
        fields.momentum_y.values()[i].add_mul_real(residual.momentum_y.values()[i], scale);
    fields.momentum_z.values_mut()[i] =
        fields.momentum_z.values()[i].add_mul_real(residual.momentum_z.values()[i], scale);
    fields.total_energy.values_mut()[i] =
        fields.total_energy.values()[i].add_mul_real(residual.total_energy.values()[i], scale);
}

fn combine_rk4_component<T: ComputeFloat>(dst: &mut [T], k1: &[T], k2: &[T], k3: &[T], k4: &[T]) {
    let sixth = 1.0 / 6.0;
    for (d, (&a, (&b, (&c, &e)))) in dst
        .iter_mut()
        .zip(k1.iter().zip(k2.iter().zip(k3.iter().zip(k4.iter()))))
    {
        let sum = a.to_real() + 2.0 * b.to_real() + 2.0 * c.to_real() + e.to_real();
        *d = T::from_real(sixth * sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{ConservedFieldsT, ConservedResidualT};
    use crate::core::approx_eq;
    use crate::field::{ConservedFields, ConservedResidual};

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

    #[test]
    fn f32_assign_axpy_matches_real_arithmetic() {
        let base = ConservedFieldsT::<f32>::uniform(
            1,
            crate::physics::ConservedState {
                density: 1.0,
                momentum: [0.0, 0.0, 0.0],
                total_energy: 2.0,
            },
        )
        .expect("base");
        let mut rhs = ConservedResidualT::<f32>::zeros(1).expect("rhs");
        rhs.density.values_mut()[0] = f32::from_real(4.0);
        let mut out = ConservedFieldsT::<f32>::uniform(
            1,
            crate::physics::ConservedState {
                density: 0.0,
                momentum: [0.0, 0.0, 0.0],
                total_energy: 0.0,
            },
        )
        .expect("out");
        out.assign_axpy(&base, &rhs, 0.5).expect("axpy");
        assert!((out.density.values()[0].to_real() - 3.0).abs() < 1.0e-5);
    }
}
