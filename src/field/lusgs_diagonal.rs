//! LU-SGS 对角隐式更新（含双时间步分母扩展）。

use crate::core::{ComputeFloat, Real};
use crate::error::{AsimuError, Result};

use super::{ConservedFieldsT, ConservedResidualT};

/// LU-SGS 对角更新标量系数（含双时间步 \(1/\Delta t_{\mathrm{phys}}\)）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LusgsDiagonalCoeffs {
    pub omega: Real,
    pub gamma: Real,
    pub min_pressure: Real,
    pub inv_dt_phys: Real,
}

impl LusgsDiagonalCoeffs {
    #[must_use]
    pub const fn steady_pseudo_time(omega: Real, gamma: Real, min_pressure: Real) -> Self {
        Self {
            omega,
            gamma,
            min_pressure,
            inv_dt_phys: 0.0,
        }
    }

    #[must_use]
    pub const fn with_inv_dt_phys(self, inv_dt_phys: Real) -> Self {
        Self {
            inv_dt_phys,
            ..self
        }
    }
}

/// LU-SGS 对角更新精度后端（仅 crate 内为 `f32` / `f64` 实现）。
pub trait LusgsDiagonalUpdateBackend: ComputeFloat {
    fn assign_lusgs_diagonal_update_impl(
        out: &mut ConservedFieldsT<Self>,
        base: &ConservedFieldsT<Self>,
        residual: &ConservedResidualT<Self>,
        sigma: &[Real],
        dt: &[Real],
        coeffs: LusgsDiagonalCoeffs,
    ) -> Result<()>;
}

impl LusgsDiagonalUpdateBackend for f64 {
    fn assign_lusgs_diagonal_update_impl(
        out: &mut ConservedFieldsT<Self>,
        base: &ConservedFieldsT<Self>,
        residual: &ConservedResidualT<Self>,
        sigma: &[Real],
        dt: &[Real],
        coeffs: LusgsDiagonalCoeffs,
    ) -> Result<()> {
        validate_lusgs_diagonal_update_args(
            out.num_cells(),
            base,
            residual,
            sigma,
            dt,
            coeffs.omega,
        )?;
        let scale = prepare_lusgs_diagonal_scales(
            base.num_cells(),
            sigma,
            dt,
            coeffs.omega,
            coeffs.inv_dt_phys,
        )?;
        crate::exec::cpu::assign_lusgs_diagonal_update(crate::exec::cpu::LusgsDiagonalUpdate {
            out: crate::exec::cpu::ConservedSoAMut {
                rho: out.density.values_mut(),
                mx: out.momentum_x.values_mut(),
                my: out.momentum_y.values_mut(),
                mz: out.momentum_z.values_mut(),
                energy: out.total_energy.values_mut(),
            },
            base: crate::exec::cpu::ConservedSoA {
                rho: base.density.values(),
                mx: base.momentum_x.values(),
                my: base.momentum_y.values(),
                mz: base.momentum_z.values(),
                energy: base.total_energy.values(),
            },
            residual: crate::exec::cpu::ConservedSoA {
                rho: residual.density.values(),
                mx: residual.momentum_x.values(),
                my: residual.momentum_y.values(),
                mz: residual.momentum_z.values(),
                energy: residual.total_energy.values(),
            },
            scale: &scale,
        });
        let _ = (coeffs.gamma, coeffs.min_pressure);
        Ok(())
    }
}

impl LusgsDiagonalUpdateBackend for f32 {
    fn assign_lusgs_diagonal_update_impl(
        out: &mut ConservedFieldsT<Self>,
        base: &ConservedFieldsT<Self>,
        residual: &ConservedResidualT<Self>,
        sigma: &[Real],
        dt: &[Real],
        coeffs: LusgsDiagonalCoeffs,
    ) -> Result<()> {
        validate_lusgs_diagonal_update_args(
            out.num_cells(),
            base,
            residual,
            sigma,
            dt,
            coeffs.omega,
        )?;
        let n = base.num_cells();
        for (i, &dt_i) in dt.iter().enumerate().take(n) {
            let scale = coeffs.omega * dt_i / (1.0 + dt_i * sigma[i] + dt_i * coeffs.inv_dt_phys);
            out.density.values_mut()[i] =
                base.density.values()[i].add_mul_real(residual.density.values()[i], scale);
            out.momentum_x.values_mut()[i] =
                base.momentum_x.values()[i].add_mul_real(residual.momentum_x.values()[i], scale);
            out.momentum_y.values_mut()[i] =
                base.momentum_y.values()[i].add_mul_real(residual.momentum_y.values()[i], scale);
            out.momentum_z.values_mut()[i] =
                base.momentum_z.values()[i].add_mul_real(residual.momentum_z.values()[i], scale);
            out.total_energy.values_mut()[i] = base.total_energy.values()[i]
                .add_mul_real(residual.total_energy.values()[i], scale);
        }
        let _ = (coeffs.gamma, coeffs.min_pressure);
        Ok(())
    }
}

/// f32 对角 LU-SGS 系数（原生 f32 \(\sigma,\Delta t\)）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LusgsDiagonalCoeffsF32 {
    pub omega: f32,
    pub inv_dt_phys: f32,
}

pub fn assign_lusgs_diagonal_update_f32(
    out: &mut ConservedFieldsT<f32>,
    base: &ConservedFieldsT<f32>,
    residual: &ConservedResidualT<f32>,
    sigma: &[f32],
    dt: &[f32],
    coeffs: LusgsDiagonalCoeffsF32,
) -> Result<()> {
    validate_lusgs_diagonal_update_args_f32(
        out.num_cells(),
        base,
        residual,
        sigma,
        dt,
        coeffs.omega,
    )?;
    let n = base.num_cells();
    for (i, &dt_i) in dt.iter().enumerate().take(n) {
        let scale = coeffs.omega * dt_i / (1.0 + dt_i * sigma[i] + dt_i * coeffs.inv_dt_phys);
        out.density.values_mut()[i] =
            base.density.values()[i] + residual.density.values()[i] * scale;
        out.momentum_x.values_mut()[i] =
            base.momentum_x.values()[i] + residual.momentum_x.values()[i] * scale;
        out.momentum_y.values_mut()[i] =
            base.momentum_y.values()[i] + residual.momentum_y.values()[i] * scale;
        out.momentum_z.values_mut()[i] =
            base.momentum_z.values()[i] + residual.momentum_z.values()[i] * scale;
        out.total_energy.values_mut()[i] =
            base.total_energy.values()[i] + residual.total_energy.values()[i] * scale;
    }
    Ok(())
}

impl<T: ComputeFloat + LusgsDiagonalUpdateBackend> ConservedFieldsT<T> {
    /// 对角 LU-SGS：`self ← base + ω·Δt_i·R / (1 + Δt_i·σ_i + Δt_i/Δt_phys)`。
    pub fn assign_lusgs_diagonal_update(
        &mut self,
        base: &Self,
        residual: &ConservedResidualT<T>,
        sigma: &[Real],
        dt: &[Real],
        coeffs: LusgsDiagonalCoeffs,
    ) -> Result<()> {
        T::assign_lusgs_diagonal_update_impl(self, base, residual, sigma, dt, coeffs)
    }
}

fn validate_lusgs_diagonal_update_args_f32(
    out_cells: usize,
    base: &ConservedFieldsT<f32>,
    residual: &ConservedResidualT<f32>,
    sigma: &[f32],
    dt: &[f32],
    omega: f32,
) -> Result<()> {
    ensure_same_size(out_cells, base.num_cells())?;
    ensure_residual_size(base.num_cells(), residual.num_cells())?;
    ensure_dt_size_f32(base.num_cells(), dt.len())?;
    if sigma.len() != base.num_cells() {
        return Err(AsimuError::Field(
            "lu_sgs: sigma 与场单元数不一致".to_string(),
        ));
    }
    if omega <= 0.0 {
        return Err(AsimuError::Field("lu_sgs: omega 须为正".to_string()));
    }
    for (i, &dt_i) in dt.iter().enumerate().take(base.num_cells()) {
        if dt_i <= 0.0 {
            return Err(AsimuError::Field(format!("lu_sgs: 单元 {i} 的 Δt 须为正")));
        }
    }
    Ok(())
}

fn validate_lusgs_diagonal_update_args<T: ComputeFloat>(
    out_cells: usize,
    base: &ConservedFieldsT<T>,
    residual: &ConservedResidualT<T>,
    sigma: &[Real],
    dt: &[Real],
    omega: Real,
) -> Result<()> {
    ensure_same_size(out_cells, base.num_cells())?;
    ensure_residual_size(base.num_cells(), residual.num_cells())?;
    ensure_dt_size(base.num_cells(), dt.len())?;
    if sigma.len() != base.num_cells() {
        return Err(AsimuError::Field(
            "lu_sgs: sigma 与场单元数不一致".to_string(),
        ));
    }
    if omega <= 0.0 {
        return Err(AsimuError::Field("lu_sgs: omega 须为正".to_string()));
    }
    for (i, &dt_i) in dt.iter().enumerate().take(base.num_cells()) {
        if dt_i <= 0.0 {
            return Err(AsimuError::Field(format!("lu_sgs: 单元 {i} 的 Δt 须为正")));
        }
    }
    Ok(())
}

fn prepare_lusgs_diagonal_scales(
    n: usize,
    sigma: &[Real],
    dt: &[Real],
    omega: Real,
    inv_dt_phys: Real,
) -> Result<Vec<Real>> {
    let mut scale = vec![0.0; n];
    for (i, &dt_i) in dt.iter().enumerate().take(n) {
        scale[i] = omega * dt_i / (1.0 + dt_i * sigma[i] + dt_i * inv_dt_phys);
    }
    Ok(scale)
}

fn ensure_dt_size_f32(fields: usize, dt_len: usize) -> Result<()> {
    if fields != dt_len {
        return Err(AsimuError::Field(format!(
            "逐单元 dt 长度 {dt_len} 与场单元数 {fields} 不一致"
        )));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{ConservedFields, ConservedFieldsT, ConservedResidual, ConservedResidualT};

    fn approx_eq(a: Real, b: Real, tol: Real) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn f64_lusgs_diagonal_update_matches_reference() {
        let n = 5;
        let base = ConservedFields::uniform(
            n,
            crate::physics::ConservedState {
                density: 1.0,
                momentum: [0.1, 0.0, 0.0],
                total_energy: 2.5,
            },
        )
        .expect("base");
        let mut residual = ConservedResidual::zeros(n).expect("residual");
        residual.density.values_mut()[2] = 0.5;
        residual.momentum_x.values_mut()[2] = 0.2;
        let sigma = vec![10.0; n];
        let dt = vec![0.01; n];
        let scale = dt[2] / (1.0 + dt[2] * sigma[2]);
        let mut out = base.clone();
        out.assign_lusgs_diagonal_update(
            &base,
            &residual,
            &sigma,
            &dt,
            LusgsDiagonalCoeffs::steady_pseudo_time(1.0, 1.4, 1.0e-6),
        )
        .expect("update");
        assert!(approx_eq(
            out.density.values()[2],
            base.density.values()[2] + scale * residual.density.values()[2],
            1.0e-12,
        ));
    }

    #[test]
    fn f32_lusgs_diagonal_update_matches_reference() {
        let n = 3;
        let base = ConservedFieldsT::<f32>::uniform(
            n,
            crate::physics::ConservedState {
                density: 1.0,
                momentum: [0.1, 0.0, 0.0],
                total_energy: 2.5,
            },
        )
        .expect("base");
        let mut residual = ConservedResidualT::<f32>::zeros(n).expect("residual");
        residual.density.values_mut()[1] = f32::from_real(0.4);
        let sigma = vec![8.0; n];
        let dt = vec![0.02; n];
        let scale = dt[1] / (1.0 + dt[1] * sigma[1]);
        let mut out = base.clone();
        out.assign_lusgs_diagonal_update(
            &base,
            &residual,
            &sigma,
            &dt,
            LusgsDiagonalCoeffs::steady_pseudo_time(1.0, 1.4, 1.0e-6),
        )
        .expect("update");
        let expected =
            base.density.values()[1].to_real() + scale * residual.density.values()[1].to_real();
        assert!((out.density.values()[1].to_real() - expected).abs() < 1.0e-5);
    }

    #[test]
    fn dual_time_inv_dt_phys_reduces_scale() {
        let n = 1;
        let base = ConservedFields::uniform(
            n,
            crate::physics::ConservedState {
                density: 1.0,
                momentum: [0.0; 3],
                total_energy: 2.5,
            },
        )
        .expect("base");
        let mut residual = ConservedResidual::zeros(n).expect("residual");
        residual.density.values_mut()[0] = 1.0;
        let sigma = vec![2.0];
        let dt = vec![0.1];
        let mut steady = base.clone();
        steady
            .assign_lusgs_diagonal_update(
                &base,
                &residual,
                &sigma,
                &dt,
                LusgsDiagonalCoeffs::steady_pseudo_time(1.0, 1.4, 1.0e-6),
            )
            .expect("steady");
        let mut dual = base.clone();
        dual.assign_lusgs_diagonal_update(
            &base,
            &residual,
            &sigma,
            &dt,
            LusgsDiagonalCoeffs::steady_pseudo_time(1.0, 1.4, 1.0e-6).with_inv_dt_phys(10.0),
        )
        .expect("dual");
        assert!(dual.density.values()[0] > base.density.values()[0]);
        assert!(dual.density.values()[0] < steady.density.values()[0]);
    }
}
