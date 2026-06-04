//! LU-SGS 隐式伪时间步：阶段 C 对角（默认）或实验性阶段 D 双扫。
//!
//! \((1/\Delta t_i + \sigma_i)\,\partial\mathbf{U}/\partial\tau \approx \mathbf{R}_i\)（\(\mathbf{R}=\mathrm{d}\mathbf{U}/\mathrm{d}t\)）
//! \(\Rightarrow \Delta\mathbf{U}_i = \omega\,\Delta t_i\,\mathbf{R}_i / (1 + \Delta t_i\,\sigma_i)\)，\(\sigma_i=(|u|+a)_i/h_i\)

#![allow(clippy::too_many_arguments)]

use tracing::info_span;

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual};

use super::common::maybe_enforce_positivity;
use super::rk4::Rk4Storage;

/// LU-SGS 伪时间步选项。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LuSgsConfig {
    /// 松弛因子 \(\omega\in(0,1]\)（默认 1）。
    pub omega: Real,
    /// `true`：i/j/k 双扫（阶段 D）；`false`：对角隐式（阶段 C，默认）。
    pub sweep: bool,
    /// 后扫邻居耦合阻尼 \(\in(0,1]\)（默认 0.5），抑制后扫过冲。
    pub sweep_backward_damping: Real,
}

impl Default for LuSgsConfig {
    fn default() -> Self {
        Self {
            omega: 1.0,
            sweep: false,
            sweep_backward_damping: 0.5,
        }
    }
}

impl LuSgsConfig {
    pub fn parse(
        omega: Option<Real>,
        sweep: Option<bool>,
        sweep_backward_damping: Option<Real>,
    ) -> Result<Self> {
        let omega = omega.unwrap_or(1.0);
        if !(0.0 < omega && omega <= 1.0) {
            return Err(AsimuError::Config(
                "[time].lusgs_omega 须在 (0, 1] 内".to_string(),
            ));
        }
        let sweep_backward_damping = sweep_backward_damping.unwrap_or(0.5);
        if !(0.0 < sweep_backward_damping && sweep_backward_damping <= 1.0) {
            return Err(AsimuError::Config(
                "[time].lusgs_sweep_backward_damping 须在 (0, 1] 内".to_string(),
            ));
        }
        Ok(Self {
            omega,
            sweep: sweep.unwrap_or(false),
            sweep_backward_damping,
        })
    }

    pub fn parse_omega(value: Option<Real>) -> Result<Self> {
        Self::parse(value, None, None)
    }
}

/// LU-SGS 步进上下文（合并多参数以满足复杂度门禁）。
pub struct LuSgsStepContext<'a> {
    pub sigma: &'a [Real],
    pub volumes: &'a [Real],
    pub config: &'a LuSgsConfig,
    pub eos: Option<&'a crate::physics::IdealGasEoS>,
    pub min_pressure: Real,
}

/// 单步对角 LU-SGS（全局 \(\Delta t\) + 逐单元 \(\sigma_i\)）。
pub fn lu_sgs_step<F>(
    fields: &mut ConservedFields,
    storage: &mut Rk4Storage,
    dt: Real,
    ctx: &LuSgsStepContext<'_>,
    mut evaluate_rhs: F,
) -> Result<()>
where
    F: FnMut(&ConservedFields, &mut ConservedResidual) -> Result<()>,
{
    let n = fields.num_cells();
    storage.ensure_capacity(n)?;
    ensure_lusgs_lengths(n, ctx.sigma.len(), ctx.volumes.len(), None)?;
    maybe_enforce_positivity(fields, ctx.eos, ctx.min_pressure);
    {
        let _span = info_span!("lu_sgs_rhs").entered();
        evaluate_rhs(fields, &mut storage.k1)?;
    }
    {
        let _span = info_span!("lu_sgs_update").entered();
        fields.assign_lusgs_diagonal_increment(
            &storage.k1,
            ctx.sigma,
            ctx.volumes,
            dt,
            ctx.config.omega,
            ctx.eos.map(|e| e.gamma).unwrap_or(1.4),
            ctx.min_pressure,
        )?;
        maybe_enforce_positivity(fields, ctx.eos, ctx.min_pressure);
    }
    Ok(())
}

/// 逐单元 \(\Delta t_i\) 的 LU-SGS（`config.sweep` 选择双扫或对角）。
pub fn lu_sgs_step_local<F>(
    fields: &mut ConservedFields,
    storage: &mut Rk4Storage,
    dt: &[Real],
    ctx: &LuSgsStepContext<'_>,
    mut evaluate_rhs: F,
) -> Result<()>
where
    F: FnMut(&ConservedFields, &mut ConservedResidual) -> Result<()>,
{
    let n = fields.num_cells();
    storage.ensure_capacity(n)?;
    ensure_lusgs_lengths(n, ctx.sigma.len(), ctx.volumes.len(), Some(dt.len()))?;
    maybe_enforce_positivity(fields, ctx.eos, ctx.min_pressure);
    storage.u0.copy_from(fields)?;
    {
        let _span = info_span!("lu_sgs_rhs").entered();
        evaluate_rhs(&storage.u0, &mut storage.k1)?;
    }
    if ctx.config.sweep {
        let _span = info_span!("lu_sgs_sweep").entered();
        // 扫掠在 compressible 层调用（需网格/边界上下文）；此处不应到达。
        return Err(AsimuError::Solver(
            "lu_sgs_step_local(sweep=true) 须由 lu_sgs_step_sweep_local 调用".to_string(),
        ));
    }
    {
        let _span = info_span!("lu_sgs_diagonal_update").entered();
        storage.stage.assign_lusgs_diagonal_update(
            &storage.u0,
            &storage.k1,
            ctx.sigma,
            dt,
            ctx.config.omega,
            ctx.eos.map(|e| e.gamma).unwrap_or(1.4),
            ctx.min_pressure,
        )?;
        fields.copy_from(&storage.stage)?;
        maybe_enforce_positivity(fields, ctx.eos, ctx.min_pressure);
    }
    Ok(())
}

/// 阶段 D：RHS + 双扫（`sweep` 闭包由 3D 求解器提供网格/边界上下文）。
pub fn lu_sgs_step_sweep_local<F, S>(
    fields: &mut ConservedFields,
    storage: &mut Rk4Storage,
    mut evaluate_rhs: F,
    mut sweep: S,
    eos: Option<&crate::physics::IdealGasEoS>,
    min_pressure: Real,
) -> Result<()>
where
    F: FnMut(&ConservedFields, &mut ConservedResidual) -> Result<()>,
    S: FnMut(&mut ConservedFields, &ConservedResidual) -> Result<()>,
{
    let n = fields.num_cells();
    storage.ensure_capacity(n)?;
    maybe_enforce_positivity(fields, eos, min_pressure);
    storage.u0.copy_from(fields)?;
    {
        let _span = info_span!("lu_sgs_rhs").entered();
        evaluate_rhs(&storage.u0, &mut storage.k1)?;
    }
    {
        let _span = info_span!("lu_sgs_sweep").entered();
        sweep(fields, &storage.k1)?;
    }
    maybe_enforce_positivity(fields, eos, min_pressure);
    Ok(())
}

fn ensure_lusgs_lengths(
    n: usize,
    sigma_len: usize,
    vol_len: usize,
    dt_len: Option<usize>,
) -> Result<()> {
    if sigma_len != n || vol_len != n {
        return Err(AsimuError::Solver(format!(
            "lu_sgs: sigma/volume 长度 {sigma_len}/{vol_len} 与单元数 {n} 不一致"
        )));
    }
    if let Some(dt_n) = dt_len
        && dt_n != n
    {
        return Err(AsimuError::Solver(format!(
            "lu_sgs_step_local: dt 长度 {dt_n} 与单元数 {n} 不一致"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::physics::ConservedState;

    #[test]
    fn lusgs_uniform_rhs_yields_zero_update() {
        let n = 4;
        let state = ConservedState {
            density: 1.2,
            momentum: [0.1, 0.0, 0.0],
            total_energy: 2.5,
        };
        let mut fields = ConservedFields::uniform(n, state).expect("fields");
        let reference = fields.clone();
        let mut storage = Rk4Storage::new(n).expect("storage");
        let sigma = vec![100.0; n];
        let volumes = vec![1.0; n];
        let dt = vec![0.01; n];
        let evaluate = |_u: &ConservedFields, r: &mut ConservedResidual| {
            r.clear();
            Ok(())
        };
        let ctx = LuSgsStepContext {
            sigma: &sigma,
            volumes: &volumes,
            config: &LuSgsConfig {
                sweep: false,
                ..LuSgsConfig::default()
            },
            eos: None,
            min_pressure: 1.0e-6,
        };
        lu_sgs_step_local(&mut fields, &mut storage, &dt, &ctx, evaluate).expect("step");
        assert!(approx_eq(
            fields.density.values()[0],
            reference.density.values()[0],
            1.0e-12
        ));
    }

    #[test]
    fn lusgs_diagonal_implicit_reduces_linear_decay() {
        let n = 1;
        let mut fields = ConservedFields::uniform(
            n,
            ConservedState {
                density: 1.0,
                momentum: [0.0, 0.0, 0.0],
                total_energy: 0.0,
            },
        )
        .expect("fields");
        let mut storage = Rk4Storage::new(n).expect("storage");
        let lambda = 2.0;
        let sigma = vec![1.0];
        let volumes = vec![1.0];
        let dt = 0.5;
        let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
            r.clear();
            for (rv, &val) in r.density.values_mut().iter_mut().zip(u.density.values()) {
                *rv = -lambda * val;
            }
            Ok(())
        };
        let ctx = LuSgsStepContext {
            sigma: &sigma,
            volumes: &volumes,
            config: &LuSgsConfig {
                sweep: false,
                ..LuSgsConfig::default()
            },
            eos: None,
            min_pressure: 1.0e-6,
        };
        lu_sgs_step(&mut fields, &mut storage, dt, &ctx, evaluate).expect("step");
        let explicit_euler = 1.0 + dt * (-lambda);
        let implicit = 1.0 + (dt * (-lambda)) / (1.0 + dt * sigma[0] / volumes[0]);
        assert!(fields.density.values()[0] > explicit_euler);
        assert!((fields.density.values()[0] - implicit).abs() < 1.0e-10);
    }
}
