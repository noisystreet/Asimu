//! 一阶前向 Euler 显式时间推进（用于与 RK4 对照排错）。
//!
//! \(\mathbf{U}^{n+1} = \mathbf{U}^n + \Delta t\,\mathbf{R}(\mathbf{U}^n)\)

use tracing::info_span;

use crate::core::{ComputeFloat, Real};
use crate::error::Result;
use crate::field::{ConservedFieldsT, ConservedResidualT};

use super::common::maybe_enforce_positivity;
use super::rk4::Rk4StorageT;

/// 单步前向 Euler（全局 \(\Delta t\)）。
pub fn euler_step<T, F>(
    fields: &mut ConservedFieldsT<T>,
    storage: &mut Rk4StorageT<T>,
    dt: Real,
    mut evaluate_rhs: F,
    eos: Option<&crate::physics::IdealGasEoS>,
    min_pressure: Real,
) -> Result<()>
where
    T: ComputeFloat,
    F: FnMut(&ConservedFieldsT<T>, &mut ConservedResidualT<T>) -> Result<()>,
{
    let n = fields.num_cells();
    storage.ensure_capacity(n)?;
    maybe_enforce_positivity(fields, eos, min_pressure);
    {
        let _span = info_span!("euler_rhs").entered();
        evaluate_rhs(fields, &mut storage.k1)?;
    }
    {
        let _span = info_span!("euler_update").entered();
        fields.add_axpy(&storage.k1, dt)?;
        maybe_enforce_positivity(fields, eos, min_pressure);
    }
    Ok(())
}

/// 逐单元 \(\Delta t_i\) 的前向 Euler（稳态当地时间步）。
pub fn euler_step_local<T, F>(
    fields: &mut ConservedFieldsT<T>,
    storage: &mut Rk4StorageT<T>,
    dt: &[Real],
    mut evaluate_rhs: F,
    eos: Option<&crate::physics::IdealGasEoS>,
    min_pressure: Real,
) -> Result<()>
where
    T: ComputeFloat,
    F: FnMut(&ConservedFieldsT<T>, &mut ConservedResidualT<T>) -> Result<()>,
{
    let n = fields.num_cells();
    storage.ensure_capacity(n)?;
    if dt.len() != n {
        return Err(crate::error::AsimuError::Solver(format!(
            "euler_step_local: dt 长度 {} 与单元数 {n} 不一致",
            dt.len()
        )));
    }
    maybe_enforce_positivity(fields, eos, min_pressure);
    let gamma = eos.map(|e| e.gamma).unwrap_or(1.4);
    storage.u0.copy_from(fields)?;
    {
        let _span = info_span!("euler_rhs").entered();
        evaluate_rhs(&storage.u0, &mut storage.k1)?;
    }
    {
        let _span = info_span!("euler_update").entered();
        storage
            .stage
            .assign_axpy_dt(&storage.u0, &storage.k1, dt, 1.0, gamma, min_pressure)?;
        fields.copy_from(&storage.stage)?;
        maybe_enforce_positivity(fields, eos, min_pressure);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::field::{ConservedFields, ConservedResidual};
    use crate::physics::ConservedState;
    use crate::solver::time::Rk4Storage;

    #[test]
    fn euler_integrates_linear_decay() {
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
        let dt = 0.5;
        let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
            r.clear();
            for (rv, &val) in r.density.values_mut().iter_mut().zip(u.density.values()) {
                *rv = -lambda * val;
            }
            Ok(())
        };
        euler_step(&mut fields, &mut storage, dt, evaluate, None, 1.0e-6).expect("euler");
        let expected = 0.0;
        assert!(approx_eq(fields.density.values()[0], expected, 1.0e-12));
    }
}
