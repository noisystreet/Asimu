//! 可压缩双时间步：物理存储项与内外循环配置（理论见 `docs/theory/dual_time_stepping.md`）。

use crate::core::{ComputeFloat, Real, log10_positive};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT};
use crate::physics::ConservedState;

/// 双时间步配置（Parse → Validate；无隐式全局状态）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualTimeConfig {
    /// 物理时间步 \(\Delta t_{\mathrm{phys}}\)（须为正）。
    pub dt_phys: Real,
    /// 每物理步伪时间迭代上限。
    pub max_inner_steps: u32,
    /// 内层 \(\log_{10}\|R_{\mathrm{eff}}\|_{\mathrm{rms}}\) 早停阈值；`None` 表示仅依 `max_inner_steps`。
    pub inner_log10_tolerance: Option<Real>,
}

impl DualTimeConfig {
    /// 解析 `[time]` 双时间步字段；`scheme = dual_time` 时由 case 层调用。
    pub fn parse(
        dt_phys: Option<Real>,
        max_inner_steps: Option<u32>,
        inner_tolerance: Option<Real>,
    ) -> Result<Self> {
        let dt_phys = dt_phys.filter(|v| *v > 0.0).ok_or_else(|| {
            AsimuError::Config(
                "time.scheme = \"dual_time\" 须设置正数 [time].dt 作为物理时间步".to_string(),
            )
        })?;
        let max_inner_steps = max_inner_steps.unwrap_or(30);
        if max_inner_steps == 0 {
            return Err(AsimuError::Config(
                "[time].max_inner_steps 须大于 0".to_string(),
            ));
        }
        Ok(Self {
            dt_phys,
            max_inner_steps,
            inner_log10_tolerance: inner_tolerance,
        })
    }

    /// \(1/\Delta t_{\mathrm{phys}}\)，用于 LU-SGS 分母扩展。
    #[must_use]
    pub fn inv_dt_phys(self) -> Real {
        1.0 / self.dt_phys
    }

    /// 内层 \(\|R_{\mathrm{eff}}\|_{\mathrm{rms}}\) 是否满足 log₁₀ 容差。
    #[must_use]
    pub fn inner_converged(self, effective_residual_rms: Real) -> bool {
        self.inner_log10_tolerance
            .is_some_and(|tol| log10_positive(effective_residual_rms) <= tol)
    }
}

/// 物理步内状态：冻结 \(U^n\) 与内层计数。
#[derive(Debug, Clone, PartialEq)]
pub struct DualTimeState<T: ComputeFloat> {
    pub u_at_physical_level: ConservedFieldsT<T>,
    pub inner_iterations: u32,
}

impl<T: ComputeFloat> DualTimeState<T> {
    pub fn new(num_cells: usize) -> Result<Self> {
        Ok(Self {
            u_at_physical_level: ConservedFieldsT::uniform(
                num_cells,
                ConservedState {
                    density: 0.0,
                    momentum: [0.0; 3],
                    total_energy: 0.0,
                },
            )?,
            inner_iterations: 0,
        })
    }

    /// 物理步初冻结 \(U^n\)。
    pub fn snapshot_u_n(&mut self, fields: &ConservedFieldsT<T>) -> Result<()> {
        self.u_at_physical_level.copy_from(fields)?;
        self.inner_iterations = 0;
        Ok(())
    }
}

/// 叠加 BDF1 物理存储项（式 (3)–(4)，与 `R_i=\mathrm dU_i/\mathrm dt` 同量纲）：
/// \(R_{\mathrm{eff},i} \leftarrow R_i - (U_i-U^n_i)/\Delta t_{\mathrm{phys}}\)。
///
/// `ConservedFields` 为单元平均守恒密度；`R_i` 已由 FVM 除以 \(V_i\)，存储项不再除 \(V_i\)。
pub fn add_physical_storage_residual<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    fields: &ConservedFieldsT<T>,
    u_at_level_n: &ConservedFieldsT<T>,
    dt_phys: Real,
) -> Result<()> {
    let n = residual.num_cells();
    if fields.num_cells() != n || u_at_level_n.num_cells() != n {
        return Err(AsimuError::Field(
            "dual_time: 场与残差单元数不一致".to_string(),
        ));
    }
    if dt_phys <= 0.0 {
        return Err(AsimuError::Field("dual_time: dt_phys 须为正".to_string()));
    }
    let inv_dt_phys = 1.0 / dt_phys;
    for i in 0..n {
        subtract_storage_component(
            residual.density.values_mut(),
            fields.density.values(),
            u_at_level_n.density.values(),
            i,
            inv_dt_phys,
        );
        subtract_storage_component(
            residual.momentum_x.values_mut(),
            fields.momentum_x.values(),
            u_at_level_n.momentum_x.values(),
            i,
            inv_dt_phys,
        );
        subtract_storage_component(
            residual.momentum_y.values_mut(),
            fields.momentum_y.values(),
            u_at_level_n.momentum_y.values(),
            i,
            inv_dt_phys,
        );
        subtract_storage_component(
            residual.momentum_z.values_mut(),
            fields.momentum_z.values(),
            u_at_level_n.momentum_z.values(),
            i,
            inv_dt_phys,
        );
        subtract_storage_component(
            residual.total_energy.values_mut(),
            fields.total_energy.values(),
            u_at_level_n.total_energy.values(),
            i,
            inv_dt_phys,
        );
    }
    Ok(())
}

#[inline]
fn subtract_storage_component<T: ComputeFloat>(
    residual: &mut [T],
    field: &[T],
    u_n: &[T],
    cell: usize,
    inv_dt_phys: Real,
) {
    let diff = field[cell].add_mul_real(u_n[cell], -1.0);
    residual[cell] = residual[cell].add_mul_real(diff, -inv_dt_phys);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{ConservedFieldsT, ConservedResidualT};
    use crate::physics::ConservedState;

    fn zero_fields<T: ComputeFloat>(n: usize) -> ConservedFieldsT<T> {
        ConservedFieldsT::uniform(
            n,
            ConservedState {
                density: 0.0,
                momentum: [0.0; 3],
                total_energy: 0.0,
            },
        )
        .expect("fields")
    }

    fn assert_component(residual: &ConservedResidualT<f64>, cell: usize, expected_rho: Real) {
        assert!(
            (residual.density.values()[cell].to_real() - expected_rho).abs() < 1.0e-12,
            "rho residual mismatch"
        );
    }

    #[test]
    fn f64_storage_subtracts_from_spatial_residual() {
        let mut fields = zero_fields::<f64>(1);
        let mut u_n = zero_fields::<f64>(1);
        fields.density.values_mut()[0] = 2.0;
        u_n.density.values_mut()[0] = 1.0;
        let mut residual = ConservedResidualT::<f64>::zeros(1).expect("res");
        residual.density.values_mut()[0] = -0.5;
        add_physical_storage_residual(&mut residual, &fields, &u_n, 0.1).expect("add");
        // -0.5 - (2-1)/0.1 = -10.5
        assert_component(&residual, 0, -10.5);
    }

    #[test]
    fn f32_storage_matches_f64_reference() {
        let mut fields = zero_fields::<f32>(2);
        let mut u_n = zero_fields::<f32>(2);
        fields.density.values_mut()[0] = 1.5_f32;
        fields.density.values_mut()[1] = 3.0_f32;
        u_n.density.values_mut()[0] = 1.0_f32;
        u_n.density.values_mut()[1] = 1.0_f32;
        let mut residual = ConservedResidualT::<f32>::zeros(2).expect("res");
        residual.density.values_mut()[0] = 0.1_f32;
        residual.density.values_mut()[1] = -0.2_f32;
        add_physical_storage_residual(&mut residual, &fields, &u_n, 0.5).expect("add");
        // 0.1 - (1.5-1.0)/0.5 = -0.9；-0.2 - (3.0-1.0)/0.5 = -4.2
        assert!((residual.density.values()[0].to_real() - (-0.9)).abs() < 1.0e-5);
        assert!((residual.density.values()[1].to_real() - (-4.2)).abs() < 1.0e-5);
    }

    #[test]
    fn parse_rejects_missing_dt() {
        let err = DualTimeConfig::parse(None, Some(10), Some(-3.0)).unwrap_err();
        assert!(err.to_string().contains("dt"));
    }

    #[test]
    fn inner_converged_respects_log10_tolerance() {
        let cfg = DualTimeConfig {
            dt_phys: 1.0e-4,
            max_inner_steps: 5,
            inner_log10_tolerance: Some(-3.0),
        };
        assert!(cfg.inner_converged(1.0e-4));
        assert!(!cfg.inner_converged(1.0e-2));
    }
}
