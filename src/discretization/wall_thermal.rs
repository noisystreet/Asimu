//! 壁面热边界：ghost 温度与粘性能量方程中的传导通量。

use crate::boundary::WallHeat;
use crate::core::Real;
use crate::discretization::gradient::VelocityGradient;
use crate::discretization::viscous::average_gradient_for_wall;
use crate::error::Result;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

/// 壁面 ghost 温度（线性面心插值 \(T_f=\frac{1}{2}(T_o+T_g)\)）。
///
/// `spacing` 为 owner 单元中心到边界面距离；`heat_flux` 为**进入流体**的热流密度 (W/m²)。
pub fn wall_ghost_temperature(
    t_owner: Real,
    heat: WallHeat,
    spacing: Real,
    viscous: Option<&ViscousPhysicsConfig>,
    eos: &IdealGasEoS,
) -> Result<Real> {
    match heat {
        WallHeat::Adiabatic => Ok(t_owner),
        WallHeat::Isothermal { temperature } => Ok(2.0 * temperature - t_owner),
        WallHeat::HeatFlux { flux } => {
            let viscous = viscous.ok_or_else(|| {
                crate::error::AsimuError::Boundary(
                    "壁面 heat_flux 须启用 [navier_stokes] 粘性物性".to_string(),
                )
            })?;
            let lambda = viscous
                .model
                .thermal_conductivity(t_owner, eos, viscous.prandtl)?;
            if lambda <= Real::EPSILON {
                return Err(crate::error::AsimuError::Boundary(
                    "壁面热流 BC：热导率无效".to_string(),
                ));
            }
            if spacing <= Real::EPSILON {
                return Err(crate::error::AsimuError::Boundary(
                    "壁面热流 BC：面间距无效".to_string(),
                ));
            }
            // \dot{q}_{\mathrm{into\,fluid}} = \lambda\,(\nabla T\cdot\mathbf{n})
            // \approx \lambda\,(T_g-T_o)/(2\delta)，\delta=\texttt{spacing}
            Ok(t_owner + 2.0 * spacing * flux / lambda)
        }
    }
}

/// 壁面 Fourier 传导项（进入流体的能量通量贡献，与 `viscous_face_flux` 能量分量同号）。
#[must_use]
pub fn wall_face_conduction(
    grad_owner: &VelocityGradient,
    grad_ghost: &VelocityGradient,
    normal: crate::core::Vector3,
    lambda: Real,
    wall_heat: WallHeat,
) -> Real {
    match wall_heat {
        WallHeat::Adiabatic => 0.0,
        WallHeat::HeatFlux { flux } => flux,
        WallHeat::Isothermal { .. } => {
            let grad = average_gradient_for_wall(grad_owner, grad_ghost);
            lambda * (grad.dt[0] * normal.x + grad.dt[1] * normal.y + grad.dt[2] * normal.z)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::gradient::VelocityGradient;
    use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

    #[test]
    fn adiabatic_wall_conduction_is_zero() {
        let grad = VelocityGradient {
            du: [0.0; 3],
            dv: [1.0, 0.0, 0.0],
            dw: [0.0; 3],
            dt: [10.0, 0.0, 0.0],
        };
        let q = wall_face_conduction(
            &grad,
            &grad,
            crate::core::Vector3::new(0.0, 1.0, 0.0),
            0.025,
            WallHeat::Adiabatic,
        );
        assert_eq!(q, 0.0);
    }

    #[test]
    fn heat_flux_wall_uses_prescribed_value() {
        let grad = VelocityGradient {
            du: [0.0; 3],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let q = wall_face_conduction(
            &grad,
            &grad,
            crate::core::Vector3::new(1.0, 0.0, 0.0),
            0.025,
            WallHeat::HeatFlux { flux: 500.0 },
        );
        assert_eq!(q, 500.0);
    }

    #[test]
    fn heat_flux_ghost_temperature_matches_fourier_relation() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous = ViscousPhysicsConfig::default();
        let t_owner = 300.0;
        let spacing = 0.1;
        let flux = 1000.0;
        let t_g = wall_ghost_temperature(
            t_owner,
            WallHeat::HeatFlux { flux },
            spacing,
            Some(&viscous),
            &eos,
        )
        .expect("t_g");
        let lambda = viscous
            .model
            .thermal_conductivity(t_owner, &eos, viscous.prandtl)
            .expect("lambda");
        let grad_n = (t_g - t_owner) / (2.0 * spacing);
        assert!((lambda * grad_n - flux).abs() < 1.0e-6);
    }
}
