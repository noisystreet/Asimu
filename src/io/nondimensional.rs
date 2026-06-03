//! 算例无量纲化：在 TOML 解析完成后对 `CaseSpec` 做一致缩放。
//!
//! 理论：[`docs/theory/nondimensional.md`](../../docs/theory/nondimensional.md) §1、§6（`apply_nondimensionalization`）。

use crate::boundary::{BoundaryKind, BoundarySet, WallHeat};
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::physics::{FreestreamParams, ReferenceScales, ViscousPhysicsConfig};

use super::CaseSpec;

fn scale_freestream(fs: &mut FreestreamParams, reference: &ReferenceScales, gamma: Real) {
    let _ = reference;
    fs.pressure = 1.0 / gamma;
    fs.temperature = 1.0;
}

fn scale_boundary_set(boundary: &mut BoundarySet, reference: &ReferenceScales) {
    for patch in boundary.patches_mut() {
        patch.kind = match &patch.kind {
            BoundaryKind::Farfield {
                mach,
                pressure,
                temperature,
                alpha,
                beta,
            } => BoundaryKind::Farfield {
                mach: *mach,
                pressure: pressure / reference.pressure,
                temperature: temperature / reference.temperature,
                alpha: *alpha,
                beta: *beta,
            },
            BoundaryKind::Inlet {
                total_pressure,
                total_temperature,
                velocity_direction,
                supersonic,
                mach,
            } => BoundaryKind::Inlet {
                total_pressure: total_pressure / reference.pressure,
                total_temperature: total_temperature / reference.temperature,
                velocity_direction: *velocity_direction,
                supersonic: *supersonic,
                mach: *mach,
            },
            BoundaryKind::TurbulentInlet {
                total_pressure,
                total_temperature,
                velocity_direction,
                turbulent_k,
                turbulent_omega,
            } => BoundaryKind::TurbulentInlet {
                total_pressure: total_pressure / reference.pressure,
                total_temperature: total_temperature / reference.temperature,
                velocity_direction: *velocity_direction,
                turbulent_k: *turbulent_k,
                turbulent_omega: *turbulent_omega,
            },
            BoundaryKind::Outlet {
                static_pressure,
                supersonic,
            } => BoundaryKind::Outlet {
                static_pressure: static_pressure / reference.pressure,
                supersonic: *supersonic,
            },
            BoundaryKind::Wall { no_slip, heat } => BoundaryKind::Wall {
                no_slip: *no_slip,
                heat: scale_wall_heat(heat, reference),
            },
            other => other.clone(),
        };
    }
}

fn scale_wall_heat(heat: &WallHeat, reference: &ReferenceScales) -> WallHeat {
    match heat {
        WallHeat::Adiabatic => WallHeat::Adiabatic,
        WallHeat::Isothermal { temperature } => WallHeat::Isothermal {
            temperature: temperature / reference.temperature,
        },
        WallHeat::HeatFlux { flux } => WallHeat::HeatFlux {
            flux: flux / reference.heat_flux_scale(),
        },
    }
}

/// 将可压缩算例切换为 \(*\) 变量求解（原地修改 `case`）。
pub fn apply_nondimensionalization(
    case: &mut CaseSpec,
    mut reference: ReferenceScales,
) -> Result<()> {
    reference.dimensional_gas_constant = case.physics.eos()?.gas_constant;

    let inv_length = 1.0 / reference.length;
    if (inv_length - 1.0).abs() > Real::EPSILON {
        case.mesh.scale_coordinates(inv_length)?;
    }

    if let Some(eos) = &mut case.physics.eos {
        eos.gas_constant = reference.nondimensional_gas_constant();
    }

    if let Some(fs) = &mut case.freestream {
        scale_freestream(fs, &reference, case.physics.eos()?.gamma);
    }
    if let Some(fs) = &mut case.fluid_initial.freestream {
        scale_freestream(fs, &reference, case.physics.eos()?.gamma);
    }

    scale_boundary_set(&mut case.boundary, &reference);

    if let Some(viscous) = &mut case.physics.viscous {
        configure_nondimensional_viscous(viscous, &reference);
    }

    case.reference = Some(reference);
    Ok(())
}

fn configure_nondimensional_viscous(
    viscous: &mut ViscousPhysicsConfig,
    reference: &ReferenceScales,
) {
    viscous.inv_reynolds = reference.inv_reynolds();
    viscous.viscosity_ref = Some(reference.viscosity);
    viscous.temperature_ref = Some(reference.temperature);
}

/// 解析 `[nondimensional]` 并在可压缩算例上应用缩放。
pub(super) fn maybe_apply_nondimensionalization(case: &mut CaseSpec, enabled: bool) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    if !case.is_compressible() {
        return Err(AsimuError::Config(
            "[nondimensional] 仅适用于可压缩算例（须配置 gamma 与 gas_constant）".to_string(),
        ));
    }
    let fs = case
        .freestream
        .or(case.fluid_initial.freestream)
        .ok_or_else(|| {
            AsimuError::Config("[nondimensional] 可压缩算例须指定 [freestream]".to_string())
        })?;
    let reference =
        ReferenceScales::from_freestream(&case.physics.eos()?, &fs, case.physics.viscous.as_ref())?;
    apply_nondimensionalization(case, reference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundaryKind;
    use crate::io::parse_case_str;

    #[test]
    fn compressible_case_defaults_to_nondimensional_without_section() {
        let case = parse_case_str(
            r#"
name = "nd_default"
[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
[physics]
gamma = 1.4
gas_constant = 287.0
prandtl = 0.72
[freestream]
pressure = 1000.0
temperature = 300.0
[euler]
flux = "hllc"
"#,
        )
        .expect("parse");
        assert!(case.reference.is_some());
        let fs = case.freestream.expect("fs");
        let eos = case.physics.eos().expect("eos");
        assert!((fs.pressure - 1.0 / eos.gamma).abs() < 1.0e-6);
    }

    #[test]
    fn nondimensional_can_be_disabled_explicitly() {
        let case = parse_case_str(
            r#"
name = "dim"
[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
pressure = 1000.0
temperature = 300.0
[euler]
flux = "hllc"
[nondimensional]
enabled = false
"#,
        )
        .expect("parse");
        assert!(case.reference.is_none());
        assert!((case.freestream.expect("fs").pressure - 1000.0).abs() < 1.0e-6);
    }

    #[test]
    fn nondimensional_freestream_pressure_is_one_over_gamma() {
        let case_str = r#"
name = "nd"
benchmark_id = "nd"

[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
lx = 1.0
ly = 1.0
lz = 1.0

[physics]
gamma = 1.4
gas_constant = 287.052871936417

[freestream]
mach = 2.0
pressure = 1000.0
temperature = 300.0

[euler]
flux = "hllc"

[nondimensional]
enabled = true
"#;
        let case = parse_case_str(case_str).expect("parse");
        let fs = case.freestream.expect("fs");
        let eos = case.physics.eos().expect("eos");
        assert!((fs.temperature - 1.0).abs() < 1.0e-12);
        assert!((fs.pressure - 1.0 / eos.gamma).abs() < 1.0e-6);
        assert!(case.reference.is_some());
    }

    #[test]
    fn wall_temperature_scales_with_reference() {
        let case = parse_case_str(
            r#"
name = "wall"
[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
[physics]
gamma = 1.4
gas_constant = 287.0
prandtl = 0.72
[freestream]
pressure = 1000.0
temperature = 300.0
[navier_stokes]
flux = "hllc"
[boundary.i_min]
kind = "wall"
no_slip = true
heat = "isothermal"
wall_temperature = 600.0
[nondimensional]
enabled = true
"#,
        )
        .expect("parse");
        let wall = &case.boundary.patches()[0].kind;
        assert!(matches!(
            wall,
            BoundaryKind::Wall {
                heat: WallHeat::Isothermal { temperature },
                ..
            } if (*temperature - 2.0).abs() < 1.0e-12
        ));
    }
}
