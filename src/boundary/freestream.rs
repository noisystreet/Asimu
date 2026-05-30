//! 用 `[freestream]` 填充可压缩边界 patch 参数。

use crate::error::Result;
use crate::physics::{FreestreamParams, IdealGasEoS};

use super::kind::BoundaryKind;
use super::patch::BoundarySet;

impl BoundarySet {
    /// 将来流静参数写入 inlet/outlet/farfield patch（CGNS 默认 BC 占位值替换）。
    pub fn apply_freestream(&mut self, fs: &FreestreamParams, eos: &IdealGasEoS) -> Result<()> {
        let direction = fs.effective_direction();
        let total_pressure = eos.stagnation_pressure(fs.pressure, fs.mach)?;
        let total_temperature = eos.stagnation_temperature(fs.temperature, fs.mach);
        for patch in self.patches_mut() {
            patch.kind = match &patch.kind {
                BoundaryKind::Farfield { .. } => BoundaryKind::Farfield {
                    mach: fs.mach,
                    pressure: fs.pressure,
                    temperature: fs.temperature,
                    alpha: fs.alpha,
                    beta: fs.beta,
                },
                BoundaryKind::Inlet { .. } => BoundaryKind::Inlet {
                    total_pressure,
                    total_temperature,
                    velocity_direction: direction,
                    mach: fs.mach,
                },
                BoundaryKind::TurbulentInlet {
                    turbulent_k,
                    turbulent_omega,
                    ..
                } => BoundaryKind::TurbulentInlet {
                    total_pressure,
                    total_temperature,
                    velocity_direction: direction,
                    turbulent_k: *turbulent_k,
                    turbulent_omega: *turbulent_omega,
                },
                BoundaryKind::Outlet { supersonic, .. } => BoundaryKind::Outlet {
                    static_pressure: fs.pressure,
                    supersonic: *supersonic,
                },
                _ => continue,
            };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundaryPatch;

    #[test]
    fn apply_freestream_updates_inlet_and_outlet() {
        let mut set = BoundarySet::new(vec![
            BoundaryPatch::new(
                "in",
                vec![],
                BoundaryKind::Inlet {
                    total_pressure: 101_325.0,
                    total_temperature: 300.0,
                    velocity_direction: [1.0, 0.0, 0.0],
                    mach: 0.0,
                },
            ),
            BoundaryPatch::new(
                "out",
                vec![],
                BoundaryKind::Outlet {
                    static_pressure: 101_325.0,
                    supersonic: false,
                },
            ),
        ]);
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 8.0,
            pressure: 1000.0,
            temperature: 300.0,
            ..FreestreamParams::default()
        };
        set.apply_freestream(&fs, &eos).expect("apply");
        let inlet = &set.patches()[0].kind;
        let outlet = &set.patches()[1].kind;
        assert!(matches!(
            inlet,
            BoundaryKind::Inlet {
                total_pressure,
                total_temperature,
                ..
            } if *total_pressure > 100_000.0 && *total_temperature > 3000.0
        ));
        assert!(matches!(
            outlet,
            BoundaryKind::Outlet {
                static_pressure: 1000.0,
                supersonic: false,
            }
        ));
    }
}
