//! 无粘可压缩 Euler 边界修正（`[euler]` 算例）。

use tracing::info;

use super::kind::BoundaryKind;
use super::patch::BoundarySet;

impl BoundarySet {
    /// `[euler]` 无粘算例：将所有壁面改为**滑移绝热壁**（`no_slip = false`）。
    ///
    /// CGNS 中 `BCWall` / `BCWallViscous` 等常映射为有滑移壁；无粘 Euler 应做法向反射、切向自由。
    pub fn apply_inviscid_euler_walls(&mut self) {
        let mut count = 0u32;
        for patch in self.patches_mut() {
            let BoundaryKind::Wall { no_slip, heat } = &patch.kind else {
                continue;
            };
            if *no_slip {
                count += 1;
            }
            patch.kind = BoundaryKind::Wall {
                no_slip: false,
                heat: *heat,
            };
        }
        if count > 0 {
            info!(
                patches = count,
                "无粘 Euler：壁面已改为滑移壁（no_slip = false）"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryPatch, WallHeat};

    #[test]
    fn apply_inviscid_euler_walls_clears_no_slip() {
        let mut set = BoundarySet::new(vec![
            BoundaryPatch::new(
                "dom-7",
                vec![],
                BoundaryKind::Wall {
                    no_slip: true,
                    heat: WallHeat::Adiabatic,
                },
            ),
            BoundaryPatch::new("sym", vec![], BoundaryKind::Symmetry),
        ]);
        set.apply_inviscid_euler_walls();
        assert!(matches!(
            &set.patches()[0].kind,
            BoundaryKind::Wall {
                no_slip: false,
                heat: WallHeat::Adiabatic,
            }
        ));
        assert!(matches!(set.patches()[1].kind, BoundaryKind::Symmetry));
    }
}
