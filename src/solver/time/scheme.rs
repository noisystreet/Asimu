//! 显式时间积分格式选择。

use crate::error::{AsimuError, Result};

/// 时间推进 / pressure-velocity 耦合格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeIntegrationScheme {
    /// 经典四阶 Runge-Kutta（默认）。
    #[default]
    Rk4,
    /// 一阶前向 Euler（排错对照用，稳定性比 RK4 差）。
    Euler,
    /// LU-SGS 隐式伪时间（默认对角；`lusgs_sweep=true` 启用双扫）。
    LuSgs,
    /// Matrix-free GMRES 隐式伪时间（LU-SGS 对角预条件器）。
    Gmres,
    /// 双时间步：首步 BDF1、之后 BDF2 + 内层 LU-SGS 伪时间（非结构可压缩）。
    DualTime,
    /// 不可压缩 SIMPLEC 稳态 pressure-velocity 路径（可压缩求解器不支持）。
    Simplec,
    /// 不可压缩 PISO/BDF1 瞬态 pressure-velocity 路径（可压缩求解器不支持）。
    Piso,
    /// 不可压缩 BDF1 瞬态动量离散（case 层映射到 PISO 压力-速度耦合）。
    Bdf1,
}

impl TimeIntegrationScheme {
    pub fn parse(name: &str) -> Result<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "rk4" | "runge_kutta_4" | "runge-kutta-4" | "rk4_4" => Ok(Self::Rk4),
            "euler" | "forward_euler" | "euler1" | "rk1" | "euler_1" => Ok(Self::Euler),
            "lu_sgs" | "lusgs" | "lu-sgs" => Ok(Self::LuSgs),
            "gmres" | "jfnk" | "matrix_free_gmres" | "matrix-free-gmres" => Ok(Self::Gmres),
            "dual_time" | "dts" | "dual-time" => Ok(Self::DualTime),
            "simplec" => Ok(Self::Simplec),
            "piso" => Ok(Self::Piso),
            "bdf1" | "backward_euler" | "implicit_euler" => Ok(Self::Bdf1),
            other => Err(AsimuError::Config(format!(
                "不支持的 time.scheme \"{other}\"（可用 rk4、euler、lu_sgs、gmres、dual_time、simplec、piso）"
            ))),
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Rk4 => "rk4",
            Self::Euler => "euler",
            Self::LuSgs => "lu_sgs",
            Self::Gmres => "gmres",
            Self::DualTime => "dual_time",
            Self::Simplec => "simplec",
            Self::Piso => "piso",
            Self::Bdf1 => "bdf1",
        }
    }

    #[must_use]
    pub const fn is_implicit_pseudo_time(self) -> bool {
        matches!(self, Self::LuSgs | Self::Gmres | Self::DualTime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lu_sgs_aliases() {
        assert_eq!(
            TimeIntegrationScheme::parse("lusgs").expect("lu_sgs"),
            TimeIntegrationScheme::LuSgs
        );
    }

    #[test]
    fn parses_gmres_aliases() {
        assert_eq!(
            TimeIntegrationScheme::parse("matrix-free-gmres").expect("gmres"),
            TimeIntegrationScheme::Gmres
        );
    }

    #[test]
    fn parses_piso_alias() {
        assert_eq!(
            TimeIntegrationScheme::parse("piso").expect("piso"),
            TimeIntegrationScheme::Piso
        );
    }

    #[test]
    fn parses_bdf1_alias() {
        assert_eq!(
            TimeIntegrationScheme::parse("bdf1").expect("bdf1"),
            TimeIntegrationScheme::Bdf1
        );
    }

    #[test]
    fn parses_simplec_alias() {
        assert_eq!(
            TimeIntegrationScheme::parse("simplec").expect("simplec"),
            TimeIntegrationScheme::Simplec
        );
    }

    #[test]
    fn parses_dual_time_aliases() {
        assert_eq!(
            TimeIntegrationScheme::parse("dts").expect("dual_time"),
            TimeIntegrationScheme::DualTime
        );
    }

    #[test]
    fn parses_euler_aliases() {
        assert_eq!(
            TimeIntegrationScheme::parse("forward_euler").expect("euler"),
            TimeIntegrationScheme::Euler
        );
        assert_eq!(
            TimeIntegrationScheme::parse("RK4").expect("rk4"),
            TimeIntegrationScheme::Rk4
        );
    }
}
