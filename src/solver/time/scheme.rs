//! 显式时间积分格式选择。

use crate::error::{AsimuError, Result};

/// 可压缩时间推进格式。
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
    /// 不可压缩 PISO smoke 路径（可压缩求解器不支持）。
    Piso,
}

impl TimeIntegrationScheme {
    pub fn parse(name: &str) -> Result<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "rk4" | "runge_kutta_4" | "runge-kutta-4" | "rk4_4" => Ok(Self::Rk4),
            "euler" | "forward_euler" | "euler1" | "rk1" | "euler_1" => Ok(Self::Euler),
            "lu_sgs" | "lusgs" | "lu-sgs" => Ok(Self::LuSgs),
            "gmres" | "jfnk" | "matrix_free_gmres" | "matrix-free-gmres" => Ok(Self::Gmres),
            "piso" => Ok(Self::Piso),
            other => Err(AsimuError::Config(format!(
                "不支持的 time.scheme \"{other}\"（可用 rk4、euler、lu_sgs、gmres、piso）"
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
            Self::Piso => "piso",
        }
    }

    #[must_use]
    pub const fn is_implicit_pseudo_time(self) -> bool {
        matches!(self, Self::LuSgs | Self::Gmres)
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
