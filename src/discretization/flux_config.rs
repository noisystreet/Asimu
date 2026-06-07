//! 无粘通量与界面重构配置。

use super::roe::RoeFluxConfig;
use super::unstructured_limiter::UnstructuredGradientLimiter;

/// 界面重构格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReconstructionKind {
    #[default]
    FirstOrder,
    Muscl,
}

/// MUSCL 斜率限制器。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlopeLimiter {
    #[default]
    Minmod,
    VanLeer,
    VanAlbada,
}

/// 无粘数值通量格式。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FluxScheme {
    Roe(RoeFluxConfig),
    Hllc,
    VanLeer,
    HanelVanLeer,
    Slau2,
}

impl Default for FluxScheme {
    fn default() -> Self {
        Self::Roe(RoeFluxConfig::default())
    }
}

/// 无粘面通量 + 重构选项。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InviscidFluxConfig {
    pub reconstruction: ReconstructionKind,
    pub limiter: SlopeLimiter,
    pub scheme: FluxScheme,
    /// 非结构二阶路径专用；结构化 MUSCL 忽略。
    pub unstructured_gradient_limiter: Option<UnstructuredGradientLimiter>,
}

impl InviscidFluxConfig {
    #[must_use]
    pub const fn roe_first_order() -> Self {
        Self {
            reconstruction: ReconstructionKind::FirstOrder,
            limiter: SlopeLimiter::Minmod,
            scheme: FluxScheme::Roe(RoeFluxConfig {
                entropy_fix: true,
                entropy_delta: None,
            }),
            unstructured_gradient_limiter: None,
        }
    }

    #[must_use]
    pub const fn muscl_roe() -> Self {
        Self {
            reconstruction: ReconstructionKind::Muscl,
            limiter: SlopeLimiter::Minmod,
            scheme: FluxScheme::Roe(RoeFluxConfig {
                entropy_fix: true,
                entropy_delta: None,
            }),
            unstructured_gradient_limiter: None,
        }
    }

    #[must_use]
    pub const fn muscl_hllc() -> Self {
        Self {
            reconstruction: ReconstructionKind::Muscl,
            limiter: SlopeLimiter::Minmod,
            scheme: FluxScheme::Hllc,
            unstructured_gradient_limiter: None,
        }
    }

    #[must_use]
    pub const fn van_leer_first_order() -> Self {
        Self {
            reconstruction: ReconstructionKind::FirstOrder,
            limiter: SlopeLimiter::Minmod,
            scheme: FluxScheme::VanLeer,
            unstructured_gradient_limiter: None,
        }
    }

    pub const fn hanel_van_leer_first_order() -> Self {
        Self {
            reconstruction: ReconstructionKind::FirstOrder,
            limiter: SlopeLimiter::Minmod,
            scheme: FluxScheme::HanelVanLeer,
            unstructured_gradient_limiter: None,
        }
    }

    #[must_use]
    pub const fn muscl_hanel_van_leer() -> Self {
        Self {
            reconstruction: ReconstructionKind::Muscl,
            limiter: SlopeLimiter::Minmod,
            scheme: FluxScheme::HanelVanLeer,
            unstructured_gradient_limiter: None,
        }
    }

    #[must_use]
    pub const fn muscl_van_leer() -> Self {
        Self {
            reconstruction: ReconstructionKind::Muscl,
            limiter: SlopeLimiter::Minmod,
            scheme: FluxScheme::VanLeer,
            unstructured_gradient_limiter: None,
        }
    }

    #[must_use]
    pub const fn slau2_first_order() -> Self {
        Self {
            reconstruction: ReconstructionKind::FirstOrder,
            limiter: SlopeLimiter::Minmod,
            scheme: FluxScheme::Slau2,
            unstructured_gradient_limiter: None,
        }
    }

    #[must_use]
    pub const fn muscl_slau2() -> Self {
        Self {
            reconstruction: ReconstructionKind::Muscl,
            limiter: SlopeLimiter::Minmod,
            scheme: FluxScheme::Slau2,
            unstructured_gradient_limiter: None,
        }
    }

    /// 一阶分段常数重构不使用斜率限制器（Godunov 型格式本身单调）。
    #[must_use]
    pub const fn uses_limiter(self) -> bool {
        matches!(self.reconstruction, ReconstructionKind::Muscl)
    }

    /// 限制器简短标识（导出元数据用）；一阶为 `"none"`。
    #[must_use]
    pub fn limiter_label(self) -> &'static str {
        if let Some(limiter) = self.unstructured_gradient_limiter {
            return limiter.label();
        }
        if !self.uses_limiter() {
            return "none";
        }
        match self.limiter {
            SlopeLimiter::Minmod => "minmod",
            SlopeLimiter::VanLeer => "van_leer",
            SlopeLimiter::VanAlbada => "van_albada",
        }
    }

    #[must_use]
    pub const fn with_unstructured_gradient_limiter(
        self,
        limiter: UnstructuredGradientLimiter,
    ) -> Self {
        Self {
            unstructured_gradient_limiter: Some(limiter),
            ..self
        }
    }

    #[must_use]
    pub const fn with_limiter(self, limiter: SlopeLimiter) -> Self {
        Self { limiter, ..self }
    }

    /// 导出/元数据用简短标识。
    #[must_use]
    pub const fn short_label(self) -> &'static str {
        match (self.reconstruction, self.scheme) {
            (ReconstructionKind::FirstOrder, FluxScheme::Roe(_)) => "roe_first_order",
            (ReconstructionKind::Muscl, FluxScheme::Hllc) => "muscl_hllc",
            (ReconstructionKind::Muscl, FluxScheme::Roe(_)) => "muscl_roe",
            (ReconstructionKind::FirstOrder, FluxScheme::Hllc) => "first_order_hllc",
            (ReconstructionKind::FirstOrder, FluxScheme::VanLeer) => "van_leer_first_order",
            (ReconstructionKind::Muscl, FluxScheme::VanLeer) => "muscl_van_leer",
            (ReconstructionKind::FirstOrder, FluxScheme::HanelVanLeer) => {
                "hanel_van_leer_first_order"
            }
            (ReconstructionKind::Muscl, FluxScheme::HanelVanLeer) => "muscl_hanel_van_leer",
            (ReconstructionKind::FirstOrder, FluxScheme::Slau2) => "slau2_first_order",
            (ReconstructionKind::Muscl, FluxScheme::Slau2) => "muscl_slau2",
        }
    }
}

impl Default for InviscidFluxConfig {
    fn default() -> Self {
        Self::roe_first_order()
    }
}
