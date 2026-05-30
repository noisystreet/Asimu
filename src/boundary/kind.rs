//! 边界条件类型（扩散 v0.2 + 可压缩 NS v0.3+）。

use crate::core::Real;

/// 壁面热边界类型。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WallHeat {
    Adiabatic,
    Isothermal { temperature: Real },
    HeatFlux { flux: Real },
}

/// 边界条件类型。
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryKind {
    // --- 扩散 (v0.2) ---
    Dirichlet {
        value: Real,
    },
    Neumann {
        flux: Real,
    },

    // --- 可压缩 NS (v0.3+) ---
    Farfield {
        mach: Real,
        pressure: Real,
        temperature: Real,
        alpha: Real,
        beta: Real,
    },
    Inlet {
        total_pressure: Real,
        total_temperature: Real,
        velocity_direction: [Real; 3],
        /// 设计 Mach（`apply_freestream` 写入；\(M\ge 1\) 时按超声速入口处理）。
        mach: Real,
    },
    Outlet {
        static_pressure: Real,
        /// 超声速出口：零梯度外推 owner（CGNS `BCOutflowSupersonic`）。
        supersonic: bool,
    },
    Wall {
        no_slip: bool,
        heat: WallHeat,
    },
    Symmetry,
    Periodic {
        partner: String,
    },
    TurbulentInlet {
        total_pressure: Real,
        total_temperature: Real,
        velocity_direction: [Real; 3],
        turbulent_k: Real,
        turbulent_omega: Real,
    },
}

impl BoundaryKind {
    #[must_use]
    pub const fn dirichlet(value: Real) -> Self {
        Self::Dirichlet { value }
    }

    #[must_use]
    pub const fn neumann(flux: Real) -> Self {
        Self::Neumann { flux }
    }

    /// 扩散 TOML 解析（向后兼容）。
    pub fn from_diffusion_toml(
        kind: &str,
        value: Option<Real>,
        flux: Option<Real>,
    ) -> Option<Self> {
        match kind {
            "dirichlet" => value.map(Self::dirichlet),
            "neumann" => flux.map(Self::neumann),
            _ => None,
        }
    }
}

/// TOML 边界表解析上下文。
#[derive(Debug, Clone, Default)]
pub struct BoundaryTomlFields<'a> {
    pub kind: &'a str,
    pub value: Option<Real>,
    pub flux: Option<Real>,
    pub mach: Option<Real>,
    pub pressure: Option<Real>,
    pub temperature: Option<Real>,
    pub alpha: Option<Real>,
    pub beta: Option<Real>,
    pub total_pressure: Option<Real>,
    pub total_temperature: Option<Real>,
    pub static_pressure: Option<Real>,
    pub velocity_direction: Option<[Real; 3]>,
    pub no_slip: Option<bool>,
    pub heat: Option<&'a str>,
    pub wall_temperature: Option<Real>,
    pub heat_flux: Option<Real>,
    pub partner: Option<&'a str>,
    pub turbulent_k: Option<Real>,
    pub turbulent_omega: Option<Real>,
}

impl BoundaryKind {
    pub fn from_toml(fields: &BoundaryTomlFields<'_>) -> Option<Self> {
        match fields.kind {
            "dirichlet" => fields.value.map(Self::dirichlet),
            "neumann" => fields.flux.map(Self::neumann),
            "farfield" => Some(Self::Farfield {
                mach: fields.mach.unwrap_or(0.0),
                pressure: fields.pressure.unwrap_or(101_325.0),
                temperature: fields.temperature.unwrap_or(288.15),
                alpha: fields.alpha.unwrap_or(0.0),
                beta: fields.beta.unwrap_or(0.0),
            }),
            "inlet" => {
                let total_pressure = fields.total_pressure?;
                let total_temperature = fields.total_temperature?;
                let velocity_direction = fields.velocity_direction.unwrap_or([1.0, 0.0, 0.0]);
                Some(Self::Inlet {
                    total_pressure,
                    total_temperature,
                    velocity_direction,
                    mach: fields.mach.unwrap_or(0.0),
                })
            }
            "outlet" => fields.static_pressure.map(|static_pressure| Self::Outlet {
                static_pressure,
                supersonic: fields.mach.is_some_and(|m| m >= 1.0),
            }),
            "wall" => {
                let no_slip = fields.no_slip.unwrap_or(true);
                let heat = match fields.heat.unwrap_or("adiabatic") {
                    "adiabatic" => WallHeat::Adiabatic,
                    "isothermal" => WallHeat::Isothermal {
                        temperature: fields.wall_temperature.unwrap_or(300.0),
                    },
                    "heat_flux" => WallHeat::HeatFlux {
                        flux: fields.heat_flux.unwrap_or(0.0),
                    },
                    _ => WallHeat::Adiabatic,
                };
                Some(Self::Wall { no_slip, heat })
            }
            "symmetry" => Some(Self::Symmetry),
            "periodic" => fields.partner.map(|partner| Self::Periodic {
                partner: partner.to_string(),
            }),
            "turbulent_inlet" => {
                let total_pressure = fields.total_pressure?;
                let total_temperature = fields.total_temperature?;
                let turbulent_k = fields.turbulent_k?;
                let turbulent_omega = fields.turbulent_omega?;
                Some(Self::TurbulentInlet {
                    total_pressure,
                    total_temperature,
                    velocity_direction: fields.velocity_direction.unwrap_or([1.0, 0.0, 0.0]),
                    turbulent_k,
                    turbulent_omega,
                })
            }
            _ => None,
        }
    }

    /// 由 CGNS `BCType_t` 映射（见 `io::cgns::zonebc`）。
    pub fn from_cgns_bctype(bctype: i32, name: &str) -> Self {
        map_cgns_bctype(bctype).with_cgns_name_note(name)
    }

    fn with_cgns_name_note(self, _name: &str) -> Self {
        self
    }

    /// 边界类型短标签（报告 / 日志用）。
    #[must_use]
    pub fn summary_label(&self) -> &'static str {
        match self {
            Self::Dirichlet { .. } => "Dirichlet",
            Self::Neumann { .. } => "Neumann",
            Self::Farfield { .. } => "Farfield",
            Self::Inlet { .. } => "Inlet",
            Self::Outlet {
                supersonic: true, ..
            } => "Outlet(supersonic)",
            Self::Outlet { .. } => "Outlet",
            Self::Wall { no_slip: true, .. } => "Wall(no-slip)",
            Self::Wall { no_slip: false, .. } => "Wall(slip)",
            Self::Symmetry => "Symmetry",
            Self::Periodic { .. } => "Periodic",
            Self::TurbulentInlet { .. } => "TurbInlet",
        }
    }

    /// 边界条件参数详情（人类可读）。
    #[must_use]
    pub fn detail_label(&self) -> String {
        match self {
            Self::Dirichlet { value } => format!("φ = {value}"),
            Self::Neumann { flux } => format!("∂φ/∂n = {flux}"),
            Self::Farfield {
                mach,
                pressure,
                temperature,
                alpha,
                beta,
            } => format!("M={mach}, p={pressure}, T={temperature}, α={alpha}, β={beta}"),
            Self::Inlet {
                total_pressure,
                total_temperature,
                velocity_direction,
                mach,
            } => format!(
                "M={mach}, p₀={total_pressure:.6e}, T₀={total_temperature:.6e}, n=[{:.4}, {:.4}, {:.4}]",
                velocity_direction[0], velocity_direction[1], velocity_direction[2]
            ),
            Self::Outlet {
                static_pressure,
                supersonic,
            } => {
                if *supersonic {
                    format!("p={static_pressure:.6e}, supersonic extrapolation")
                } else {
                    format!("p={static_pressure:.6e}")
                }
            }
            Self::Wall { no_slip, heat } => {
                let slip = if *no_slip { "no-slip" } else { "slip" };
                let heat_label = match heat {
                    WallHeat::Adiabatic => "adiabatic",
                    WallHeat::Isothermal { temperature } => {
                        return format!("{slip}, isothermal T={temperature}");
                    }
                    WallHeat::HeatFlux { flux } => {
                        return format!("{slip}, heat_flux={flux}");
                    }
                };
                format!("{slip}, {heat_label}")
            }
            Self::Symmetry => "symmetry plane".to_string(),
            Self::Periodic { partner } => format!("partner={partner}"),
            Self::TurbulentInlet {
                total_pressure,
                total_temperature,
                velocity_direction,
                turbulent_k,
                turbulent_omega,
            } => format!(
                "p₀={total_pressure:.6e}, T₀={total_temperature:.6e}, n=[{:.4}, {:.4}, {:.4}], k={turbulent_k:.6e}, ω={turbulent_omega:.6e}",
                velocity_direction[0], velocity_direction[1], velocity_direction[2]
            ),
        }
    }
}

fn map_cgns_bctype(bctype: i32) -> BoundaryKind {
    if bctype == cgns_bc::BC_FARFIELD {
        return BoundaryKind::Farfield {
            mach: 0.0,
            pressure: 101_325.0,
            temperature: 288.15,
            alpha: 0.0,
            beta: 0.0,
        };
    }
    if bctype == cgns_bc::BC_EXTRAPOLATE || is_outflow(bctype) {
        return default_outflow(bctype == cgns_bc::BC_OUTFLOW_SUPERSONIC);
    }
    if is_inflow(bctype) {
        return default_inlet();
    }
    if bctype == cgns_bc::BC_SYMMETRY_PLANE || bctype == cgns_bc::BC_SYMMETRY_POLAR {
        return BoundaryKind::Symmetry;
    }
    if bctype == cgns_bc::BC_WALL_INVISCID {
        return default_wall(false);
    }
    if is_viscous_wall(bctype) {
        return default_wall(true);
    }
    default_wall(true)
}

fn default_inlet() -> BoundaryKind {
    BoundaryKind::Inlet {
        total_pressure: 101_325.0,
        total_temperature: 300.0,
        velocity_direction: [1.0, 0.0, 0.0],
        mach: 0.0,
    }
}

fn default_outflow(supersonic: bool) -> BoundaryKind {
    BoundaryKind::Outlet {
        static_pressure: 101_325.0,
        supersonic,
    }
}

fn default_wall(no_slip: bool) -> BoundaryKind {
    BoundaryKind::Wall {
        no_slip,
        heat: WallHeat::Adiabatic,
    }
}

fn is_inflow(bctype: i32) -> bool {
    matches!(
        bctype,
        cgns_bc::BC_INFLOW | cgns_bc::BC_INFLOW_SUBSONIC | cgns_bc::BC_INFLOW_SUPERSONIC
    )
}

fn is_outflow(bctype: i32) -> bool {
    matches!(
        bctype,
        cgns_bc::BC_OUTFLOW | cgns_bc::BC_OUTFLOW_SUBSONIC | cgns_bc::BC_OUTFLOW_SUPERSONIC
    )
}

fn is_viscous_wall(bctype: i32) -> bool {
    matches!(
        bctype,
        cgns_bc::BC_WALL
            | cgns_bc::BC_WALL_VISCOUS
            | cgns_bc::BC_WALL_VISCOUS_HEAT_FLUX
            | cgns_bc::BC_WALL_VISCOUS_ISOTHERMAL
    )
}

/// CGNS BCType 常量（与 `cgnslib.h` 一致子集）。
pub mod cgns_bc {
    pub const BC_EXTRAPOLATE: i32 = 6;
    pub const BC_FARFIELD: i32 = 7;
    pub const BC_INFLOW: i32 = 9;
    pub const BC_INFLOW_SUBSONIC: i32 = 10;
    pub const BC_INFLOW_SUPERSONIC: i32 = 11;
    pub const BC_OUTFLOW: i32 = 13;
    pub const BC_OUTFLOW_SUBSONIC: i32 = 14;
    pub const BC_OUTFLOW_SUPERSONIC: i32 = 15;
    pub const BC_SYMMETRY_PLANE: i32 = 16;
    pub const BC_SYMMETRY_POLAR: i32 = 17;
    pub const BC_WALL: i32 = 20;
    pub const BC_WALL_INVISCID: i32 = 21;
    pub const BC_WALL_VISCOUS: i32 = 22;
    pub const BC_WALL_VISCOUS_HEAT_FLUX: i32 = 23;
    pub const BC_WALL_VISCOUS_ISOTHERMAL: i32 = 24;
    pub const BC_FAMILY_SPECIFIED: i32 = 25;
}
