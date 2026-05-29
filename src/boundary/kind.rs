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
    Dirichlet { value: Real },
    Neumann { flux: Real },

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
    },
    Outlet {
        static_pressure: Real,
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
    pub fn from_diffusion_toml(kind: &str, value: Option<Real>, flux: Option<Real>) -> Option<Self> {
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
                })
            }
            "outlet" => fields
                .static_pressure
                .map(|static_pressure| Self::Outlet { static_pressure }),
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
        match bctype {
            x if x == cgns_bc::BC_WALL => Self::Wall {
                no_slip: true,
                heat: WallHeat::Adiabatic,
            },
            x if x == cgns_bc::BC_FARFIELD => Self::Farfield {
                mach: 0.0,
                pressure: 101_325.0,
                temperature: 288.15,
                alpha: 0.0,
                beta: 0.0,
            },
            x if x == cgns_bc::BC_SYMMETRY_PLANE => Self::Symmetry,
            x if x == cgns_bc::BC_INFLOW => Self::Inlet {
                total_pressure: 101_325.0,
                total_temperature: 300.0,
                velocity_direction: [1.0, 0.0, 0.0],
            },
            x if x == cgns_bc::BC_OUTFLOW => Self::Outlet {
                static_pressure: 101_325.0,
            },
            _ => Self::Wall {
                no_slip: true,
                heat: WallHeat::Adiabatic,
            },
        }
        .with_cgns_name_note(name)
    }

    fn with_cgns_name_note(self, _name: &str) -> Self {
        self
    }
}

/// CGNS BCType 常量（与 `cgnslib.h` 一致子集）。
pub mod cgns_bc {
    pub const BC_WALL: i32 = 20;
    pub const BC_FARFIELD: i32 = 21;
    pub const BC_SYMMETRY_PLANE: i32 = 22;
    pub const BC_INFLOW: i32 = 23;
    pub const BC_OUTFLOW: i32 = 24;
}
