use crate::core::Real;
use crate::linalg::{GmresConfig, PcgConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompressiblePressureLinearSolverKind {
    Gmres,
    Pcg,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressiblePressureLinearSolverConfig {
    pub kind: IncompressiblePressureLinearSolverKind,
    pub max_iters: usize,
    pub tolerance: Real,
    pub gmres_restart: usize,
}

impl IncompressiblePressureLinearSolverConfig {
    #[must_use]
    pub fn gmres_config(self) -> GmresConfig {
        GmresConfig {
            restart: self.gmres_restart,
            max_iters: self.max_iters,
            tolerance: self.tolerance,
        }
    }

    #[must_use]
    pub fn pcg_config(self) -> PcgConfig {
        PcgConfig {
            max_iters: self.max_iters,
            tolerance: self.tolerance,
        }
    }
}

impl Default for IncompressiblePressureLinearSolverConfig {
    fn default() -> Self {
        Self {
            kind: IncompressiblePressureLinearSolverKind::Pcg,
            max_iters: 500,
            tolerance: 1.0e-10,
            gmres_restart: 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IncompressibleLinearSolverConfig {
    pub momentum: GmresConfig,
    pub pressure: IncompressiblePressureLinearSolverConfig,
}
