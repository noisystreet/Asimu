//! asimu — Rust 计算流体力学求解器库。
//!
//! 库用户请使用 [`mesh`](mesh)、[`solver`](solver) 等模块；
//! CLI 编排见 [`app`](app)（应用层，semver 独立于数值 API）。
//!
//! SPDX-License-Identifier: Apache-2.0 OR MIT

pub mod app;
pub mod config;
pub mod core;
pub mod discretization;
pub mod error;
pub mod field;
pub mod io;
pub mod linalg;
pub mod mesh;
pub mod solver;

/// 常用类型 re-export，便于库集成。
pub mod prelude {
    pub use crate::config::{AppConfig, SolverConfig};
    pub use crate::core::{CellId, Real, Vector3, approx_eq};
    pub use crate::error::{AsimuError, Result};
    pub use crate::field::ScalarField;
    pub use crate::mesh::Mesh;
    pub use crate::solver::{
        SolveResult, Solver, SolverState, SteadyStateIntegrator, TimeIntegrator,
    };
}
