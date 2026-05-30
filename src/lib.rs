//! asimu — Rust 计算流体力学求解器库。
//!
//! 库用户请使用 [`mesh`](mesh)、[`solver`](solver) 等模块；
//! CLI 编排见 [`app`](app)（应用层，semver 独立于数值 API）。
//!
//! SPDX-License-Identifier: Apache-2.0 OR MIT

pub mod app;
pub mod boundary;
pub mod case;
pub mod config;
pub mod core;
pub mod discretization;
pub mod error;
pub mod field;
pub mod io;
pub mod linalg;
pub mod mesh;
pub mod physics;
pub mod solver;

/// 常用类型 re-export，便于库集成。
pub mod prelude {
    pub use crate::boundary::{BoundaryKind, BoundaryPatch, BoundaryRegistry, BoundarySet};
    pub use crate::config::{AppConfig, SolverConfig};
    pub use crate::core::{CellId, FaceId, Real, Vector3, approx_eq};
    pub use crate::discretization::{
        BoundaryGhosts1d, FaceFluxInput, FluxScheme, InviscidFlux, InviscidFluxConfig,
        ReconstructionKind, RoeFluxConfig, SlopeLimiter, apply_boundary_conditions,
        apply_compressible_boundary_conditions, apply_dirichlet, apply_neumann,
        assemble_diffusion_1d, assemble_inviscid_residual_1d, assemble_inviscid_residual_3d,
        face_inviscid_flux, hllc_flux, reconstruct_face_states, reconstruct_first_order, roe_flux,
    };
    pub use crate::error::{AsimuError, Result};
    pub use crate::field::{
        ConservedFields, ConservedResidual, Fields, FluidInitialConfig, InitialKind, InitialSet,
        ScalarField, ScalarInitial,
    };
    pub use crate::io::{CaseMesh, CaseSpec, load_case, load_conserved_fields};
    pub use crate::linalg::LinearSystem;
    pub use crate::mesh::{
        BoundaryMesh, BoundaryMesh3d, Mesh, StructuredMesh, StructuredMesh1d, StructuredMesh2d,
        StructuredMesh3d,
    };
    pub use crate::physics::{FreestreamParams, IdealGasEoS, PhysicsConfig, PrimitiveState};
    pub use crate::solver::{
        CompressibleAdvanceContext1d, CompressibleAdvanceContext3d, CompressibleEulerConfig,
        CompressibleEulerSolver, CompressibleStepInfo, Rk4Storage, RungeKutta4Config,
        RungeKutta4Integrator, SolveResult, Solver, SolverState, SteadyStateIntegrator,
        TimeIntegrator, max_wave_speed, rk4_step,
    };

    #[cfg(feature = "io-vtk")]
    pub use crate::io::{VtmBlock, VtsLoadResult, load_vts, write_vtm, write_vts};

    #[cfg(feature = "io-cgns")]
    pub use crate::io::{
        CgnsLoadResult, CgnsMultiLoadResult, CgnsZoneInfo, export_cgns_to_vtm, export_cgns_to_vts,
        export_cgns_zone_to_vts, list_cgns_zones, load_cgns_all_zones, load_cgns_zone,
    };
}
