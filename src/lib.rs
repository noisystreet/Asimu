//! asimu — Rust 计算流体力学求解器库。
//!
//! 库用户请使用 [`mesh`](mesh)、[`solver`](solver) 等模块；
//! CLI 编排见 [`app`](app)（应用层，semver 独立于数值 API）。
//!
//! SPDX-License-Identifier: Apache-2.0 OR MIT

pub mod app;
pub mod boundary;
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
    pub use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet, BoundaryRegistry};
    pub use crate::config::{AppConfig, SolverConfig};
    pub use crate::core::{CellId, FaceId, Real, Vector3, approx_eq};
    pub use crate::discretization::{
        apply_boundary_conditions, apply_compressible_boundary_conditions, apply_dirichlet,
        apply_neumann, assemble_diffusion_1d,
    };
    pub use crate::error::{AsimuError, Result};
    pub use crate::field::{
        ConservedFields, Fields, FluidInitialConfig, InitialKind, InitialSet, ScalarField,
        ScalarInitial,
    };
    pub use crate::io::{CaseMesh, CaseSpec, load_case, load_conserved_fields};
    pub use crate::linalg::LinearSystem;
    pub use crate::mesh::{
        BoundaryMesh, BoundaryMesh3d, Mesh, StructuredMesh, StructuredMesh1d, StructuredMesh2d,
        StructuredMesh3d,
    };
    pub use crate::physics::{FreestreamParams, IdealGasEoS, PhysicsConfig, PrimitiveState};
    pub use crate::solver::{
        SolveResult, Solver, SolverState, SteadyStateIntegrator, TimeIntegrator,
    };

    #[cfg(feature = "io-vtk")]
    pub use crate::io::{VtmBlock, VtsLoadResult, load_vts, write_vtm, write_vts};

    #[cfg(feature = "io-cgns")]
    pub use crate::io::{
        CgnsLoadResult, CgnsMultiLoadResult, CgnsZoneInfo, export_cgns_to_vtm, export_cgns_to_vts,
        export_cgns_zone_to_vts, list_cgns_zones, load_cgns_all_zones, load_cgns_zone,
    };
}
