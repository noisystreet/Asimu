//! 边界条件数据与调度（v0.2）。
//!
//! 架构对照 CFL3D：
//! - **数据**：`BoundaryPatch` / `BoundaryKind`（类比 `cfl3d.inp` BC 段 + `ibcinfo`）
//! - **调度**：`BoundaryRegistry`（类比 `bc.F`）
//! - **数值施加**：[`crate::discretization::bc`]（类比 `bc1000.F` / `bc2004.F` 等）
//!
//! 理论：[`docs/theory/boundary_conditions.md`](../../docs/theory/boundary_conditions.md)

mod kind;
mod patch;
mod registry;

pub use kind::{BoundaryKind, BoundaryTomlFields, WallHeat, cgns_bc};
pub use patch::{BoundaryPatch, BoundarySet};
pub use registry::{BcHandler, BoundaryRegistry};
