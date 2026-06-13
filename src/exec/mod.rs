//! CPU/GPU 执行后端（ADR 0003 / ADR 0013 / ADR 0017）。
//!
//! v0.x：`cpu` 子模块提供可选 SIMD 热算子（feature `simd-fvm`）；
//! `ExecutionContext` 统一 scatter 调度（E0 串行回退；E1 `ParallelUnsafeAtomics` + rayon）。
//!
//! ADR 0013：本模块允许 scatter atomic 等 approved `unsafe`（主 crate 仍 forbid）。

#![allow(unsafe_code)]

pub mod batch;
pub mod cpu;

mod backend_state;
mod context;
mod device;
#[cfg(feature = "cuda")]
pub mod gpu;
mod metrics;
mod scratch;

#[cfg(feature = "parallel-fvm")]
pub mod parallel;

pub mod scatter;

mod idwls;
#[cfg(feature = "cuda")]
pub mod inviscid;
mod spmv;

pub use batch::ExecFaceBatchStatic4;
pub use context::{
    EXEC_SCATTER_PARALLEL_MIN_FACES, ExecConfig, ExecutionContext, ResolvedScatterMode, ScatterMode,
};
pub use device::{
    ExecBackend, ExecCpuPolicy, ExecDevice, cpu_policy_for_device, default_cpu_policy,
    exec_backend_view, legacy_backend_to_parts, parse_exec_backend,
};
pub use metrics::MeshExecMetrics;
#[cfg(feature = "parallel-fvm")]
pub use scratch::{ColoredViscousFaceBuffer, ColoredViscousFaceFlux, ColoredViscousFaceGeom};
pub use scratch::{ExecScratch, IdwlsRhsBuffer};
pub use spmv::CsrSpmvView;
