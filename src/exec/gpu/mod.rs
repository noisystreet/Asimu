//! GPU 执行后端（feature-gated；ADR 0017）。

#[cfg(feature = "cuda")]
pub mod cuda;
