//! `exec` 对 [`crate::core::exec_device`] 的再导出（ADR 0017）。

pub use crate::core::exec_device::{
    ExecBackend, ExecCpuPolicy, ExecDevice, cpu_policy_for_device, default_cpu_policy,
    exec_backend_view, legacy_backend_to_parts, parse_exec_backend,
};
