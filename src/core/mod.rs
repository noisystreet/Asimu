//! 核心数值类型与常量。
//!
//! 理论参考：[`docs/theory/`](../../docs/theory/README.md)

pub mod convergence;
pub mod exec_device;
pub mod id;
pub mod precision;
pub mod real;
pub mod timing;

pub use convergence::{
    compressible_log10_tolerance_met, incompressible_steady_convergence_window,
    log10_residual_converged,
};
pub use exec_device::{
    ExecBackend, ExecCpuPolicy, ExecDevice, cpu_policy_for_device, default_cpu_policy,
    exec_backend_view, legacy_backend_to_parts, parse_exec_backend,
};
pub use id::{CellId, FaceId, NodeId};
pub use precision::{ComputeFloat, ComputePrecision, parse_compute_precision};
pub use real::{
    Real, RealOps, approx_eq, format_log_fixed4, format_log_fixed5, format_log_sci4,
    log10_positive, residual_converged,
};
pub use timing::elapsed_ms;

/// 三维向量（占位，后续扩展为 SIMD 友好布局）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3 {
    pub x: Real,
    pub y: Real,
    pub z: Real,
}

impl Vector3 {
    #[must_use]
    pub const fn new(x: Real, y: Real, z: Real) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    pub fn magnitude(self) -> Real {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_magnitude() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        assert!(approx_eq(v.magnitude(), 5.0, 1.0e-12));
    }

    #[test]
    fn cell_id_orders() {
        assert!(CellId(1) < CellId(2));
    }
}
