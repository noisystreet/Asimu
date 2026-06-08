//! 步间复用缓冲（ADR 0013 E2/E3：着色桶 flat buffer、IDWLS RHS 等）。

mod idwls;

#[cfg(feature = "parallel-fvm")]
mod colored_viscous;

pub use idwls::IdwlsRhsBuffer;

#[cfg(feature = "parallel-fvm")]
pub use colored_viscous::{
    ColoredViscousFaceBuffer, ColoredViscousFaceFlux, ColoredViscousFaceGeom,
};

use super::metrics::MeshExecMetrics;

/// LU-SGS 步内由 [`ExecutionContext`](super::context::ExecutionContext) 持有。
#[derive(Debug, Default)]
pub struct ExecScratch {
    idwls: IdwlsRhsBuffer,
    #[cfg(feature = "parallel-fvm")]
    colored_viscous: ColoredViscousFaceBuffer,
}

impl ExecScratch {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 按 mesh 规模预分配（case 层构造 `ExecutionContext` 时调用一次）。
    #[must_use]
    pub fn with_metrics(metrics: MeshExecMetrics) -> Self {
        Self {
            idwls: IdwlsRhsBuffer::with_capacity(metrics.num_cells),
            #[cfg(feature = "parallel-fvm")]
            colored_viscous: ColoredViscousFaceBuffer::with_capacity(metrics.max_bucket_faces),
        }
    }

    #[must_use]
    pub fn idwls(&self) -> &IdwlsRhsBuffer {
        &self.idwls
    }

    #[must_use]
    pub fn idwls_mut(&mut self) -> &mut IdwlsRhsBuffer {
        &mut self.idwls
    }

    #[cfg(feature = "parallel-fvm")]
    #[must_use]
    pub fn colored_viscous(&self) -> &ColoredViscousFaceBuffer {
        &self.colored_viscous
    }

    #[cfg(feature = "parallel-fvm")]
    #[must_use]
    pub fn colored_viscous_mut(&mut self) -> &mut ColoredViscousFaceBuffer {
        &mut self.colored_viscous
    }
}
