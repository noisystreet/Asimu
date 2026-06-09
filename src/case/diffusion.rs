//! 1D 稳态扩散算例编排。

use tracing::info;

use crate::case::{CaseRunKind, CaseRunResult};
use crate::core::Real;
use crate::discretization::{apply_boundary_conditions, assemble_diffusion_1d};
use crate::error::Result;
use crate::io::CaseSpec;
use crate::linalg::LinearSystem;

/// 扩散运行指标。
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionRunMetrics {
    pub max_abs_error: Real,
    pub l2_error: Real,
}

pub fn run(case: &CaseSpec) -> Result<CaseRunResult> {
    let mesh = case.mesh.as_1d()?;
    let n = mesh.num_cells();
    let mut system = LinearSystem::zeros(n)?;
    assemble_diffusion_1d(mesh, &mut system, case.diffusivity())?;
    apply_boundary_conditions(mesh, &case.boundary, &mut system, case.diffusivity())?;
    let solution = system.solve_tridiagonal()?;

    let metrics = measure_against_linear_exact(mesh, &solution);
    info!(
        max_abs_error = metrics.max_abs_error,
        l2_error = metrics.l2_error,
        cells = n,
        "1D 扩散稳态求解完成"
    );

    Ok(CaseRunResult {
        name: case.name.clone(),
        benchmark_id: case.benchmark_id.clone(),
        kind: CaseRunKind::Diffusion1dSteady,
        summary: format!(
            "1D 扩散稳态：max|e|={:.6e} L2={:.6e} (n={n})",
            metrics.max_abs_error, metrics.l2_error
        ),
        diffusion: Some(metrics),
        sod: None,
        compressible_3d: None,
        incompressible_3d: None,
    })
}

fn measure_against_linear_exact(
    mesh: &crate::mesh::StructuredMesh1d,
    solution: &[Real],
) -> DiffusionRunMetrics {
    let dx = mesh.dx();
    let origin = mesh.origin;
    let mut max_abs_error: Real = 0.0;
    let mut l2_sum = 0.0;
    for (i, &phi) in solution.iter().enumerate() {
        let x = origin + (i as Real + 0.5) * dx;
        let exact = x / mesh.length;
        let err = (phi - exact).abs();
        max_abs_error = max_abs_error.max(err);
        l2_sum += err * err;
    }
    DiffusionRunMetrics {
        max_abs_error,
        l2_error: (l2_sum / solution.len() as Real).sqrt(),
    }
}
