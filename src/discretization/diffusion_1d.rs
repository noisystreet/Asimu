//! 1D 稳态扩散 FVM 装配（内部面）。

use crate::core::Real;
use crate::error::Result;
use crate::linalg::LinearSystem;
use crate::mesh::StructuredMesh1d;

/// 装配 \(-\nabla\cdot(D\nabla\phi)\) 的内部面通量（均匀 1D 网格）。
pub fn assemble_diffusion_1d(
    mesh: &StructuredMesh1d,
    system: &mut LinearSystem,
    diffusivity: Real,
) -> Result<()> {
    let n = mesh.num_cells();
    if system.len() != n {
        return Err(crate::error::AsimuError::Linalg(format!(
            "线性系统尺寸 {} 与网格单元数 {n} 不一致",
            system.len()
        )));
    }
    let g = diffusivity / mesh.dx();
    for i in 0..n.saturating_sub(1) {
        system.add_coupling(i, i, g);
        system.add_coupling(i, i + 1, -g);
        system.add_coupling(i + 1, i, -g);
        system.add_coupling(i + 1, i + 1, g);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_row_has_neumann_sum_zero_coefficients() {
        let mesh = StructuredMesh1d::new("line", 4, 0.0, 1.0).expect("mesh");
        let mut system = LinearSystem::zeros(4).expect("system");
        assemble_diffusion_1d(&mesh, &mut system, 1.0).expect("assemble");
        let g = 1.0 / mesh.dx();
        assert!((system.diag()[1] - 2.0 * g).abs() < 1.0e-12);
        assert!((system.lower()[1] + g).abs() < 1.0e-12);
        assert!((system.upper()[1] + g).abs() < 1.0e-12);
    }
}
