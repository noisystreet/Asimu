//! 空间离散算子（v0.2 骨架）。
//!
//! 理论：[`docs/theory/fvm_diffusion.md`](../../docs/theory/fvm_diffusion.md)

use crate::core::Real;
use crate::error::Result;
use crate::field::ScalarField;
use crate::linalg::LinearSystem;
use crate::mesh::Mesh;

/// 占位装配入口：验证 field / mesh / system 尺寸一致。
///
/// v0.2 后续 PR 实现 1D FVM 扩散装配；当前仅清零 RHS。
pub fn assemble_diffusion_placeholder(
    mesh: &Mesh,
    field: &ScalarField,
    system: &mut LinearSystem,
    diffusivity: Real,
) -> Result<()> {
    let _ = diffusivity;
    debug_assert_eq!(mesh.cell_count, field.len());
    debug_assert_eq!(field.len(), system.len());
    for value in system.rhs_mut() {
        *value = 0.0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::ScalarField;

    #[test]
    fn placeholder_assemble_succeeds_on_matching_sizes() {
        let mesh = Mesh::new("line", 4).expect("mesh");
        let field = ScalarField::uniform(4, 0.0).expect("field");
        let mut system = LinearSystem::new(vec![1.0; 4]).expect("system");
        assemble_diffusion_placeholder(&mesh, &field, &mut system, 1.0).expect("assemble");
        assert!(system.rhs().iter().all(|&v| v == 0.0));
    }
}
