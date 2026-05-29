//! 边界条件数值施加（类比 CFL3D `bcXXXX.F`）。
//!
//! 调度入口：[`apply_boundary_conditions`]；内部按 [`BoundaryRegistry::handler_for`] 分派。

use crate::boundary::{BcHandler, BoundaryKind, BoundaryPatch, BoundaryRegistry, BoundarySet};
use crate::core::{CellId, FaceId, Real};
use crate::error::Result;
use crate::linalg::LinearSystem;
use crate::mesh::BoundaryMesh;

/// 强 Dirichlet：将单元行置为 \(\phi_i = \phi_b\)（仅用于行替换场景）。
pub fn apply_dirichlet(system: &mut LinearSystem, cell: CellId, value: Real) -> Result<()> {
    let row = usize::try_from(cell.index()).map_err(|_| {
        crate::error::AsimuError::Boundary(format!("单元索引越界: {}", cell.index()))
    })?;
    system.set_dirichlet_row(row, value);
    Ok(())
}

/// 面 Dirichlet（ghost 单元）：\(\phi_g = 2\phi_b - \phi_{\text{owner}}\)。
pub fn apply_dirichlet_face(
    system: &mut LinearSystem,
    mesh: &dyn BoundaryMesh,
    face: FaceId,
    diffusivity: Real,
    value: Real,
) -> Result<()> {
    let owner = mesh.face_owner(face)?;
    let row = usize::try_from(owner.index()).map_err(|_| {
        crate::error::AsimuError::Boundary(format!("单元索引越界: {}", owner.index()))
    })?;
    let spacing = mesh.face_spacing(face)?;
    if spacing <= 0.0 {
        return Err(crate::error::AsimuError::Boundary(
            "面间距必须大于 0".to_string(),
        ));
    }
    let conductance = 2.0 * diffusivity / spacing;
    system.add_diagonal(row, conductance);
    system.add_rhs(row, conductance * value);
    Ok(())
}

/// Neumann：\(-D \partial\phi/\partial n = q\)，通过 ghost 单元消元。
pub fn apply_neumann(
    system: &mut LinearSystem,
    mesh: &dyn BoundaryMesh,
    face: FaceId,
    diffusivity: Real,
    flux: Real,
) -> Result<()> {
    let owner = mesh.face_owner(face)?;
    let row = usize::try_from(owner.index()).map_err(|_| {
        crate::error::AsimuError::Boundary(format!("单元索引越界: {}", owner.index()))
    })?;
    let spacing = mesh.face_spacing(face)?;
    if spacing <= 0.0 {
        return Err(crate::error::AsimuError::Boundary(
            "面间距必须大于 0".to_string(),
        ));
    }
    let conductance = diffusivity / spacing;
    system.add_diagonal(row, conductance);
    system.add_rhs(row, flux);
    Ok(())
}

fn apply_patch(
    mesh: &dyn BoundaryMesh,
    patch: &BoundaryPatch,
    system: &mut LinearSystem,
    diffusivity: Real,
) -> Result<()> {
    let handler = BoundaryRegistry::handler_for(&patch.kind);
    for &face in &patch.face_ids {
        match handler {
            BcHandler::DiffusionDirichlet => {
                let BoundaryKind::Dirichlet { value } = patch.kind else {
                    unreachable!("handler/kind mismatch");
                };
                apply_dirichlet_face(system, mesh, face, diffusivity, value)?;
            }
            BcHandler::DiffusionNeumann => {
                let BoundaryKind::Neumann { flux } = patch.kind else {
                    unreachable!("handler/kind mismatch");
                };
                apply_neumann(system, mesh, face, diffusivity, flux)?;
            }
            _ => {
                return Err(crate::error::AsimuError::Boundary(format!(
                    "扩散求解器不支持边界类型 {:?}",
                    patch.kind
                )));
            }
        }
    }
    Ok(())
}

/// 按 patch 顺序施加全部边界条件（类比 CFL3D `bc.F` 主循环）。
pub fn apply_boundary_conditions(
    mesh: &dyn BoundaryMesh,
    patches: &BoundarySet,
    system: &mut LinearSystem,
    diffusivity: Real,
) -> Result<()> {
    BoundaryRegistry::validate_patches(patches.patches())?;
    for patch in patches.patches() {
        apply_patch(mesh, patch, system, diffusivity)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::mesh::StructuredMesh1d;

    #[test]
    fn dirichlet_overrides_row() {
        let mut system = LinearSystem::zeros(3).expect("system");
        system.add_coupling(0, 0, 2.0);
        system.add_coupling(0, 1, -1.0);
        apply_dirichlet(&mut system, CellId(0), 5.0).expect("bc");
        assert_eq!(system.diag()[0], 1.0);
        assert_eq!(system.rhs()[0], 5.0);
        assert_eq!(system.upper()[0], 0.0);
    }

    #[test]
    fn neumann_adds_conductance_and_flux() {
        let mesh = StructuredMesh1d::new("line", 4, 0.0, 1.0).expect("mesh");
        let mut system = LinearSystem::zeros(4).expect("system");
        apply_neumann(&mut system, &mesh, StructuredMesh1d::left_face(), 2.0, 1.0)
            .expect("neumann");
        let g = 2.0 / (mesh.dx() * 0.5);
        assert!((system.diag()[0] - g).abs() < 1.0e-12);
        assert!((system.rhs()[0] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn apply_set_with_assembly_enforces_dirichlet_faces() {
        let mesh = StructuredMesh1d::new("line", 4, 0.0, 1.0).expect("mesh");
        let mut system = LinearSystem::zeros(4).expect("system");
        crate::discretization::assemble_diffusion_1d(&mesh, &mut system, 1.0).expect("asm");
        let patches = BoundarySet::new(vec![
            BoundaryPatch::new(
                "left",
                vec![StructuredMesh1d::left_face()],
                BoundaryKind::dirichlet(0.0),
            ),
            BoundaryPatch::new(
                "right",
                vec![StructuredMesh1d::right_face()],
                BoundaryKind::dirichlet(1.0),
            ),
        ]);
        apply_boundary_conditions(&mesh, &patches, &mut system, 1.0).expect("bc");
        let x = system.solve_tridiagonal().expect("solve");
        // n=4 均匀网格解析离散解 φ₀ = 1/14（面 Dirichlet ghost 法）
        assert!((x[0] - 1.0 / 14.0).abs() < 1.0e-10);
        assert!((x[3] - 13.0 / 14.0).abs() < 1.0e-10);
    }
}
