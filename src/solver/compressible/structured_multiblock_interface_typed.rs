//! 结构化 typed 多块 1-to-1 共享接口通量 impl（ADR 0019 S2-a）。

use tracing::info_span;

use crate::core::{Real, Vector3};
use crate::discretization::GhostCellState;
use crate::discretization::face_flux_typed::face_inviscid_flux_first_order_boundary_soa_f32;
use crate::discretization::inviscid_f32::{FaceNormalF32, InviscidFluxF32};
use crate::error::Result;
use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};
use crate::solver::compressible::multiblock_interface::{
    InterfaceInviscidFlux, InterfaceResidualContribution, SharedInterfaceResidualParams,
    StructuredMultiblockInterfaceTyped, compute_shared_interface_residuals,
};

impl StructuredMultiblockInterfaceTyped for f64 {
    fn compute_shared_interface_residuals(
        params: &SharedInterfaceResidualParams<'_>,
    ) -> Result<Vec<Vec<InterfaceResidualContribution>>> {
        compute_shared_interface_residuals(params)
    }

    fn apply_interface_residuals(
        residual: &mut ConservedResidualT<f64>,
        contributions: &[InterfaceResidualContribution],
    ) -> Result<()> {
        for contribution in contributions {
            let InterfaceInviscidFlux::Real(flux) = &contribution.flux else {
                return Err(crate::error::AsimuError::Solver(
                    "f64 接口残差修正收到非 Real 通量".to_string(),
                ));
            };
            residual.add_flux_to_cell(
                contribution.cell,
                flux.mass,
                flux.momentum,
                flux.energy,
                contribution.scale,
            )?;
        }
        Ok(())
    }
}

impl StructuredMultiblockInterfaceTyped for f32 {
    fn compute_shared_interface_residuals(
        params: &SharedInterfaceResidualParams<'_>,
    ) -> Result<Vec<Vec<InterfaceResidualContribution>>> {
        let mut primitives = Vec::with_capacity(params.blocks.len());
        for (block, fields) in params.blocks.iter().zip(params.snapshots.iter()) {
            let _span = info_span!(
                "fill_interface_primitives_typed",
                block = %block.name,
                cells = block.mesh.num_cells(),
                precision = "f32",
            )
            .entered();
            let fields_f32 = ConservedFieldsT::<f32>::from_real_fields(fields)?;
            let mut prim = PrimitiveFieldsT::<f32>::zeros(block.mesh.num_cells())?;
            prim.fill_from_conserved(&fields_f32, params.eos, p_floor(params.freestream))?;
            primitives.push(prim);
        }
        let mut out = new_contribution_buffers(params.blocks.len(), params.shared_faces);
        for face in params.shared_faces {
            add_shared_interface_face_f32(params, &primitives, face, &mut out)?;
        }
        Ok(out)
    }

    fn apply_interface_residuals(
        residual: &mut ConservedResidualT<f32>,
        contributions: &[InterfaceResidualContribution],
    ) -> Result<()> {
        for contribution in contributions {
            let InterfaceInviscidFlux::F32(flux) = &contribution.flux else {
                return Err(crate::error::AsimuError::Solver(
                    "f32 接口残差修正收到非 F32 通量".to_string(),
                ));
            };
            crate::discretization::compressible::residual::add_inviscid_flux_f32_to_cell(
                residual,
                contribution.cell,
                flux,
                contribution.scale as f32,
            )?;
        }
        Ok(())
    }
}

fn new_contribution_buffers(
    blocks: usize,
    shared_faces: &[crate::solver::compressible::multiblock::SharedInterfaceFace],
) -> Vec<Vec<InterfaceResidualContribution>> {
    let mut counts = vec![0usize; blocks];
    for face in shared_faces {
        counts[face.owner_block_index] += 1;
        counts[face.donor_block_index] += 1;
    }
    counts.into_iter().map(Vec::with_capacity).collect()
}

fn add_shared_interface_face_f32(
    params: &SharedInterfaceResidualParams<'_>,
    primitives: &[PrimitiveFieldsT<f32>],
    face: &crate::solver::compressible::multiblock::SharedInterfaceFace,
    out: &mut [Vec<InterfaceResidualContribution>],
) -> Result<()> {
    let exterior = params.snapshots[face.donor_block_index].cell_state(face.donor_cell)?;
    let ghost = GhostCellState {
        conserved: exterior,
    };
    let flux = face_inviscid_flux_first_order_boundary_soa_f32(
        &primitives[face.owner_block_index],
        face.owner_cell,
        &ghost,
        face_normal_f32(face.normal),
        params.eos,
        params.inviscid,
        p_floor(params.freestream),
    )?;
    push_interface_contribution_f32(out, face, flux);
    Ok(())
}

fn push_interface_contribution_f32(
    out: &mut [Vec<InterfaceResidualContribution>],
    face: &crate::solver::compressible::multiblock::SharedInterfaceFace,
    flux: InviscidFluxF32,
) {
    let flux_tag = InterfaceInviscidFlux::F32(flux);
    out[face.owner_block_index].push(InterfaceResidualContribution {
        cell: face.owner_cell,
        flux: flux_tag.clone(),
        scale: face.owner_scale,
    });
    out[face.donor_block_index].push(InterfaceResidualContribution {
        cell: face.donor_cell,
        flux: flux_tag,
        scale: face.donor_scale,
    });
}

fn face_normal_f32(normal: Vector3) -> FaceNormalF32 {
    [normal.x as f32, normal.y as f32, normal.z as f32]
}

fn p_floor(freestream: &crate::physics::FreestreamParams) -> Real {
    crate::field::positivity_pressure_floor(freestream.pressure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundarySet;
    use crate::core::{ComputeFloat, approx_eq};
    use crate::discretization::inviscid_f32::inviscid_flux_f32_to_real;
    use crate::field::{ConservedFields, ConservedResidual, ConservedResidualT};
    use crate::mesh::{
        MultiBlockStructuredMesh3d, StructuredBlockInterface3d, StructuredIndexRange3d,
        StructuredMesh3d,
    };
    use crate::physics::{FreestreamParams, IdealGasEoS};
    use crate::solver::compressible::multiblock::build_multiblock_interface_metadata;
    use crate::solver::compressible::{
        CompressibleEulerConfig, CompressibleEulerSolver, MultiblockStructuredDriverInput,
        run_multiblock_structured_typed_with_observer,
    };
    use crate::solver::time::RungeKutta4Config;

    fn block(name: &str, nx: usize) -> StructuredMesh3d {
        StructuredMesh3d::uniform_box(name, nx, 1, 1, nx as Real, 1.0, 1.0).expect("block")
    }

    fn two_block_interface_mesh() -> MultiBlockStructuredMesh3d {
        MultiBlockStructuredMesh3d::with_interfaces(
            "pair",
            vec![block("a", 1), block("b", 1)],
            vec![StructuredBlockInterface3d {
                owner_block: "a".to_string(),
                donor_block: "b".to_string(),
                owner_range: StructuredIndexRange3d {
                    imin: 2,
                    imax: 2,
                    jmin: 1,
                    jmax: 2,
                    kmin: 1,
                    kmax: 2,
                },
                donor_range: StructuredIndexRange3d {
                    imin: 1,
                    imax: 1,
                    jmin: 1,
                    jmax: 2,
                    kmin: 1,
                    kmax: 2,
                },
                transform: [1, 2, 3],
            }],
        )
        .expect("mesh")
    }

    fn uniform_freestream_fields(
        mesh: &StructuredMesh3d,
        fs: &FreestreamParams,
    ) -> ConservedFields {
        let eos = IdealGasEoS::AIR_STANDARD;
        ConservedFields::from_freestream(mesh.num_cells(), &eos, fs).expect("fields")
    }

    #[test]
    fn f32_shared_interface_flux_matches_f64_on_uniform_freestream() {
        let mesh = two_block_interface_mesh();
        let metadata = build_multiblock_interface_metadata(&mesh).expect("metadata");
        assert_eq!(metadata.shared_faces.len(), 1);
        let fs = FreestreamParams::default();
        let eos = IdealGasEoS::AIR_STANDARD;
        let snapshots: Vec<ConservedFields> = mesh
            .blocks()
            .iter()
            .map(|block| uniform_freestream_fields(&block.mesh, &fs))
            .collect();
        let params = SharedInterfaceResidualParams {
            blocks: mesh.blocks(),
            shared_faces: &metadata.shared_faces,
            snapshots: &snapshots,
            eos: &eos,
            freestream: &fs,
            inviscid: &crate::discretization::InviscidFluxConfig::default(),
        };
        let f64_contribs = compute_shared_interface_residuals(&params).expect("f64");
        let f32_contribs = f32::compute_shared_interface_residuals(&params).expect("f32");
        assert_eq!(f64_contribs.len(), f32_contribs.len());
        for (f64_block, f32_block) in f64_contribs.iter().zip(f32_contribs.iter()) {
            assert_eq!(f64_block.len(), f32_block.len());
            for (f64_c, f32_c) in f64_block.iter().zip(f32_block.iter()) {
                assert_eq!(f64_c.cell, f32_c.cell);
                assert!(approx_eq(f64_c.scale, f32_c.scale, 1.0e-6));
                let InterfaceInviscidFlux::Real(f64_flux) = &f64_c.flux else {
                    panic!("expected f64 flux");
                };
                let InterfaceInviscidFlux::F32(f32_flux) = &f32_c.flux else {
                    panic!("expected f32 flux");
                };
                let mapped = inviscid_flux_f32_to_real(*f32_flux);
                assert!(approx_eq(f64_flux.mass, mapped.mass, 1.0e-3));
                for axis in 0..3 {
                    assert!(approx_eq(
                        f64_flux.momentum[axis],
                        mapped.momentum[axis],
                        1.0e-3
                    ));
                }
                assert!(approx_eq(f64_flux.energy, mapped.energy, 1.0e-3));
            }
        }
    }

    #[test]
    fn f32_apply_interface_residual_matches_f64_on_uniform_freestream() {
        let mesh = two_block_interface_mesh();
        let metadata = build_multiblock_interface_metadata(&mesh).expect("metadata");
        let fs = FreestreamParams::default();
        let eos = IdealGasEoS::AIR_STANDARD;
        let snapshots: Vec<ConservedFields> = mesh
            .blocks()
            .iter()
            .map(|block| uniform_freestream_fields(&block.mesh, &fs))
            .collect();
        let params = SharedInterfaceResidualParams {
            blocks: mesh.blocks(),
            shared_faces: &metadata.shared_faces,
            snapshots: &snapshots,
            eos: &eos,
            freestream: &fs,
            inviscid: &crate::discretization::InviscidFluxConfig::default(),
        };
        let f64_contribs = compute_shared_interface_residuals(&params).expect("f64");
        let f32_contribs = f32::compute_shared_interface_residuals(&params).expect("f32");
        for block_index in 0..mesh.num_blocks() {
            let n = mesh.blocks()[block_index].mesh.num_cells();
            let mut res_f64 = ConservedResidual::zeros(n).expect("res f64");
            crate::solver::compressible::multiblock_interface::apply_interface_residuals(
                &mut res_f64,
                &f64_contribs[block_index],
            )
            .expect("apply f64");
            let mut res_f32 = ConservedResidualT::<f32>::zeros(n).expect("res f32");
            f32::apply_interface_residuals(&mut res_f32, &f32_contribs[block_index])
                .expect("apply f32");
            for cell in 0..n {
                assert!(approx_eq(
                    res_f64.density.values()[cell],
                    res_f32.density.values()[cell].to_real(),
                    1.0e-3
                ));
            }
        }
    }

    #[test]
    fn f32_multiblock_typed_step_matches_f64_on_uniform_freestream() {
        let mesh = two_block_interface_mesh();
        let fs = FreestreamParams::default();
        let eos = IdealGasEoS::AIR_STANDARD;
        let initial: Vec<ConservedFields> = mesh
            .blocks()
            .iter()
            .map(|block| uniform_freestream_fields(&block.mesh, &fs))
            .collect();
        let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
            time: RungeKutta4Config {
                dt: 1.0e-4,
                max_steps: 1,
            },
            ..CompressibleEulerConfig::default()
        });
        let boundary = BoundarySet::new(vec![]);
        let run = |precision: &str| {
            let input = MultiblockStructuredDriverInput {
                solver: &solver,
                eos: &eos,
                freestream: &fs,
                mesh: &mesh,
                global_boundary: &boundary,
                reference: None,
                residual_tolerance: None,
                initial_fields: initial.clone(),
            };
            match precision {
                "f32" => run_multiblock_structured_typed_with_observer::<f32>(input, |_| Ok(())),
                _ => run_multiblock_structured_typed_with_observer::<f64>(input, |_| Ok(())),
            }
        };
        let (hist_f32, fields_f32) = run("f32").expect("f32");
        let (hist_f64, fields_f64) = run("f64").expect("f64");
        assert!(approx_eq(
            hist_f32[0].residual_rms,
            hist_f64[0].residual_rms,
            1.0e-5
        ));
        for (f32_block, f64_block) in fields_f32.iter().zip(fields_f64.iter()) {
            for (rho_f32, rho_f64) in f32_block
                .density
                .values()
                .iter()
                .zip(f64_block.density.values())
            {
                let rel = (rho_f32.to_real() - rho_f64).abs() / rho_f64.max(1.0e-12);
                assert!(rel < 1.0e-3, "rel={rel}");
            }
        }
    }
}
