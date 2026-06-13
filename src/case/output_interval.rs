//! Case 层间隔输出编排：统一判定与可压 / 不可压写出入口。

use std::path::PathBuf;

use tracing::info_span;

use crate::error::Result;
use crate::io::CaseSpec;
use crate::mesh::{MultiBlockStructuredMesh3d, StructuredMesh3d, UnstructuredMesh3d};
use crate::solver::{
    CompressibleMultiblockStepView, CompressibleUnstructuredStepView,
    IncompressiblePressureVelocityStepView,
};

pub use super::output_3d::interval_output_due;

/// 多块结构化可压缩：间隔残差 + 流场 CGNS。
pub fn maybe_write_compressible_structured_interval(
    case: &CaseSpec,
    mesh: &MultiBlockStructuredMesh3d,
    step: CompressibleMultiblockStepView<'_>,
) -> Result<Vec<PathBuf>> {
    if !interval_output_due(case, step.info.step) {
        return Ok(Vec::new());
    }
    let _span = info_span!(
        "maybe_write_compressible_structured_interval",
        step = step.info.step,
        blocks = step.fields.len()
    )
    .entered();
    let mut written =
        super::output_3d::maybe_write_residual_outputs(case, step.history, step.info)?;
    if let Some(path) =
        super::output_3d::maybe_write_interval_flow_snapshot(case, mesh, step.fields, step.info)?
    {
        written.push(path);
    }
    Ok(written)
}

/// 非结构可压缩：间隔残差 + 流场 CGNS。
pub fn maybe_write_compressible_unstructured_interval(
    case: &CaseSpec,
    mesh: &UnstructuredMesh3d,
    step: CompressibleUnstructuredStepView<'_>,
) -> Result<Vec<PathBuf>> {
    if !interval_output_due(case, step.info.step) {
        return Ok(Vec::new());
    }
    let _span = info_span!(
        "maybe_write_compressible_unstructured_interval",
        step = step.info.step,
        cells = mesh.num_cells()
    )
    .entered();
    let mut written = super::output_3d::write_residual_outputs(case, step.history)?;
    let Some(output) = &case.output else {
        return Ok(written);
    };
    let Some(base) = output.solution_cgns.as_ref() else {
        return Ok(written);
    };
    let name = super::output_3d::flow_cgns_name_for_step(base, step.info.step);
    let cgns_path =
        crate::io::resolve_case_output_path(case.case_dir.as_deref(), &output.dir, &name)?;
    super::compressible_unstructured_3d::write_unstructured_interval_flow(
        case,
        mesh,
        step.fields,
        step.info.physical_time,
        cgns_path.clone(),
    )?;
    written.push(cgns_path);
    Ok(written)
}

/// 不可压缩 SIMPLEC/PISO：间隔残差 + 流场 CGNS。
pub fn maybe_write_incompressible_interval(
    case: &CaseSpec,
    mesh: &StructuredMesh3d,
    step: IncompressiblePressureVelocityStepView<'_>,
) -> Result<Vec<PathBuf>> {
    if !interval_output_due(case, step.info.step) {
        return Ok(Vec::new());
    }
    let _span = info_span!("maybe_write_incompressible_interval", step = step.info.step).entered();
    let mut written =
        super::incompressible_3d::write_incompressible_residual_outputs(case, step.history)?;
    let Some(output) = &case.output else {
        return Ok(written);
    };
    let Some(base) = output.solution_cgns.as_ref() else {
        return Ok(written);
    };
    let name = super::output_3d::flow_cgns_name_for_step(base, step.info.step);
    let path = crate::io::resolve_case_output_path(case.case_dir.as_deref(), &output.dir, &name)?;
    super::incompressible_3d::write_incompressible_interval_flow_cgns(
        case,
        mesh,
        step.fields,
        step.info.nondimensional_time,
        &path,
    )?;
    written.push(path);
    Ok(written)
}
