//! 不可压缩 3D I0 占位求解器：初始化字段并写出流场。

use std::path::PathBuf;

#[cfg(not(feature = "io-cgns"))]
use tracing::warn;
use tracing::{info, info_span};

use crate::core::format_log_sci4;
use crate::error::{AsimuError, Result};
use crate::field::IncompressibleFields;
#[cfg(feature = "io-cgns")]
use crate::field::ScalarField;
use crate::io::{CaseSpec, resolve_case_output_path};
#[cfg(feature = "io-cgns")]
use crate::io::{
    StructuredVertexSolution, VertexScalarFieldView, write_structured_vertex_solution_cgns,
};
use crate::mesh::StructuredMesh3d;

use super::{CaseRunKind, CaseRunResult};

#[derive(Debug, Clone, PartialEq)]
pub struct Incompressible3dRunMetrics {
    pub steps: u64,
    pub physical_time: f64,
    pub written: Vec<PathBuf>,
}

pub fn run(case: &CaseSpec) -> Result<CaseRunResult> {
    let mesh = case.mesh.as_3d()?;
    let config = case
        .incompressible
        .as_ref()
        .ok_or_else(|| AsimuError::Config("不可压缩算例须包含 [incompressible] 段".to_string()))?;
    let steps = case.time.max_steps.unwrap_or(1);
    let dt = case.time.dt.unwrap_or(0.0);
    let physical_time = dt * steps as f64;
    let fields = IncompressibleFields::uniform(mesh.num_cells(), config.pressure, config.velocity)?;
    fields.validate_len(mesh.num_cells())?;

    let written = write_outputs(case, mesh, &fields, physical_time)?;
    info!(
        steps,
        t = %format_log_sci4(physical_time),
        "不可压缩 3D I0 placeholder 完成"
    );
    Ok(CaseRunResult {
        name: case.name.clone(),
        benchmark_id: case.benchmark_id.clone(),
        kind: CaseRunKind::Incompressible3dSteady,
        summary: format!("incompressible_3d_i0 steps={steps}"),
        diffusion: None,
        sod: None,
        compressible_3d: None,
        incompressible_3d: Some(Incompressible3dRunMetrics {
            steps,
            physical_time,
            written,
        }),
    })
}

fn write_outputs(
    case: &CaseSpec,
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    physical_time: f64,
) -> Result<Vec<PathBuf>> {
    let Some(output) = &case.output else {
        return Ok(Vec::new());
    };
    let mut written = Vec::new();
    if let Some(name) = &output.solution_cgns {
        let path = resolve_case_output_path(case.case_dir.as_deref(), &output.dir, name)?;
        write_incompressible_cgns(&path, mesh, fields, physical_time)?;
        info!(path = %path.display(), "已写出不可压缩流场 CGNS");
        written.push(path);
    }
    Ok(written)
}

#[cfg(feature = "io-cgns")]
fn write_incompressible_cgns(
    path: &std::path::Path,
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    physical_time: f64,
) -> Result<()> {
    let _span = info_span!("write_incompressible_cgns", path = %path.display()).entered();
    let vertex = gather_incompressible_vertex_fields(mesh, fields)?;
    let views = [
        VertexScalarFieldView {
            name: "Pressure",
            values: &vertex.pressure,
        },
        VertexScalarFieldView {
            name: "VelocityX",
            values: &vertex.velocity_x,
        },
        VertexScalarFieldView {
            name: "VelocityY",
            values: &vertex.velocity_y,
        },
        VertexScalarFieldView {
            name: "VelocityZ",
            values: &vertex.velocity_z,
        },
    ];
    write_structured_vertex_solution_cgns(
        path,
        mesh,
        StructuredVertexSolution {
            physical_time,
            fields: &views,
        },
    )
}

#[cfg(not(feature = "io-cgns"))]
fn write_incompressible_cgns(
    path: &std::path::Path,
    _mesh: &StructuredMesh3d,
    _fields: &IncompressibleFields,
    _physical_time: f64,
) -> Result<()> {
    let _span = info_span!("write_incompressible_cgns", path = %path.display()).entered();
    warn!("solution_cgns 须启用 feature io-cgns");
    Ok(())
}

#[cfg(feature = "io-cgns")]
struct IncompressibleVertexFields {
    pressure: Vec<f64>,
    velocity_x: Vec<f64>,
    velocity_y: Vec<f64>,
    velocity_z: Vec<f64>,
}

#[cfg(feature = "io-cgns")]
fn gather_incompressible_vertex_fields(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
) -> Result<IncompressibleVertexFields> {
    fields.validate_len(mesh.num_cells())?;
    Ok(IncompressibleVertexFields {
        pressure: gather_vertex_scalar(mesh, &fields.pressure)?,
        velocity_x: gather_vertex_scalar(mesh, &fields.velocity_x)?,
        velocity_y: gather_vertex_scalar(mesh, &fields.velocity_y)?,
        velocity_z: gather_vertex_scalar(mesh, &fields.velocity_z)?,
    })
}

#[cfg(feature = "io-cgns")]
fn gather_vertex_scalar(mesh: &StructuredMesh3d, field: &ScalarField) -> Result<Vec<f64>> {
    if field.len() != mesh.num_cells() {
        return Err(AsimuError::Field(format!(
            "不可压缩输出字段长度 {} 与单元数 {} 不一致",
            field.len(),
            mesh.num_cells()
        )));
    }
    let mut out = Vec::with_capacity(mesh.num_nodes());
    for k in 0..=mesh.nz {
        for j in 0..=mesh.ny {
            for i in 0..=mesh.nx {
                let mut sum = 0.0;
                let mut count = 0usize;
                let k0 = k.saturating_sub(1);
                let j0 = j.saturating_sub(1);
                let i0 = i.saturating_sub(1);
                let k1 = k.min(mesh.nz - 1);
                let j1 = j.min(mesh.ny - 1);
                let i1 = i.min(mesh.nx - 1);
                for ck in k0..=k1 {
                    for cj in j0..=j1 {
                        for ci in i0..=i1 {
                            sum += field.values()[mesh.cell_index(ci, cj, ck)];
                            count += 1;
                        }
                    }
                }
                out.push(sum / count.max(1) as f64);
            }
        }
    }
    Ok(out)
}
