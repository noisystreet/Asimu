//! CGNS 流场解写出（结构化 zone + 单元中心原始变量）。

#![allow(unsafe_code)]

use std::ffi::{CStr, CString};
use std::path::Path;

use crate::error::{AsimuError, Result};
use crate::field::ConservedFields;
use crate::io::limits::{io_error, validate_input_path};
use crate::mesh::StructuredMesh3d;
use crate::physics::{IdealGasEoS, PrimitiveState};

use super::ffi::{CG_OK, asimu_cg_write_structured_flow, cg_get_error};
use super::read::CGNS_LOCK;

/// 将 3D 守恒场写出为 CGNS（坐标 @ Vertex，ρ/u/v/w/p @ CellCenter）。
pub fn write_flow_cgns(
    path: &Path,
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    physical_time: f64,
) -> Result<()> {
    validate_input_path(path)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                io_error(
                    e.kind(),
                    format!("无法创建 CGNS 输出目录 {}: {e}", parent.display()),
                )
            })?;
        }
    }
    if fields.num_cells() != mesh.num_cells() {
        return Err(AsimuError::Field(format!(
            "场单元数 {} 与网格 {} 不一致",
            fields.num_cells(),
            mesh.num_cells()
        )));
    }

    let (rho, u, v, w, p) = gather_cell_primitives(mesh, fields, eos)?;

    let cpath = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| io_error(std::io::ErrorKind::InvalidInput, "CGNS 路径含内嵌 NUL 字节"))?;
    let base = CString::new("Base").expect("base");
    let zone =
        CString::new(mesh.name.as_str()).unwrap_or_else(|_| CString::new("Zone").expect("zone"));

    let _guard = CGNS_LOCK.lock().expect("CGNS lock");
    let err = unsafe {
        asimu_cg_write_structured_flow(
            cpath.as_ptr(),
            base.as_ptr(),
            zone.as_ptr(),
            mesh.nx as i32,
            mesh.ny as i32,
            mesh.nz as i32,
            mesh.points_x.as_ptr(),
            mesh.points_y.as_ptr(),
            mesh.points_z.as_ptr(),
            rho.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w.as_ptr(),
            p.as_ptr(),
            physical_time,
        )
    };
    check_cg(err)
}

type CellPrimitiveArrays = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn gather_cell_primitives(
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
) -> Result<CellPrimitiveArrays> {
    let n = mesh.num_cells();
    let mut rho = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    let mut p = Vec::with_capacity(n);
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let prim = fields.primitive_at(mesh.cell_index(i, j, k), eos)?;
                push_primitive(&mut rho, &mut u, &mut v, &mut w, &mut p, &prim);
            }
        }
    }
    Ok((rho, u, v, w, p))
}

#[cfg(test)]
/// 单元标量场按 CGNS 节点索引平均到顶点（历史 Vertex 写出测试用）。
fn scatter_cell_scalar_to_vertices(mesh: &StructuredMesh3d, cell: &[f64]) -> Vec<f64> {
    let npts = mesh.num_nodes();
    let mut node = vec![0.0; npts];
    let mut count = vec![0u32; npts];
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let c = cell[mesh.cell_index(i, j, k)];
                for dk in 0..2 {
                    for dj in 0..2 {
                        for di in 0..2 {
                            let idx = mesh.node_index(i + di, j + dj, k + dk);
                            node[idx] += c;
                            count[idx] += 1;
                        }
                    }
                }
            }
        }
    }
    for (value, n) in node.iter_mut().zip(count.iter()) {
        if *n > 0 {
            *value /= f64::from(*n);
        }
    }
    node
}

fn push_primitive(
    rho: &mut Vec<f64>,
    u: &mut Vec<f64>,
    v: &mut Vec<f64>,
    w: &mut Vec<f64>,
    p: &mut Vec<f64>,
    prim: &PrimitiveState,
) {
    rho.push(prim.density);
    u.push(prim.velocity[0]);
    v.push(prim.velocity[1]);
    w.push(prim.velocity[2]);
    p.push(prim.pressure);
}

fn check_cg(err: i32) -> Result<()> {
    if err == CG_OK {
        Ok(())
    } else {
        let msg = unsafe {
            CStr::from_ptr(cg_get_error())
                .to_string_lossy()
                .into_owned()
        };
        Err(AsimuError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("CGNS 写出失败: {msg}"),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::StructuredMesh3d;

    #[test]
    fn scatter_uniform_cell_field_to_vertices() {
        let mesh = StructuredMesh3d::new(
            "box",
            2,
            2,
            1,
            vec![
                0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
                1.0, 2.0,
            ],
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                1.0, 1.0,
            ],
            vec![0.0; 18],
        )
        .expect("mesh");
        let cell = vec![3.0; mesh.num_cells()];
        let node = scatter_cell_scalar_to_vertices(&mesh, &cell);
        assert_eq!(node.len(), mesh.num_nodes());
        assert!(node.iter().all(|&v| (v - 3.0).abs() < 1.0e-12));
    }
}
