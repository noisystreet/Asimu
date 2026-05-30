//! CGNS 流场解写出（结构化 zone + 顶点原始变量）。

#![allow(unsafe_code)]

use std::ffi::{CStr, CString};
use std::path::Path;

use crate::error::{AsimuError, Result};
use crate::field::ConservedFields;
use crate::io::limits::{io_error, validate_input_path};
use crate::io::vertex_field::gather_vertex_primitives;
use crate::mesh::StructuredMesh3d;
use crate::physics::IdealGasEoS;

use super::ffi::{CG_OK, asimu_cg_write_structured_flow, cg_get_error};
use super::read::CGNS_LOCK;

/// 将 3D 守恒场写出为 CGNS（坐标与 ρ/u/v/w/p 均在 Vertex；单元值经邻点平均）。
pub fn write_flow_cgns(
    path: &Path,
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    physical_time: f64,
    min_pressure: f64,
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

    let (rho, u, v, w, p) = gather_vertex_primitives(mesh, fields, eos, min_pressure)?;
    let npts = mesh.num_nodes();
    for (name, data) in [
        ("Density", rho.len()),
        ("VelocityX", u.len()),
        ("Pressure", p.len()),
    ] {
        if data != npts {
            return Err(AsimuError::Field(format!(
                "CGNS Vertex 场 {name} 长度 {data} 与网格顶点数 {npts} 不一致"
            )));
        }
    }

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

#[cfg(all(test, feature = "io-cgns"))]
mod cgns_write_tests {
    use super::*;
    use crate::field::ConservedFields;
    use crate::mesh::StructuredMesh3d;
    use crate::physics::{FreestreamParams, IdealGasEoS};
    use std::process::Command;

    #[test]
    fn flow_cgns_writes_vertex_grid_location() {
        let mesh = StructuredMesh3d::uniform_box("t", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let dir = std::env::temp_dir().join("asimu_cgns_vertex_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("flow.cgns");
        write_flow_cgns(&path, &mesh, &fields, &eos, 0.0, 1.0e-6).expect("write");

        let script = format!(
            r#"
import h5py
p = {path:?}
with h5py.File(p, "r") as f:
    found = {{}}
    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        if name.endswith("FlowSolution/GridLocation/ data"):
            found["gl"] = name
        elif name.endswith("FlowSolution/Density/ data"):
            found["rho"] = name
        elif name.endswith("GridCoordinates/CoordinateX/ data"):
            found["cx"] = name
    f.visititems(visit)
    assert "rho" in found and "cx" in found, found
    rho = f[found["rho"]].shape
    cx = f[found["cx"]].shape
    assert rho == cx, (rho, cx)
    if "gl" in found:
        gl = bytes(f[found["gl"]][()]).decode("ascii").replace("\x00", "").strip()
        assert gl == "Vertex", gl
print("ok")
"#,
            path = path.display().to_string()
        );
        let status = Command::new("python3")
            .arg("-c")
            .arg(&script)
            .status()
            .expect("python");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(status.success(), "python verify failed");
    }
}
