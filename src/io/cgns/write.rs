//! CGNS 流场解写出（结构化 zone + 顶点原始变量）。

#![allow(unsafe_code)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;

use crate::error::{AsimuError, Result};
use crate::field::ConservedFields;
use crate::io::limits::{io_error, validate_input_path};
use crate::io::vertex_field::{gather_unstructured_cell_primitives, gather_vertex_primitives};
use crate::mesh::{CellKind, MultiBlockStructuredMesh3d, StructuredMesh3d, UnstructuredMesh3d};
use crate::physics::IdealGasEoS;

use super::ffi::{
    CG_OK, CgSize, asimu_cg_write_multiblock_structured_flow, asimu_cg_write_structured_flow,
    asimu_cg_write_structured_solution_fields, asimu_cg_write_unstructured_flow, cg_get_error,
};
use super::read::CGNS_LOCK;

struct VertexFlowArrays {
    rho: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    w: Vec<f64>,
    p: Vec<f64>,
    mach: Vec<f64>,
    temperature: Vec<f64>,
}

/// 通用 CGNS Vertex 标量场视图。
pub struct VertexScalarFieldView<'a> {
    pub name: &'a str,
    pub values: &'a [f64],
}

/// 单 Zone 结构化 CGNS 输出字段集合。
pub struct StructuredVertexSolution<'a> {
    pub physical_time: f64,
    pub fields: &'a [VertexScalarFieldView<'a>],
}

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
    create_output_parent(path)?;
    let arrays = prepare_vertex_flow_arrays(mesh, fields, eos, min_pressure)?;

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
            arrays.rho.as_ptr(),
            arrays.u.as_ptr(),
            arrays.v.as_ptr(),
            arrays.w.as_ptr(),
            arrays.p.as_ptr(),
            arrays.mach.as_ptr(),
            arrays.temperature.as_ptr(),
            physical_time,
        )
    };
    check_cg(err)
}

/// 将非结构 3D 守恒场写出为 CGNS（坐标 @ Vertex；ρ/u/v/w/p/Mach/T @ CellCenter；按单元类型分 section）。
pub fn write_flow_cgns_unstructured(
    path: &Path,
    mesh: &UnstructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    physical_time: f64,
    min_pressure: f64,
) -> Result<()> {
    validate_input_path(path)?;
    create_output_parent(path)?;
    if fields.num_cells() != mesh.num_cells() {
        return Err(AsimuError::Field(format!(
            "场单元数 {} 与网格 {} 不一致",
            fields.num_cells(),
            mesh.num_cells()
        )));
    }

    let (rho, u, v, w, p, mach, temperature) =
        gather_unstructured_cell_primitives(fields, eos, min_pressure)?;
    validate_cell_center_flow_len(mesh, &rho, &u, &p, &mach, &temperature)?;

    let num_nodes = mesh.num_nodes();
    let num_cells = mesh.num_cells();
    let mut points_x = Vec::with_capacity(num_nodes);
    let mut points_y = Vec::with_capacity(num_nodes);
    let mut points_z = Vec::with_capacity(num_nodes);
    for point in mesh.points() {
        points_x.push(point[0]);
        points_y.push(point[1]);
        points_z.push(point[2]);
    }

    let sections = build_fixed_element_sections(mesh);
    let section_names = sections
        .iter()
        .map(|section| {
            CString::new(section.name.as_str()).map_err(|_| {
                io_error(
                    std::io::ErrorKind::InvalidInput,
                    format!("CGNS section 名含内嵌 NUL 字节: {}", section.name),
                )
            })
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let name_ptrs = section_names
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<*const c_char>>();
    let element_types: Vec<i32> = sections
        .iter()
        .map(|section| section.element_type)
        .collect();
    let section_starts: Vec<i32> = sections.iter().map(|section| section.start).collect();
    let section_ends: Vec<i32> = sections.iter().map(|section| section.end).collect();
    let connectivity_ptrs: Vec<*const CgSize> = sections
        .iter()
        .map(|section| section.connectivity.as_ptr())
        .collect();

    let cpath = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| io_error(std::io::ErrorKind::InvalidInput, "CGNS 路径含内嵌 NUL 字节"))?;
    let base = CString::new("Base").expect("base");
    let zone = CString::new(mesh.name()).unwrap_or_else(|_| CString::new("Zone").expect("zone"));

    let _guard = CGNS_LOCK.lock().expect("CGNS lock");
    let err = unsafe {
        asimu_cg_write_unstructured_flow(
            cpath.as_ptr(),
            base.as_ptr(),
            zone.as_ptr(),
            i32_from_usize(num_nodes, "num_nodes")?,
            i32_from_usize(num_cells, "num_cells")?,
            points_x.as_ptr(),
            points_y.as_ptr(),
            points_z.as_ptr(),
            i32_from_usize(sections.len(), "section_count")?,
            name_ptrs.as_ptr(),
            element_types.as_ptr(),
            section_starts.as_ptr(),
            section_ends.as_ptr(),
            connectivity_ptrs.as_ptr(),
            rho.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w.as_ptr(),
            p.as_ptr(),
            mach.as_ptr(),
            temperature.as_ptr(),
            physical_time,
        )
    };
    check_cg(err)
}

/// 将任意 Vertex 标量字段写出为单 Zone 结构化 CGNS。
pub fn write_structured_vertex_solution_cgns(
    path: &Path,
    mesh: &StructuredMesh3d,
    solution: StructuredVertexSolution<'_>,
) -> Result<()> {
    validate_input_path(path)?;
    create_output_parent(path)?;
    validate_vertex_solution_fields(mesh, solution.fields)?;

    let cpath = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| io_error(std::io::ErrorKind::InvalidInput, "CGNS 路径含内嵌 NUL 字节"))?;
    let base = CString::new("Base").expect("base");
    let zone =
        CString::new(mesh.name.as_str()).unwrap_or_else(|_| CString::new("Zone").expect("zone"));
    let names = solution
        .fields
        .iter()
        .map(|field| {
            CString::new(field.name).map_err(|_| {
                io_error(
                    std::io::ErrorKind::InvalidInput,
                    format!("CGNS 字段名含内嵌 NUL 字节: {}", field.name),
                )
            })
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let name_ptrs = names
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<*const c_char>>();
    let value_ptrs = solution
        .fields
        .iter()
        .map(|field| field.values.as_ptr())
        .collect::<Vec<*const f64>>();

    let _guard = CGNS_LOCK.lock().expect("CGNS lock");
    let err = unsafe {
        asimu_cg_write_structured_solution_fields(
            cpath.as_ptr(),
            base.as_ptr(),
            zone.as_ptr(),
            mesh.nx as c_int,
            mesh.ny as c_int,
            mesh.nz as c_int,
            mesh.points_x.as_ptr(),
            mesh.points_y.as_ptr(),
            mesh.points_z.as_ptr(),
            solution.fields.len() as c_int,
            name_ptrs.as_ptr().cast::<*const c_char>(),
            value_ptrs.as_ptr(),
            solution.physical_time,
        )
    };
    check_cg(err)
}

/// 将多块 3D 守恒场写出为单个 CGNS 文件（每个 block 一个 Structured Zone）。
pub fn write_multiblock_flow_cgns(
    path: &Path,
    mesh: &MultiBlockStructuredMesh3d,
    fields: &[ConservedFields],
    eos: &IdealGasEoS,
    physical_time: f64,
    min_pressure: f64,
) -> Result<()> {
    validate_input_path(path)?;
    create_output_parent(path)?;
    if fields.len() != mesh.num_blocks() {
        return Err(AsimuError::Field(format!(
            "多块流场数量 {} 与 block 数 {} 不一致",
            fields.len(),
            mesh.num_blocks()
        )));
    }

    let mut names = Vec::with_capacity(mesh.num_blocks());
    let mut name_ptrs = Vec::with_capacity(mesh.num_blocks());
    let mut nx = Vec::with_capacity(mesh.num_blocks());
    let mut ny = Vec::with_capacity(mesh.num_blocks());
    let mut nz = Vec::with_capacity(mesh.num_blocks());
    let mut arrays = Vec::with_capacity(mesh.num_blocks());

    for (block, field) in mesh.blocks().iter().zip(fields.iter()) {
        names.push(
            CString::new(block.name.as_str())
                .unwrap_or_else(|_| CString::new("Zone").expect("zone")),
        );
        nx.push(block.mesh.nx as c_int);
        ny.push(block.mesh.ny as c_int);
        nz.push(block.mesh.nz as c_int);
        arrays.push(prepare_vertex_flow_arrays(
            &block.mesh,
            field,
            eos,
            min_pressure,
        )?);
    }
    for name in &names {
        name_ptrs.push(name.as_ptr());
    }

    let point_x: Vec<*const f64> = mesh
        .blocks()
        .iter()
        .map(|block| block.mesh.points_x.as_ptr())
        .collect();
    let point_y: Vec<*const f64> = mesh
        .blocks()
        .iter()
        .map(|block| block.mesh.points_y.as_ptr())
        .collect();
    let point_z: Vec<*const f64> = mesh
        .blocks()
        .iter()
        .map(|block| block.mesh.points_z.as_ptr())
        .collect();
    let rho = array_ptrs(&arrays, |a| a.rho.as_ptr());
    let u = array_ptrs(&arrays, |a| a.u.as_ptr());
    let v = array_ptrs(&arrays, |a| a.v.as_ptr());
    let w = array_ptrs(&arrays, |a| a.w.as_ptr());
    let p = array_ptrs(&arrays, |a| a.p.as_ptr());
    let mach = array_ptrs(&arrays, |a| a.mach.as_ptr());
    let temperature = array_ptrs(&arrays, |a| a.temperature.as_ptr());

    let cpath = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| io_error(std::io::ErrorKind::InvalidInput, "CGNS 路径含内嵌 NUL 字节"))?;
    let base = CString::new("Base").expect("base");
    let _guard = CGNS_LOCK.lock().expect("CGNS lock");
    let err = unsafe {
        asimu_cg_write_multiblock_structured_flow(
            cpath.as_ptr(),
            base.as_ptr(),
            mesh.num_blocks() as c_int,
            name_ptrs.as_ptr().cast::<*const c_char>(),
            nx.as_ptr(),
            ny.as_ptr(),
            nz.as_ptr(),
            point_x.as_ptr(),
            point_y.as_ptr(),
            point_z.as_ptr(),
            rho.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w.as_ptr(),
            p.as_ptr(),
            mach.as_ptr(),
            temperature.as_ptr(),
            physical_time,
        )
    };
    check_cg(err)
}

fn create_output_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            io_error(
                e.kind(),
                format!("无法创建 CGNS 输出目录 {}: {e}", parent.display()),
            )
        })?;
    }
    Ok(())
}

fn prepare_vertex_flow_arrays(
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: f64,
) -> Result<VertexFlowArrays> {
    if fields.num_cells() != mesh.num_cells() {
        return Err(AsimuError::Field(format!(
            "场单元数 {} 与网格 {} 不一致",
            fields.num_cells(),
            mesh.num_cells()
        )));
    }
    let (rho, u, v, w, p, mach, temperature) =
        gather_vertex_primitives(mesh, fields, eos, min_pressure)?;
    validate_vertex_flow_len(mesh, &rho, &u, &p, &mach, &temperature)?;
    Ok(VertexFlowArrays {
        rho,
        u,
        v,
        w,
        p,
        mach,
        temperature,
    })
}

struct UnstructuredCgnsSection {
    name: String,
    element_type: i32,
    start: i32,
    end: i32,
    connectivity: Vec<CgSize>,
}

fn build_fixed_element_sections(mesh: &UnstructuredMesh3d) -> Vec<UnstructuredCgnsSection> {
    let mut sections: Vec<UnstructuredCgnsSection> = Vec::new();
    for (index, cell) in mesh.cells().iter().enumerate() {
        let element_id = i32::try_from(index + 1).expect("CGNS element id");
        let mut nodes = Vec::with_capacity(cell.kind.node_count());
        for node in &cell.nodes {
            nodes.push(i32::try_from(node.index() + 1).expect("CGNS 节点索引") as CgSize);
        }
        if let Some(last) = sections.last_mut()
            && last.element_type == cell.kind.cgns_element_type()
            && last.end + 1 == element_id
        {
            last.end = element_id;
            last.connectivity.extend_from_slice(&nodes);
            continue;
        }
        sections.push(UnstructuredCgnsSection {
            name: format!("{}_{element_id}", cell_kind_section_prefix(cell.kind)),
            element_type: cell.kind.cgns_element_type(),
            start: element_id,
            end: element_id,
            connectivity: nodes,
        });
    }
    sections
}

fn cell_kind_section_prefix(kind: CellKind) -> &'static str {
    match kind {
        CellKind::Tet => "Tet",
        CellKind::Hex => "Hex",
        CellKind::Pyramid => "Pyramid",
        CellKind::Prism => "Prism",
    }
}

fn i32_from_usize(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| AsimuError::Mesh(format!("CGNS {label}={value} 超出 i32 范围")))
}

fn validate_cell_center_flow_len(
    mesh: &UnstructuredMesh3d,
    rho: &[f64],
    u: &[f64],
    p: &[f64],
    mach: &[f64],
    temperature: &[f64],
) -> Result<()> {
    let ncells = mesh.num_cells();
    for (name, data) in [
        ("Density", rho.len()),
        ("VelocityX", u.len()),
        ("Pressure", p.len()),
        ("MachNumber", mach.len()),
        ("Temperature", temperature.len()),
    ] {
        if data != ncells {
            return Err(AsimuError::Field(format!(
                "CGNS CellCenter 场 {name} 长度 {data} 与网格单元数 {ncells} 不一致"
            )));
        }
    }
    Ok(())
}

fn validate_vertex_flow_len(
    mesh: &StructuredMesh3d,
    rho: &[f64],
    u: &[f64],
    p: &[f64],
    mach: &[f64],
    temperature: &[f64],
) -> Result<()> {
    let npts = mesh.num_nodes();
    for (name, data) in [
        ("Density", rho.len()),
        ("VelocityX", u.len()),
        ("Pressure", p.len()),
        ("MachNumber", mach.len()),
        ("Temperature", temperature.len()),
    ] {
        if data != npts {
            return Err(AsimuError::Field(format!(
                "CGNS Vertex 场 {name} 长度 {data} 与网格顶点数 {npts} 不一致"
            )));
        }
    }
    Ok(())
}

fn validate_vertex_solution_fields(
    mesh: &StructuredMesh3d,
    fields: &[VertexScalarFieldView<'_>],
) -> Result<()> {
    if fields.is_empty() {
        return Err(AsimuError::Field(
            "CGNS Vertex 输出至少需要一个物理量".to_string(),
        ));
    }
    let npts = mesh.num_nodes();
    for field in fields {
        if field.name.trim().is_empty() {
            return Err(AsimuError::Field("CGNS 字段名不能为空".to_string()));
        }
        if field.values.len() != npts {
            return Err(AsimuError::Field(format!(
                "CGNS Vertex 场 {} 长度 {} 与网格顶点数 {npts} 不一致",
                field.name,
                field.values.len()
            )));
        }
    }
    Ok(())
}

fn array_ptrs(
    arrays: &[VertexFlowArrays],
    ptr: impl Fn(&VertexFlowArrays) -> *const f64,
) -> Vec<*const f64> {
    arrays.iter().map(ptr).collect()
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
    use crate::mesh::{MultiBlockStructuredMesh3d, StructuredMesh3d};
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
        elif name.endswith("FlowSolution/MachNumber/ data"):
            found["mach"] = name
        elif name.endswith("FlowSolution/Temperature/ data"):
            found["temp"] = name
        elif name.endswith("GridCoordinates/CoordinateX/ data"):
            found["cx"] = name
    f.visititems(visit)
    assert "rho" in found and "mach" in found and "temp" in found and "cx" in found, found
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

    #[test]
    fn multiblock_flow_cgns_writes_multiple_zones() {
        let a = StructuredMesh3d::uniform_box("a", 1, 1, 1, 1.0, 1.0, 1.0).expect("a");
        let b = StructuredMesh3d::uniform_box("b", 1, 1, 1, 1.0, 1.0, 1.0).expect("b");
        let mesh = MultiBlockStructuredMesh3d::new("multi", vec![a, b]).expect("multi");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let fields = mesh
            .blocks()
            .iter()
            .map(|block| ConservedFields::from_freestream(block.mesh.num_cells(), &eos, &fs))
            .collect::<Result<Vec<_>>>()
            .expect("fields");
        let dir = std::env::temp_dir().join("asimu_cgns_multiblock_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("flow.cgns");
        write_multiblock_flow_cgns(&path, &mesh, &fields, &eos, 0.0, 1.0e-6).expect("write");

        let script = format!(
            r#"
import h5py
p = {path:?}
found = {{"rho": 0, "cx": 0}}
with h5py.File(p, "r") as f:
    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        if name.endswith("FlowSolution/Density/ data"):
            found["rho"] += 1
        elif name.endswith("GridCoordinates/CoordinateX/ data"):
            found["cx"] += 1
    f.visititems(visit)
assert found == {{"rho": 2, "cx": 2}}, found
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

    #[test]
    fn unstructured_flow_cgns_writes_cell_center_grid_location() {
        use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

        let mesh = UnstructuredMesh3d::new(
            "mixed",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            vec![
                UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("tet"),
                UnstructuredCell::new(CellKind::Prism, vec![1, 4, 2, 5, 6, 3]).expect("wedge"),
            ],
        )
        .expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fields =
            ConservedFields::from_freestream(mesh.num_cells(), &eos, &FreestreamParams::default())
                .expect("fields");
        let dir = std::env::temp_dir().join("asimu_cgns_unstructured_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("flow.cgns");
        write_flow_cgns_unstructured(&path, &mesh, &fields, &eos, 0.0, 1.0e-6).expect("write");

        let loaded =
            super::super::unstructured::load_cgns_unstructured_zone(&path, 1).expect("load");
        assert_eq!(loaded.mesh.num_cells(), 2);
        assert_eq!(loaded.mesh.num_nodes(), 7);

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
    assert f[found["rho"]].shape[0] == 2, f[found["rho"]].shape
    assert f[found["cx"]].shape[0] == 7, f[found["cx"]].shape
    if "gl" in found:
        gl = bytes(f[found["gl"]][()]).decode("ascii").replace("\x00", "").strip()
        assert gl == "CellCenter", gl
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
