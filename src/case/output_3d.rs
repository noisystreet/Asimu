//! 3D 可压缩算例输出（残差 CSV / 曲线图 / 流场 CGNS；可选 VTU/VTS）。

use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::{info, warn};

use crate::core::{Real, format_log_sci4};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, positivity_pressure_floor};
#[cfg(feature = "io-cgns")]
use crate::io::write_flow_cgns;
use crate::io::{CaseSpec, resolve_case_output_path, write_residual_csv};
use crate::mesh::StructuredMesh3d;
use crate::physics::IdealGasEoS;
use crate::solver::CompressibleStepInfo;

pub fn write_compressible_3d_outputs(
    case: &CaseSpec,
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    history: &[CompressibleStepInfo],
) -> Result<Vec<PathBuf>> {
    let Some(output) = &case.output else {
        return Ok(Vec::new());
    };
    let mut written = Vec::new();

    if let Some(name) = &output.residual_csv {
        let path = resolve_case_output_path(case.case_dir.as_deref(), &output.dir, name)?;
        write_residual_csv(&path, history)?;
        info!(path = %path.display(), "已写出残差 CSV");
        written.push(path.clone());

        if let Some(plot_name) = &output.residual_plot {
            let plot_path =
                resolve_case_output_path(case.case_dir.as_deref(), &output.dir, plot_name)?;
            if let Err(err) = try_plot_residual(&path, &plot_path) {
                warn!(error = %err, "残差曲线图未生成（需 python3 + matplotlib）");
            } else {
                info!(path = %plot_path.display(), "已写出残差曲线图");
                written.push(plot_path);
            }
        }
    }

    if let Some(name) = &output.solution_cgns {
        let path = resolve_case_output_path(case.case_dir.as_deref(), &output.dir, name)?;
        let physical_time = history.last().map(|s| s.physical_time).unwrap_or(0.0);
        write_solution_flow(&path, mesh, fields, eos, physical_time, case)?;
        info!(path = %path.display(), t = %format_log_sci4(physical_time), "已写出流场 CGNS");
        written.push(path.clone());
        #[cfg(feature = "io-vtk")]
        if output.solution_vtk {
            written.push(flow_vtu_path(&path));
            written.push(flow_vts_path(&path));
        }
    }

    Ok(written)
}

/// 若当前步满足间隔条件，写出流场快照 CGNS。
pub fn maybe_write_flow_snapshot(
    case: &CaseSpec,
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    info: &CompressibleStepInfo,
) -> Result<Option<PathBuf>> {
    let output = match &case.output {
        Some(o) if o.wants_interval_flow() => o,
        _ => return Ok(None),
    };
    let every = output.solution_every.expect("wants_interval_flow");
    if info.step % every != 0 {
        return Ok(None);
    }
    let base = output.solution_cgns.as_ref().expect("wants_interval_flow");
    let name = flow_cgns_name_for_step(base, info.step);
    let path = resolve_case_output_path(case.case_dir.as_deref(), &output.dir, &name)?;
    write_solution_flow(&path, mesh, fields, eos, info.physical_time, case)?;
    #[cfg(feature = "io-vtk")]
    if output.solution_vtk {
        info!(
            cgns = %path.display(),
            vtu = %flow_vtu_path(&path).display(),
            step = info.step,
            t = %format_log_sci4(info.physical_time),
            every,
            "已写出间隔流场（ParaView 请打开 .vtu）"
        );
    } else {
        info!(
            path = %path.display(),
            step = info.step,
            t = %format_log_sci4(info.physical_time),
            every,
            "已写出间隔流场 CGNS"
        );
    }
    #[cfg(all(feature = "io-cgns", not(feature = "io-vtk")))]
    info!(
        path = %path.display(),
        step = info.step,
        t = %format_log_sci4(info.physical_time),
        every,
        "已写出间隔流场 CGNS"
    );
    Ok(Some(path))
}

/// 与 CGNS 流场文件同目录、同主文件名的 `.vts` 路径。
#[must_use]
pub fn flow_vts_path(cgns: &Path) -> PathBuf {
    cgns.with_extension("vts")
}

/// 与 CGNS 流场文件同目录、同主文件名的 `.vtu` 路径（ParaView 推荐）。
#[must_use]
pub fn flow_vtu_path(cgns: &Path) -> PathBuf {
    cgns.with_extension("vtu")
}

fn write_solution_flow(
    path: &Path,
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    physical_time: Real,
    case: &CaseSpec,
) -> Result<()> {
    let p_floor = case
        .freestream
        .or(case.fluid_initial.freestream)
        .map(|f| positivity_pressure_floor(f.pressure))
        .unwrap_or(1.0e-6);
    #[cfg(feature = "io-cgns")]
    {
        write_flow_cgns(path, mesh, fields, eos, physical_time, p_floor)?;
    }
    #[cfg(not(feature = "io-cgns"))]
    {
        let _ = (path, mesh, fields, eos, physical_time, p_floor);
        warn!("solution_cgns 须启用 feature io-cgns");
    }
    #[cfg(feature = "io-vtk")]
    if case.output.as_ref().is_some_and(|o| o.solution_vtk) {
        let vtu = flow_vtu_path(path);
        crate::io::write_flow_vtu(&vtu, mesh, fields, eos, p_floor)?;
        info!(
            path = %vtu.display(),
            "已写出流场 VTU（ParaView 请优先打开此文件）"
        );
        let vts = flow_vts_path(path);
        crate::io::write_flow_vts(&vts, mesh, fields, eos, p_floor)?;
        info!(
            path = %vts.display(),
            "已写出流场 VTS（备用；Coloring 选 Cell Data → Density）"
        );
    }
    Ok(())
}

/// 由 `flow.cgns` 或 `snapshots/flow.cgns` 生成间隔输出文件名。
#[must_use]
pub fn flow_cgns_name_for_step(base: &str, step: u64) -> String {
    let path = Path::new(base);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("flow");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("cgns");
    let file_name = format!("{stem}_step{step:06}.{ext}");
    match path
        .parent()
        .and_then(|p| p.to_str())
        .filter(|p| !p.is_empty())
    {
        Some(dir) => format!("{dir}/{file_name}"),
        None => file_name,
    }
}

fn try_plot_residual(csv: &Path, png: &Path) -> Result<()> {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/plot_residual.py");
    if !script.is_file() {
        return Err(AsimuError::Exec("plot_residual.py 不存在".to_string()));
    }
    if let Some(parent) = png.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let status = Command::new("python3")
        .arg(&script)
        .arg(csv)
        .arg("--output")
        .arg(png)
        .status()
        .map_err(|e| AsimuError::Exec(format!("调用 python3 失败: {e}")))?;
    if status.success() {
        Ok(())
    } else {
        Err(AsimuError::Exec(format!(
            "plot_residual.py 退出码 {:?}",
            status.code()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_snapshot_name_from_base() {
        assert_eq!(
            flow_cgns_name_for_step("flow.cgns", 10),
            "flow_step000010.cgns"
        );
        assert_eq!(
            flow_cgns_name_for_step("out/solution.cgns", 123),
            "out/solution_step000123.cgns"
        );
    }
}
