//! 独立网格预检工具（计算前几何 / 度量 / 边界检查）。
//!
//! ```bash
//! cargo run --bin mesh_check --features io-cgns-vts -- case_cylinder/case.toml
//! cargo run --bin mesh_check --features io-cgns-vts -- cylinder.cgns --zone 1
//! cargo run --bin mesh_check --features io-cgns-vts -- cylinder.cgns --list-zones
//! cargo run --bin mesh_check --features io-cgns-vts -- case.toml --strict
//! ```

use std::path::{Path, PathBuf};
use std::process;

use clap::Parser;

use asimu::error::Result;
use asimu::io::{CaseMesh, list_cgns_zones, load_case, load_cgns_zone};
use asimu::mesh::{
    MeshCheckOptions, MeshCheckReport, MeshCheckReportDisplay, MeshMetricMode, StructuredMesh,
    check_mesh1d, check_mesh2d, check_mesh3d, check_multiblock_mesh3d,
};

#[derive(Debug, Parser)]
#[command(
    name = "mesh_check",
    about = "asimu 网格预检：计算前检查几何、度量与边界 patch",
    version
)]
struct Args {
    /// case.toml / mesh.cgns / mesh.vts
    input: PathBuf,

    /// CGNS zone 索引（从 1 开始）
    #[arg(long)]
    zone: Option<usize>,

    /// 列出 CGNS 文件内所有 zone
    #[arg(long)]
    list_zones: bool,

    /// 警告也视为失败（适合 CI）
    #[arg(long)]
    strict: bool,

    /// 度量模式覆盖（cartesian | curvilinear；CGNS 默认 curvilinear）
    #[arg(long, value_name = "MODE")]
    metric: Option<String>,
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(&args) {
        eprintln!("错误: {err}");
        process::exit(1);
    }
}

fn run(args: &Args) -> Result<()> {
    let opts = MeshCheckOptions {
        strict: args.strict,
    };

    let ext = args
        .input
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "toml" => {
            let case = load_case(&args.input)?;
            let source = args.input.display().to_string();
            let mut report = match &case.mesh {
                CaseMesh::Structured1d(mesh) => check_mesh1d(mesh, source),
                CaseMesh::Structured3d(mesh) => check_mesh3d(mesh, Some(&case.boundary), source)?,
                CaseMesh::MultiBlockStructured3d(mesh) => check_multiblock_mesh3d(mesh, source)?,
            };
            report.boundary_note =
                Some("已按 case.toml 解析（含 [freestream] / [euler] 等修正）".to_string());
            print_report(&report, opts)
        }
        "cgns" => check_cgns(&args.input, args, opts),
        "vts" => check_vts(&args.input, opts),
        other => Err(asimu::error::AsimuError::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("不支持的文件类型 \"{other}\"（支持 .toml / .cgns / .vts）"),
        ))),
    }
}

fn check_cgns(path: &Path, args: &Args, opts: MeshCheckOptions) -> Result<()> {
    if args.list_zones {
        let zones = list_cgns_zones(path)?;
        println!("CGNS zones in {}:", path.display());
        for info in zones {
            println!(
                "  [{:>2}] {:<24} cells={}×{}×{}",
                info.index, info.name, info.nx, info.ny, info.nz
            );
        }
        return Ok(());
    }

    let zone_index = args.zone.unwrap_or(1);
    let loaded = load_cgns_zone(path, zone_index)?;
    let mut mesh = match loaded.mesh {
        StructuredMesh::D3(mesh) => mesh,
        _ => {
            return Err(asimu::error::AsimuError::Mesh(
                "CGNS zone 须为 3D structured".to_string(),
            ));
        }
    };

    let metric_mode =
        parse_metric_override(args.metric.as_deref())?.unwrap_or(MeshMetricMode::Curvilinear);
    mesh.set_metric_mode(metric_mode);
    mesh.rebuild_metric_cache_if_needed()?;

    let source = format!(
        "CGNS {} zone {}/{}",
        path.display(),
        loaded.zone.index,
        loaded.zone.name
    );
    let report = check_mesh3d(&mesh, Some(&loaded.boundary), source)?;
    let mut report = report;
    report.boundary_note =
        Some("来自 CGNS ZoneBC（默认占位参数；未应用 case.toml 修正）".to_string());
    print_report(&report, opts)
}

#[cfg(feature = "io-vtk")]
fn check_vts(path: &Path, opts: MeshCheckOptions) -> Result<()> {
    use asimu::io::load_vts;

    let loaded = load_vts(path)?;
    let mut mesh = match loaded.mesh {
        StructuredMesh::D3(mesh) => mesh,
        StructuredMesh::D2(mesh) => {
            let report = check_mesh2d(&mesh, format!("VTS {}", path.display()));
            return print_report(&report, opts);
        }
    };
    mesh.set_metric_mode(MeshMetricMode::Curvilinear);
    mesh.rebuild_metric_cache_if_needed()?;
    let report = check_mesh3d(&mesh, None, format!("VTS {}", path.display()))?;
    print_report(&report, opts)
}

#[cfg(not(feature = "io-vtk"))]
fn check_vts(path: &Path, _opts: MeshCheckOptions) -> Result<()> {
    let _ = path;
    Err(asimu::error::AsimuError::Config(
        "VTS 读入须启用 feature io-vtk（建议使用 --features io-cgns-vts）".to_string(),
    ))
}

fn parse_metric_override(raw: Option<&str>) -> Result<Option<MeshMetricMode>> {
    match raw {
        None => Ok(None),
        Some("cartesian") => Ok(Some(MeshMetricMode::Cartesian)),
        Some("curvilinear") => Ok(Some(MeshMetricMode::Curvilinear)),
        Some(other) => Err(asimu::error::AsimuError::Config(format!(
            "不支持的 --metric \"{other}\"（支持 cartesian | curvilinear）"
        ))),
    }
}

fn print_report(report: &MeshCheckReport, opts: MeshCheckOptions) -> Result<()> {
    println!("{}", MeshCheckReportDisplay { report, opts });
    if !report.passed_with(opts) {
        process::exit(2);
    }
    Ok(())
}
