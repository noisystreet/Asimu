//! 结构化 / CGNS / VTS / case 网格探测与诊断报告。
//!
//! 完整预检请使用独立工具 `mesh_check`：
//! ```bash
//! cargo run --bin mesh_check -- case_cylinder/case.toml
//! ```
//!
//! ```bash
//! cargo run --example mesh_probe -- cylinder.cgns
//! cargo run --example mesh_probe -- mesh.vts
//! cargo run --example mesh_probe -- case.toml
//! cargo run --example mesh_probe -- mesh.cgns --zone 2
//! cargo run --example mesh_probe -- mesh.cgns --list-zones
//! ```

use std::path::{Path, PathBuf};
use std::process;

use asimu::io::{
    list_cgns_zones, load_case, load_cgns_zone, load_vts, report_case_mesh, report_cgns_zone,
    report_vts,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("用法: mesh_probe <file.cgns|file.vts|case.toml> [--zone N] [--list-zones]");
        process::exit(2);
    }
    let path = PathBuf::from(&args[1]);
    let list_zones = args.iter().any(|arg| arg == "--list-zones");
    let zone = parse_zone_flag(&args[2..]);

    if let Err(err) = run(&path, list_zones, zone) {
        eprintln!("错误: {err}");
        process::exit(1);
    }
}

fn parse_zone_flag(args: &[String]) -> Option<usize> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--zone" {
            return iter.next().and_then(|s| s.parse().ok()).filter(|z| *z > 0);
        }
    }
    None
}

fn run(path: &Path, list_zones: bool, zone: Option<usize>) -> asimu::error::Result<()> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "cgns" => probe_cgns(path, list_zones, zone),
        "vts" => probe_vts(path),
        "toml" => probe_case(path),
        other => Err(asimu::error::AsimuError::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("不支持的文件类型 \"{other}\"（支持 .cgns / .vts / .toml）"),
        ))),
    }
}

fn probe_cgns(path: &Path, list_zones: bool, zone: Option<usize>) -> asimu::error::Result<()> {
    if list_zones {
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

    let zone_index = zone.unwrap_or(1);
    let loaded = load_cgns_zone(path, zone_index)?;
    let report = report_cgns_zone(&loaded);
    print!("{report}");
    Ok(())
}

fn probe_vts(path: &Path) -> asimu::error::Result<()> {
    let loaded = load_vts(path)?;
    let report = report_vts(&loaded, path);
    print!("{report}");
    Ok(())
}

fn probe_case(path: &Path) -> asimu::error::Result<()> {
    let case = load_case(path)?;
    let report = report_case_mesh(path.display().to_string(), &case.mesh, &case.boundary);
    print!("{report}");
    Ok(())
}
