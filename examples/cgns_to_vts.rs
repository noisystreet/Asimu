//! CGNS 结构化 zone → VTS / VTM 转换。
//!
//! ```bash
//! # 全部 zone → .vtm + 子 VTS（ParaView 打开 .vtm）
//! cargo run --example cgns_to_vts --features io-cgns-vts -- \
//!   /path/to/mesh.cgns /path/to/out.vts
//!
//! # 单 zone → 单个 .vts
//! cargo run --example cgns_to_vts --features io-cgns-vts -- \
//!   /path/to/mesh.cgns /path/to/out.vts --zone 1
//!
//! # 每 zone 一个 .vts（输出目录，无 .vtm）
//! cargo run --example cgns_to_vts --features io-cgns-vts -- \
//!   /path/to/mesh.cgns /path/to/out_dir/
//! ```

use std::path::{Path, PathBuf};

use asimu::io::{export_cgns_to_vtm, export_cgns_zone_to_vts, list_cgns_zones};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("用法: cgns_to_vts <input.cgns> <output.vts|out_dir/> [--zone N]");
        std::process::exit(1);
    }
    let input = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let zone = parse_zone_flag(&args[3..]);

    if let Err(err) = run(&input, &output, zone) {
        eprintln!("错误: {err}");
        std::process::exit(1);
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

fn run(input: &Path, output: &Path, zone: Option<usize>) -> asimu::error::Result<()> {
    if output
        .extension()
        .is_some_and(|ext| ext == "vts" || ext == "vtm")
    {
        if let Some(zone_index) = zone {
            let loaded = export_cgns_zone_to_vts(input, zone_index, output)?;
            print_zone_ok(&loaded, output);
            return Ok(());
        }
        let loaded = export_cgns_to_vtm(input, output)?;
        let cells: usize = loaded.zones.iter().map(|z| z.mesh.num_cells()).sum();
        let vtm = loaded
            .vtm_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        println!(
            "OK  {} zones → {} (total cells={})",
            loaded.zones.len(),
            vtm,
            cells
        );
        println!("    ParaView 请打开上述 .vtm 文件");
        return Ok(());
    }

    std::fs::create_dir_all(output)?;
    let zones = list_cgns_zones(input)?;
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("mesh");
    for info in zones {
        if let Some(only) = zone {
            if info.index != only {
                continue;
            }
        }
        let out = output.join(format!("{stem}_zone{:02}.vts", info.index));
        let loaded = export_cgns_zone_to_vts(input, info.index, &out)?;
        print_zone_ok(&loaded, &out);
    }
    Ok(())
}

fn print_zone_ok(loaded: &asimu::io::CgnsLoadResult, output: &Path) {
    println!(
        "OK  zone {}/{} → {}",
        loaded.zone.index,
        loaded.zone.name,
        output.display()
    );
    if let asimu::mesh::StructuredMesh::D3(m) = &loaded.mesh {
        println!(
            "    cells={} nodes={} ({}×{}×{})",
            m.num_cells(),
            m.num_nodes(),
            m.nx,
            m.ny,
            m.nz
        );
    }
}
