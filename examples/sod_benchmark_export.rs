//! Sod 激波管 benchmark：导出数值/精确解对比文本，供 matplotlib 绘图。
//!
//! ```bash
//! cargo run --example sod_benchmark_export -- sod_profile.txt
//! cargo run --example sod_benchmark_export -- sod_profile.txt --cells 200 --time 0.2
//!
//! python3 scripts/plot_sod_benchmark.py sod_profile.txt -o sod_compare.png
//! ```

use std::env;
use std::path::PathBuf;
use std::process;

use asimu::solver::{SodBenchmarkConfig, run_sod_benchmark, write_sod_profile};

fn main() {
    if let Err(err) = run() {
        eprintln!("错误: {err}");
        process::exit(1);
    }
}

fn run() -> asimu::error::Result<()> {
    let (output, config) = parse_args().map_err(asimu::error::AsimuError::Config)?;
    let result = run_sod_benchmark(&config)?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    write_sod_profile(&output, &config, &result)?;
    println!("OK  {}", output.display());
    println!(
        "    ncells={} t={:.4} steps={} L1={:.6} L2={:.6}",
        config.ncells, result.final_time, result.steps, result.l1_density, result.l2_density
    );
    println!(
        "    绘图: python3 scripts/plot_sod_benchmark.py {}",
        output.display()
    );
    Ok(())
}

fn parse_args() -> Result<(PathBuf, SodBenchmarkConfig), String> {
    let mut args = env::args().skip(1);
    let output = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sod_profile.txt"));
    let mut config = SodBenchmarkConfig::default();
    let tail: Vec<String> = args.collect();
    let mut i = 0;
    while i < tail.len() {
        match tail[i].as_str() {
            "--cells" => {
                i += 1;
                config.ncells = tail
                    .get(i)
                    .ok_or("缺少 --cells 参数")?
                    .parse()
                    .map_err(|_| "--cells 须为正整数".to_string())?;
            }
            "--time" => {
                i += 1;
                config.final_time = tail
                    .get(i)
                    .ok_or("缺少 --time 参数")?
                    .parse()
                    .map_err(|_| "--time 须为正数".to_string())?;
            }
            "--length" => {
                i += 1;
                config.length = tail
                    .get(i)
                    .ok_or("缺少 --length 参数")?
                    .parse()
                    .map_err(|_| "--length 须为正数".to_string())?;
            }
            "--diaphragm" => {
                i += 1;
                config.diaphragm = tail
                    .get(i)
                    .ok_or("缺少 --diaphragm 参数")?
                    .parse()
                    .map_err(|_| "--diaphragm 须为数值".to_string())?;
            }
            flag => return Err(format!("未知参数: {flag}")),
        }
        i += 1;
    }
    if config.ncells == 0 {
        return Err("--cells 须大于 0".to_string());
    }
    Ok((output, config))
}
