//! Sod 激波管 benchmark：导出 MUSCL+Roe 与 MUSCL+HLLC 数值解及精确解对比文本。
//!
//! 默认使用 **van Albada** 斜率限制器（MUSCL 重构）。
//!
//! ```bash
//! cargo run --example sod_benchmark_export -- sod_compare.txt
//! cargo run --example sod_benchmark_export -- sod_compare.txt --cells 200 --time 0.2
//! cargo run --example sod_benchmark_export -- sod_compare.txt --limiter minmod
//!
//! python3 scripts/plot_sod_benchmark.py sod_compare.txt -o sod_compare.png
//! ```

use std::env;
use std::path::PathBuf;
use std::process;

use asimu::discretization::{InviscidFluxConfig, SlopeLimiter};
use asimu::solver::{
    SodBenchmarkConfig, run_sod_benchmark, write_sod_compare_profile, write_sod_profile,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("错误: {err}");
        process::exit(1);
    }
}

fn run() -> asimu::error::Result<()> {
    let (output, base, limiter, mode) = parse_args().map_err(asimu::error::AsimuError::Config)?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    match mode {
        ExportMode::Compare => {
            let roe_config = muscl_config(InviscidFluxConfig::muscl_roe(), limiter, base);
            let muscl_config = muscl_config(InviscidFluxConfig::muscl_hllc(), limiter, base);
            let roe_result = run_sod_benchmark(&roe_config)?;
            let muscl_result = run_sod_benchmark(&muscl_config)?;
            write_sod_compare_profile(
                &output,
                &base,
                (&roe_config, &roe_result),
                (&muscl_config, &muscl_result),
            )?;
            println!("OK  {}", output.display());
            println!(
                "    ncells={} t={:.4} limiter={}",
                base.ncells,
                roe_result.final_time,
                limiter_name(limiter)
            );
            println!(
                "    MUSCL+Roe:   steps={} L1={:.6} L2={:.6}",
                roe_result.steps, roe_result.l1_density, roe_result.l2_density
            );
            println!(
                "    MUSCL+HLLC:  steps={} L1={:.6} L2={:.6}",
                muscl_result.steps, muscl_result.l1_density, muscl_result.l2_density
            );
        }
        ExportMode::Single(config) => {
            let result = run_sod_benchmark(&config)?;
            write_sod_profile(&output, &config, &result)?;
            println!("OK  {}", output.display());
            println!(
                "    scheme={} limiter={} ncells={} t={:.4} steps={} L1={:.6} L2={:.6}",
                config.inviscid.short_label(),
                config.inviscid.limiter_label(),
                config.ncells,
                result.final_time,
                result.steps,
                result.l1_density,
                result.l2_density
            );
        }
    }

    println!(
        "    绘图: python3 scripts/plot_sod_benchmark.py {}",
        output.display()
    );
    Ok(())
}

fn muscl_config(
    preset: InviscidFluxConfig,
    limiter: SlopeLimiter,
    base: SodBenchmarkConfig,
) -> SodBenchmarkConfig {
    SodBenchmarkConfig {
        inviscid: preset.with_limiter(limiter),
        ..base
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ExportMode {
    Compare,
    Single(SodBenchmarkConfig),
}

struct CliState {
    base: SodBenchmarkConfig,
    limiter: SlopeLimiter,
    mode: ExportMode,
}

fn limiter_name(limiter: SlopeLimiter) -> &'static str {
    InviscidFluxConfig::muscl_roe()
        .with_limiter(limiter)
        .limiter_label()
}

fn parse_limiter(name: &str) -> Result<SlopeLimiter, String> {
    match name {
        "minmod" => Ok(SlopeLimiter::Minmod),
        "van_leer" | "vanleer" | "van-leer" => Ok(SlopeLimiter::VanLeer),
        "van_albada" | "vanalbada" | "van-albada" => Ok(SlopeLimiter::VanAlbada),
        other => Err(format!(
            "未知 limiter: {other}（可选 minmod | van_leer | van_albada）"
        )),
    }
}

fn parse_scheme(scheme: &str, limiter: SlopeLimiter) -> Result<InviscidFluxConfig, String> {
    match scheme {
        "roe" | "muscl_roe" | "muscl-roe" => {
            Ok(InviscidFluxConfig::muscl_roe().with_limiter(limiter))
        }
        "roe_first_order" | "first_order_roe" => Ok(InviscidFluxConfig::roe_first_order()),
        "muscl_hllc" | "muscl-hllc" => Ok(InviscidFluxConfig::muscl_hllc().with_limiter(limiter)),
        other => Err(format!(
            "未知 scheme: {other}（可选 roe | muscl_hllc | roe_first_order）"
        )),
    }
}

fn next_token<'a>(tail: &'a [String], i: &mut usize, label: &str) -> Result<&'a str, String> {
    *i += 1;
    tail.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("缺少 {label} 参数"))
}

fn parse_next_usize(tail: &[String], i: &mut usize, label: &str) -> Result<usize, String> {
    next_token(tail, i, label)?
        .parse()
        .map_err(|_| format!("{label} 须为正整数"))
}

fn parse_next_float(tail: &[String], i: &mut usize, label: &str) -> Result<f64, String> {
    next_token(tail, i, label)?
        .parse()
        .map_err(|_| format!("{label} 须为数值"))
}

fn apply_numeric_flag(
    state: &mut CliState,
    flag: &str,
    tail: &[String],
    i: &mut usize,
) -> Result<bool, String> {
    match flag {
        "--cells" => state.base.ncells = parse_next_usize(tail, i, "--cells")?,
        "--time" => state.base.final_time = parse_next_float(tail, i, "--time")?,
        "--length" => state.base.length = parse_next_float(tail, i, "--length")?,
        "--diaphragm" => state.base.diaphragm = parse_next_float(tail, i, "--diaphragm")?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn apply_mode_flag(
    state: &mut CliState,
    flag: &str,
    tail: &[String],
    i: &mut usize,
) -> Result<bool, String> {
    match flag {
        "--limiter" => state.limiter = parse_limiter(next_token(tail, i, "--limiter")?)?,
        "--compare" => state.mode = ExportMode::Compare,
        "--scheme" => {
            let inviscid = parse_scheme(next_token(tail, i, "--scheme")?, state.limiter)?;
            state.mode = ExportMode::Single(SodBenchmarkConfig {
                inviscid,
                ..state.base
            });
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn apply_cli_flag(
    state: &mut CliState,
    flag: &str,
    tail: &[String],
    i: &mut usize,
) -> Result<(), String> {
    if apply_numeric_flag(state, flag, tail, i)? || apply_mode_flag(state, flag, tail, i)? {
        return Ok(());
    }
    Err(format!("未知参数: {flag}"))
}

fn finalize_mode(state: &mut CliState) {
    if let ExportMode::Single(ref mut config) = state.mode {
        *config = SodBenchmarkConfig {
            inviscid: config.inviscid,
            ..state.base
        };
    }
}

fn parse_args() -> Result<(PathBuf, SodBenchmarkConfig, SlopeLimiter, ExportMode), String> {
    let mut args = env::args().skip(1);
    let output = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sod_compare.txt"));
    let mut state = CliState {
        base: SodBenchmarkConfig::default(),
        limiter: SlopeLimiter::VanAlbada,
        mode: ExportMode::Compare,
    };
    let tail: Vec<String> = args.collect();
    let mut i = 0;
    while i < tail.len() {
        apply_cli_flag(&mut state, &tail[i], &tail, &mut i)?;
        i += 1;
    }
    if state.base.ncells == 0 {
        return Err("--cells 须大于 0".to_string());
    }
    finalize_mode(&mut state);
    Ok((output, state.base, state.limiter, state.mode))
}
