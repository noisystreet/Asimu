//! 构建脚本：`io-cgns` 链接 CGNS；`cuda` 用 `nvcc` 预编译 PTX（ADR 0017 G1）。

use std::path::PathBuf;
use std::process::Command;

fn main() {
    register_cuda_cfgs();
    build_cgns_shim();
    build_cuda_kernels();
}

fn register_cuda_cfgs() {
    println!("cargo::rustc-check-cfg=cfg(cuda_kernels_built)");
    println!("cargo::rustc-check-cfg=cfg(cuda_kernels_disabled)");
}

fn build_cgns_shim() {
    if std::env::var("CARGO_FEATURE_IO_CGNS").is_err() {
        return;
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_IO_CGNS");
    println!("cargo:rerun-if-changed=src/io/cgns/cgns_shim.c");
    cc::Build::new()
        .file("src/io/cgns/cgns_shim.c")
        .compile("asimu_cgns_shim");
    println!("cargo:rustc-link-lib=cgns");
}

fn build_cuda_kernels() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-changed=kernels/cuda/inviscid_first_order_f32.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let ptx_path = out_dir.join("inviscid_first_order_f32.ptx");
    let src = "kernels/cuda/inviscid_first_order_f32.cu";

    let nvcc = std::env::var("CUDA_NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let status = Command::new(&nvcc)
        .args(["--ptx", "-O3", "--use_fast_math", "-o"])
        .arg(&ptx_path)
        .arg(src)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!(
                "cargo:rustc-env=CUDA_PTX_INVISCID_F32={}",
                ptx_path.display()
            );
            println!("cargo:rustc-cfg=cuda_kernels_built");
        }
        Ok(s) => {
            println!(
                "cargo:warning=CUDA kernel 编译失败（exit={}）；GPU 热路径不可用",
                s.code().unwrap_or(-1)
            );
            println!("cargo:rustc-cfg=cuda_kernels_disabled");
        }
        Err(e) => {
            println!("cargo:warning=未找到 nvcc（{e}）；GPU 热路径不可用");
            println!("cargo:rustc-cfg=cuda_kernels_disabled");
        }
    }
}
