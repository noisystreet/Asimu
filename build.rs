//! 构建脚本：`io-cgns` 链接 CGNS；`cuda` 用 `nvcc` 预编译 PTX（ADR 0017 G1+G2）。

use std::path::{Path, PathBuf};
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
    println!("cargo:rerun-if-changed=kernels/cuda/viscous_interior_f32.cu");
    println!("cargo:rerun-if-changed=kernels/cuda/idwls_viscous_rhs_f32.cu");

    println!("cargo:rerun-if-changed=kernels/cuda/spectral_radius_unstructured_f32.cu");
    println!("cargo:rerun-if-changed=kernels/cuda/lusgs_diagonal_f32.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let nvcc = std::env::var("CUDA_NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let inviscid_ok = compile_cuda_ptx(
        &nvcc,
        &out_dir,
        "kernels/cuda/inviscid_first_order_f32.cu",
        "inviscid_first_order_f32.ptx",
        "CUDA_PTX_INVISCID_F32",
    );
    let viscous_ok = compile_cuda_ptx(
        &nvcc,
        &out_dir,
        "kernels/cuda/viscous_interior_f32.cu",
        "viscous_interior_f32.ptx",
        "CUDA_PTX_VISCOUS_F32",
    );
    let idwls_ok = compile_cuda_ptx(
        &nvcc,
        &out_dir,
        "kernels/cuda/idwls_viscous_rhs_f32.cu",
        "idwls_viscous_rhs_f32.ptx",
        "CUDA_PTX_IDWLS_F32",
    );
    let spectral_ok = compile_cuda_ptx(
        &nvcc,
        &out_dir,
        "kernels/cuda/spectral_radius_unstructured_f32.cu",
        "spectral_radius_unstructured_f32.ptx",
        "CUDA_PTX_SPECTRAL_RADIUS_F32",
    );
    let lusgs_ok = compile_cuda_ptx(
        &nvcc,
        &out_dir,
        "kernels/cuda/lusgs_diagonal_f32.cu",
        "lusgs_diagonal_f32.ptx",
        "CUDA_PTX_LUSGS_DIAGONAL_F32",
    );

    if inviscid_ok && viscous_ok && idwls_ok && spectral_ok && lusgs_ok {
        println!("cargo:rustc-cfg=cuda_kernels_built");
    } else {
        println!("cargo:rustc-cfg=cuda_kernels_disabled");
    }
}

fn compile_cuda_ptx(nvcc: &str, out_dir: &Path, src: &str, ptx_name: &str, env_key: &str) -> bool {
    let ptx_path = out_dir.join(ptx_name);
    let status = Command::new(nvcc)
        .args(["--ptx", "-O3", "--use_fast_math", "-o"])
        .arg(&ptx_path)
        .arg(src)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-env={env_key}={}", ptx_path.display());
            true
        }
        Ok(s) => {
            println!(
                "cargo:warning=CUDA kernel 编译失败 {src}（exit={}）",
                s.code().unwrap_or(-1)
            );
            false
        }
        Err(e) => {
            println!("cargo:warning=未找到 nvcc（{e}）；跳过 {src}");
            false
        }
    }
}
