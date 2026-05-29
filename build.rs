//! 可选 feature `io-cgns` 链接系统 CGNS 库（`libcgns-dev`）。

fn main() {
    if std::env::var("CARGO_FEATURE_IO_CGNS").is_err() {
        return;
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_IO_CGNS");
    println!("cargo:rustc-link-lib=cgns");
}
