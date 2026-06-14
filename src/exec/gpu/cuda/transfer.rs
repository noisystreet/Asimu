//! CUDA H2D/D2H 传输（Chrome trace：`cuda_h2d` / `cuda_d2h`）。

use std::mem::size_of;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
use tracing::info_span;

use crate::error::{AsimuError, Result};

#[inline]
fn byte_len<T>(count: usize) -> usize {
    count.saturating_mul(size_of::<T>())
}

/// 单次 `memcpy_htod`（带 `cuda_h2d` span）。
pub fn memcpy_htod<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    label: &'static str,
    src: &[T],
    dst: &mut CudaSlice<T>,
) -> Result<()> {
    let _span = info_span!(
        "cuda_h2d",
        label,
        bytes = byte_len::<T>(src.len()),
        elements = src.len(),
    )
    .entered();
    stream
        .memcpy_htod(src, dst)
        .map_err(|e| AsimuError::Exec(format!("CUDA H2D `{label}` 失败: {e:?}")))
}

/// 单次 `clone_htod`（带 `cuda_h2d` span）。
pub fn clone_htod<T: DeviceRepr + Clone>(
    stream: &Arc<CudaStream>,
    label: &'static str,
    host: &[T],
) -> Result<CudaSlice<T>> {
    let _span = info_span!(
        "cuda_h2d",
        label,
        bytes = byte_len::<T>(host.len()),
        elements = host.len(),
    )
    .entered();
    stream
        .clone_htod(host)
        .map_err(|e| AsimuError::Exec(format!("CUDA H2D `{label}` 失败: {e:?}")))
}

/// 单次 `clone_dtoh`（带 `cuda_d2h` span）。
pub fn clone_dtoh<T: DeviceRepr + Clone>(
    stream: &Arc<CudaStream>,
    label: &'static str,
    src: &CudaSlice<T>,
) -> Result<Vec<T>> {
    let n = src.len();
    let _span = info_span!("cuda_d2h", label, bytes = byte_len::<T>(n), elements = n,).entered();
    stream
        .clone_dtoh(src)
        .map_err(|e| AsimuError::Exec(format!("CUDA D2H `{label}` 失败: {e:?}")))
}

/// 合并多次 H2D 为一条 trace（如 SoA 多分量上传）。
pub fn h2d_batch<F>(label: &'static str, bytes: usize, elements: usize, f: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    let _span = info_span!("cuda_h2d", label, bytes, elements).entered();
    f()
}

/// 合并多次 D2H 为一条 trace。
pub fn d2h_batch<F>(label: &'static str, bytes: usize, elements: usize, f: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    let _span = info_span!("cuda_d2h", label, bytes, elements).entered();
    f()
}

/// 无 span 的内部 memcpy（须在 `h2d_batch` 闭包内调用）。
pub(crate) fn memcpy_htod_unchecked<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    src: &[T],
    dst: &mut CudaSlice<T>,
) -> Result<()> {
    stream
        .memcpy_htod(src, dst)
        .map_err(|e| AsimuError::Exec(format!("CUDA H2D 失败: {e:?}")))
}

/// 无 span 的内部 dtoh（须在 `d2h_batch` 闭包内调用）。
pub(crate) fn clone_dtoh_unchecked<T: DeviceRepr + Clone>(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<T>,
) -> Result<Vec<T>> {
    stream
        .clone_dtoh(src)
        .map_err(|e| AsimuError::Exec(format!("CUDA D2H 失败: {e:?}")))
}
