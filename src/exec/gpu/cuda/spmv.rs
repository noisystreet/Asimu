//! cuSPARSE CSR SpMV（ADR 0017 G3；f64）。

use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::cusparse::{result as cusparse_result, sys as cusparse_sys};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use tracing::info_span;

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::exec::CsrSpmvView;

/// 步间 CSR 结构缓存（pattern 不变时复用 device 索引与 workspace）。
pub struct CudaCsrSpmvCache {
    structure_key: Option<u64>,
    nrows: usize,
    ncols: usize,
    nnz: usize,
    d_row_ptr: Option<CudaSlice<i32>>,
    d_col_idx: Option<CudaSlice<i32>>,
    d_values: Option<CudaSlice<f64>>,
    sp_mat: Option<CusparseSpMatDescr>,
    workspace: Option<CudaSlice<u8>>,
    workspace_bytes: usize,
}

impl CudaCsrSpmvCache {
    pub fn new() -> Self {
        Self {
            structure_key: None,
            nrows: 0,
            ncols: 0,
            nnz: 0,
            d_row_ptr: None,
            d_col_idx: None,
            d_values: None,
            sp_mat: None,
            workspace: None,
            workspace_bytes: 0,
        }
    }
}

impl Default for CudaCsrSpmvCache {
    fn default() -> Self {
        Self::new()
    }
}

struct CusparseSpMatDescr {
    raw: cusparse_sys::cusparseSpMatDescr_t,
}

// SAFETY: CUDA 资源绑定在构造 `CudaBackendState` 的线程与 device 0 默认 stream。
unsafe impl Send for CusparseSpMatDescr {}
unsafe impl Sync for CusparseSpMatDescr {}
unsafe impl Send for CudaCsrSpmvCache {}
unsafe impl Sync for CudaCsrSpmvCache {}

impl Drop for CusparseSpMatDescr {
    fn drop(&mut self) {
        // SAFETY: `raw` 由 `cusparseCreateCsr` 创建且仅在此处销毁。
        unsafe {
            let _ = cusparse_sys::cusparseDestroySpMat(self.raw).result();
        }
    }
}

/// \(y \leftarrow A x\)（cuSPARSE generic CSR；`Real` = f64）。
pub fn csr_spmv_f64(
    stream: &Arc<CudaStream>,
    handle: CusparseHandle,
    cache: &mut CudaCsrSpmvCache,
    matrix: &CsrSpmvView<'_>,
    x: &[Real],
    y: &mut [Real],
) -> Result<()> {
    let _span = info_span!(
        "cuda_csr_spmv",
        nrows = matrix.nrows,
        ncols = matrix.ncols,
        nnz = matrix.values.len(),
    )
    .entered();

    bind_cusparse_stream(handle, stream)?;
    ensure_csr_structure(stream, handle, cache, matrix)?;

    let d_values = cache.d_values.as_mut().expect("values after ensure");
    stream
        .memcpy_htod(matrix.values, d_values)
        .map_err(|e| AsimuError::Exec(format!("CUDA CSR values H2D 失败: {e:?}")))?;
    {
        let sp_mat = cache
            .sp_mat
            .as_ref()
            .ok_or_else(|| AsimuError::Exec("CUDA CSR 描述符未初始化".to_string()))?;
        let (values_dev, _sync) = d_values.device_ptr(stream);
        // SAFETY: `d_values` 与 `sp_mat` 同一次 ensure 分配，长度 = nnz。
        unsafe {
            cusparse_sys::cusparseSpMatSetValues(sp_mat.raw, values_dev as *mut core::ffi::c_void)
                .result()
                .map_err(map_cusparse_err)?;
        }
    }

    let d_x = stream
        .clone_htod(x)
        .map_err(|e| AsimuError::Exec(format!("CUDA SpMV x H2D 失败: {e:?}")))?;
    let mut d_y = stream
        .alloc_zeros::<f64>(matrix.nrows)
        .map_err(|e| AsimuError::Exec(format!("CUDA SpMV y 分配失败: {e:?}")))?;

    run_cusparse_spmv(stream, handle, cache, &d_x, &mut d_y)?;

    let host_y = stream
        .clone_dtoh(&d_y)
        .map_err(|e| AsimuError::Exec(format!("CUDA SpMV y D2H 失败: {e:?}")))?;
    y.copy_from_slice(host_y.as_slice());
    Ok(())
}

fn bind_cusparse_stream(handle: CusparseHandle, stream: &Arc<CudaStream>) -> Result<()> {
    // SAFETY: handle 在 `CudaBackendState` 生命周期内有效。
    unsafe {
        cusparse_sys::cusparseSetStream(handle.0, stream.cu_stream() as cusparse_sys::cudaStream_t)
            .result()
            .map_err(map_cusparse_err)?;
    }
    Ok(())
}

fn csr_structure_key(matrix: &CsrSpmvView<'_>) -> u64 {
    let nnz = matrix.values.len() as u64;
    let row_ptr_addr = std::ptr::from_ref(matrix.row_ptr).addr() as u64;
    let col_idx_addr = std::ptr::from_ref(matrix.col_idx).addr() as u64;
    row_ptr_addr
        .wrapping_mul(31)
        .wrapping_add(col_idx_addr)
        .wrapping_add((matrix.nrows as u64) << 32)
        .wrapping_add(matrix.ncols as u64)
        .wrapping_add(nnz.wrapping_mul(17))
}

fn ensure_csr_structure(
    stream: &Arc<CudaStream>,
    handle: CusparseHandle,
    cache: &mut CudaCsrSpmvCache,
    matrix: &CsrSpmvView<'_>,
) -> Result<()> {
    let key = csr_structure_key(matrix);
    if cache.structure_key == Some(key) {
        return Ok(());
    }
    cache.structure_key = Some(key);
    cache.nrows = matrix.nrows;
    cache.ncols = matrix.ncols;
    cache.nnz = matrix.values.len();
    cache.sp_mat = None;
    cache.workspace = None;
    cache.workspace_bytes = 0;

    let row_ptr_i32 = usize_to_i32(matrix.row_ptr, "row_ptr")?;
    let col_idx_i32 = usize_to_i32(matrix.col_idx, "col_idx")?;
    cache.d_row_ptr = Some(
        stream
            .clone_htod(&row_ptr_i32)
            .map_err(|e| AsimuError::Exec(format!("CUDA CSR row_ptr H2D 失败: {e:?}")))?,
    );
    cache.d_col_idx = Some(
        stream
            .clone_htod(&col_idx_i32)
            .map_err(|e| AsimuError::Exec(format!("CUDA CSR col_idx H2D 失败: {e:?}")))?,
    );
    cache.d_values = Some(
        stream
            .alloc_zeros::<f64>(cache.nnz)
            .map_err(|e| AsimuError::Exec(format!("CUDA CSR values 分配失败: {e:?}")))?,
    );

    let sp_mat = create_csr_descr(
        stream,
        matrix.nrows,
        matrix.ncols,
        cache.nnz,
        cache.d_row_ptr.as_ref().expect("row_ptr"),
        cache.d_col_idx.as_ref().expect("col_idx"),
        cache.d_values.as_ref().expect("values"),
    )?;
    ensure_workspace(stream, handle, cache, &sp_mat)?;
    cache.sp_mat = Some(sp_mat);
    Ok(())
}

fn usize_to_i32(values: &[usize], label: &str) -> Result<Vec<i32>> {
    values
        .iter()
        .map(|&v| {
            i32::try_from(v).map_err(|_| {
                AsimuError::Linalg(format!("CSR {label} 索引 {v} 超出 cuSPARSE i32 范围"))
            })
        })
        .collect()
}

fn create_csr_descr(
    stream: &Arc<CudaStream>,
    nrows: usize,
    ncols: usize,
    nnz: usize,
    d_row_ptr: &CudaSlice<i32>,
    d_col_idx: &CudaSlice<i32>,
    d_values: &CudaSlice<f64>,
) -> Result<CusparseSpMatDescr> {
    let mut descr = MaybeUninit::uninit();
    let rows = i64::try_from(nrows)
        .map_err(|_| AsimuError::Linalg("CSR 行数超出 i64 范围".to_string()))?;
    let cols = i64::try_from(ncols)
        .map_err(|_| AsimuError::Linalg("CSR 列数超出 i64 范围".to_string()))?;
    let nnz_i64 =
        i64::try_from(nnz).map_err(|_| AsimuError::Linalg("CSR nnz 超出 i64 范围".to_string()))?;
    let (row_ptr_dev, _row_sync) = d_row_ptr.device_ptr(stream);
    let (col_idx_dev, _col_sync) = d_col_idx.device_ptr(stream);
    let (values_dev, _val_sync) = d_values.device_ptr(stream);
    // SAFETY: device 指针来自 cudarc 分配；布局与 `cusparseCreateCsr` 契约一致。
    unsafe {
        cusparse_sys::cusparseCreateCsr(
            descr.as_mut_ptr(),
            rows,
            cols,
            nnz_i64,
            row_ptr_dev as *mut core::ffi::c_void,
            col_idx_dev as *mut core::ffi::c_void,
            values_dev as *mut core::ffi::c_void,
            cusparse_sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusparse_sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusparse_sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
            cusparse_sys::cudaDataType::CUDA_R_64F,
        )
        .result()
        .map_err(map_cusparse_err)?;
        Ok(CusparseSpMatDescr {
            raw: descr.assume_init(),
        })
    }
}

fn ensure_workspace(
    stream: &Arc<CudaStream>,
    handle: CusparseHandle,
    cache: &mut CudaCsrSpmvCache,
    sp_mat: &CusparseSpMatDescr,
) -> Result<()> {
    let alpha = 1.0_f64;
    let beta = 0.0_f64;
    let mut buffer_size = 0usize;
    let d_x = stream
        .alloc_zeros::<f64>(cache.ncols)
        .map_err(|e| AsimuError::Exec(format!("CUDA SpMV workspace probe x 失败: {e:?}")))?;
    let mut d_y = stream
        .alloc_zeros::<f64>(cache.nrows)
        .map_err(|e| AsimuError::Exec(format!("CUDA SpMV workspace probe y 失败: {e:?}")))?;
    let vec_x = create_const_dn_vec(stream, cache.ncols, &d_x)?;
    let vec_y = create_dn_vec(stream, cache.nrows, &mut d_y)?;
    // SAFETY: 临时描述符仅用于 bufferSize 查询。
    unsafe {
        cusparse_sys::cusparseSpMV_bufferSize(
            handle.0,
            cusparse_sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            (&alpha as *const f64).cast(),
            sp_mat.raw,
            vec_x,
            (&beta as *const f64).cast(),
            vec_y,
            cusparse_sys::cudaDataType::CUDA_R_64F,
            cusparse_sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
            &mut buffer_size,
        )
        .result()
        .map_err(map_cusparse_err)?;
        destroy_const_dn_vec(vec_x)?;
        destroy_mut_dn_vec(vec_y)?;
    }
    if buffer_size == 0 {
        cache.workspace = None;
        cache.workspace_bytes = 0;
        return Ok(());
    }
    cache.workspace = Some(
        stream
            .alloc_zeros::<u8>(buffer_size)
            .map_err(|e| AsimuError::Exec(format!("CUDA SpMV workspace 分配失败: {e:?}")))?,
    );
    cache.workspace_bytes = buffer_size;
    Ok(())
}

fn run_cusparse_spmv(
    stream: &Arc<CudaStream>,
    handle: CusparseHandle,
    cache: &CudaCsrSpmvCache,
    d_x: &CudaSlice<f64>,
    d_y: &mut CudaSlice<f64>,
) -> Result<()> {
    let sp_mat = cache
        .sp_mat
        .as_ref()
        .ok_or_else(|| AsimuError::Exec("CUDA CSR 描述符未初始化".to_string()))?;
    let alpha = 1.0_f64;
    let beta = 0.0_f64;
    let vec_x = create_const_dn_vec(stream, cache.ncols, d_x)?;
    let vec_y = create_dn_vec(stream, cache.nrows, d_y)?;
    let workspace_ptr = if let Some(ws) = cache.workspace.as_ref() {
        let (ptr, _sync) = ws.device_ptr(stream);
        ptr as *mut core::ffi::c_void
    } else {
        core::ptr::null_mut()
    };
    // SAFETY: alpha/beta、描述符与 workspace 布局符合 cuSPARSE generic SpMV 契约。
    unsafe {
        cusparse_sys::cusparseSpMV(
            handle.0,
            cusparse_sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            (&alpha as *const f64).cast(),
            sp_mat.raw,
            vec_x,
            (&beta as *const f64).cast(),
            vec_y,
            cusparse_sys::cudaDataType::CUDA_R_64F,
            cusparse_sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
            workspace_ptr,
        )
        .result()
        .map_err(map_cusparse_err)?;
        destroy_const_dn_vec(vec_x)?;
        destroy_mut_dn_vec(vec_y)?;
    }
    Ok(())
}

fn create_const_dn_vec(
    stream: &Arc<CudaStream>,
    size: usize,
    values: &CudaSlice<f64>,
) -> Result<cusparse_sys::cusparseConstDnVecDescr_t> {
    let mut descr = MaybeUninit::uninit();
    let size_i64 = i64::try_from(size)
        .map_err(|_| AsimuError::Linalg("稠密向量长度超出 i64 范围".to_string()))?;
    let (values_dev, _sync) = values.device_ptr(stream);
    unsafe {
        cusparse_sys::cusparseCreateConstDnVec(
            descr.as_mut_ptr(),
            size_i64,
            values_dev as *const core::ffi::c_void,
            cusparse_sys::cudaDataType::CUDA_R_64F,
        )
        .result()
        .map_err(map_cusparse_err)?;
        Ok(descr.assume_init())
    }
}

fn create_dn_vec(
    stream: &Arc<CudaStream>,
    size: usize,
    values: &mut CudaSlice<f64>,
) -> Result<cusparse_sys::cusparseDnVecDescr_t> {
    let mut descr = MaybeUninit::uninit();
    let size_i64 = i64::try_from(size)
        .map_err(|_| AsimuError::Linalg("稠密向量长度超出 i64 范围".to_string()))?;
    let (values_dev, _sync) = values.device_ptr_mut(stream);
    unsafe {
        cusparse_sys::cusparseCreateDnVec(
            descr.as_mut_ptr(),
            size_i64,
            values_dev as *mut core::ffi::c_void,
            cusparse_sys::cudaDataType::CUDA_R_64F,
        )
        .result()
        .map_err(map_cusparse_err)?;
        Ok(descr.assume_init())
    }
}

fn destroy_const_dn_vec(descr: cusparse_sys::cusparseConstDnVecDescr_t) -> Result<()> {
    unsafe {
        cusparse_sys::cusparseDestroyDnVec(descr)
            .result()
            .map_err(map_cusparse_err)
    }
}

fn destroy_mut_dn_vec(descr: cusparse_sys::cusparseDnVecDescr_t) -> Result<()> {
    unsafe {
        cusparse_sys::cusparseDestroyDnVec(descr)
            .result()
            .map_err(map_cusparse_err)
    }
}

fn map_cusparse_err(err: cusparse_result::CusparseError) -> AsimuError {
    AsimuError::Exec(format!("cuSPARSE 失败: {err:?}"))
}

pub fn try_create_cusparse_handle() -> Result<CusparseHandle> {
    cusparse_result::create()
        .map(CusparseHandle)
        .map_err(map_cusparse_err)
}

pub fn destroy_cusparse_handle(handle: CusparseHandle) -> Result<()> {
    // SAFETY: handle 仅由 `try_create_cusparse_handle` 构造且单次销毁。
    unsafe { cusparse_result::destroy(handle.0).map_err(map_cusparse_err) }
}

/// cuSPARSE 句柄（CUDA 驱动资源；算例级单线程使用）。
#[derive(Clone, Copy)]
pub struct CusparseHandle(cusparse_sys::cusparseHandle_t);

// SAFETY: 与 `CudaBackendState` 同线程、同 device 使用。
unsafe impl Send for CusparseHandle {}
unsafe impl Sync for CusparseHandle {}
