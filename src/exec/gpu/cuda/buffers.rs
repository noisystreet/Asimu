//! 设备侧场缓冲（原始变量 SoA + 残差 SoA）。

use std::mem::size_of;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};

use super::transfer::{
    clone_dtoh_unchecked, d2d_batch, d2h_batch, h2d_batch, memcpy_dtod_unchecked,
    memcpy_htod_unchecked,
};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};

/// 与 CUDA kernel `FaceGeom` 一致的设备布局。
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeviceFaceGeom {
    pub owner: u32,
    pub neighbor: u32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub owner_scale: f32,
    pub neighbor_scale: f32,
}

unsafe impl DeviceRepr for DeviceFaceGeom {}

/// 步间常驻：原始变量 + 残差 device 缓冲。
pub struct CudaFieldBuffers {
    pub(crate) prim_rho: CudaSlice<f32>,
    pub(crate) prim_p: CudaSlice<f32>,
    pub(crate) prim_ux: CudaSlice<f32>,
    pub(crate) prim_uy: CudaSlice<f32>,
    pub(crate) prim_uz: CudaSlice<f32>,
    pub(crate) res_rho: CudaSlice<f32>,
    pub(crate) res_mx: CudaSlice<f32>,
    pub(crate) res_my: CudaSlice<f32>,
    pub(crate) res_mz: CudaSlice<f32>,
    pub(crate) res_e: CudaSlice<f32>,
    pub(crate) cons_rho: CudaSlice<f32>,
    pub(crate) cons_mx: CudaSlice<f32>,
    pub(crate) cons_my: CudaSlice<f32>,
    pub(crate) cons_mz: CudaSlice<f32>,
    pub(crate) cons_e: CudaSlice<f32>,
    pub(crate) cons_u_n_rho: CudaSlice<f32>,
    pub(crate) cons_u_n_mx: CudaSlice<f32>,
    pub(crate) cons_u_n_my: CudaSlice<f32>,
    pub(crate) cons_u_n_mz: CudaSlice<f32>,
    pub(crate) cons_u_n_e: CudaSlice<f32>,
    pub(crate) num_cells: usize,
}

impl CudaFieldBuffers {
    #[must_use]
    pub(crate) fn num_cells(&self) -> usize {
        self.num_cells
    }

    pub fn try_new(stream: &Arc<CudaStream>, num_cells: usize) -> Result<Self> {
        if num_cells == 0 {
            return Err(AsimuError::Field(
                "CUDA 场缓冲需要 num_cells > 0".to_string(),
            ));
        }
        let prim_res = alloc_prim_and_residual_buffers(stream, num_cells)?;
        let cons = alloc_conserved_buffers(stream, num_cells)?;
        let cons_u_n = alloc_conserved_buffers(stream, num_cells)?;
        Ok(Self {
            prim_rho: prim_res.prim_rho,
            prim_p: prim_res.prim_p,
            prim_ux: prim_res.prim_ux,
            prim_uy: prim_res.prim_uy,
            prim_uz: prim_res.prim_uz,
            res_rho: prim_res.res_rho,
            res_mx: prim_res.res_mx,
            res_my: prim_res.res_my,
            res_mz: prim_res.res_mz,
            res_e: prim_res.res_e,
            cons_rho: cons.cons_rho,
            cons_mx: cons.cons_mx,
            cons_my: cons.cons_my,
            cons_mz: cons.cons_mz,
            cons_e: cons.cons_e,
            cons_u_n_rho: cons_u_n.cons_rho,
            cons_u_n_mx: cons_u_n.cons_mx,
            cons_u_n_my: cons_u_n.cons_my,
            cons_u_n_mz: cons_u_n.cons_mz,
            cons_u_n_e: cons_u_n.cons_e,
            num_cells,
        })
    }

    pub fn upload_primitives(
        &mut self,
        stream: &Arc<CudaStream>,
        primitives: &PrimitiveFieldsT<f32>,
    ) -> Result<()> {
        let n = primitives.num_cells();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "primitive 长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        h2d_batch("field_primitives", n * 5 * size_of::<f32>(), n, || {
            memcpy_htod_unchecked(stream, primitives.density.values(), &mut self.prim_rho)?;
            memcpy_htod_unchecked(stream, primitives.pressure.values(), &mut self.prim_p)?;
            memcpy_htod_unchecked(stream, primitives.velocity_x.values(), &mut self.prim_ux)?;
            memcpy_htod_unchecked(stream, primitives.velocity_y.values(), &mut self.prim_uy)?;
            memcpy_htod_unchecked(stream, primitives.velocity_z.values(), &mut self.prim_uz)?;
            Ok(())
        })
    }

    pub fn zero_residual(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        zero_slice(stream, &mut self.res_rho)?;
        zero_slice(stream, &mut self.res_mx)?;
        zero_slice(stream, &mut self.res_my)?;
        zero_slice(stream, &mut self.res_mz)?;
        zero_slice(stream, &mut self.res_e)?;
        Ok(())
    }

    pub fn download_residual(
        &self,
        stream: &Arc<CudaStream>,
        residual: &mut ConservedResidualT<f32>,
    ) -> Result<()> {
        let n = residual.num_cells();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "残差长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        d2h_batch("field_residual", n * 5 * size_of::<f32>(), n, || {
            dtoh_into_unchecked(stream, &self.res_rho, residual.density.values_mut())?;
            dtoh_into_unchecked(stream, &self.res_mx, residual.momentum_x.values_mut())?;
            dtoh_into_unchecked(stream, &self.res_my, residual.momentum_y.values_mut())?;
            dtoh_into_unchecked(stream, &self.res_mz, residual.momentum_z.values_mut())?;
            dtoh_into_unchecked(stream, &self.res_e, residual.total_energy.values_mut())?;
            Ok(())
        })
    }

    /// 物理步初 D2D 快照：\(U^n \leftarrow U\)（device 守恒缓冲）。
    pub fn snapshot_u_n_on_device(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        let n = self.num_cells;
        let bytes = n
            .checked_mul(5 * size_of::<f32>())
            .ok_or_else(|| AsimuError::Field("dual_time U^n D2D 字节数溢出".to_string()))?;
        d2d_batch("field_conserved_u_n_snapshot", bytes, n, || {
            memcpy_dtod_unchecked(stream, &self.cons_rho, &mut self.cons_u_n_rho)?;
            memcpy_dtod_unchecked(stream, &self.cons_mx, &mut self.cons_u_n_mx)?;
            memcpy_dtod_unchecked(stream, &self.cons_my, &mut self.cons_u_n_my)?;
            memcpy_dtod_unchecked(stream, &self.cons_mz, &mut self.cons_u_n_mz)?;
            memcpy_dtod_unchecked(stream, &self.cons_e, &mut self.cons_u_n_e)?;
            Ok(())
        })
    }

    pub fn download_u_n(
        &self,
        stream: &Arc<CudaStream>,
        fields: &mut ConservedFieldsT<f32>,
    ) -> Result<()> {
        let n = fields.num_cells();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "U^n 下载长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        d2h_batch("field_conserved_u_n", n * 5 * size_of::<f32>(), n, || {
            dtoh_into_unchecked(stream, &self.cons_u_n_rho, fields.density.values_mut())?;
            dtoh_into_unchecked(stream, &self.cons_u_n_mx, fields.momentum_x.values_mut())?;
            dtoh_into_unchecked(stream, &self.cons_u_n_my, fields.momentum_y.values_mut())?;
            dtoh_into_unchecked(stream, &self.cons_u_n_mz, fields.momentum_z.values_mut())?;
            dtoh_into_unchecked(stream, &self.cons_u_n_e, fields.total_energy.values_mut())?;
            Ok(())
        })
    }

    pub fn upload_conserved(
        &mut self,
        stream: &Arc<CudaStream>,
        fields: &ConservedFieldsT<f32>,
    ) -> Result<()> {
        let n = fields.num_cells();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "守恒场长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        h2d_batch("field_conserved", n * 5 * size_of::<f32>(), n, || {
            memcpy_htod_unchecked(stream, fields.density.values(), &mut self.cons_rho)?;
            memcpy_htod_unchecked(stream, fields.momentum_x.values(), &mut self.cons_mx)?;
            memcpy_htod_unchecked(stream, fields.momentum_y.values(), &mut self.cons_my)?;
            memcpy_htod_unchecked(stream, fields.momentum_z.values(), &mut self.cons_mz)?;
            memcpy_htod_unchecked(stream, fields.total_energy.values(), &mut self.cons_e)?;
            Ok(())
        })
    }

    pub fn download_conserved(
        &self,
        stream: &Arc<CudaStream>,
        fields: &mut ConservedFieldsT<f32>,
    ) -> Result<()> {
        let n = fields.num_cells();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "守恒场长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        d2h_batch("field_conserved", n * 5 * size_of::<f32>(), n, || {
            dtoh_into_unchecked(stream, &self.cons_rho, fields.density.values_mut())?;
            dtoh_into_unchecked(stream, &self.cons_mx, fields.momentum_x.values_mut())?;
            dtoh_into_unchecked(stream, &self.cons_my, fields.momentum_y.values_mut())?;
            dtoh_into_unchecked(stream, &self.cons_mz, fields.momentum_z.values_mut())?;
            dtoh_into_unchecked(stream, &self.cons_e, fields.total_energy.values_mut())?;
            Ok(())
        })
    }

    pub fn upload_full_residual(
        &mut self,
        stream: &Arc<CudaStream>,
        residual: &ConservedResidualT<f32>,
    ) -> Result<()> {
        let n = residual.num_cells();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "残差长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        h2d_batch("field_residual_full", n * 5 * size_of::<f32>(), n, || {
            memcpy_htod_unchecked(stream, residual.density.values(), &mut self.res_rho)?;
            memcpy_htod_unchecked(stream, residual.momentum_x.values(), &mut self.res_mx)?;
            memcpy_htod_unchecked(stream, residual.momentum_y.values(), &mut self.res_my)?;
            memcpy_htod_unchecked(stream, residual.momentum_z.values(), &mut self.res_mz)?;
            memcpy_htod_unchecked(stream, residual.total_energy.values(), &mut self.res_e)?;
            Ok(())
        })
    }

    pub fn upload_momentum_energy_residual(
        &mut self,
        stream: &Arc<CudaStream>,
        residual: &ConservedResidualT<f32>,
    ) -> Result<()> {
        let n = residual.num_cells();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "残差长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        h2d_batch(
            "field_residual_momentum_energy",
            n * 4 * size_of::<f32>(),
            n,
            || {
                memcpy_htod_unchecked(stream, residual.momentum_x.values(), &mut self.res_mx)?;
                memcpy_htod_unchecked(stream, residual.momentum_y.values(), &mut self.res_my)?;
                memcpy_htod_unchecked(stream, residual.momentum_z.values(), &mut self.res_mz)?;
                memcpy_htod_unchecked(stream, residual.total_energy.values(), &mut self.res_e)?;
                Ok(())
            },
        )
    }

    pub fn download_momentum_energy_residual(
        &self,
        stream: &Arc<CudaStream>,
        residual: &mut ConservedResidualT<f32>,
    ) -> Result<()> {
        let n = residual.num_cells();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "残差长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        d2h_batch(
            "field_residual_momentum_energy",
            n * 4 * size_of::<f32>(),
            n,
            || {
                dtoh_into_unchecked(stream, &self.res_mx, residual.momentum_x.values_mut())?;
                dtoh_into_unchecked(stream, &self.res_my, residual.momentum_y.values_mut())?;
                dtoh_into_unchecked(stream, &self.res_mz, residual.momentum_z.values_mut())?;
                dtoh_into_unchecked(stream, &self.res_e, residual.total_energy.values_mut())?;
                Ok(())
            },
        )
    }
}

fn dtoh_into_unchecked(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut [f32],
) -> Result<()> {
    let host = clone_dtoh_unchecked(stream, src)?;
    dst.copy_from_slice(host.as_slice());
    Ok(())
}

fn zero_slice(stream: &Arc<CudaStream>, buf: &mut CudaSlice<f32>) -> Result<()> {
    stream
        .memset_zeros(buf)
        .map_err(|e| AsimuError::Exec(format!("CUDA memset 失败: {e:?}")))
}

struct PrimResidualBuffers {
    prim_rho: CudaSlice<f32>,
    prim_p: CudaSlice<f32>,
    prim_ux: CudaSlice<f32>,
    prim_uy: CudaSlice<f32>,
    prim_uz: CudaSlice<f32>,
    res_rho: CudaSlice<f32>,
    res_mx: CudaSlice<f32>,
    res_my: CudaSlice<f32>,
    res_mz: CudaSlice<f32>,
    res_e: CudaSlice<f32>,
}

struct ConservedBuffers {
    cons_rho: CudaSlice<f32>,
    cons_mx: CudaSlice<f32>,
    cons_my: CudaSlice<f32>,
    cons_mz: CudaSlice<f32>,
    cons_e: CudaSlice<f32>,
}

fn alloc_zeros_f32(stream: &Arc<CudaStream>, n: usize) -> Result<CudaSlice<f32>> {
    stream
        .alloc_zeros::<f32>(n)
        .map_err(|e| AsimuError::Exec(format!("CUDA 分配失败: {e:?}")))
}

fn alloc_prim_and_residual_buffers(
    stream: &Arc<CudaStream>,
    num_cells: usize,
) -> Result<PrimResidualBuffers> {
    Ok(PrimResidualBuffers {
        prim_rho: alloc_zeros_f32(stream, num_cells)?,
        prim_p: alloc_zeros_f32(stream, num_cells)?,
        prim_ux: alloc_zeros_f32(stream, num_cells)?,
        prim_uy: alloc_zeros_f32(stream, num_cells)?,
        prim_uz: alloc_zeros_f32(stream, num_cells)?,
        res_rho: alloc_zeros_f32(stream, num_cells)?,
        res_mx: alloc_zeros_f32(stream, num_cells)?,
        res_my: alloc_zeros_f32(stream, num_cells)?,
        res_mz: alloc_zeros_f32(stream, num_cells)?,
        res_e: alloc_zeros_f32(stream, num_cells)?,
    })
}

fn alloc_conserved_buffers(stream: &Arc<CudaStream>, num_cells: usize) -> Result<ConservedBuffers> {
    Ok(ConservedBuffers {
        cons_rho: alloc_zeros_f32(stream, num_cells)?,
        cons_mx: alloc_zeros_f32(stream, num_cells)?,
        cons_my: alloc_zeros_f32(stream, num_cells)?,
        cons_mz: alloc_zeros_f32(stream, num_cells)?,
        cons_e: alloc_zeros_f32(stream, num_cells)?,
    })
}

#[cfg(all(test, feature = "cuda"))]
mod gpu_tests {
    use std::sync::Arc;

    use cudarc::driver::CudaContext;

    use super::*;
    use crate::core::ComputeFloat;
    use crate::exec::gpu::cuda::CudaBackendState;
    use crate::field::ConservedFieldsT;
    use crate::physics::ConservedState;

    #[test]
    #[ignore = "gpu"]
    fn cuda_snapshot_u_n_preserves_physical_level_after_cons_update() {
        let ctx = Arc::new(CudaContext::new(0).expect("CUDA 设备"));
        let stream = ctx.default_stream();
        let mut fields = CudaFieldBuffers::try_new(&stream, 1).expect("buffers");
        let state = ConservedState {
            density: 1.0,
            momentum: [0.2, 0.0, 0.0],
            total_energy: 2.5,
        };
        let base = ConservedFieldsT::<f32>::uniform(1, state).expect("base");
        fields.upload_conserved(&stream, &base).expect("upload");
        fields.snapshot_u_n_on_device(&stream).expect("snapshot");

        let mut perturbed = base.clone();
        perturbed.density.values_mut()[0] = f32::from_real(1.5);
        fields
            .upload_conserved(&stream, &perturbed)
            .expect("perturb cons");

        let mut u_n = base.clone();
        fields
            .download_u_n(&stream, &mut u_n)
            .expect("download u_n");
        assert!((u_n.density.values()[0] - base.density.values()[0]).abs() < 1.0e-6);

        let mut cons = base.clone();
        fields
            .download_conserved(&stream, &mut cons)
            .expect("download cons");
        assert!((cons.density.values()[0] - perturbed.density.values()[0]).abs() < 1.0e-6);
    }

    #[test]
    #[ignore = "gpu"]
    fn cuda_backend_snapshot_u_n_matches_host_after_d2h() {
        let mut backend = CudaBackendState::try_new().expect("backend");
        let state = ConservedState {
            density: 1.1,
            momentum: [0.0, 0.3, 0.0],
            total_energy: 2.7,
        };
        let host = ConservedFieldsT::<f32>::uniform(1, state).expect("host");
        backend
            .snapshot_u_n_on_device(&host)
            .expect("device snapshot");
        let mut u_n = ConservedFieldsT::<f32>::uniform(
            1,
            ConservedState {
                density: 0.0,
                momentum: [0.0; 3],
                total_energy: 0.0,
            },
        )
        .expect("out");
        backend.download_u_n_on_device(&mut u_n).expect("d2h u_n");
        assert!((u_n.density.values()[0] - host.density.values()[0]).abs() < 1.0e-6);
        assert!(backend.u_n_on_device());
    }
}
