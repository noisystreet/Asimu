//! 谱半径静态拓扑 device 缓存。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::spectral_radius_topology::{DeviceSpectralGhostPrim, ExecSpectralRadiusTopology};
use super::transfer::{clone_dtoh_unchecked, clone_htod, d2h_batch, memcpy_htod};
use crate::error::{AsimuError, Result};

/// 步间常驻：内面/边界面几何 + CSR 关联。
pub struct CudaSpectralMeshDeviceCache {
    interior: CudaSlice<super::spectral_radius_topology::DeviceSpectralInteriorFace>,
    boundary: CudaSlice<super::spectral_radius_topology::DeviceSpectralBoundaryFace>,
    owner_offsets: CudaSlice<u32>,
    owner_indices: CudaSlice<u32>,
    neighbor_offsets: CudaSlice<u32>,
    neighbor_indices: CudaSlice<u32>,
    boundary_offsets: CudaSlice<u32>,
    boundary_indices: CudaSlice<u32>,
    num_cells: usize,
    boundary_face_count: usize,
    boundary_ghosts: CudaSlice<DeviceSpectralGhostPrim>,
    diffusivity: CudaSlice<f32>,
    sigma: CudaSlice<f32>,
    cell_dts: CudaSlice<f32>,
    min_dt_scratch: CudaSlice<f32>,
}

impl CudaSpectralMeshDeviceCache {
    pub fn try_upload(stream: &Arc<CudaStream>, topo: &ExecSpectralRadiusTopology) -> Result<Self> {
        let n = topo.num_cells;
        if n == 0 {
            return Err(AsimuError::Field(
                "CUDA 谱半径拓扑需要 num_cells > 0".to_string(),
            ));
        }
        let nb = topo.boundary_faces.len().max(1);
        let ghost_pad = vec![DeviceSpectralGhostPrim::default(); nb];
        Ok(Self {
            interior: upload_slice(stream, "init_spectral_interior_faces", &topo.interior_faces)?,
            boundary: upload_slice(stream, "init_spectral_boundary_faces", &topo.boundary_faces)?,
            owner_offsets: upload_slice(
                stream,
                "init_spectral_owner_offsets",
                &topo.owner_offsets,
            )?,
            owner_indices: upload_slice(
                stream,
                "init_spectral_owner_indices",
                &topo.owner_indices,
            )?,
            neighbor_offsets: upload_slice(
                stream,
                "init_spectral_neighbor_offsets",
                &topo.neighbor_offsets,
            )?,
            neighbor_indices: upload_slice(
                stream,
                "init_spectral_neighbor_indices",
                &topo.neighbor_indices,
            )?,
            boundary_offsets: upload_slice(
                stream,
                "init_spectral_boundary_offsets",
                &topo.boundary_offsets,
            )?,
            boundary_indices: upload_slice(
                stream,
                "init_spectral_boundary_indices",
                &topo.boundary_indices,
            )?,
            num_cells: n,
            boundary_face_count: topo.boundary_faces.len(),
            boundary_ghosts: upload_slice(stream, "init_spectral_boundary_ghosts", &ghost_pad)?,
            diffusivity: stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| AsimuError::Exec(format!("CUDA 谱半径 diff 分配失败: {e:?}")))?,
            sigma: stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| AsimuError::Exec(format!("CUDA 谱半径 sigma 分配失败: {e:?}")))?,
            cell_dts: stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| AsimuError::Exec(format!("CUDA 谱半径 cell_dts 分配失败: {e:?}")))?,
            min_dt_scratch: stream
                .alloc_zeros::<f32>(1)
                .map_err(|e| AsimuError::Exec(format!("CUDA min_dt scratch 分配失败: {e:?}")))?,
        })
    }

    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    pub fn upload_boundary_ghosts(
        &mut self,
        stream: &Arc<CudaStream>,
        ghosts: &[DeviceSpectralGhostPrim],
    ) -> Result<()> {
        if ghosts.is_empty() {
            return Ok(());
        }
        if ghosts.len() != self.boundary_face_count {
            self.boundary_face_count = ghosts.len();
            self.boundary_ghosts = clone_htod(stream, "spectral_boundary_ghosts_resize", ghosts)?;
            return Ok(());
        }
        memcpy_htod(
            stream,
            "spectral_boundary_ghosts",
            ghosts,
            &mut self.boundary_ghosts,
        )
    }

    pub fn upload_diffusivity(
        &mut self,
        stream: &Arc<CudaStream>,
        diff: Option<&[f32]>,
    ) -> Result<()> {
        match diff {
            Some(values) => {
                if values.len() != self.num_cells {
                    return Err(AsimuError::Field(format!(
                        "diffusivity 长度 {} 与单元数 {} 不一致",
                        values.len(),
                        self.num_cells
                    )));
                }
                memcpy_htod(
                    stream,
                    "spectral_diffusivity",
                    values,
                    &mut self.diffusivity,
                )
            }
            None => stream
                .memset_zeros(&mut self.diffusivity)
                .map_err(|e| AsimuError::Exec(format!("CUDA 谱半径 diff 清零失败: {e:?}"))),
        }
    }

    pub fn download_timestep(
        &self,
        stream: &Arc<CudaStream>,
        sigma_host: &mut [f32],
        cell_dts_host: &mut [f32],
    ) -> Result<()> {
        if sigma_host.len() != self.num_cells || cell_dts_host.len() != self.num_cells {
            return Err(AsimuError::Field(format!(
                "host timestep 长度须为 {}",
                self.num_cells
            )));
        }
        let bytes = self
            .num_cells
            .checked_mul(4)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| AsimuError::Field("timestep D2H 字节数溢出".to_string()))?;
        d2h_batch("spectral_timestep", bytes, self.num_cells, || {
            let sigma_flat = clone_dtoh_unchecked(stream, &self.sigma)?;
            let dts_flat = clone_dtoh_unchecked(stream, &self.cell_dts)?;
            sigma_host.copy_from_slice(sigma_flat.as_slice());
            cell_dts_host.copy_from_slice(dts_flat.as_slice());
            Ok(())
        })
    }

    pub fn download_min_cell_dt(
        &mut self,
        stream: &Arc<CudaStream>,
        spectral_module: &super::module::CudaSpectralRadiusModule,
    ) -> Result<f32> {
        super::spectral_radius::launch_init_min_positive_scratch(
            stream,
            &spectral_module.init_min_positive_scratch,
            &mut self.min_dt_scratch,
        )?;
        let num_cells = self.num_cells as u32;
        let cell_dts = &self.cell_dts;
        let min_scratch = &mut self.min_dt_scratch;
        super::spectral_radius::launch_min_positive_cell_dt(
            stream,
            &spectral_module.min_positive_dt,
            num_cells,
            cell_dts,
            min_scratch,
        )?;
        let min_host = super::transfer::clone_dtoh(stream, "spectral_min_cell_dt", min_scratch)?;
        let min_dt = min_host.first().copied().unwrap_or(0.0);
        if !min_dt.is_finite() || min_dt <= 0.0 {
            return Err(AsimuError::Field(
                "CUDA min_positive_cell_dt 未得到有效正有限 Δt".to_string(),
            ));
        }
        Ok(min_dt)
    }

    #[allow(dead_code)]
    pub fn download_sigma(&self, stream: &Arc<CudaStream>, host: &mut [f32]) -> Result<()> {
        if host.len() != self.num_cells {
            return Err(AsimuError::Field(format!(
                "host sigma 长度须为 {}",
                self.num_cells
            )));
        }
        let bytes = self
            .num_cells
            .checked_mul(4)
            .ok_or_else(|| AsimuError::Field("谱半径 D2H 字节数溢出".to_string()))?;
        d2h_batch("spectral_radius_sigma", bytes, self.num_cells, || {
            let flat = clone_dtoh_unchecked(stream, &self.sigma)?;
            host.copy_from_slice(flat.as_slice());
            Ok(())
        })
    }

    pub(crate) fn owner_offsets(&self) -> &CudaSlice<u32> {
        &self.owner_offsets
    }

    pub(crate) fn owner_indices(&self) -> &CudaSlice<u32> {
        &self.owner_indices
    }

    pub(crate) fn neighbor_offsets(&self) -> &CudaSlice<u32> {
        &self.neighbor_offsets
    }

    pub(crate) fn neighbor_indices(&self) -> &CudaSlice<u32> {
        &self.neighbor_indices
    }

    pub(crate) fn boundary_offsets(&self) -> &CudaSlice<u32> {
        &self.boundary_offsets
    }

    pub(crate) fn boundary_indices(&self) -> &CudaSlice<u32> {
        &self.boundary_indices
    }

    pub(crate) fn interior(
        &self,
    ) -> &CudaSlice<super::spectral_radius_topology::DeviceSpectralInteriorFace> {
        &self.interior
    }

    pub(crate) fn boundary(
        &self,
    ) -> &CudaSlice<super::spectral_radius_topology::DeviceSpectralBoundaryFace> {
        &self.boundary
    }

    pub(crate) fn boundary_ghosts(&self) -> &CudaSlice<DeviceSpectralGhostPrim> {
        &self.boundary_ghosts
    }

    pub(crate) fn boundary_ghosts_mut(&mut self) -> &mut CudaSlice<DeviceSpectralGhostPrim> {
        &mut self.boundary_ghosts
    }

    pub(crate) fn diffusivity(&self) -> &CudaSlice<f32> {
        &self.diffusivity
    }

    pub(crate) fn diffusivity_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.diffusivity
    }

    pub(crate) fn sigma(&self) -> &CudaSlice<f32> {
        &self.sigma
    }

    pub(crate) fn cell_dts(&self) -> &CudaSlice<f32> {
        &self.cell_dts
    }

    pub(crate) fn sigma_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.sigma
    }
}

fn upload_slice<T: cudarc::driver::DeviceRepr + Clone + Default>(
    stream: &Arc<CudaStream>,
    label: &'static str,
    host: &[T],
) -> Result<CudaSlice<T>> {
    if host.is_empty() {
        let pad = vec![T::default()];
        return clone_htod(stream, label, &pad);
    }
    clone_htod(stream, label, host)
}
