//! IDWLS 粘性 RHS 静态拓扑 device 缓存。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};

use super::idwls_topology::ExecIdwlsViscousTopology;
use super::transfer::{clone_dtoh_unchecked, clone_htod, d2h_batch, memcpy_htod};
use crate::discretization::unstructured_idwls_exec_topo::IdwlsGhostSampleHost;
use crate::error::{AsimuError, Result};

/// 步间常驻：内面/边界面几何 + CSR 关联。
pub struct CudaIdwlsMeshDeviceCache {
    interior: CudaSlice<super::idwls_topology::DeviceIdwlsInteriorFace>,
    boundary: CudaSlice<super::idwls_topology::DeviceIdwlsBoundaryFace>,
    owner_offsets: CudaSlice<u32>,
    owner_indices: CudaSlice<u32>,
    neighbor_offsets: CudaSlice<u32>,
    neighbor_indices: CudaSlice<u32>,
    boundary_offsets: CudaSlice<u32>,
    boundary_indices: CudaSlice<u32>,
    num_cells: usize,
    boundary_face_count: usize,
    boundary_ghosts: CudaSlice<IdwlsGhostSampleHost>,
    temperature: CudaSlice<f32>,
}

/// Host 侧粘性 IDWLS RHS 输出槽（D2H 目标）。
pub struct IdwlsViscousRhsHostOut<'a> {
    pub bu: &'a mut [[f32; 3]],
    pub bv: &'a mut [[f32; 3]],
    pub bw: &'a mut [[f32; 3]],
    pub bt: &'a mut [[f32; 3]],
}

/// 每单元 3 分量 RHS（device SoA flat）。
pub struct CudaIdwlsRhsDeviceBuffers {
    bu: CudaSlice<f32>,
    bv: CudaSlice<f32>,
    bw: CudaSlice<f32>,
    bt: CudaSlice<f32>,
    num_cells: usize,
}

impl CudaIdwlsMeshDeviceCache {
    pub fn try_upload(
        stream: &Arc<CudaStream>,
        topo: &ExecIdwlsViscousTopology,
    ) -> Result<(Self, CudaIdwlsRhsDeviceBuffers)> {
        let n = topo.num_cells;
        if n == 0 {
            return Err(AsimuError::Field(
                "CUDA IDWLS 拓扑需要 num_cells > 0".to_string(),
            ));
        }
        let rhs = alloc_rhs_device_buffers(stream, n)?;
        let mesh = upload_static_mesh_cache(stream, topo, n)?;
        Ok((mesh, rhs))
    }

    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    pub fn upload_temperature(&mut self, stream: &Arc<CudaStream>, temp: &[f32]) -> Result<()> {
        if temp.len() != self.num_cells {
            return Err(AsimuError::Field(format!(
                "温度长度 {} 与单元数 {} 不一致",
                temp.len(),
                self.num_cells
            )));
        }
        memcpy_htod(stream, "idwls_temperature", temp, &mut self.temperature)
    }

    pub fn upload_boundary_ghosts(
        &mut self,
        stream: &Arc<CudaStream>,
        ghosts: &[IdwlsGhostSampleHost],
    ) -> Result<()> {
        if ghosts.is_empty() {
            return Ok(());
        }
        if ghosts.len() != self.boundary_face_count {
            self.boundary_face_count = ghosts.len();
            self.boundary_ghosts = clone_htod(stream, "idwls_boundary_ghosts_resize", ghosts)?;
            return Ok(());
        }
        memcpy_htod(
            stream,
            "idwls_boundary_ghosts",
            ghosts,
            &mut self.boundary_ghosts,
        )
    }

    pub(crate) fn interior(&self) -> &CudaSlice<super::idwls_topology::DeviceIdwlsInteriorFace> {
        &self.interior
    }

    pub(crate) fn boundary(&self) -> &CudaSlice<super::idwls_topology::DeviceIdwlsBoundaryFace> {
        &self.boundary
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

    pub(crate) fn boundary_ghosts(&self) -> &CudaSlice<IdwlsGhostSampleHost> {
        &self.boundary_ghosts
    }

    pub(crate) fn temperature(&self) -> &CudaSlice<f32> {
        &self.temperature
    }
}

impl CudaIdwlsRhsDeviceBuffers {
    #[must_use]
    pub(crate) fn num_cells(&self) -> usize {
        self.num_cells
    }

    pub(crate) fn bu_slice(&self) -> &CudaSlice<f32> {
        &self.bu
    }

    pub(crate) fn bv_slice(&self) -> &CudaSlice<f32> {
        &self.bv
    }

    pub(crate) fn bw_slice(&self) -> &CudaSlice<f32> {
        &self.bw
    }

    pub(crate) fn bt_slice(&self) -> &CudaSlice<f32> {
        &self.bt
    }

    pub(crate) fn bu_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.bu
    }

    pub(crate) fn bv_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.bv
    }

    pub(crate) fn bw_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.bw
    }

    pub(crate) fn bt_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.bt
    }

    pub fn download_into(
        &self,
        stream: &Arc<CudaStream>,
        out: IdwlsViscousRhsHostOut<'_>,
    ) -> Result<()> {
        let n = self.num_cells;
        if out.bu.len() != n || out.bv.len() != n || out.bw.len() != n || out.bt.len() != n {
            return Err(AsimuError::Field(format!("host IDWLS RHS 长度须为 {n}")));
        }
        let rhs_bytes = n
            .checked_mul(3)
            .and_then(|x| x.checked_mul(4))
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| AsimuError::Field("IDWLS RHS D2H 字节数溢出".to_string()))?;
        d2h_batch("idwls_rhs", rhs_bytes, n, || {
            unpack_component(stream, &self.bu, out.bu)?;
            unpack_component(stream, &self.bv, out.bv)?;
            unpack_component(stream, &self.bw, out.bw)?;
            unpack_component(stream, &self.bt, out.bt)?;
            Ok(())
        })
    }
}

fn unpack_component(
    stream: &Arc<CudaStream>,
    device: &CudaSlice<f32>,
    host: &mut [[f32; 3]],
) -> Result<()> {
    let flat = clone_dtoh_unchecked(stream, device)?;
    for (i, row) in host.iter_mut().enumerate() {
        let base = i * 3;
        row[0] = flat[base];
        row[1] = flat[base + 1];
        row[2] = flat[base + 2];
    }
    Ok(())
}

fn alloc_rhs_device_buffers(
    stream: &Arc<CudaStream>,
    num_cells: usize,
) -> Result<CudaIdwlsRhsDeviceBuffers> {
    let rhs_len = num_cells
        .checked_mul(3)
        .ok_or_else(|| AsimuError::Field("IDWLS RHS 长度溢出".to_string()))?;
    let alloc_one = || -> Result<CudaSlice<f32>> {
        stream
            .alloc_zeros::<f32>(rhs_len)
            .map_err(|e| AsimuError::Exec(format!("CUDA IDWLS RHS 分配失败: {e:?}")))
    };
    Ok(CudaIdwlsRhsDeviceBuffers {
        bu: alloc_one()?,
        bv: alloc_one()?,
        bw: alloc_one()?,
        bt: alloc_one()?,
        num_cells,
    })
}

fn upload_static_mesh_cache(
    stream: &Arc<CudaStream>,
    topo: &ExecIdwlsViscousTopology,
    num_cells: usize,
) -> Result<CudaIdwlsMeshDeviceCache> {
    let nb = topo.boundary_faces.len().max(1);
    let ghost_pad = vec![IdwlsGhostSampleHost::default(); nb];
    Ok(CudaIdwlsMeshDeviceCache {
        interior: upload_slice(stream, "init_idwls_interior_faces", &topo.interior_faces)?,
        boundary: upload_slice(stream, "init_idwls_boundary_faces", &topo.boundary_faces)?,
        owner_offsets: upload_slice(stream, "init_idwls_owner_offsets", &topo.owner_offsets)?,
        owner_indices: upload_slice(stream, "init_idwls_owner_indices", &topo.owner_indices)?,
        neighbor_offsets: upload_slice(
            stream,
            "init_idwls_neighbor_offsets",
            &topo.neighbor_offsets,
        )?,
        neighbor_indices: upload_slice(
            stream,
            "init_idwls_neighbor_indices",
            &topo.neighbor_indices,
        )?,
        boundary_offsets: upload_slice(
            stream,
            "init_idwls_boundary_offsets",
            &topo.boundary_offsets,
        )?,
        boundary_indices: upload_slice(
            stream,
            "init_idwls_boundary_indices",
            &topo.boundary_indices,
        )?,
        num_cells,
        boundary_face_count: topo.boundary_faces.len(),
        boundary_ghosts: upload_slice(stream, "init_idwls_boundary_ghosts", &ghost_pad)?,
        temperature: stream
            .alloc_zeros::<f32>(num_cells)
            .map_err(|e| AsimuError::Exec(format!("CUDA 温度缓冲分配失败: {e:?}")))?,
    })
}

fn upload_slice<T: DeviceRepr + Clone + Default>(
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
