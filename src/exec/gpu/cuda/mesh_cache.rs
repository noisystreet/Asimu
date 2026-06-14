//! 内面拓扑 device 缓存（init 一次 H2D）。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::buffers::DeviceFaceGeom;
use super::face_geom::ExecInteriorFaceTopology;
use super::transfer::clone_htod;
use crate::error::{AsimuError, Result};

/// 着色桶面索引与静态几何的 device 快照。
pub struct CudaMeshDeviceCache {
    face_geom: CudaSlice<DeviceFaceGeom>,
    bucket_faces: Vec<CudaSlice<u32>>,
    bucket_lens: Vec<u32>,
}

impl CudaMeshDeviceCache {
    pub fn try_upload(stream: &Arc<CudaStream>, topo: &ExecInteriorFaceTopology) -> Result<Self> {
        let face_geom_host: Vec<DeviceFaceGeom> = topo
            .faces
            .iter()
            .map(|f| DeviceFaceGeom {
                owner: f.owner,
                neighbor: f.neighbor,
                nx: f.nx,
                ny: f.ny,
                nz: f.nz,
                owner_scale: f.owner_scale,
                neighbor_scale: f.neighbor_scale,
            })
            .collect();
        let face_geom = clone_htod(stream, "init_inviscid_face_geom", &face_geom_host)?;
        let mut bucket_faces = Vec::with_capacity(topo.color_buckets.len());
        let mut bucket_lens = Vec::with_capacity(topo.color_buckets.len());
        for bucket in &topo.color_buckets {
            bucket_lens.push(bucket.face_indices.len() as u32);
            bucket_faces.push(clone_htod(
                stream,
                "init_inviscid_color_bucket",
                &bucket.face_indices,
            )?);
        }
        Ok(Self {
            face_geom,
            bucket_faces,
            bucket_lens,
        })
    }

    pub fn face_geom(&self) -> &CudaSlice<DeviceFaceGeom> {
        &self.face_geom
    }

    pub fn bucket_faces(&self, color: usize) -> Result<&CudaSlice<u32>> {
        self.bucket_faces
            .get(color)
            .ok_or_else(|| AsimuError::Exec(format!("CUDA 着色桶索引越界: color={color}")))
    }

    pub fn bucket_len(&self, color: usize) -> Result<u32> {
        self.bucket_lens
            .get(color)
            .copied()
            .ok_or_else(|| AsimuError::Exec(format!("CUDA 着色桶长度越界: color={color}")))
    }

    pub fn num_colors(&self) -> usize {
        self.bucket_faces.len()
    }
}
