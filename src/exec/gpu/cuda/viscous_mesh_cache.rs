//! 粘性内面拓扑 device 缓存（着色桶静态；face_geom 每步刷新）。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::transfer::{clone_htod, memcpy_htod};
use super::viscous_face_geom::{DeviceViscousFaceGeom, ExecViscousInteriorTopology};
use crate::error::{AsimuError, Result};

/// 着色桶索引（与无粘共用 coloring，步间可缓存）。
pub struct CudaViscousBucketCache {
    bucket_faces: Vec<CudaSlice<u32>>,
    bucket_lens: Vec<u32>,
}

impl CudaViscousBucketCache {
    pub fn try_upload(
        stream: &Arc<CudaStream>,
        topo: &ExecViscousInteriorTopology,
    ) -> Result<Self> {
        let mut bucket_faces = Vec::with_capacity(topo.color_buckets.len());
        let mut bucket_lens = Vec::with_capacity(topo.color_buckets.len());
        for bucket in &topo.color_buckets {
            bucket_lens.push(bucket.face_indices.len() as u32);
            bucket_faces.push(clone_htod(
                stream,
                "init_viscous_color_bucket",
                &bucket.face_indices,
            )?);
        }
        Ok(Self {
            bucket_faces,
            bucket_lens,
        })
    }

    pub fn bucket_faces(&self, color: usize) -> Result<&CudaSlice<u32>> {
        self.bucket_faces
            .get(color)
            .ok_or_else(|| AsimuError::Exec(format!("CUDA 粘性着色桶索引越界: color={color}")))
    }

    pub fn bucket_len(&self, color: usize) -> Result<u32> {
        self.bucket_lens
            .get(color)
            .copied()
            .ok_or_else(|| AsimuError::Exec(format!("CUDA 粘性着色桶长度越界: color={color}")))
    }

    pub fn num_colors(&self) -> usize {
        self.bucket_faces.len()
    }
}

/// 每步上传的粘性面几何（含 \(\mu,\lambda\)）。
pub struct CudaViscousFaceGeomBuffer {
    face_geom: CudaSlice<DeviceViscousFaceGeom>,
}

impl CudaViscousFaceGeomBuffer {
    pub fn try_upload(stream: &Arc<CudaStream>, faces: &[DeviceViscousFaceGeom]) -> Result<Self> {
        Ok(Self {
            face_geom: clone_htod(stream, "init_viscous_face_geom", faces)?,
        })
    }

    pub fn refresh(
        &mut self,
        stream: &Arc<CudaStream>,
        faces: &[DeviceViscousFaceGeom],
    ) -> Result<()> {
        if self.face_geom.len() == faces.len() {
            memcpy_htod(stream, "viscous_face_geom", faces, &mut self.face_geom)?;
            return Ok(());
        }
        self.face_geom = clone_htod(stream, "viscous_face_geom_resize", faces)?;
        Ok(())
    }

    pub fn face_geom(&self) -> &CudaSlice<DeviceViscousFaceGeom> {
        &self.face_geom
    }
}
