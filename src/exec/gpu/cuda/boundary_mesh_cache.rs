//! 边界面静态拓扑 device 缓存 + 每步 ghost H2D。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::boundary_face_geom::{
    ExecInviscidBoundaryFaceStatic, ExecInviscidBoundaryTopology, ExecViscousBoundaryFaceStatic,
    ExecViscousBoundaryTopology, ViscousBoundaryGhostHost,
};
use super::spectral_radius_topology::DeviceSpectralGhostPrim;
use super::transfer::{clone_htod, memcpy_htod};
use crate::error::{AsimuError, Result};

pub struct CudaInviscidBoundaryMeshCache {
    faces: CudaSlice<ExecInviscidBoundaryFaceStatic>,
    ghosts: CudaSlice<DeviceSpectralGhostPrim>,
    num_faces: usize,
}

impl CudaInviscidBoundaryMeshCache {
    pub fn try_upload(
        stream: &Arc<CudaStream>,
        topo: &ExecInviscidBoundaryTopology,
    ) -> Result<Self> {
        let num_faces = topo.num_faces();
        let ghost_pad = if num_faces == 0 {
            vec![DeviceSpectralGhostPrim::default()]
        } else {
            vec![DeviceSpectralGhostPrim::default(); num_faces]
        };
        Ok(Self {
            faces: if num_faces == 0 {
                clone_htod(
                    stream,
                    "init_inviscid_boundary_faces_empty",
                    &[ExecInviscidBoundaryFaceStatic {
                        owner: 0,
                        nx: 0.0,
                        ny: 0.0,
                        nz: 0.0,
                        owner_scale: 0.0,
                    }],
                )?
            } else {
                clone_htod(stream, "init_inviscid_boundary_faces", &topo.faces)?
            },
            ghosts: clone_htod(stream, "init_inviscid_boundary_ghosts", &ghost_pad)?,
            num_faces,
        })
    }

    pub fn faces(&self) -> &CudaSlice<ExecInviscidBoundaryFaceStatic> {
        &self.faces
    }

    pub fn ghosts(&self) -> &CudaSlice<DeviceSpectralGhostPrim> {
        &self.ghosts
    }

    pub(crate) fn ghosts_mut(&mut self) -> &mut CudaSlice<DeviceSpectralGhostPrim> {
        &mut self.ghosts
    }

    pub fn upload_ghosts(
        &mut self,
        stream: &Arc<CudaStream>,
        ghosts: &[DeviceSpectralGhostPrim],
    ) -> Result<()> {
        if ghosts.is_empty() {
            return Ok(());
        }
        if ghosts.len() != self.num_faces {
            return Err(AsimuError::Field(format!(
                "无粘边界面 ghost 数量 {} 与拓扑 {} 不一致",
                ghosts.len(),
                self.num_faces
            )));
        }
        memcpy_htod(stream, "inviscid_boundary_ghosts", ghosts, &mut self.ghosts)
    }
}

pub struct CudaViscousBoundaryMeshCache {
    faces: CudaSlice<ExecViscousBoundaryFaceStatic>,
    ghosts: CudaSlice<ViscousBoundaryGhostHost>,
    num_faces: usize,
}

impl CudaViscousBoundaryMeshCache {
    pub fn try_upload(
        stream: &Arc<CudaStream>,
        topo: &ExecViscousBoundaryTopology,
    ) -> Result<Self> {
        let num_faces = topo.num_faces();
        let ghost_pad = if num_faces == 0 {
            vec![ViscousBoundaryGhostHost::default()]
        } else {
            vec![ViscousBoundaryGhostHost::default(); num_faces]
        };
        Ok(Self {
            faces: if num_faces == 0 {
                clone_htod(
                    stream,
                    "init_viscous_boundary_faces_empty",
                    &[ExecViscousBoundaryFaceStatic {
                        owner: 0,
                        nx: 0.0,
                        ny: 0.0,
                        nz: 0.0,
                        owner_scale: 0.0,
                        spacing: 0.0,
                        flags: 0,
                        wall_param: 0.0,
                    }],
                )?
            } else {
                clone_htod(stream, "init_viscous_boundary_faces", &topo.faces)?
            },
            ghosts: clone_htod(stream, "init_viscous_boundary_ghosts", &ghost_pad)?,
            num_faces,
        })
    }

    pub fn faces(&self) -> &CudaSlice<ExecViscousBoundaryFaceStatic> {
        &self.faces
    }

    pub fn ghosts(&self) -> &CudaSlice<ViscousBoundaryGhostHost> {
        &self.ghosts
    }

    pub(crate) fn ghosts_mut(&mut self) -> &mut CudaSlice<ViscousBoundaryGhostHost> {
        &mut self.ghosts
    }

    pub fn upload_ghosts(
        &mut self,
        stream: &Arc<CudaStream>,
        ghosts: &[ViscousBoundaryGhostHost],
    ) -> Result<()> {
        if ghosts.is_empty() {
            return Ok(());
        }
        if ghosts.len() != self.num_faces {
            return Err(AsimuError::Field(format!(
                "粘性边界面 ghost 数量 {} 与拓扑 {} 不一致",
                ghosts.len(),
                self.num_faces
            )));
        }
        memcpy_htod(stream, "viscous_boundary_ghosts", ghosts, &mut self.ghosts)
    }
}
