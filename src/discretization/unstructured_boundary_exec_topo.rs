//! 非结构边界面 CUDA exec 拓扑与每步 ghost 打包（P2）。

#[cfg(feature = "cuda")]
use crate::boundary::WallHeat;
#[cfg(feature = "cuda")]
use crate::core::Real;
#[cfg(feature = "cuda")]
use crate::discretization::BoundaryGhostBuffer;
#[cfg(feature = "cuda")]
use crate::discretization::unstructured_face_cache::UnstructuredBoundaryViscousKind;
use crate::discretization::unstructured_face_cache_f32::UnstructuredFaceTopologyF32;
#[cfg(feature = "cuda")]
use crate::discretization::unstructured_spectral_exec_topo::SpectralGhostPrimHost;
#[cfg(feature = "cuda")]
use crate::error::{AsimuError, Result};
#[cfg(feature = "cuda")]
use crate::exec::gpu::cuda::{
    ExecInviscidBoundaryFaceStatic, ExecInviscidBoundaryTopology, ExecViscousBoundaryFaceStatic,
    ExecViscousBoundaryTopology, ViscousBoundaryGhostHost,
};
#[cfg(feature = "cuda")]
use crate::field::primitive_from_conserved_relaxed_f32_from_state;
#[cfg(feature = "cuda")]
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

#[cfg(feature = "cuda")]
fn unit_normal(nx: f32, ny: f32, nz: f32) -> (f32, f32, f32) {
    let mag = (nx * nx + ny * ny + nz * nz).sqrt();
    if mag > 1.0e-30 {
        let inv = 1.0 / mag;
        (nx * inv, ny * inv, nz * inv)
    } else {
        (nx, ny, nz)
    }
}

#[cfg(feature = "cuda")]
fn encode_viscous_boundary_flags(kind: UnstructuredBoundaryViscousKind) -> (u32, f32) {
    let mut flags = 0u32;
    if kind.is_wall {
        flags |= 1;
    }
    if kind.no_slip {
        flags |= 2;
    }
    let mut wall_param = 0.0_f32;
    if let Some(heat) = kind.wall_heat {
        flags |= 4;
        match heat {
            WallHeat::Adiabatic => {}
            WallHeat::HeatFlux { flux } => {
                flags |= 8;
                wall_param = flux as f32;
            }
            WallHeat::Isothermal { temperature } => {
                flags |= 16;
                wall_param = temperature as f32;
            }
        }
    }
    (flags, wall_param)
}

/// 构建无粘边界面 CUDA 拓扑（静态几何；init 一次 H2D）。
#[cfg(feature = "cuda")]
#[must_use]
pub fn build_cuda_inviscid_boundary_topology(
    topology_f32: &UnstructuredFaceTopologyF32,
) -> ExecInviscidBoundaryTopology {
    let faces = topology_f32
        .boundary
        .iter()
        .map(|face| {
            let (nx, ny, nz) = unit_normal(face.normal[0], face.normal[1], face.normal[2]);
            ExecInviscidBoundaryFaceStatic {
                owner: face.owner as u32,
                nx,
                ny,
                nz,
                owner_scale: face.owner_rhs_scale,
            }
        })
        .collect();
    ExecInviscidBoundaryTopology { faces }
}

/// 构建粘性边界面 CUDA 拓扑（静态几何；init 一次 H2D）。
#[cfg(feature = "cuda")]
#[must_use]
pub fn build_cuda_viscous_boundary_topology(
    topology_f32: &UnstructuredFaceTopologyF32,
) -> ExecViscousBoundaryTopology {
    let faces = topology_f32
        .boundary
        .iter()
        .map(|face| {
            let (nx, ny, nz) = unit_normal(face.normal[0], face.normal[1], face.normal[2]);
            let (flags, wall_param) = encode_viscous_boundary_flags(face.viscous);
            ExecViscousBoundaryFaceStatic {
                owner: face.owner as u32,
                nx,
                ny,
                nz,
                owner_scale: face.owner_rhs_scale,
                spacing: face.spacing,
                flags,
                wall_param,
            }
        })
        .collect();
    ExecViscousBoundaryTopology { faces }
}

/// 无粘边界面 ghost 原变量（对齐谱半径 CUDA 布局）。
#[cfg(feature = "cuda")]
pub fn prepare_inviscid_boundary_ghost_prims_f32(
    topology_f32: &UnstructuredFaceTopologyF32,
    ghosts: &BoundaryGhostBuffer,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<Vec<SpectralGhostPrimHost>> {
    let mut out = Vec::with_capacity(topology_f32.boundary.len());
    for face in &topology_f32.boundary {
        let ghost = ghosts.get_face(face.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "无粘边界面 CUDA FaceId({}) 缺少 ghost",
                face.face.index()
            ))
        })?;
        let prim =
            primitive_from_conserved_relaxed_f32_from_state(eos, &ghost.conserved, min_pressure)?;
        out.push(SpectralGhostPrimHost {
            rho: prim.density,
            pressure: prim.pressure,
            u: prim.velocity[0],
            v: prim.velocity[1],
            w: prim.velocity[2],
        });
    }
    Ok(out)
}

/// 粘性边界面 ghost 原变量 + 静温。
#[cfg(feature = "cuda")]
pub fn prepare_viscous_boundary_ghost_prims_f32(
    topology_f32: &UnstructuredFaceTopologyF32,
    ghosts: &BoundaryGhostBuffer,
    eos: &IdealGasEoS,
    viscous: &ViscousPhysicsConfig,
    min_pressure: Real,
) -> Result<Vec<ViscousBoundaryGhostHost>> {
    let mut out = Vec::with_capacity(topology_f32.boundary.len());
    for face in &topology_f32.boundary {
        let ghost = ghosts.get_face(face.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "粘性边界面 CUDA FaceId({}) 缺少 ghost",
                face.face.index()
            ))
        })?;
        let prim =
            primitive_from_conserved_relaxed_f32_from_state(eos, &ghost.conserved, min_pressure)?;
        let temperature =
            viscous.static_temperature_f32(prim.pressure, prim.density.max(1.0e-30_f32), eos);
        out.push(ViscousBoundaryGhostHost {
            rho: prim.density,
            pressure: prim.pressure,
            u: prim.velocity[0],
            v: prim.velocity[1],
            w: prim.velocity[2],
            temperature,
        });
    }
    Ok(out)
}

/// IDWLS 粘性边界面 ghost 样本（每步 H2D；与 `gradient_unstructured_f32_cuda` 一致）。
#[cfg(feature = "cuda")]
pub fn prepare_idwls_boundary_ghost_samples_f32(
    topology_f32: &UnstructuredFaceTopologyF32,
    ghosts: &BoundaryGhostBuffer,
    eos: &IdealGasEoS,
    viscous: &ViscousPhysicsConfig,
    min_pressure: Real,
) -> Result<Vec<crate::discretization::unstructured_idwls_exec_topo::IdwlsGhostSampleHost>> {
    use crate::discretization::unstructured_idwls_exec_topo::IdwlsGhostSampleHost;
    let mut out = Vec::with_capacity(topology_f32.boundary.len());
    for face in &topology_f32.boundary {
        let ghost = ghosts.get_face(face.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "IDWLS 边界面 CUDA FaceId({}) 缺少 ghost",
                face.face.index()
            ))
        })?;
        let prim =
            primitive_from_conserved_relaxed_f32_from_state(eos, &ghost.conserved, min_pressure)?;
        let t = viscous.static_temperature_f32(prim.pressure, prim.density.max(1.0e-30_f32), eos);
        out.push(IdwlsGhostSampleHost {
            u: prim.velocity[0],
            v: prim.velocity[1],
            w: prim.velocity[2],
            t,
        });
    }
    Ok(out)
}
