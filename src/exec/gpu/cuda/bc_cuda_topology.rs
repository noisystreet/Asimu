//! 可压缩 BC device 静态拓扑（与 `boundary_bc_f32.cu` 布局一致）。

use crate::boundary::{BoundaryKind, BoundaryPatch, BoundaryRegistry, BoundarySet, WallHeat};
use crate::discretization::unstructured_face_cache_f32::UnstructuredFaceTopologyF32;
use crate::error::{AsimuError, Result};
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

/// BC 种类（与 CUDA `BC_KIND_*` 一致）。
pub const BC_KIND_WALL: u32 = 1;
pub const BC_KIND_FARFIELD: u32 = 2;
pub const BC_KIND_INLET: u32 = 3;
pub const BC_KIND_OUTLET: u32 = 4;
pub const BC_KIND_SYMMETRY: u32 = 5;
pub const BC_KIND_COPY_OWNER: u32 = 6;

/// 壁面热模式（flags bit 1–2）。
pub const BC_WALL_HEAT_ADIABATIC: u32 = 0;
pub const BC_WALL_HEAT_ISOTHERMAL: u32 = 1;
pub const BC_WALL_HEAT_FLUX: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceBcPatchParams {
    pub kind: u32,
    pub flags: u32,
    pub f0: f32,
    pub f1: f32,
    pub f2: f32,
    pub f3: f32,
    pub f4: f32,
    pub f5: f32,
    pub f6: f32,
    pub f7: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceBcFaceStatic {
    pub owner: u32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub spacing: f32,
    pub patch_index: u32,
}

unsafe impl cudarc::driver::DeviceRepr for DeviceBcPatchParams {}
unsafe impl cudarc::driver::DeviceRepr for DeviceBcFaceStatic {}

#[derive(Debug, Clone)]
pub struct ExecCompressibleBcTopology {
    pub faces: Vec<DeviceBcFaceStatic>,
    pub patches: Vec<DeviceBcPatchParams>,
}

impl ExecCompressibleBcTopology {
    #[must_use]
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }
}

/// 当前 CUDA BC kernel 是否覆盖全部 patch（否则回退 CPU ghost + H2D）。
#[must_use]
pub fn cuda_compressible_bc_supported(patches: &BoundarySet) -> bool {
    for patch in patches.patches() {
        if !patch_cuda_bc_supported(&patch.kind) {
            return false;
        }
    }
    true
}

fn patch_cuda_bc_supported(kind: &BoundaryKind) -> bool {
    match kind {
        BoundaryKind::Farfield { .. }
        | BoundaryKind::Inlet { .. }
        | BoundaryKind::Outlet { .. }
        | BoundaryKind::Wall { .. }
        | BoundaryKind::Symmetry
        | BoundaryKind::Periodic { .. } => true,
        BoundaryKind::Dirichlet { .. }
        | BoundaryKind::Neumann { .. }
        | BoundaryKind::IncompressibleVelocityInlet { .. }
        | BoundaryKind::IncompressiblePressureOutlet { .. }
        | BoundaryKind::MovingWall { .. }
        | BoundaryKind::TurbulentInlet { .. } => false,
    }
}

fn encode_wall_patch(
    no_slip: bool,
    heat: WallHeat,
    viscous: Option<&ViscousPhysicsConfig>,
    eos: &IdealGasEoS,
) -> Result<DeviceBcPatchParams> {
    let mut flags = if no_slip { 1u32 } else { 0u32 };
    let mut f0 = 0.0_f32;
    let mut f1 = 0.0_f32;
    match heat {
        WallHeat::Adiabatic => {
            flags |= BC_WALL_HEAT_ADIABATIC << 1;
        }
        WallHeat::Isothermal { temperature } => {
            flags |= BC_WALL_HEAT_ISOTHERMAL << 1;
            f0 = temperature as f32;
        }
        WallHeat::HeatFlux { flux } => {
            flags |= BC_WALL_HEAT_FLUX << 1;
            f0 = flux as f32;
            let viscous = viscous.ok_or_else(|| {
                AsimuError::Boundary("壁面 heat_flux 须启用 [navier_stokes] 粘性物性".to_string())
            })?;
            let t_ref = viscous.temperature_ref.unwrap_or(300.0) as f32;
            f1 = viscous.thermal_conductivity_coefficient(t_ref as f64, eos)? as f32;
            if f1 <= 0.0_f32 {
                return Err(AsimuError::Boundary(
                    "CUDA 壁面热流 BC：热导率无效".to_string(),
                ));
            }
        }
    }
    Ok(DeviceBcPatchParams {
        kind: BC_KIND_WALL,
        flags,
        f0,
        f1,
        ..DeviceBcPatchParams::default()
    })
}

fn encode_patch_params(
    patch: &BoundaryPatch,
    viscous: Option<&ViscousPhysicsConfig>,
    eos: &IdealGasEoS,
) -> Result<DeviceBcPatchParams> {
    let handler = BoundaryRegistry::handler_for(&patch.kind);
    match (&handler, &patch.kind) {
        (_, BoundaryKind::Wall { no_slip, heat }) => {
            encode_wall_patch(*no_slip, *heat, viscous, eos)
        }
        (
            _,
            BoundaryKind::Farfield {
                mach,
                pressure,
                temperature,
                alpha,
                beta,
            },
        ) => {
            use crate::physics::FreestreamParams;
            let dir = FreestreamParams {
                mach: *mach,
                pressure: *pressure,
                temperature: *temperature,
                alpha: *alpha,
                beta: *beta,
                velocity_direction: [1.0, 0.0, 0.0],
            }
            .effective_direction();
            Ok(DeviceBcPatchParams {
                kind: BC_KIND_FARFIELD,
                f0: *pressure as f32,
                f1: *temperature as f32,
                f2: *mach as f32,
                f3: *alpha as f32,
                f4: *beta as f32,
                f5: dir[0] as f32,
                f6: dir[1] as f32,
                f7: dir[2] as f32,
                ..DeviceBcPatchParams::default()
            })
        }
        (
            _,
            BoundaryKind::Inlet {
                total_pressure,
                total_temperature,
                velocity_direction,
                supersonic,
                ..
            },
        ) => Ok(DeviceBcPatchParams {
            kind: BC_KIND_INLET,
            flags: u32::from(*supersonic),
            f0: *total_pressure as f32,
            f1: *total_temperature as f32,
            f2: velocity_direction[0] as f32,
            f3: velocity_direction[1] as f32,
            f4: velocity_direction[2] as f32,
            ..DeviceBcPatchParams::default()
        }),
        (
            _,
            BoundaryKind::Outlet {
                static_pressure,
                supersonic,
                ..
            },
        ) => Ok(DeviceBcPatchParams {
            kind: BC_KIND_OUTLET,
            flags: u32::from(*supersonic),
            f0: *static_pressure as f32,
            ..DeviceBcPatchParams::default()
        }),
        (_, BoundaryKind::Symmetry) => Ok(DeviceBcPatchParams {
            kind: BC_KIND_SYMMETRY,
            ..DeviceBcPatchParams::default()
        }),
        (_, BoundaryKind::Periodic { .. }) => Ok(DeviceBcPatchParams {
            kind: BC_KIND_COPY_OWNER,
            ..DeviceBcPatchParams::default()
        }),
        _ => Err(AsimuError::Boundary(format!(
            "patch \"{}\" 类型暂不支持 CUDA BC",
            patch.name
        ))),
    }
}

/// 按 `face_topology_f32.boundary` 顺序构建 BC 面静态表 + patch 参数表。
pub fn build_cuda_compressible_bc_topology(
    face_topology_f32: &UnstructuredFaceTopologyF32,
    patches: &BoundarySet,
    viscous: Option<&ViscousPhysicsConfig>,
    eos: &IdealGasEoS,
) -> Result<ExecCompressibleBcTopology> {
    BoundaryRegistry::validate_patches(patches.patches())?;
    if !cuda_compressible_bc_supported(patches) {
        return Err(AsimuError::Boundary(
            "边界 patch 含 CUDA BC 未覆盖类型".to_string(),
        ));
    }
    let patch_params: Vec<DeviceBcPatchParams> = patches
        .patches()
        .iter()
        .map(|p| encode_patch_params(p, viscous, eos))
        .collect::<Result<Vec<_>>>()?;
    let mut face_patch = vec![None; face_topology_f32.boundary.len().max(1)];
    for (pi, patch) in patches.patches().iter().enumerate() {
        for &face in &patch.face_ids {
            let idx = face.index() as usize;
            if idx >= face_patch.len() {
                face_patch.resize(idx + 1, None);
            }
            face_patch[idx] = Some(pi as u32);
        }
    }
    let mut faces = Vec::with_capacity(face_topology_f32.boundary.len());
    for bface in &face_topology_f32.boundary {
        let patch_index = face_patch
            .get(bface.face.index() as usize)
            .copied()
            .flatten()
            .ok_or_else(|| {
                AsimuError::Boundary(format!(
                    "边界面 FaceId({}) 未分配 patch",
                    bface.face.index()
                ))
            })?;
        let mag = (bface.normal[0] * bface.normal[0]
            + bface.normal[1] * bface.normal[1]
            + bface.normal[2] * bface.normal[2])
            .sqrt();
        let (nx, ny, nz) = if mag > 1.0e-30_f32 {
            let inv = 1.0_f32 / mag;
            (
                bface.normal[0] * inv,
                bface.normal[1] * inv,
                bface.normal[2] * inv,
            )
        } else {
            (bface.normal[0], bface.normal[1], bface.normal[2])
        };
        faces.push(DeviceBcFaceStatic {
            owner: bface.owner as u32,
            nx,
            ny,
            nz,
            spacing: bface.spacing,
            patch_index,
        });
    }
    Ok(ExecCompressibleBcTopology {
        faces,
        patches: patch_params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet, WallHeat};
    use crate::core::FaceId;

    #[test]
    fn cuda_compressible_bc_supported_for_standard_patches() {
        let patches = BoundarySet::new(vec![
            BoundaryPatch::new(
                "wall",
                vec![FaceId::new(0)],
                BoundaryKind::Wall {
                    no_slip: true,
                    heat: WallHeat::Adiabatic,
                },
            ),
            BoundaryPatch::new(
                "farfield",
                vec![FaceId::new(1)],
                BoundaryKind::Farfield {
                    mach: 0.5,
                    pressure: 101325.0,
                    temperature: 300.0,
                    alpha: 0.0,
                    beta: 0.0,
                },
            ),
        ]);
        assert!(cuda_compressible_bc_supported(&patches));
    }

    #[test]
    fn cuda_compressible_bc_not_supported_for_diffusion_dirichlet() {
        let patches = BoundarySet::new(vec![BoundaryPatch::new(
            "dirichlet",
            vec![FaceId::new(0)],
            BoundaryKind::Dirichlet { value: 1.0 },
        )]);
        assert!(!cuda_compressible_bc_supported(&patches));
    }
}
