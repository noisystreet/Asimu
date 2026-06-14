//! 非结构一阶无粘残差 f32 装配（读取 `face_topology_f32` 预打包几何；scatter 全 f32）。

use crate::discretization::face_flux_typed::{
    face_inviscid_flux_first_order_boundary_soa_f32,
    face_inviscid_flux_first_order_interior_soa_f32,
};
use crate::discretization::residual::{
    accumulate_boundary_face_f32, accumulate_interior_face_f32, is_degenerate_volume_f32,
};
use crate::discretization::vec3_from_f32;
use crate::error::{AsimuError, Result};
use crate::field::ConservedResidualT;

use super::InviscidAssemblyUnstructuredTypedParams;

pub(super) fn assemble_first_order_interior_faces_serial_f32(
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
) -> Result<()> {
    for face in &params.mesh_cache.face_topology_f32.interior {
        if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
            continue;
        }
        if is_degenerate_volume_f32(face.owner_volume)
            || is_degenerate_volume_f32(face.neighbor_volume)
        {
            continue;
        }
        let flux = face_inviscid_flux_first_order_interior_soa_f32(
            face.owner,
            face.neighbor,
            params.primitives,
            vec3_from_f32(face.normal),
            params.eos,
            params.config,
        )?;
        accumulate_interior_face_f32(
            residual,
            face.owner,
            face.neighbor,
            &flux,
            face.area,
            face.owner_volume,
            face.neighbor_volume,
        )?;
    }
    Ok(())
}

pub(super) fn assemble_boundary_faces_first_order_f32(
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
) -> Result<()> {
    for bface in &params.mesh_cache.face_topology_f32.boundary {
        if bface.owner_rhs_scale == 0.0 {
            continue;
        }
        if is_degenerate_volume_f32(bface.owner_volume) {
            continue;
        }
        let ghost = params.ghosts.get_face(bface.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "边界面 FaceId({}) 缺少 ghost 状态",
                bface.face.index()
            ))
        })?;
        let flux = face_inviscid_flux_first_order_boundary_soa_f32(
            params.primitives,
            bface.owner,
            &ghost,
            vec3_from_f32(bface.normal),
            params.eos,
            params.config,
            params.min_pressure,
        )?;
        accumulate_boundary_face_f32(residual, bface.owner, &flux, bface.area, bface.owner_volume)?;
    }
    Ok(())
}
