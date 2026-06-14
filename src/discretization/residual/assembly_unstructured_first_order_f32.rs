//! 非结构一阶无粘残差 f32 装配（读取 `face_topology_f32` 预打包几何）。

use crate::core::Real;
use crate::discretization::face_flux_typed::InviscidFaceFluxTyped;
use crate::discretization::residual::{
    accumulate_boundary_face_typed, accumulate_interior_face_typed, is_degenerate_volume,
};
use crate::discretization::vec3_from_f32;
use crate::error::{AsimuError, Result};
use crate::field::ConservedResidualT;

use super::InviscidAssemblyUnstructuredTypedParams;
use super::first_order_typed::first_order_interior_flux;

pub(super) fn assemble_first_order_interior_faces_serial_f32(
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, f32>,
) -> Result<()> {
    for face in &params.mesh_cache.face_topology_f32.interior {
        if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
            continue;
        }
        if is_degenerate_volume(face.owner_volume as Real)
            || is_degenerate_volume(face.neighbor_volume as Real)
        {
            continue;
        }
        let flux = first_order_interior_flux(
            params.primitives,
            face.owner,
            face.neighbor,
            vec3_from_f32(face.normal),
            params.eos,
            params.config,
        )?;
        accumulate_interior_face_typed(
            residual,
            face.owner,
            face.neighbor,
            &flux,
            face.area as Real,
            face.owner_volume as Real,
            face.neighbor_volume as Real,
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
        if is_degenerate_volume(bface.owner_volume as Real) {
            continue;
        }
        let ghost = params.ghosts.get_face(bface.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "边界面 FaceId({}) 缺少 ghost 状态",
                bface.face.index()
            ))
        })?;
        let flux = f32::first_order_boundary_soa(
            params.primitives,
            bface.owner,
            &ghost,
            vec3_from_f32(bface.normal),
            params.eos,
            params.config,
            params.min_pressure,
        )?;
        accumulate_boundary_face_typed(
            residual,
            bface.owner,
            &flux,
            bface.area as Real,
            bface.owner_volume as Real,
        )?;
    }
    Ok(())
}
