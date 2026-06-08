//! 非结构粘性内面 P7：面心预平均 SoA 填充与 SIMD gather。

use crate::core::Real;
use crate::discretization::gradient::VelocityGradientSlices;
use crate::discretization::viscous::ViscousFaceAveragedLane;

use super::ViscousAssemblyUnstructuredParams;
#[cfg(not(feature = "simd-fvm"))]
use super::ViscousAssemblyUnstructuredScratch;

/// 单内面面心预平均（cell SoA 直读，供 remainder / 非 SIMD 路径）。
#[cfg(feature = "simd-fvm")]
pub(super) fn face_averaged_lane_at(
    face_idx: usize,
    params: &ViscousAssemblyUnstructuredParams<'_>,
) -> ViscousFaceAveragedLane {
    let face = &params.face_topology.interior[face_idx];
    let prim = params.primitives;
    let grad = params.gradients.velocity_gradient_slices();
    average_face_lane(
        face.owner,
        face.neighbor,
        prim.velocity_x.values(),
        prim.velocity_y.values(),
        prim.velocity_z.values(),
        &grad,
    )
}

#[cfg(not(feature = "simd-fvm"))]
pub(super) fn fill_face_averaged_viscous_soa(
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) {
    let num_faces = params.face_topology.interior.len();
    scratch.ensure_face_averaged(num_faces);
    let prim = params.primitives;
    let ux = prim.velocity_x.values();
    let uy = prim.velocity_y.values();
    let uz = prim.velocity_z.values();
    let grad = params.gradients.velocity_gradient_slices();
    let interior = &params.face_topology.interior;
    let lanes = &mut scratch.face_averaged.lanes;

    #[cfg(feature = "parallel-fvm")]
    {
        crate::exec::parallel::par_for_each_zip_mut2(lanes, interior, |lane, face| {
            if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
                return;
            }
            *lane = average_face_lane(face.owner, face.neighbor, ux, uy, uz, &grad);
        });
    }

    #[cfg(not(feature = "parallel-fvm"))]
    {
        for (lane, face) in lanes.iter_mut().zip(interior.iter()) {
            if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
                continue;
            }
            *lane = average_face_lane(face.owner, face.neighbor, ux, uy, uz, &grad);
        }
    }
}

#[inline(always)]
fn average_face_lane(
    owner: usize,
    neighbor: usize,
    ux: &[Real],
    uy: &[Real],
    uz: &[Real],
    grad: &VelocityGradientSlices<'_>,
) -> ViscousFaceAveragedLane {
    let half = 0.5;
    ViscousFaceAveragedLane {
        ux: half * (ux[owner] + ux[neighbor]),
        uy: half * (uy[owner] + uy[neighbor]),
        uz: half * (uz[owner] + uz[neighbor]),
        du_dx: half * (grad.du_dx[owner] + grad.du_dx[neighbor]),
        du_dy: half * (grad.du_dy[owner] + grad.du_dy[neighbor]),
        du_dz: half * (grad.du_dz[owner] + grad.du_dz[neighbor]),
        dv_dx: half * (grad.dv_dx[owner] + grad.dv_dx[neighbor]),
        dv_dy: half * (grad.dv_dy[owner] + grad.dv_dy[neighbor]),
        dv_dz: half * (grad.dv_dz[owner] + grad.dv_dz[neighbor]),
        dw_dx: half * (grad.dw_dx[owner] + grad.dw_dx[neighbor]),
        dw_dy: half * (grad.dw_dy[owner] + grad.dw_dy[neighbor]),
        dw_dz: half * (grad.dw_dz[owner] + grad.dw_dz[neighbor]),
        dt_dx: half * (grad.dt_dx[owner] + grad.dt_dx[neighbor]),
        dt_dy: half * (grad.dt_dy[owner] + grad.dt_dy[neighbor]),
        dt_dz: half * (grad.dt_dz[owner] + grad.dt_dz[neighbor]),
    }
}
