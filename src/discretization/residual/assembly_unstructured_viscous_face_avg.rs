//! 非结构粘性内面 P7：面心预平均 SoA 填充与 SIMD gather。

use crate::core::Real;
use crate::discretization::gradient::VelocityGradientSlices;
use crate::discretization::viscous::ViscousFaceAveragedLane;

use super::{ViscousAssemblyUnstructuredParams, ViscousAssemblyUnstructuredScratch};

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
        use rayon::prelude::*;
        lanes
            .par_iter_mut()
            .zip(interior.par_iter())
            .for_each(|(lane, face)| {
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

#[cfg(feature = "simd-fvm")]
pub(super) fn gather_viscous_face_batch4_from_face_averaged(
    geom: crate::exec::cpu::ViscousFaceBatchGeom,
    face_indices: [usize; 4],
    lanes: &[ViscousFaceAveragedLane],
) -> crate::exec::cpu::ViscousFaceGather4 {
    use crate::exec::cpu::ViscousFaceGather4;
    let mut g = ViscousFaceGather4::default();
    for lane in 0..4 {
        let avg = lanes[face_indices[lane]];
        g.ux_o[lane] = avg.ux;
        g.ux_n[lane] = avg.ux;
        g.uy_o[lane] = avg.uy;
        g.uy_n[lane] = avg.uy;
        g.uz_o[lane] = avg.uz;
        g.uz_n[lane] = avg.uz;
        g.du_dx_o[lane] = avg.du_dx;
        g.du_dx_n[lane] = avg.du_dx;
        g.du_dy_o[lane] = avg.du_dy;
        g.du_dy_n[lane] = avg.du_dy;
        g.du_dz_o[lane] = avg.du_dz;
        g.du_dz_n[lane] = avg.du_dz;
        g.dv_dx_o[lane] = avg.dv_dx;
        g.dv_dx_n[lane] = avg.dv_dx;
        g.dv_dy_o[lane] = avg.dv_dy;
        g.dv_dy_n[lane] = avg.dv_dy;
        g.dv_dz_o[lane] = avg.dv_dz;
        g.dv_dz_n[lane] = avg.dv_dz;
        g.dw_dx_o[lane] = avg.dw_dx;
        g.dw_dx_n[lane] = avg.dw_dx;
        g.dw_dy_o[lane] = avg.dw_dy;
        g.dw_dy_n[lane] = avg.dw_dy;
        g.dw_dz_o[lane] = avg.dw_dz;
        g.dw_dz_n[lane] = avg.dw_dz;
        g.dt_dx_o[lane] = avg.dt_dx;
        g.dt_dx_n[lane] = avg.dt_dx;
        g.dt_dy_o[lane] = avg.dt_dy;
        g.dt_dy_n[lane] = avg.dt_dy;
        g.dt_dz_o[lane] = avg.dt_dz;
        g.dt_dz_n[lane] = avg.dt_dz;
        g.nx[lane] = geom.nx[lane];
        g.ny[lane] = geom.ny[lane];
        g.nz[lane] = geom.nz[lane];
        g.mu[lane] = geom.mu[lane];
        g.lambda[lane] = geom.lambda[lane];
    }
    g
}
