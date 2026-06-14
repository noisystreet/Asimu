// ADR 0017 P4：非结构粘性 IDWLS RHS 单元并行累加（f32）。
// 数值语义对齐 gradient_unstructured_f32.rs accumulate_lsq_rhs_f32。

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

struct IdwlsInteriorFace {
    uint32_t owner;
    uint32_t neighbor;
    float lsq_dr[3];
    float lsq_w;
};

struct IdwlsBoundaryFace {
    uint32_t owner;
    float lsq_dr[3];
    float lsq_w;
};

struct IdwlsGhostSample {
    float u;
    float v;
    float w;
    float t;
};

__device__ inline void accumulate_rhs_component(float *rhs3, const float *dr, float w, float delta) {
    if (w <= 0.0f) {
        return;
    }
    float coeff = w * delta;
    rhs3[0] += coeff * dr[0];
    rhs3[1] += coeff * dr[1];
    rhs3[2] += coeff * dr[2];
}

extern "C" __global__ void idwls_viscous_accumulate_cell_f32(
    uint32_t num_cells, const uint32_t *__restrict__ owner_off,
    const uint32_t *__restrict__ owner_idx, const uint32_t *__restrict__ neighbor_off,
    const uint32_t *__restrict__ neighbor_idx, const uint32_t *__restrict__ boundary_off,
    const uint32_t *__restrict__ boundary_idx, const IdwlsInteriorFace *__restrict__ interior,
    const IdwlsBoundaryFace *__restrict__ boundary, const IdwlsGhostSample *__restrict__ ghosts,
    const float *__restrict__ ux, const float *__restrict__ uy, const float *__restrict__ uz,
    const float *__restrict__ temperature, float *__restrict__ bu, float *__restrict__ bv,
    float *__restrict__ bw, float *__restrict__ bt) {
    uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) {
        return;
    }

    float bu_acc[3] = {0.0f, 0.0f, 0.0f};
    float bv_acc[3] = {0.0f, 0.0f, 0.0f};
    float bw_acc[3] = {0.0f, 0.0f, 0.0f};
    float bt_acc[3] = {0.0f, 0.0f, 0.0f};

    for (uint32_t k = owner_off[cell]; k < owner_off[cell + 1]; ++k) {
        uint32_t fi = owner_idx[k];
        const IdwlsInteriorFace &f = interior[fi];
        float u_o = ux[f.owner];
        float v_o = uy[f.owner];
        float w_o = uz[f.owner];
        float t_o = temperature[f.owner];
        float u_n = ux[f.neighbor];
        float v_n = uy[f.neighbor];
        float w_n = uz[f.neighbor];
        float t_n = temperature[f.neighbor];
        accumulate_rhs_component(bu_acc, f.lsq_dr, f.lsq_w, u_n - u_o);
        accumulate_rhs_component(bv_acc, f.lsq_dr, f.lsq_w, v_n - v_o);
        accumulate_rhs_component(bw_acc, f.lsq_dr, f.lsq_w, w_n - w_o);
        accumulate_rhs_component(bt_acc, f.lsq_dr, f.lsq_w, t_n - t_o);
    }

    for (uint32_t k = neighbor_off[cell]; k < neighbor_off[cell + 1]; ++k) {
        uint32_t fi = neighbor_idx[k];
        const IdwlsInteriorFace &f = interior[fi];
        float dr_n[3] = {-f.lsq_dr[0], -f.lsq_dr[1], -f.lsq_dr[2]};
        float u_o = ux[f.owner];
        float v_o = uy[f.owner];
        float w_o = uz[f.owner];
        float t_o = temperature[f.owner];
        float u_n = ux[f.neighbor];
        float v_n = uy[f.neighbor];
        float w_n = uz[f.neighbor];
        float t_n = temperature[f.neighbor];
        accumulate_rhs_component(bu_acc, dr_n, f.lsq_w, u_o - u_n);
        accumulate_rhs_component(bv_acc, dr_n, f.lsq_w, v_o - v_n);
        accumulate_rhs_component(bw_acc, dr_n, f.lsq_w, w_o - w_n);
        accumulate_rhs_component(bt_acc, dr_n, f.lsq_w, t_o - t_n);
    }

    for (uint32_t k = boundary_off[cell]; k < boundary_off[cell + 1]; ++k) {
        uint32_t bi = boundary_idx[k];
        const IdwlsBoundaryFace &f = boundary[bi];
        const IdwlsGhostSample &g = ghosts[bi];
        float u_o = ux[f.owner];
        float v_o = uy[f.owner];
        float w_o = uz[f.owner];
        float t_o = temperature[f.owner];
        accumulate_rhs_component(bu_acc, f.lsq_dr, f.lsq_w, g.u - u_o);
        accumulate_rhs_component(bv_acc, f.lsq_dr, f.lsq_w, g.v - v_o);
        accumulate_rhs_component(bw_acc, f.lsq_dr, f.lsq_w, g.w - w_o);
        accumulate_rhs_component(bt_acc, f.lsq_dr, f.lsq_w, g.t - t_o);
    }

    uint32_t base = cell * 3u;
    bu[base] = bu_acc[0];
    bu[base + 1] = bu_acc[1];
    bu[base + 2] = bu_acc[2];
    bv[base] = bv_acc[0];
    bv[base + 1] = bv_acc[1];
    bv[base + 2] = bv_acc[2];
    bw[base] = bw_acc[0];
    bw[base + 1] = bw_acc[1];
    bw[base + 2] = bw_acc[2];
    bt[base] = bt_acc[0];
    bt[base + 1] = bt_acc[1];
    bt[base + 2] = bt_acc[2];
}

struct LsqMatrixF32 {
    float a_xx;
    float a_xy;
    float a_xz;
    float a_yy;
    float a_yz;
    float a_zz;
};

__device__ inline bool solve_symmetric_3x3_f32(const LsqMatrixF32 &a, const float rhs[3],
                                               float out[3]) {
    float c_xx = a.a_yy * a.a_zz - a.a_yz * a.a_yz;
    float c_xy = a.a_xz * a.a_yz - a.a_xy * a.a_zz;
    float c_xz = a.a_xy * a.a_yz - a.a_xz * a.a_yy;
    float c_yy = a.a_xx * a.a_zz - a.a_xz * a.a_xz;
    float c_yz = a.a_xy * a.a_xz - a.a_xx * a.a_yz;
    float c_zz = a.a_xx * a.a_yy - a.a_xy * a.a_xy;
    float det = a.a_xx * c_xx + a.a_xy * c_xy + a.a_xz * c_xz;
    if (fabsf(det) <= FLT_EPSILON) {
        return false;
    }
    float inv_det = 1.0f / det;
    out[0] = (c_xx * rhs[0] + c_xy * rhs[1] + c_xz * rhs[2]) * inv_det;
    out[1] = (c_xy * rhs[0] + c_yy * rhs[1] + c_yz * rhs[2]) * inv_det;
    out[2] = (c_xz * rhs[0] + c_yz * rhs[1] + c_zz * rhs[2]) * inv_det;
    return true;
}

extern "C" __global__ void idwls_solve_gradient_cell_f32(
    uint32_t num_cells, const LsqMatrixF32 *__restrict__ lsq_geometry, const float *__restrict__ bu,
    const float *__restrict__ bv, const float *__restrict__ bw, const float *__restrict__ bt,
    float *__restrict__ du_dx, float *__restrict__ du_dy, float *__restrict__ du_dz,
    float *__restrict__ dv_dx, float *__restrict__ dv_dy, float *__restrict__ dv_dz,
    float *__restrict__ dw_dx, float *__restrict__ dw_dy, float *__restrict__ dw_dz,
    float *__restrict__ dt_dx, float *__restrict__ dt_dy, float *__restrict__ dt_dz) {
    uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) {
        return;
    }
    const LsqMatrixF32 &geom = lsq_geometry[cell];
    uint32_t base = cell * 3u;
    float rhs_u[3] = {bu[base], bu[base + 1], bu[base + 2]};
    float rhs_v[3] = {bv[base], bv[base + 1], bv[base + 2]};
    float rhs_w[3] = {bw[base], bw[base + 1], bw[base + 2]};
    float rhs_t[3] = {bt[base], bt[base + 1], bt[base + 2]};
    float du[3], dv[3], dw[3], dt[3];
    if (!solve_symmetric_3x3_f32(geom, rhs_u, du) || !solve_symmetric_3x3_f32(geom, rhs_v, dv) ||
        !solve_symmetric_3x3_f32(geom, rhs_w, dw) || !solve_symmetric_3x3_f32(geom, rhs_t, dt)) {
        du_dx[cell] = 0.0f;
        du_dy[cell] = 0.0f;
        du_dz[cell] = 0.0f;
        dv_dx[cell] = 0.0f;
        dv_dy[cell] = 0.0f;
        dv_dz[cell] = 0.0f;
        dw_dx[cell] = 0.0f;
        dw_dy[cell] = 0.0f;
        dw_dz[cell] = 0.0f;
        dt_dx[cell] = 0.0f;
        dt_dy[cell] = 0.0f;
        dt_dz[cell] = 0.0f;
        return;
    }
    du_dx[cell] = du[0];
    du_dy[cell] = du[1];
    du_dz[cell] = du[2];
    dv_dx[cell] = dv[0];
    dv_dy[cell] = dv[1];
    dv_dz[cell] = dv[2];
    dw_dx[cell] = dw[0];
    dw_dy[cell] = dw[1];
    dw_dz[cell] = dw[2];
    dt_dx[cell] = dt[0];
    dt_dy[cell] = dt[1];
    dt_dz[cell] = dt[2];
}
