// ADR 0017 G2：非结构粘性内面通量 + scatter（f32）。
// 数值语义对齐 src/discretization/viscous_f32.rs fused_interior_viscous_face_flux_averaged_f32。

#include <cuda_runtime.h>
#include <stdint.h>

struct ViscousFaceGeom {
    uint32_t owner;
    uint32_t neighbor;
    float nx;
    float ny;
    float nz;
    float mu;
    float lambda;
    float owner_scale;
    float neighbor_scale;
};

struct ViscousFlux4 {
    float mx;
    float my;
    float mz;
    float energy;
};

__device__ inline void normalize3(float &x, float &y, float &z) {
    float mag = sqrtf(x * x + y * y + z * z);
    if (mag > 1.0e-30f) {
        float inv = 1.0f / mag;
        x *= inv;
        y *= inv;
        z *= inv;
    }
}

__device__ inline ViscousFlux4 fused_viscous_flux_averaged(
    float ux, float uy, float uz, float du_dx, float du_dy, float du_dz, float dv_dx, float dv_dy,
    float dv_dz, float dw_dx, float dw_dy, float dw_dz, float dt_dx, float dt_dy, float dt_dz,
    float nx, float ny, float nz, float mu, float lambda) {
    float div_u = du_dx + dv_dy + dw_dz;
    float two_thirds = 2.0f / 3.0f;
    float tau_xx = mu * (2.0f * du_dx - two_thirds * div_u);
    float tau_yy = mu * (2.0f * dv_dy - two_thirds * div_u);
    float tau_zz = mu * (2.0f * dw_dz - two_thirds * div_u);
    float tau_xy = mu * (du_dy + dv_dx);
    float tau_xz = mu * (du_dz + dw_dx);
    float tau_yz = mu * (dv_dz + dw_dy);

    float tau_dot_n0 = tau_xx * nx + tau_xy * ny + tau_xz * nz;
    float tau_dot_n1 = tau_xy * nx + tau_yy * ny + tau_yz * nz;
    float tau_dot_n2 = tau_xz * nx + tau_yz * ny + tau_zz * nz;
    float heat_flux = lambda * (dt_dx * nx + dt_dy * ny + dt_dz * nz);
    float energy_flux = -(heat_flux + tau_dot_n0 * ux + tau_dot_n1 * uy + tau_dot_n2 * uz);

    ViscousFlux4 out;
    out.mx = -tau_dot_n0;
    out.my = -tau_dot_n1;
    out.mz = -tau_dot_n2;
    out.energy = energy_flux;
    return out;
}

__device__ inline void scatter_viscous(float *res_mx, float *res_my, float *res_mz, float *res_e,
                                       uint32_t owner, uint32_t neighbor, float os, float ns,
                                       const ViscousFlux4 &f) {
    res_mx[owner] += os * f.mx;
    res_my[owner] += os * f.my;
    res_mz[owner] += os * f.mz;
    res_e[owner] += os * f.energy;
    res_mx[neighbor] += ns * f.mx;
    res_my[neighbor] += ns * f.my;
    res_mz[neighbor] += ns * f.mz;
    res_e[neighbor] += ns * f.energy;
}

extern "C" __global__ void viscous_interior_bucket_f32(
    const uint32_t *__restrict__ bucket_faces, uint32_t num_faces,
    const ViscousFaceGeom *__restrict__ face_geom, const float *__restrict__ prim_ux,
    const float *__restrict__ prim_uy, const float *__restrict__ prim_uz,
    const float *__restrict__ du_dx, const float *__restrict__ du_dy, const float *__restrict__ du_dz,
    const float *__restrict__ dv_dx, const float *__restrict__ dv_dy, const float *__restrict__ dv_dz,
    const float *__restrict__ dw_dx, const float *__restrict__ dw_dy, const float *__restrict__ dw_dz,
    const float *__restrict__ dt_dx, const float *__restrict__ dt_dy, const float *__restrict__ dt_dz,
    float *__restrict__ res_mx, float *__restrict__ res_my, float *__restrict__ res_mz,
    float *__restrict__ res_e) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_faces) {
        return;
    }
    uint32_t face_idx = bucket_faces[i];
    ViscousFaceGeom g = face_geom[face_idx];
    if (g.owner_scale == 0.0f && g.neighbor_scale == 0.0f) {
        return;
    }
    uint32_t o = g.owner;
    uint32_t n = g.neighbor;
    float half = 0.5f;
    float ux = half * (prim_ux[o] + prim_ux[n]);
    float uy = half * (prim_uy[o] + prim_uy[n]);
    float uz = half * (prim_uz[o] + prim_uz[n]);
    float du_dx_a = half * (du_dx[o] + du_dx[n]);
    float du_dy_a = half * (du_dy[o] + du_dy[n]);
    float du_dz_a = half * (du_dz[o] + du_dz[n]);
    float dv_dx_a = half * (dv_dx[o] + dv_dx[n]);
    float dv_dy_a = half * (dv_dy[o] + dv_dy[n]);
    float dv_dz_a = half * (dv_dz[o] + dv_dz[n]);
    float dw_dx_a = half * (dw_dx[o] + dw_dx[n]);
    float dw_dy_a = half * (dw_dy[o] + dw_dy[n]);
    float dw_dz_a = half * (dw_dz[o] + dw_dz[n]);
    float dt_dx_a = half * (dt_dx[o] + dt_dx[n]);
    float dt_dy_a = half * (dt_dy[o] + dt_dy[n]);
    float dt_dz_a = half * (dt_dz[o] + dt_dz[n]);
    float nx = g.nx;
    float ny = g.ny;
    float nz = g.nz;
    normalize3(nx, ny, nz);
    ViscousFlux4 flux = fused_viscous_flux_averaged(
        ux, uy, uz, du_dx_a, du_dy_a, du_dz_a, dv_dx_a, dv_dy_a, dv_dz_a, dw_dx_a, dw_dy_a, dw_dz_a,
        dt_dx_a, dt_dy_a, dt_dz_a, nx, ny, nz, g.mu, g.lambda);
    scatter_viscous(res_mx, res_my, res_mz, res_e, o, n, g.owner_scale, g.neighbor_scale, flux);
}

// 数值语义对齐 prepare_unstructured_viscous_transport_f32 + fill_face_transport_coefficients_for_topology。
struct ViscousTransportParams {
    uint32_t model_kind;
    float mu_const;
    float lambda_const;
    float mu_ref;
    float t_ref;
    float sutherland_s;
    float prandtl;
    float viscosity_ref_scale;
    float temperature_ref;
    float cp;
};

__device__ inline float dimensional_temperature_f32(float t_static, float temperature_ref) {
    return temperature_ref > 0.0f ? t_static * temperature_ref : t_static;
}

__device__ inline float dynamic_viscosity_sutherland_f32(float t_dim, float mu_ref, float t_ref,
                                                         float sutherland_s) {
    float tr = t_dim / t_ref;
    return mu_ref * powf(tr, 1.5f) * (t_ref + sutherland_s) / (t_dim + sutherland_s);
}

__device__ inline void cell_transport_f32(float t_static, const ViscousTransportParams &p, float &mu,
                                          float &lambda) {
    if (p.model_kind == 0u) {
        mu = p.mu_const;
        lambda = p.lambda_const;
        return;
    }
    float t_dim = dimensional_temperature_f32(t_static, p.temperature_ref);
    float mu_cell = dynamic_viscosity_sutherland_f32(t_dim, p.mu_ref, p.t_ref, p.sutherland_s);
    mu = mu_cell;
    lambda = mu_cell * p.cp / p.prandtl;
    if (p.viscosity_ref_scale > 0.0f) {
        mu *= p.viscosity_ref_scale;
        lambda *= p.viscosity_ref_scale;
    }
}

extern "C" __global__ void viscous_face_transport_f32(
    ViscousFaceGeom *__restrict__ face_geom, uint32_t num_faces,
    const float *__restrict__ temperatures, ViscousTransportParams params) {
    uint32_t face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_faces) {
        return;
    }
    ViscousFaceGeom g = face_geom[face_idx];
    if (g.owner_scale == 0.0f && g.neighbor_scale == 0.0f) {
        return;
    }
    if (params.model_kind == 0u) {
        g.mu = params.mu_const;
        g.lambda = params.lambda_const;
        face_geom[face_idx] = g;
        return;
    }
    float mu_o = 0.0f;
    float lambda_o = 0.0f;
    float mu_n = 0.0f;
    float lambda_n = 0.0f;
    cell_transport_f32(temperatures[g.owner], params, mu_o, lambda_o);
    cell_transport_f32(temperatures[g.neighbor], params, mu_n, lambda_n);
    g.mu = 0.5f * (mu_o + mu_n);
    g.lambda = 0.5f * (lambda_o + lambda_n);
    face_geom[face_idx] = g;
}
