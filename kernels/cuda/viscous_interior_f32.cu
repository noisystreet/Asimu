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

struct ViscousBoundaryFaceGeom {
    uint32_t owner;
    float nx;
    float ny;
    float nz;
    float owner_scale;
    float spacing;
    uint32_t flags;
    float wall_param;
};

struct ViscousBoundaryGhostPrim {
    float rho;
    float pressure;
    float u;
    float v;
    float w;
    float temperature;
};

__device__ inline void scatter_viscous_boundary_atomic(float *res_mx, float *res_my, float *res_mz,
                                                       float *res_e, uint32_t owner, float os,
                                                       const ViscousFlux4 &f) {
    atomicAdd(&res_mx[owner], os * f.mx);
    atomicAdd(&res_my[owner], os * f.my);
    atomicAdd(&res_mz[owner], os * f.mz);
    atomicAdd(&res_e[owner], os * f.energy);
}

__device__ inline void wall_extrapolated_gradient_f32(
    float du_o[3], float dv_o[3], float dw_o[3], float dt_o[3], float u_o[3], float u_g[3],
    float t_o, float t_g, float nx, float ny, float nz, float spacing, float du_g[3], float dv_g[3],
    float dw_g[3], float dt_g[3]) {
    if (spacing <= 1.0e-30f) {
        for (int k = 0; k < 3; ++k) {
            du_g[k] = du_o[k];
            dv_g[k] = dv_o[k];
            dw_g[k] = dw_o[k];
            dt_g[k] = dt_o[k];
        }
        return;
    }
    float inv_two_delta = 1.0f / (2.0f * spacing);
    for (int k = 0; k < 3; ++k) {
        du_g[k] = du_o[k];
        dv_g[k] = dv_o[k];
        dw_g[k] = dw_o[k];
        dt_g[k] = dt_o[k];
    }
    float dudn = (u_g[0] - u_o[0]) * inv_two_delta;
    float grad_n = du_o[0] * nx + du_o[1] * ny + du_o[2] * nz;
    float corr = dudn - grad_n;
    du_g[0] += corr * nx;
    du_g[1] += corr * ny;
    du_g[2] += corr * nz;
    dudn = (u_g[1] - u_o[1]) * inv_two_delta;
    grad_n = dv_o[0] * nx + dv_o[1] * ny + dv_o[2] * nz;
    corr = dudn - grad_n;
    dv_g[0] += corr * nx;
    dv_g[1] += corr * ny;
    dv_g[2] += corr * nz;
    dudn = (u_g[2] - u_o[2]) * inv_two_delta;
    grad_n = dw_o[0] * nx + dw_o[1] * ny + dw_o[2] * nz;
    corr = dudn - grad_n;
    dw_g[0] += corr * nx;
    dw_g[1] += corr * ny;
    dw_g[2] += corr * nz;
    float dtdn = (t_g - t_o) * inv_two_delta;
    float grad_t_n = dt_o[0] * nx + dt_o[1] * ny + dt_o[2] * nz;
    float corr_t = dtdn - grad_t_n;
    dt_g[0] += corr_t * nx;
    dt_g[1] += corr_t * ny;
    dt_g[2] += corr_t * nz;
}

__device__ inline void average_gradient_f32(const float du_l[3], const float dv_l[3],
                                            const float dw_l[3], const float dt_l[3],
                                            const float du_r[3], const float dv_r[3],
                                            const float dw_r[3], const float dt_r[3],
                                            float du_a[3], float dv_a[3], float dw_a[3],
                                            float dt_a[3]) {
    float half = 0.5f;
    for (int k = 0; k < 3; ++k) {
        du_a[k] = half * (du_l[k] + du_r[k]);
        dv_a[k] = half * (dv_l[k] + dv_r[k]);
        dw_a[k] = half * (dw_l[k] + dw_r[k]);
        dt_a[k] = half * (dt_l[k] + dt_r[k]);
    }
}

__device__ inline float wall_heat_flux_into_fluid_f32(float t_owner, float t_ghost, float spacing,
                                                      float lambda, uint32_t flags,
                                                      float wall_param) {
    if ((flags & 4u) == 0u) {
        return 0.0f;
    }
    if ((flags & 8u) != 0u) {
        return wall_param;
    }
    if ((flags & 16u) != 0u) {
        if (spacing <= 1.0e-30f) {
            return 0.0f;
        }
        return lambda * (wall_param - t_owner) / spacing;
    }
    return 0.0f;
}

// 粘性边界面通量（读 device 梯度；ghost 每步 H2D）。
extern "C" __global__ void viscous_boundary_f32(
    const ViscousBoundaryFaceGeom *__restrict__ faces, uint32_t num_faces,
    const ViscousBoundaryGhostPrim *__restrict__ ghosts, const float *__restrict__ prim_ux,
    const float *__restrict__ prim_uy, const float *__restrict__ prim_uz,
    const float *__restrict__ temperatures, const float *__restrict__ du_dx,
    const float *__restrict__ du_dy, const float *__restrict__ du_dz,
    const float *__restrict__ dv_dx, const float *__restrict__ dv_dy, const float *__restrict__ dv_dz,
    const float *__restrict__ dw_dx, const float *__restrict__ dw_dy, const float *__restrict__ dw_dz,
    const float *__restrict__ dt_dx, const float *__restrict__ dt_dy, const float *__restrict__ dt_dz,
    float *__restrict__ res_mx, float *__restrict__ res_my, float *__restrict__ res_mz,
    float *__restrict__ res_e, ViscousTransportParams params) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_faces) {
        return;
    }
    ViscousBoundaryFaceGeom g = faces[i];
    if (g.owner_scale == 0.0f) {
        return;
    }
    uint32_t o = g.owner;
    float nx = g.nx;
    float ny = g.ny;
    float nz = g.nz;
    normalize3(nx, ny, nz);
    float u_o[3] = {prim_ux[o], prim_uy[o], prim_uz[o]};
    ViscousBoundaryGhostPrim gh = ghosts[i];
    float u_g[3] = {gh.u, gh.v, gh.w};
    float du_o[3] = {du_dx[o], du_dy[o], du_dz[o]};
    float dv_o[3] = {dv_dx[o], dv_dy[o], dv_dz[o]};
    float dw_o[3] = {dw_dx[o], dw_dy[o], dw_dz[o]};
    float dt_o[3] = {dt_dx[o], dt_dy[o], dt_dz[o]};
    float du_g[3], dv_g[3], dw_g[3], dt_g[3];
    uint32_t flags = g.flags;
    float t_o = temperatures[o];
    float t_g = gh.temperature;
    if ((flags & 1u) != 0u) {
        wall_extrapolated_gradient_f32(du_o, dv_o, dw_o, dt_o, u_o, u_g, t_o, t_g, nx, ny, nz,
                                       g.spacing, du_g, dv_g, dw_g, dt_g);
    } else {
        for (int k = 0; k < 3; ++k) {
            du_g[k] = du_o[k];
            dv_g[k] = dv_o[k];
            dw_g[k] = dw_o[k];
            dt_g[k] = dt_o[k];
        }
    }
    float mu_o = 0.0f;
    float lambda_o = 0.0f;
    float mu_g = 0.0f;
    float lambda_g = 0.0f;
    cell_transport_f32(t_o, params, mu_o, lambda_o);
    cell_transport_f32(t_g, params, mu_g, lambda_g);
    float mu = 0.5f * (mu_o + mu_g);
    float lambda = 0.5f * (lambda_o + lambda_g);
    float du_a[3], dv_a[3], dw_a[3], dt_a[3];
    average_gradient_f32(du_o, dv_o, dw_o, dt_o, du_g, dv_g, dw_g, dt_g, du_a, dv_a, dw_a, dt_a);
    float half = 0.5f;
    float ux = half * (u_o[0] + u_g[0]);
    float uy = half * (u_o[1] + u_g[1]);
    float uz = half * (u_o[2] + u_g[2]);
    ViscousFlux4 flux = fused_viscous_flux_averaged(
        ux, uy, uz, du_a[0], du_a[1], du_a[2], dv_a[0], dv_a[1], dv_a[2], dw_a[0], dw_a[1], dw_a[2],
        dt_a[0], dt_a[1], dt_a[2], nx, ny, nz, mu, lambda);
    if ((flags & 2u) != 0u) {
        flux.energy = lambda * (dt_a[0] * nx + dt_a[1] * ny + dt_a[2] * nz);
    }
    if ((flags & 4u) != 0u) {
        flux.energy = wall_heat_flux_into_fluid_f32(t_o, t_g, g.spacing, lambda, flags, g.wall_param);
    }
    scatter_viscous_boundary_atomic(res_mx, res_my, res_mz, res_e, o, g.owner_scale, flux);
}
