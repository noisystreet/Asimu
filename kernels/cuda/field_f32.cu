// 守恒场 → 原变量 / 单元粘性扩散系数（P5 device 路径）。

#include <cuda_runtime.h>
#include <stdint.h>

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

__device__ inline float sutherland_mu(float t_dim, float mu_ref, float t_ref, float s) {
    float tr = t_dim / t_ref;
    return mu_ref * powf(tr, 1.5f) * (t_ref + s) / (t_dim + s);
}

__device__ inline float static_temperature_f32(float pressure, float rho, float gamma, float gas_r,
                                               float nondim) {
    float r = rho > 1.0e-30f ? rho : 1.0e-30f;
    if (nondim > 0.0f) {
        return pressure / r * gamma;
    }
    return pressure / (r * gas_r);
}

extern "C" __global__ void fill_primitives_from_conserved_f32(uint32_t num_cells, float gamma,
                                                              float min_pressure,
                                                              const float *__restrict__ cons_rho,
                                                              const float *__restrict__ cons_mx,
                                                              const float *__restrict__ cons_my,
                                                              const float *__restrict__ cons_mz,
                                                              const float *__restrict__ cons_e,
                                                              float *__restrict__ prim_rho,
                                                              float *__restrict__ prim_p,
                                                              float *__restrict__ prim_ux,
                                                              float *__restrict__ prim_uy,
                                                              float *__restrict__ prim_uz) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) {
        return;
    }
    float rho = cons_rho[i];
    if (rho <= 0.0f) {
        return;
    }
    float inv_rho = 1.0f / rho;
    float ux = cons_mx[i] * inv_rho;
    float uy = cons_my[i] * inv_rho;
    float uz = cons_mz[i] * inv_rho;
    float ke = 0.5f * rho * (ux * ux + uy * uy + uz * uz);
    float internal = cons_e[i] - ke;
    if (internal <= 0.0f) {
        return;
    }
    float p = (gamma - 1.0f) * internal;
    if (p < min_pressure) {
        p = min_pressure;
    }
    prim_rho[i] = rho;
    prim_p[i] = p;
    prim_ux[i] = ux;
    prim_uy[i] = uy;
    prim_uz[i] = uz;
}

extern "C" __global__ void cell_viscous_diffusivity_max_f32(
    uint32_t num_cells, float gamma, float gas_r, float nondim_flag, ViscousTransportParams transport,
    const float *__restrict__ prim_rho, const float *__restrict__ prim_p,
    float *__restrict__ diffusivity_out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) {
        return;
    }
    float rho = prim_rho[i] > 1.0e-30f ? prim_rho[i] : 1.0e-30f;
    float p = prim_p[i] > 1.0e-30f ? prim_p[i] : 1.0e-30f;
    float t_star = static_temperature_f32(p, rho, gamma, gas_r, nondim_flag);
    float t_dim = transport.temperature_ref > 0.0f ? t_star * transport.temperature_ref : t_star;
    float mu;
    if (transport.model_kind == 0u) {
        mu = transport.mu_const;
    } else {
        mu = sutherland_mu(t_dim, transport.mu_ref, transport.t_ref, transport.sutherland_s);
        if (transport.viscosity_ref_scale > 0.0f) {
            mu *= transport.viscosity_ref_scale;
        }
    }
    float nu = mu / rho;
    float alpha = mu * transport.cp / (rho * transport.prandtl);
    if (transport.viscosity_ref_scale > 0.0f) {
        alpha *= transport.viscosity_ref_scale;
    }
    diffusivity_out[i] = nu > alpha ? nu : alpha;
}

extern "C" __global__ void cell_static_temperature_f32(uint32_t num_cells, float gamma, float gas_r,
                                                       float nondim_flag,
                                                       const float *__restrict__ prim_rho,
                                                       const float *__restrict__ prim_p,
                                                       float *__restrict__ temp_out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) {
        return;
    }
    float rho = prim_rho[i] > 1.0e-30f ? prim_rho[i] : 1.0e-30f;
    float p = prim_p[i];
    if (nondim_flag > 0.0f) {
        temp_out[i] = p / rho * gamma;
    } else {
        temp_out[i] = p / (rho * gas_r);
    }
}

struct BoundaryConservedGhost {
    float rho;
    float mx;
    float my;
    float mz;
    float e;
};

struct IdwlsGhostSample {
    float u;
    float v;
    float w;
    float t;
};

struct SpectralGhostPrim {
    float rho;
    float pressure;
    float u;
    float v;
    float w;
};

struct ViscousBoundaryGhost {
    float rho;
    float pressure;
    float u;
    float v;
    float w;
    float temperature;
};

extern "C" __global__ void fill_boundary_ghost_buffers_from_conserved_f32(
    uint32_t num_faces, float gamma, float min_pressure, float gas_r, float nondim_flag,
    const BoundaryConservedGhost *__restrict__ cons_in, IdwlsGhostSample *__restrict__ idwls_out,
    SpectralGhostPrim *__restrict__ inviscid_out, SpectralGhostPrim *__restrict__ spectral_out,
    ViscousBoundaryGhost *__restrict__ viscous_out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_faces) {
        return;
    }
    float rho = cons_in[i].rho;
    if (rho <= 0.0f) {
        return;
    }
    float inv_rho = 1.0f / rho;
    float ux = cons_in[i].mx * inv_rho;
    float uy = cons_in[i].my * inv_rho;
    float uz = cons_in[i].mz * inv_rho;
    float ke = 0.5f * rho * (ux * ux + uy * uy + uz * uz);
    float internal = cons_in[i].e - ke;
    if (internal <= 0.0f) {
        return;
    }
    float p = (gamma - 1.0f) * internal;
    if (p < min_pressure) {
        p = min_pressure;
    }
    float t = static_temperature_f32(p, rho, gamma, gas_r, nondim_flag);
    idwls_out[i].u = ux;
    idwls_out[i].v = uy;
    idwls_out[i].w = uz;
    idwls_out[i].t = t;
    inviscid_out[i].rho = rho;
    inviscid_out[i].pressure = p;
    inviscid_out[i].u = ux;
    inviscid_out[i].v = uy;
    inviscid_out[i].w = uz;
    spectral_out[i].rho = rho;
    spectral_out[i].pressure = p;
    spectral_out[i].u = ux;
    spectral_out[i].v = uy;
    spectral_out[i].w = uz;
    viscous_out[i].rho = rho;
    viscous_out[i].pressure = p;
    viscous_out[i].u = ux;
    viscous_out[i].v = uy;
    viscous_out[i].w = uz;
    viscous_out[i].temperature = t;
}

extern "C" __global__ void enforce_conserved_positivity_f32(uint32_t num_cells, float gamma,
                                                            float min_pressure,
                                                            float *__restrict__ cons_rho,
                                                            float *__restrict__ cons_mx,
                                                            float *__restrict__ cons_my,
                                                            float *__restrict__ cons_mz,
                                                            float *__restrict__ cons_e) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) {
        return;
    }
    float rho = cons_rho[i];
    if (!isfinite(rho) || rho <= 0.0f) {
        return;
    }
    float mx = cons_mx[i];
    float my = cons_my[i];
    float mz = cons_mz[i];
    float e = cons_e[i];
    if (!isfinite(mx) || !isfinite(my) || !isfinite(mz) || !isfinite(e)) {
        return;
    }
    float ke = 0.5f * (mx * mx + my * my + mz * mz) / rho;
    float min_internal = min_pressure > 0.0f ? min_pressure / (gamma - 1.0f) : 0.0f;
    float e_min = ke + min_internal;
    if (e < e_min) {
        cons_e[i] = e_min;
    }
}
