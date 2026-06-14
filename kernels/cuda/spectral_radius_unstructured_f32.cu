// ADR 0017：非结构单元谱半径 GPU 累加（f32 原变量；单元累加 f64 保抛物项）。
// 数值语义对齐 spectral_radius_unstructured_f32.rs。

#include <cuda_runtime.h>
#include <stdint.h>

struct SpectralInteriorFace {
    uint32_t owner;
    uint32_t neighbor;
    float nx;
    float ny;
    float nz;
    float area;
    float inv_owner_volume;
    float inv_neighbor_volume;
    float owner_volume;
    float neighbor_volume;
};

struct SpectralBoundaryFace {
    uint32_t owner;
    float nx;
    float ny;
    float nz;
    float area;
    float inv_owner_volume;
    float owner_volume;
};

struct SpectralGhostPrim {
    float rho;
    float pressure;
    float u;
    float v;
    float w;
};

__device__ inline double normal_speed_plus_sound(double rho, double pressure, float ux, float uy,
                                               float uz, float nx, float ny, float nz,
                                               float gamma) {
    rho = fmax(rho, 1.0e-30);
    pressure = fmax(pressure, 1.0e-30);
    double u_n = (double)ux * nx + (double)uy * ny + (double)uz * nz;
    double a = sqrt((double)gamma * pressure / rho);
    return fabs(u_n) + a;
}

__device__ inline double face_spectral_radius(double rho_l, double p_l, float ux_l, float uy_l,
                                              float uz_l, double rho_r, double p_r, float ux_r,
                                              float uy_r, float uz_r, float nx, float ny, float nz,
                                              float gamma) {
    double lam_l =
        normal_speed_plus_sound(rho_l, p_l, ux_l, uy_l, uz_l, nx, ny, nz, gamma);
    double lam_r =
        normal_speed_plus_sound(rho_r, p_r, ux_r, uy_r, uz_r, nx, ny, nz, gamma);
    return 0.5 * (lam_l + lam_r);
}

__device__ inline void add_hyperbolic(double *sigma, double radius, float area, float inv_vol) {
    if (inv_vol > 0.0f) {
        *sigma += radius * (double)area * (double)inv_vol;
    }
}

__device__ inline void add_parabolic(double *sigma, float diff, float area, float volume) {
    const float factor = 6.0f;
    const float vol_eps = 1.0e-30f;
    if (diff > 0.0f && area > 1.0e-30f && volume > vol_eps) {
        double a = (double)area;
        double v = (double)volume;
        *sigma += (double)factor * (double)diff * a * a / (v * v);
    }
}

extern "C" __global__ void spectral_radius_accumulate_cell_f32(
    uint32_t num_cells, float gamma, uint32_t viscous_enabled,
    const uint32_t *__restrict__ owner_off, const uint32_t *__restrict__ owner_idx,
    const uint32_t *__restrict__ neighbor_off, const uint32_t *__restrict__ neighbor_idx,
    const uint32_t *__restrict__ boundary_off, const uint32_t *__restrict__ boundary_idx,
    const SpectralInteriorFace *__restrict__ interior,
    const SpectralBoundaryFace *__restrict__ boundary,
    const SpectralGhostPrim *__restrict__ ghosts, const float *__restrict__ rho,
    const float *__restrict__ pressure, const float *__restrict__ ux, const float *__restrict__ uy,
    const float *__restrict__ uz, const float *__restrict__ diffusivity, float *__restrict__ sigma_out) {
    uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) {
        return;
    }

    double sigma = 0.0;
    float diff = 0.0f;
    if (viscous_enabled != 0u) {
        diff = diffusivity[cell];
    }

    for (uint32_t k = owner_off[cell]; k < owner_off[cell + 1]; ++k) {
        uint32_t fi = owner_idx[k];
        const SpectralInteriorFace &f = interior[fi];
        double radius = face_spectral_radius(
            (double)rho[f.owner], (double)pressure[f.owner], ux[f.owner], uy[f.owner], uz[f.owner],
            (double)rho[f.neighbor], (double)pressure[f.neighbor], ux[f.neighbor], uy[f.neighbor],
            uz[f.neighbor], f.nx, f.ny, f.nz, gamma);
        add_hyperbolic(&sigma, radius, f.area, f.inv_owner_volume);
        if (viscous_enabled != 0u) {
            add_parabolic(&sigma, diff, f.area, f.owner_volume);
        }
    }

    for (uint32_t k = neighbor_off[cell]; k < neighbor_off[cell + 1]; ++k) {
        uint32_t fi = neighbor_idx[k];
        const SpectralInteriorFace &f = interior[fi];
        double radius = face_spectral_radius(
            (double)rho[f.owner], (double)pressure[f.owner], ux[f.owner], uy[f.owner], uz[f.owner],
            (double)rho[f.neighbor], (double)pressure[f.neighbor], ux[f.neighbor], uy[f.neighbor],
            uz[f.neighbor], f.nx, f.ny, f.nz, gamma);
        add_hyperbolic(&sigma, radius, f.area, f.inv_neighbor_volume);
        if (viscous_enabled != 0u) {
            add_parabolic(&sigma, diff, f.area, f.neighbor_volume);
        }
    }

    for (uint32_t k = boundary_off[cell]; k < boundary_off[cell + 1]; ++k) {
        uint32_t bi = boundary_idx[k];
        const SpectralBoundaryFace &f = boundary[bi];
        const SpectralGhostPrim &g = ghosts[bi];
        double radius = face_spectral_radius(
            (double)rho[f.owner], (double)pressure[f.owner], ux[f.owner], uy[f.owner], uz[f.owner],
            (double)g.rho, (double)g.pressure, g.u, g.v, g.w, f.nx, f.ny, f.nz, gamma);
        add_hyperbolic(&sigma, radius, f.area, f.inv_owner_volume);
        if (viscous_enabled != 0u) {
            add_parabolic(&sigma, diff, f.area, f.owner_volume);
        }
    }

    const double eps = 2.220446049250313e-16;
    sigma_out[cell] = (float)fmax(sigma, eps);
}

// dt_i = CFL / sigma_i；fixed_dt>0 时全单元填 fixed_dt（对齐 finalize_cell_dts_from_sigma_f32）。
extern "C" __global__ void finalize_cell_dts_f32(uint32_t num_cells, float cfl, float fixed_dt,
                                                 uint32_t use_fixed_dt,
                                                 const float *__restrict__ sigma,
                                                 float *__restrict__ cell_dts) {
    uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) {
        return;
    }
    if (use_fixed_dt != 0u) {
        cell_dts[cell] = fixed_dt;
        return;
    }
    float s = sigma[cell];
    if (s <= 0.0f) {
        cell_dts[cell] = 0.0f;
        return;
    }
    cell_dts[cell] = cfl / s;
}

// device 上求 cell_dts 正有限最小值（单 float D2H；对齐 min_positive_dt_f32）。
__device__ inline void atomic_min_positive_f32(float *addr, float val) {
    if (!isfinite(val) || val <= 0.0f) {
        return;
    }
    int *addr_as_i = (int *)addr;
    int old = *addr_as_i;
    while (val < __int_as_float(old)) {
        int assumed = old;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
        if (assumed == old) {
            break;
        }
    }
}

extern "C" __global__ void min_positive_cell_dt_f32(uint32_t num_cells,
                                                    const float *__restrict__ cell_dts,
                                                    float *__restrict__ min_out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) {
        return;
    }
    atomic_min_positive_f32(min_out, cell_dts[i]);
}
