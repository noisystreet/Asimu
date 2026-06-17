// 非结构 f32 LU-SGS 前/后扫（CellId 序；单线程串行 Gauss-Seidel）。
// 线搜索 stabilize 在 host 侧完成（`lu_sgs_common::stabilize_sweep_update_f32`）。

#include <cuda_runtime.h>
#include <stdint.h>

__device__ inline float implicit_scale_f32(float dt, float sigma, float omega,
                                           float inv_dt_phys) {
    return omega * dt / (1.0f + dt * sigma + dt * inv_dt_phys);
}

__device__ inline void cons_vec_f32(const float *rho, const float *mx, const float *my,
                                    const float *mz, const float *e, uint32_t i, float out[5]) {
    out[0] = rho[i];
    out[1] = mx[i];
    out[2] = my[i];
    out[3] = mz[i];
    out[4] = e[i];
}

__device__ inline void write_cons_f32(float *rho, float *mx, float *my, float *mz, float *e,
                                      uint32_t i, const float lane[5]) {
    rho[i] = lane[0];
    mx[i] = lane[1];
    my[i] = lane[2];
    mz[i] = lane[3];
    e[i] = lane[4];
}

__device__ inline void fill_prim_cell_f32(uint32_t i, float gamma, float min_pressure,
                                          const float *cons_rho, const float *cons_mx,
                                          const float *cons_my, const float *cons_mz,
                                          const float *cons_e, float *prim_rho, float *prim_p,
                                          float *prim_ux, float *prim_uy, float *prim_uz) {
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

__device__ inline float normal_speed_plus_sound_f32(float rho, float pressure,
                                                    const float velocity[3],
                                                    const float normal[3], float gamma) {
    rho = rho > 1.0e-30f ? rho : 1.0e-30f;
    float u_n = velocity[0] * normal[0] + velocity[1] * normal[1] + velocity[2] * normal[2];
    float a = sqrtf(gamma * (pressure > 1.0e-30f ? pressure : 1.0e-30f) / rho);
    return fabsf(u_n) + a;
}

__device__ inline float face_spectral_radius_f32(const float left[3], float lrho, float lp,
                                                 const float right[3], float rrho, float rp,
                                                 const float normal[3], float gamma) {
    float lam_l = normal_speed_plus_sound_f32(lrho, lp, left, normal, gamma);
    float lam_r = normal_speed_plus_sound_f32(rrho, rp, right, normal, gamma);
    return 0.5f * (lam_l + lam_r);
}

__device__ inline bool is_physical_conserved_f32(const float lane[5], float gamma,
                                                 float min_pressure) {
    float rho = lane[0];
    if (rho <= 0.0f || !isfinite(rho) || !isfinite(lane[4])) {
        return false;
    }
    float ke = 0.5f * (lane[1] * lane[1] + lane[2] * lane[2] + lane[3] * lane[3]) / rho;
    float min_internal = min_pressure > 0.0f ? min_pressure : 0.0f;
    min_internal /= (gamma - 1.0f);
    float internal = lane[4] - ke;
    return isfinite(internal) && internal > min_internal;
}

__device__ inline void apply_increment_lane_f32(const float base[5], const float inc[5],
                                                float factor, float out[5]) {
    out[0] = base[0] + inc[0] * factor;
    out[1] = base[1] + inc[1] * factor;
    out[2] = base[2] + inc[2] * factor;
    out[3] = base[3] + inc[3] * factor;
    out[4] = base[4] + inc[4] * factor;
}

__device__ inline float max_physical_increment_scale_f32(const float base[5], const float inc[5],
                                                         float scale, float gamma,
                                                         float min_pressure) {
    if (scale <= 0.0f) {
        return 0.0f;
    }
    float trial_full[5];
    apply_increment_lane_f32(base, inc, scale, trial_full);
    if (is_physical_conserved_f32(trial_full, gamma, min_pressure)) {
        return scale;
    }
    float alpha = 0.5f;
    for (int k = 0; k < 12; ++k) {
        float trial_scale = alpha * scale;
        apply_increment_lane_f32(base, inc, trial_scale, trial_full);
        if (is_physical_conserved_f32(trial_full, gamma, min_pressure)) {
            return trial_scale;
        }
        alpha *= 0.5f;
    }
    return 0.0f;
}

__device__ inline void add_coupling_delta_f32(float source[5], uint32_t cell, uint32_t neighbor,
                                              float area, const float normal[3], float volume,
                                              float gamma, const float *cons_rho,
                                              const float *cons_mx, const float *cons_my,
                                              const float *cons_mz, const float *cons_e,
                                              const float *u0_rho, const float *u0_mx,
                                              const float *u0_my, const float *u0_mz,
                                              const float *u0_e, const float *prim_rho,
                                              const float *prim_p, const float *prim_ux,
                                              const float *prim_uy, const float *prim_uz) {
    float left_v[3] = {prim_ux[cell], prim_uy[cell], prim_uz[cell]};
    float right_v[3] = {prim_ux[neighbor], prim_uy[neighbor], prim_uz[neighbor]};
    float lambda = face_spectral_radius_f32(left_v, prim_rho[cell], prim_p[cell], right_v,
                                            prim_rho[neighbor], prim_p[neighbor], normal, gamma);
    float vol = volume > 1.0e-30f ? volume : 1.0e-30f;
    float coef = area * lambda / vol;
    float cur[5];
    float old[5];
    cons_vec_f32(cons_rho, cons_mx, cons_my, cons_mz, cons_e, neighbor, cur);
    cons_vec_f32(u0_rho, u0_mx, u0_my, u0_mz, u0_e, neighbor, old);
    for (int q = 0; q < 5; ++q) {
        source[q] -= coef * (cur[q] - old[q]);
    }
}

extern "C" __global__ void lusgs_sweep_unstructured_serial_f32(
    uint32_t num_cells, float omega, float gamma, float min_pressure, float inv_dt_phys,
    float backward_damping, const uint32_t *__restrict__ cell_offsets,
    const uint32_t *__restrict__ neighbors, const float *__restrict__ areas,
    const float *__restrict__ normals, const float *__restrict__ volumes,
    const float *__restrict__ sigma, const float *__restrict__ cell_dts,
    const float *__restrict__ res_rho, const float *__restrict__ res_mx,
    const float *__restrict__ res_my, const float *__restrict__ res_mz,
    const float *__restrict__ res_e, const float *__restrict__ u0_rho,
    const float *__restrict__ u0_mx, const float *__restrict__ u0_my,
    const float *__restrict__ u0_mz, const float *__restrict__ u0_e, float *__restrict__ cons_rho,
    float *__restrict__ cons_mx, float *__restrict__ cons_my, float *__restrict__ cons_mz,
    float *__restrict__ cons_e, float *__restrict__ prim_rho, float *__restrict__ prim_p,
    float *__restrict__ prim_ux, float *__restrict__ prim_uy, float *__restrict__ prim_uz) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (uint32_t cell = 0; cell < num_cells; ++cell) {
        float scale = implicit_scale_f32(cell_dts[cell], sigma[cell], omega, inv_dt_phys);
        float source[5] = {res_rho[cell], res_mx[cell], res_my[cell], res_mz[cell], res_e[cell]};
        for (uint32_t k = cell_offsets[cell]; k < cell_offsets[cell + 1]; ++k) {
            uint32_t nb = neighbors[k];
            if (nb >= cell) {
                continue;
            }
            const float *n = &normals[k * 3];
            add_coupling_delta_f32(source, cell, nb, areas[k], n, volumes[cell], gamma, cons_rho,
                                   cons_mx, cons_my, cons_mz, cons_e, u0_rho, u0_mx, u0_my, u0_mz,
                                   u0_e, prim_rho, prim_p, prim_ux, prim_uy, prim_uz);
        }
        float base[5];
        cons_vec_f32(cons_rho, cons_mx, cons_my, cons_mz, cons_e, cell, base);
        float effective =
            max_physical_increment_scale_f32(base, source, scale, gamma, min_pressure);
        if (effective > 0.0f) {
            float out[5];
            apply_increment_lane_f32(base, source, effective, out);
            write_cons_f32(cons_rho, cons_mx, cons_my, cons_mz, cons_e, cell, out);
        }
        fill_prim_cell_f32(cell, gamma, min_pressure, cons_rho, cons_mx, cons_my, cons_mz, cons_e,
                           prim_rho, prim_p, prim_ux, prim_uy, prim_uz);
    }

    for (int cell_i = (int)num_cells - 1; cell_i >= 0; --cell_i) {
        uint32_t cell = (uint32_t)cell_i;
        float scale = implicit_scale_f32(cell_dts[cell], sigma[cell], omega, inv_dt_phys);
        float source[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t k = cell_offsets[cell]; k < cell_offsets[cell + 1]; ++k) {
            uint32_t nb = neighbors[k];
            if (nb <= cell) {
                continue;
            }
            const float *n = &normals[k * 3];
            add_coupling_delta_f32(source, cell, nb, areas[k], n, volumes[cell], gamma, cons_rho,
                                   cons_mx, cons_my, cons_mz, cons_e, u0_rho, u0_mx, u0_my, u0_mz,
                                   u0_e, prim_rho, prim_p, prim_ux, prim_uy, prim_uz);
        }
        bool any = false;
        for (int q = 0; q < 5; ++q) {
            if (fabsf(source[q]) > 1.0e-30f) {
                any = true;
                break;
            }
        }
        if (!any) {
            continue;
        }
        for (int q = 0; q < 5; ++q) {
            source[q] *= backward_damping;
        }
        float base[5];
        cons_vec_f32(cons_rho, cons_mx, cons_my, cons_mz, cons_e, cell, base);
        float effective =
            max_physical_increment_scale_f32(base, source, scale, gamma, min_pressure);
        if (effective > 0.0f) {
            float out[5];
            apply_increment_lane_f32(base, source, effective, out);
            write_cons_f32(cons_rho, cons_mx, cons_my, cons_mz, cons_e, cell, out);
        }
        fill_prim_cell_f32(cell, gamma, min_pressure, cons_rho, cons_mx, cons_my, cons_mz, cons_e,
                           prim_rho, prim_p, prim_ux, prim_uy, prim_uz);
    }
}

// 并行检查守恒场是否全部满足正性；`any_bad[0]` 非 0 表示存在非法单元。
extern "C" __global__ void lusgs_any_nonphysical_conserved_f32(
    uint32_t num_cells, float gamma, float min_pressure, const float *__restrict__ cons_rho,
    const float *__restrict__ cons_mx, const float *__restrict__ cons_my,
    const float *__restrict__ cons_mz, const float *__restrict__ cons_e, int *__restrict__ any_bad) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) {
        return;
    }
    float lane[5] = {cons_rho[i], cons_mx[i], cons_my[i], cons_mz[i], cons_e[i]};
    if (!is_physical_conserved_f32(lane, gamma, min_pressure)) {
        atomicExch(any_bad, 1);
    }
}
