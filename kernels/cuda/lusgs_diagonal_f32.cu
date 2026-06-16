// LU-SGS 对角更新（f32）：对齐 field/lusgs_diagonal.rs assign_lusgs_diagonal_update_f32。
// out[i] = base[i] + residual[i] * (omega * dt_i / (1 + dt_i * sigma[i] + dt_i * inv_dt_phys))

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void lusgs_diagonal_update_f32(uint32_t num_cells, float omega,
                                                     float inv_dt_phys,
                                                     const float *__restrict__ base_rho,
                                                     const float *__restrict__ base_mx,
                                                     const float *__restrict__ base_my,
                                                     const float *__restrict__ base_mz,
                                                     const float *__restrict__ base_e,
                                                     const float *__restrict__ res_rho,
                                                     const float *__restrict__ res_mx,
                                                     const float *__restrict__ res_my,
                                                     const float *__restrict__ res_mz,
                                                     const float *__restrict__ res_e,
                                                     const float *__restrict__ sigma,
                                                     const float *__restrict__ cell_dts,
                                                     float *__restrict__ out_rho,
                                                     float *__restrict__ out_mx,
                                                     float *__restrict__ out_my,
                                                     float *__restrict__ out_mz,
                                                     float *__restrict__ out_e) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) {
        return;
    }
    float dt_i = cell_dts[i];
    float scale = omega * dt_i / (1.0f + dt_i * sigma[i] + dt_i * inv_dt_phys);
    out_rho[i] = base_rho[i] + res_rho[i] * scale;
    out_mx[i] = base_mx[i] + res_mx[i] * scale;
    out_my[i] = base_my[i] + res_my[i] * scale;
    out_mz[i] = base_mz[i] + res_mz[i] * scale;
    out_e[i] = base_e[i] + res_e[i] * scale;
}

// 密度残差 RMS：sum_sq = Σ ρ̇²，host 侧 sqrt(sum_sq / n)。
extern "C" __global__ void residual_density_sum_sq_f32(const float *__restrict__ res_rho,
                                                       uint32_t num_cells,
                                                       float *__restrict__ sum_sq_out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) {
        return;
    }
    float r = res_rho[i];
    atomicAdd(sum_sq_out, r * r);
}
