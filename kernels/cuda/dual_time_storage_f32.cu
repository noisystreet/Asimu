// 双时间步 BDF1 物理存储项（f32）：对齐 solver/time/dual_time.rs add_physical_storage_residual。
// R_eff,i -= (U_i - U^n_i) / (V_i * dt_phys)

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void dual_time_storage_f32(uint32_t num_cells, float inv_dt_phys,
                                                 const float *__restrict__ cons_rho,
                                                 const float *__restrict__ cons_mx,
                                                 const float *__restrict__ cons_my,
                                                 const float *__restrict__ cons_mz,
                                                 const float *__restrict__ cons_e,
                                                 const float *__restrict__ u_n_rho,
                                                 const float *__restrict__ u_n_mx,
                                                 const float *__restrict__ u_n_my,
                                                 const float *__restrict__ u_n_mz,
                                                 const float *__restrict__ u_n_e,
                                                 float *__restrict__ res_rho,
                                                 float *__restrict__ res_mx,
                                                 float *__restrict__ res_my,
                                                 float *__restrict__ res_mz,
                                                 float *__restrict__ res_e,
                                                 const float *__restrict__ volumes) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) {
        return;
    }
    float inv_vol_dt = inv_dt_phys / volumes[i];
    res_rho[i] -= (cons_rho[i] - u_n_rho[i]) * inv_vol_dt;
    res_mx[i] -= (cons_mx[i] - u_n_mx[i]) * inv_vol_dt;
    res_my[i] -= (cons_my[i] - u_n_my[i]) * inv_vol_dt;
    res_mz[i] -= (cons_mz[i] - u_n_mz[i]) * inv_vol_dt;
    res_e[i] -= (cons_e[i] - u_n_e[i]) * inv_vol_dt;
}
