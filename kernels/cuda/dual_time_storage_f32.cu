// 双时间步 BDF1 物理存储项（f32）：对齐 solver/time/dual_time.rs add_physical_storage_residual。
// R_eff,i += (U_i - U^n_i) / dt_phys（单元平均守恒量，不再除 V_i）

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
                                                 float *__restrict__ res_e) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) {
        return;
    }
    res_rho[i] += (cons_rho[i] - u_n_rho[i]) * inv_dt_phys;
    res_mx[i] += (cons_mx[i] - u_n_mx[i]) * inv_dt_phys;
    res_my[i] += (cons_my[i] - u_n_my[i]) * inv_dt_phys;
    res_mz[i] += (cons_mz[i] - u_n_mz[i]) * inv_dt_phys;
    res_e[i] += (cons_e[i] - u_n_e[i]) * inv_dt_phys;
}
