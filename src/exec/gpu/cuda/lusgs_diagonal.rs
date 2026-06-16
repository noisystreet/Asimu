//! LU-SGS 对角更新 CUDA launch。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use tracing::info_span;

use super::buffers::CudaFieldBuffers;
use crate::error::{AsimuError, Result};

const BLOCK_THREADS: u32 = 256;

pub fn launch_lusgs_diagonal_update(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    fields: &CudaFieldBuffers,
    sigma: &CudaSlice<f32>,
    cell_dts: &CudaSlice<f32>,
    omega: f32,
    inv_dt_phys: f32,
) -> Result<()> {
    let num_cells = fields.num_cells() as u32;
    let _span = info_span!(
        "cuda_lusgs_diagonal_update",
        cells = num_cells,
        inv_dt_phys = inv_dt_phys,
    )
    .entered();
    let num_blocks = num_cells.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(&omega);
    builder.arg(&inv_dt_phys);
    builder.arg(&fields.cons_rho);
    builder.arg(&fields.cons_mx);
    builder.arg(&fields.cons_my);
    builder.arg(&fields.cons_mz);
    builder.arg(&fields.cons_e);
    builder.arg(&fields.res_rho);
    builder.arg(&fields.res_mx);
    builder.arg(&fields.res_my);
    builder.arg(&fields.res_mz);
    builder.arg(&fields.res_e);
    builder.arg(sigma);
    builder.arg(cell_dts);
    builder.arg(&fields.cons_rho);
    builder.arg(&fields.cons_mx);
    builder.arg(&fields.cons_my);
    builder.arg(&fields.cons_mz);
    builder.arg(&fields.cons_e);
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA LU-SGS 对角 kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}

pub fn launch_residual_density_sum_sq(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    res_rho: &CudaSlice<f32>,
    num_cells: u32,
    sum_sq_out: &mut CudaSlice<f32>,
) -> Result<()> {
    let _span = info_span!("cuda_residual_density_sum_sq", cells = num_cells).entered();
    stream
        .memset_zeros(sum_sq_out)
        .map_err(|e| AsimuError::Exec(format!("CUDA memset sum_sq 失败: {e:?}")))?;
    let num_blocks = num_cells.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(res_rho);
    builder.arg(&num_cells);
    builder.arg(sum_sq_out);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!("CUDA 密度残差 RMS kernel launch 失败: {e:?}"))
        })?;
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod gpu_tests {
    use std::sync::Arc;

    use cudarc::driver::CudaContext;

    use super::*;
    use crate::core::ComputeFloat;
    use crate::exec::gpu::cuda::buffers::CudaFieldBuffers;
    use crate::exec::gpu::cuda::module::CudaLusgsModule;
    use crate::exec::gpu::cuda::transfer::memcpy_htod;
    use crate::field::{
        ConservedFieldsT, ConservedResidualT, LusgsDiagonalCoeffsF32,
        assign_lusgs_diagonal_update_f32,
    };
    use crate::physics::ConservedState;

    fn assert_conserved_close(gpu: &ConservedFieldsT<f32>, cpu: &ConservedFieldsT<f32>, tol: f32) {
        let n = gpu.num_cells();
        assert_eq!(n, cpu.num_cells());
        for i in 0..n {
            assert!(
                (gpu.density.values()[i] - cpu.density.values()[i]).abs() < tol,
                "density[{i}]"
            );
            assert!(
                (gpu.momentum_x.values()[i] - cpu.momentum_x.values()[i]).abs() < tol,
                "mx[{i}]"
            );
            assert!(
                (gpu.momentum_y.values()[i] - cpu.momentum_y.values()[i]).abs() < tol,
                "my[{i}]"
            );
            assert!(
                (gpu.momentum_z.values()[i] - cpu.momentum_z.values()[i]).abs() < tol,
                "mz[{i}]"
            );
            assert!(
                (gpu.total_energy.values()[i] - cpu.total_energy.values()[i]).abs() < tol,
                "e[{i}]"
            );
        }
    }

    #[test]
    #[ignore = "gpu"]
    fn cuda_lusgs_diagonal_matches_cpu_with_inv_dt_phys() {
        let n = 3;
        let ctx = Arc::new(CudaContext::new(0).expect("CUDA 设备"));
        let stream = ctx.default_stream();
        let module = CudaLusgsModule::try_load(&ctx).expect("LU-SGS 模块");
        let mut fields = CudaFieldBuffers::try_new(&stream, n).expect("field 缓冲");

        let state = ConservedState {
            density: 1.2,
            momentum: [0.3, -0.1, 0.05],
            total_energy: 2.8,
        };
        let base = ConservedFieldsT::<f32>::uniform(n, state).expect("base");
        let mut residual = ConservedResidualT::<f32>::zeros(n).expect("residual");
        residual.density.values_mut()[0] = f32::from_real(0.5);
        residual.density.values_mut()[1] = f32::from_real(-0.2);
        residual.momentum_x.values_mut()[2] = f32::from_real(0.15);
        residual.total_energy.values_mut()[1] = f32::from_real(0.08);

        let sigma = vec![8.0_f32, 12.0, 6.0];
        let dt = vec![0.02_f32, 0.015, 0.025];
        let inv_dt_phys = 10.0_f32;
        let omega = 1.0_f32;

        fields.upload_conserved(&stream, &base).expect("H2D base");
        fields
            .upload_full_residual(&stream, &residual)
            .expect("H2D residual");

        let mut sigma_dev = stream.alloc_zeros::<f32>(n).expect("sigma device");
        memcpy_htod(&stream, "test_sigma", &sigma, &mut sigma_dev).expect("sigma H2D");
        let mut cell_dts_dev = stream.alloc_zeros::<f32>(n).expect("cell_dts device");
        memcpy_htod(&stream, "test_cell_dts", &dt, &mut cell_dts_dev).expect("cell_dts H2D");

        launch_lusgs_diagonal_update(
            &stream,
            &module.diagonal_update,
            &fields,
            &sigma_dev,
            &cell_dts_dev,
            omega,
            inv_dt_phys,
        )
        .expect("kernel");

        stream.synchronize().expect("sync");

        let mut gpu_out = base.clone();
        fields
            .download_conserved(&stream, &mut gpu_out)
            .expect("D2H out");

        let mut cpu_out = base.clone();
        assign_lusgs_diagonal_update_f32(
            &mut cpu_out,
            &base,
            &residual,
            &sigma,
            &dt,
            LusgsDiagonalCoeffsF32 { omega, inv_dt_phys },
        )
        .expect("cpu");

        assert_conserved_close(&gpu_out, &cpu_out, 1.0e-5);
    }

    #[test]
    #[ignore = "gpu"]
    fn cuda_lusgs_diagonal_inv_dt_phys_reduces_update() {
        let n = 1;
        let ctx = Arc::new(CudaContext::new(0).expect("CUDA 设备"));
        let stream = ctx.default_stream();
        let module = CudaLusgsModule::try_load(&ctx).expect("LU-SGS 模块");

        let state = ConservedState {
            density: 1.0,
            momentum: [0.0; 3],
            total_energy: 2.5,
        };
        let base = ConservedFieldsT::<f32>::uniform(n, state).expect("base");
        let mut residual = ConservedResidualT::<f32>::zeros(n).expect("residual");
        residual.density.values_mut()[0] = f32::from_real(1.0);

        let sigma = vec![2.0_f32];
        let dt = vec![0.1_f32];
        let omega = 1.0_f32;

        let run = |inv_dt_phys: f32| -> f32 {
            let mut fields = CudaFieldBuffers::try_new(&stream, n).expect("buffers");
            fields.upload_conserved(&stream, &base).expect("base");
            fields
                .upload_full_residual(&stream, &residual)
                .expect("res");
            let mut sigma_dev = stream.alloc_zeros::<f32>(n).expect("sigma");
            memcpy_htod(&stream, "sigma", &sigma, &mut sigma_dev).expect("sigma");
            let mut cell_dts_dev = stream.alloc_zeros::<f32>(n).expect("dts");
            memcpy_htod(&stream, "dts", &dt, &mut cell_dts_dev).expect("dts");
            launch_lusgs_diagonal_update(
                &stream,
                &module.diagonal_update,
                &fields,
                &sigma_dev,
                &cell_dts_dev,
                omega,
                inv_dt_phys,
            )
            .expect("kernel");
            stream.synchronize().expect("sync");
            let mut out = base.clone();
            fields.download_conserved(&stream, &mut out).expect("d2h");
            out.density.values()[0]
        };

        let steady = run(0.0);
        let dual = run(10.0);
        assert!(dual > base.density.values()[0]);
        assert!(dual < steady);
    }
}
