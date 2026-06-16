//! 双时间步物理存储项 CUDA launch。

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tracing::info_span;

use super::buffers::CudaFieldBuffers;
use crate::error::{AsimuError, Result};

const BLOCK_THREADS: u32 = 256;

pub fn launch_dual_time_storage(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    fields: &CudaFieldBuffers,
    inv_dt_phys: f32,
) -> Result<()> {
    let num_cells = fields.num_cells() as u32;
    let _span = info_span!(
        "cuda_dual_time_storage",
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
    builder.arg(&inv_dt_phys);
    builder.arg(&fields.cons_rho);
    builder.arg(&fields.cons_mx);
    builder.arg(&fields.cons_my);
    builder.arg(&fields.cons_mz);
    builder.arg(&fields.cons_e);
    builder.arg(&fields.cons_u_n_rho);
    builder.arg(&fields.cons_u_n_mx);
    builder.arg(&fields.cons_u_n_my);
    builder.arg(&fields.cons_u_n_mz);
    builder.arg(&fields.cons_u_n_e);
    builder.arg(&fields.res_rho);
    builder.arg(&fields.res_mx);
    builder.arg(&fields.res_my);
    builder.arg(&fields.res_mz);
    builder.arg(&fields.res_e);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!("CUDA 双时间存储项 kernel launch 失败: {e:?}"))
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
    use crate::exec::gpu::cuda::module::CudaDualTimeModule;
    use crate::field::{ConservedFieldsT, ConservedResidualT};
    use crate::physics::ConservedState;
    use crate::solver::time::add_physical_storage_residual;

    #[test]
    #[ignore = "gpu"]
    fn cuda_dual_time_storage_matches_cpu() {
        let n = 2;
        let ctx = Arc::new(CudaContext::new(0).expect("CUDA 设备"));
        let stream = ctx.default_stream();
        let module = CudaDualTimeModule::try_load(&ctx).expect("dual_time 模块");
        let mut fields = CudaFieldBuffers::try_new(&stream, n).expect("buffers");

        let state = ConservedState {
            density: 1.0,
            momentum: [0.1, 0.0, 0.0],
            total_energy: 2.5,
        };
        let u = ConservedFieldsT::<f32>::uniform(n, state).expect("u");
        let mut u_n = u.clone();
        u_n.density.values_mut()[0] = f32::from_real(0.8);
        u_n.density.values_mut()[1] = f32::from_real(0.9);

        let mut u_curr = u.clone();
        u_curr.density.values_mut()[0] = f32::from_real(1.2);
        u_curr.density.values_mut()[1] = f32::from_real(1.1);

        let mut residual_host = ConservedResidualT::<f32>::zeros(n).expect("res");
        residual_host.density.values_mut()[0] = f32::from_real(-0.5);
        residual_host.density.values_mut()[1] = f32::from_real(0.3);

        let dt_phys = 0.1_f32;

        fields.upload_conserved(&stream, &u_n).expect("u_n to cons");
        fields.snapshot_u_n_on_device(&stream).expect("snapshot");
        fields.upload_conserved(&stream, &u_curr).expect("u_curr");
        fields
            .upload_full_residual(&stream, &residual_host)
            .expect("res");

        launch_dual_time_storage(&stream, &module.storage, &fields, 1.0 / dt_phys).expect("kernel");
        stream.synchronize().expect("sync");
        let mut residual_gpu = residual_host.clone();
        fields
            .download_residual(&stream, &mut residual_gpu)
            .expect("d2h res");

        let mut residual_cpu = residual_host;
        add_physical_storage_residual(&mut residual_cpu, &u_curr, &u_n, f64::from(dt_phys))
            .expect("cpu");

        for i in 0..n {
            assert!(
                (residual_gpu.density.values()[i] - residual_cpu.density.values()[i]).abs()
                    < 1.0e-5,
                "cell {i}"
            );
        }
    }
}
