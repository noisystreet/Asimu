//! LU-SGS 扫掠 CUDA 参数与 launch。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use tracing::info_span;

use super::buffers::CudaFieldBuffers;
use super::lusgs_sweep_mesh_cache::CudaLusgsSweepMeshDeviceCache;
use crate::discretization::unstructured_lusgs_sweep_exec_topo::LuSgsSweepHostTopology;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};

/// device 扫掠 launch 标量参数。
pub struct LusgsSweepCudaScalars {
    pub omega: f32,
    pub gamma: f32,
    pub min_pressure: f32,
    pub inv_dt_phys: f32,
    pub backward_damping: f32,
}

/// device 扫掠 launch 缓冲引用。
pub struct LusgsSweepCudaLaunchBuffers<'a> {
    pub fields: &'a CudaFieldBuffers,
    pub sweep_mesh: &'a CudaLusgsSweepMeshDeviceCache,
    pub sigma: &'a CudaSlice<f32>,
    pub cell_dts: &'a CudaSlice<f32>,
    pub u0_rho: &'a CudaSlice<f32>,
    pub u0_mx: &'a CudaSlice<f32>,
    pub u0_my: &'a CudaSlice<f32>,
    pub u0_mz: &'a CudaSlice<f32>,
    pub u0_e: &'a CudaSlice<f32>,
}

/// host 侧扫掠 + stabilize 输入。
pub struct LusgsSweepCudaHostInput<'a> {
    pub fields: &'a mut ConservedFieldsT<f32>,
    pub u0: &'a ConservedFieldsT<f32>,
    pub residual: &'a mut ConservedResidualT<f32>,
    pub sweep_topo: &'a LuSgsSweepHostTopology,
    pub topo_key: usize,
    pub primitives: &'a PrimitiveFieldsT<f32>,
    pub host_sigma: &'a [f32],
    pub host_cell_dts: &'a [f32],
    pub host_volumes: &'a [f32],
    pub local_time_step: bool,
    pub scalars: LusgsSweepCudaScalars,
}

/// 图着色 wavefront：按颜色批并行前/后扫（生产路径）。
pub fn launch_lusgs_sweep_wavefront(
    stream: &Arc<CudaStream>,
    forward_fn: &CudaFunction,
    backward_fn: &CudaFunction,
    buffers: &LusgsSweepCudaLaunchBuffers<'_>,
    scalars: &LusgsSweepCudaScalars,
) -> Result<()> {
    let num_cells = buffers.fields.num_cells() as u32;
    let num_colors = buffers.sweep_mesh.num_colors();
    let _span = info_span!(
        "cuda_lusgs_sweep_wavefront",
        cells = num_cells,
        colors = num_colors,
        inv_dt_phys = scalars.inv_dt_phys,
    )
    .entered();
    let offsets = buffers.sweep_mesh.host_color_offsets();
    for color in 0..num_colors as usize {
        let begin = offsets[color];
        let end = offsets[color + 1];
        let count = end - begin;
        if count == 0 {
            continue;
        }
        launch_sweep_color_kernel(stream, forward_fn, begin, count, buffers, scalars, true)?;
    }
    for color in (0..num_colors as usize).rev() {
        let begin = offsets[color];
        let end = offsets[color + 1];
        let count = end - begin;
        if count == 0 {
            continue;
        }
        launch_sweep_color_kernel(stream, backward_fn, begin, count, buffers, scalars, false)?;
    }
    Ok(())
}

fn launch_sweep_color_kernel(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    color_begin: u32,
    num_color_cells: u32,
    buffers: &LusgsSweepCudaLaunchBuffers<'_>,
    scalars: &LusgsSweepCudaScalars,
    forward: bool,
) -> Result<()> {
    let num_blocks = num_color_cells.div_ceil(SWEEP_BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (SWEEP_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&color_begin);
    builder.arg(&num_color_cells);
    builder.arg(buffers.sweep_mesh.color_cells());
    builder.arg(&scalars.omega);
    builder.arg(&scalars.gamma);
    builder.arg(&scalars.min_pressure);
    builder.arg(&scalars.inv_dt_phys);
    if !forward {
        builder.arg(&scalars.backward_damping);
    }
    builder.arg(buffers.sweep_mesh.cell_offsets());
    builder.arg(buffers.sweep_mesh.neighbors());
    builder.arg(buffers.sweep_mesh.areas());
    builder.arg(buffers.sweep_mesh.normals());
    builder.arg(buffers.sweep_mesh.volumes());
    builder.arg(buffers.sigma);
    builder.arg(buffers.cell_dts);
    if forward {
        builder.arg(&buffers.fields.res_rho);
        builder.arg(&buffers.fields.res_mx);
        builder.arg(&buffers.fields.res_my);
        builder.arg(&buffers.fields.res_mz);
        builder.arg(&buffers.fields.res_e);
    }
    builder.arg(buffers.u0_rho);
    builder.arg(buffers.u0_mx);
    builder.arg(buffers.u0_my);
    builder.arg(buffers.u0_mz);
    builder.arg(buffers.u0_e);
    builder.arg(&buffers.fields.cons_rho);
    builder.arg(&buffers.fields.cons_mx);
    builder.arg(&buffers.fields.cons_my);
    builder.arg(&buffers.fields.cons_mz);
    builder.arg(&buffers.fields.cons_e);
    builder.arg(&buffers.fields.prim_rho);
    builder.arg(&buffers.fields.prim_p);
    builder.arg(&buffers.fields.prim_ux);
    builder.arg(&buffers.fields.prim_uy);
    builder.arg(&buffers.fields.prim_uz);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!(
                "CUDA LU-SGS {}扫 kernel launch 失败: {e:?}",
                if forward { "前" } else { "后" }
            ))
        })?;
    }
    Ok(())
}

pub fn launch_lusgs_sweep_unstructured_serial(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    buffers: &LusgsSweepCudaLaunchBuffers<'_>,
    scalars: &LusgsSweepCudaScalars,
) -> Result<()> {
    let num_cells = buffers.fields.num_cells() as u32;
    let _span = info_span!(
        "cuda_lusgs_sweep_unstructured",
        cells = num_cells,
        inv_dt_phys = scalars.inv_dt_phys,
    )
    .entered();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(&scalars.omega);
    builder.arg(&scalars.gamma);
    builder.arg(&scalars.min_pressure);
    builder.arg(&scalars.inv_dt_phys);
    builder.arg(&scalars.backward_damping);
    builder.arg(buffers.sweep_mesh.cell_offsets());
    builder.arg(buffers.sweep_mesh.neighbors());
    builder.arg(buffers.sweep_mesh.areas());
    builder.arg(buffers.sweep_mesh.normals());
    builder.arg(buffers.sweep_mesh.volumes());
    builder.arg(buffers.sigma);
    builder.arg(buffers.cell_dts);
    builder.arg(&buffers.fields.res_rho);
    builder.arg(&buffers.fields.res_mx);
    builder.arg(&buffers.fields.res_my);
    builder.arg(&buffers.fields.res_mz);
    builder.arg(&buffers.fields.res_e);
    builder.arg(buffers.u0_rho);
    builder.arg(buffers.u0_mx);
    builder.arg(buffers.u0_my);
    builder.arg(buffers.u0_mz);
    builder.arg(buffers.u0_e);
    builder.arg(&buffers.fields.cons_rho);
    builder.arg(&buffers.fields.cons_mx);
    builder.arg(&buffers.fields.cons_my);
    builder.arg(&buffers.fields.cons_mz);
    builder.arg(&buffers.fields.cons_e);
    builder.arg(&buffers.fields.prim_rho);
    builder.arg(&buffers.fields.prim_p);
    builder.arg(&buffers.fields.prim_ux);
    builder.arg(&buffers.fields.prim_uy);
    builder.arg(&buffers.fields.prim_uz);
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA LU-SGS 扫掠 kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}

const SWEEP_BLOCK_THREADS: u32 = 256;

/// 并行检查 device 守恒场是否全部正性；`any_bad[0]==0` 表示可跳过 host stabilize。
pub fn launch_lusgs_any_nonphysical_conserved(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    fields: &CudaFieldBuffers,
    gamma: f32,
    min_pressure: f32,
    any_bad: &mut CudaSlice<i32>,
) -> Result<()> {
    let num_cells = fields.num_cells() as u32;
    let _span = info_span!("cuda_lusgs_any_nonphysical", cells = num_cells,).entered();
    stream
        .memset_zeros(any_bad)
        .map_err(|e| AsimuError::Exec(format!("CUDA any_nonphysical memset 失败: {e:?}")))?;
    let num_blocks = num_cells.div_ceil(SWEEP_BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (SWEEP_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(&gamma);
    builder.arg(&min_pressure);
    builder.arg(&fields.cons_rho);
    builder.arg(&fields.cons_mx);
    builder.arg(&fields.cons_my);
    builder.arg(&fields.cons_mz);
    builder.arg(&fields.cons_e);
    builder.arg(any_bad);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!("CUDA LU-SGS 正性检查 kernel launch 失败: {e:?}"))
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::CudaContext;

    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::FaceId;
    use crate::discretization::unstructured_lusgs_sweep_exec_topo::LuSgsSweepHostTopology;
    use crate::exec::gpu::cuda::buffers::CudaFieldBuffers;
    use crate::exec::gpu::cuda::lusgs_sweep_mesh_cache::{
        CudaLusgsSweepMeshDeviceCache, upload_u0_snapshot,
    };
    use crate::exec::gpu::cuda::module::CudaLusgsModule;
    use crate::exec::gpu::cuda::transfer::memcpy_htod;
    use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
    use crate::physics::{ConservedState, IdealGasEoS};

    fn closed_tet_mesh() -> UnstructuredMesh3d {
        UnstructuredMesh3d::new(
            "tet",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell")],
        )
        .expect("mesh")
    }

    fn tet_sweep_topo(mesh: &UnstructuredMesh3d) -> LuSgsSweepHostTopology {
        let faces = (0..mesh.num_faces())
            .map(|f| FaceId(f as u32))
            .collect::<Vec<_>>();
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "far",
            faces,
            BoundaryKind::Farfield {
                mach: 0.1,
                pressure: 1.0,
                temperature: 1.0,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        let cache = crate::discretization::UnstructuredSolverMeshCache::from_mesh(mesh, &boundary)
            .expect("cache");
        LuSgsSweepHostTopology::from_mesh_and_couplings(mesh, &cache.lusgs_couplings_f32)
    }

    fn assert_conserved_close(a: &ConservedFieldsT<f32>, b: &ConservedFieldsT<f32>, tol: f32) {
        let n = a.num_cells();
        assert_eq!(n, b.num_cells());
        for i in 0..n {
            assert!(
                (a.density.values()[i] - b.density.values()[i]).abs() < tol,
                "rho[{i}]"
            );
            assert!(
                (a.momentum_x.values()[i] - b.momentum_x.values()[i]).abs() < tol,
                "mx[{i}]"
            );
            assert!(
                (a.momentum_y.values()[i] - b.momentum_y.values()[i]).abs() < tol,
                "my[{i}]"
            );
            assert!(
                (a.momentum_z.values()[i] - b.momentum_z.values()[i]).abs() < tol,
                "mz[{i}]"
            );
            assert!(
                (a.total_energy.values()[i] - b.total_energy.values()[i]).abs() < tol,
                "e[{i}]"
            );
        }
    }

    #[test]
    #[ignore = "gpu"]
    fn cuda_lusgs_sweep_wavefront_matches_serial_on_tet() {
        let mesh = closed_tet_mesh();
        let topo = tet_sweep_topo(&mesh);
        let n = mesh.num_cells();
        let ctx = Arc::new(CudaContext::new(0).expect("CUDA 设备"));
        let stream = ctx.default_stream();
        let module = CudaLusgsModule::try_load(&ctx).expect("LU-SGS 模块");
        let sweep_mesh = CudaLusgsSweepMeshDeviceCache::try_upload(&stream, &topo).expect("mesh");

        let state = ConservedState {
            density: 1.2,
            momentum: [0.3, -0.1, 0.05],
            total_energy: 2.8,
        };
        let base = ConservedFieldsT::<f32>::uniform(n, state).expect("base");
        let u0 = base.clone();
        let mut residual = ConservedResidualT::<f32>::zeros(n).expect("residual");
        residual.density.values_mut()[0] = 0.05_f32;

        let eos = IdealGasEoS {
            gamma: 1.4,
            gas_constant: 287.0,
        };
        let mut primitives = PrimitiveFieldsT::<f32>::zeros(n).expect("prim");
        primitives
            .fill_from_conserved(&base, &eos, 1.0e-6)
            .expect("prim from cons");

        let sigma = vec![8.0_f32; n];
        let cell_dts = vec![0.02_f32; n];
        let scalars = LusgsSweepCudaScalars {
            omega: 1.0,
            gamma: 1.4,
            min_pressure: 1.0e-6,
            inv_dt_phys: 0.0,
            backward_damping: 0.5,
        };

        let mut sigma_dev = stream.alloc_zeros::<f32>(n).expect("sigma");
        memcpy_htod(&stream, "sigma", &sigma, &mut sigma_dev).expect("sigma H2D");
        let mut cell_dts_dev = stream.alloc_zeros::<f32>(n).expect("dts");
        memcpy_htod(&stream, "dts", &cell_dts, &mut cell_dts_dev).expect("dts H2D");

        let mut u0_rho = stream.alloc_zeros::<f32>(n).expect("u0_rho");
        let mut u0_mx = stream.alloc_zeros::<f32>(n).expect("u0_mx");
        let mut u0_my = stream.alloc_zeros::<f32>(n).expect("u0_my");
        let mut u0_mz = stream.alloc_zeros::<f32>(n).expect("u0_mz");
        let mut u0_e = stream.alloc_zeros::<f32>(n).expect("u0_e");
        upload_u0_snapshot(
            &stream,
            &u0,
            &mut u0_rho,
            &mut u0_mx,
            &mut u0_my,
            &mut u0_mz,
            &mut u0_e,
        )
        .expect("u0");

        let run_sweep = |serial: bool| -> ConservedFieldsT<f32> {
            let mut fields = CudaFieldBuffers::try_new(&stream, n).expect("buffers");
            fields.upload_conserved(&stream, &base).expect("cons");
            fields
                .upload_full_residual(&stream, &residual)
                .expect("res");
            fields
                .upload_primitives(&stream, &primitives)
                .expect("prim");
            let launch_bufs = LusgsSweepCudaLaunchBuffers {
                fields: &fields,
                sweep_mesh: &sweep_mesh,
                sigma: &sigma_dev,
                cell_dts: &cell_dts_dev,
                u0_rho: &u0_rho,
                u0_mx: &u0_mx,
                u0_my: &u0_my,
                u0_mz: &u0_mz,
                u0_e: &u0_e,
            };
            if serial {
                launch_lusgs_sweep_unstructured_serial(
                    &stream,
                    &module.sweep_unstructured_serial,
                    &launch_bufs,
                    &scalars,
                )
                .expect("serial");
            } else {
                launch_lusgs_sweep_wavefront(
                    &stream,
                    &module.sweep_forward_color,
                    &module.sweep_backward_color,
                    &launch_bufs,
                    &scalars,
                )
                .expect("wavefront");
            }
            stream.synchronize().expect("sync");
            let mut out = base.clone();
            fields.download_conserved(&stream, &mut out).expect("d2h");
            out
        };

        let serial_out = run_sweep(true);
        let wave_out = run_sweep(false);
        assert_conserved_close(&serial_out, &wave_out, 1.0e-5);
        assert!(topo.cell_coloring.num_colors >= 1);
    }
}
