# 变更日志

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- **ADR 0017 CUDA + LU-SGS**：`case/validate` 允许 `backend=cuda` + `time.scheme=lu_sgs`（须 `local_time_step=true`）；LU-SGS 步末 `mark_cuda_primitives_stale`；benchmark `dual_ellipsoid/case_cuda_lusgs_f32.toml`；单 tet validate 与 GPU smoke（`#[ignore=gpu]`）。
- **ADR 0017 G3 完成**：cuSPARSE CSR SpMV 经 `ExecutionContext::csr_spmv` 分发（f64）；CUDA 侧 CSR 结构缓存与 workspace；`cpu_csr_spmv_matches_cuda_csr_spmv`（`#[ignore=gpu]`）。
- **ADR 0017 G2 完成**：非结构 f32 CUDA 粘性内面着色桶 kernel（`viscous_interior_f32.cu`）；梯度 H2D + 动量/能量残差累加 scatter；`case/validate` 允许 `backend=cuda` + 粘性/Navier-Stokes（`rk4`/`euler`）；benchmark `dual_ellipsoid/case_cuda_f32.toml`；单 tet CPU≈CUDA 粘性单测与 case GPU smoke（`#[ignore=gpu]`）。
- **ADR 0017 G1 完成**：非结构 f32 CUDA 一阶无粘 Roe/HVL 着色桶 kernel；`sync_to_host` / `sync_to_device` 骨架与 `primitives_dirty` 步间缓冲；驱动层 `mark_cuda_primitives_stale`；`make check-cuda` / `make test-cuda`；benchmark `unstructured_cuda_freestream`；case 层 CUDA validate / GPU smoke 测试。
- 非结构 f32 热路径原生原变量恢复：`primitive_from_conserved_relaxed_f32` / `PrimitiveFillFromConserved`；ghost 边界面单次 `primitive_from_conserved_relaxed_f32_from_state`。
- f32 Riemann 求解器法向 API：`FaceNormalF32`（`[f32; 3]`）；一阶/MUSCL 装配直接传 `face_topology_f32.normal`，消除 `vec3_from_f32`→`Vector3` 往返。
- f32 Sutherland 输运系数：`ViscousPhysicsConfig::face_transport_coefficients_f32`、`static_temperature_f32`；粘性装配/谱半径/边界面通量全 f32 温度链。
- 非结构 f32 谱半径：面循环 f32 原变量与法向；单元 \(\sigma_i\) 以 f64 累加后输出 `Vec<f32>`（保留粘性抛物项）；`cell_viscous_diffusivity_max_f32` 返回 `Vec<f32>`。
- 非结构 f32 CFL / 当地时间步：`UnstructuredTimestepBuffers`（`sigma_f32` / `cell_dts_f32` / `volumes_f32`）；`finalize_cell_dts_from_sigma_f32`、`min_positive_dt_f32`；显式推进 `euler_step_local_f32` / `rk4_step_local_f32`；LU-SGS 对角更新 `assign_lusgs_diagonal_update_f32` 与 `UnstructuredLusgsDiagonalUpdate` 精度分发。
- 非结构 f32 LU-SGS 正性限制：`field::positivity` 新增 `state_after_increment_f32`、`is_physical_conserved_f32`、`max_physical_increment_scale_f32`；扫掠/稳定化全 f32 增量（`apply_limited_cell_increment_f32`、`stabilize_sweep_update_f32`），消除 `increment_real_from_f32` 往返。
- 非结构 f32 粘性边界面法向：`viscous_boundary_f32` / 装配层使用 `FaceNormalF32`（`[f32; 3]`），直接传 `face_topology_f32.normal`。
- 非结构 f32 LU-SGS 扫掠 `source` / 耦合差分全 f32（`residual_cell_vector_f32`、`conserved_vector_f32`）；`LuSgsSweepUnstructuredF32Input` 预打包 \(\sigma,\Delta t_i,\omega,\gamma\)。
- 非结构 f32 MUSCL 内面在 `parallel-fvm` 下走着色桶 `scatter_inviscid_pairs_f32`；粘性 f32 内面 scatter 改为 `InteriorViscousScatterGeomF32` / `ViscousScatterOpF32` 原生热路径。
- 非结构 f32 MUSCL 限制器样本（`cell_gradient_samples_f32`）、LU-SGS 预打包耦合（`lusgs_couplings_f32`）与无粘 scatter 原生热路径（`InviscidFluxF32` / `InviscidScatterOpF32`）；新增 `lu_sgs_sweep_unstructured_f32` 与驱动层 `compressible_unstructured_lusgs_typed` 精度分发。
- **非结构 f32 一阶无粘 SIMD**（ADR 0016 P5，`simd-fvm`）：`exec::cpu` 新增 Roe/HVL `batch4` f32 内核（HVL 亚音速分裂 `f32x4`）；`assembly_unstructured_inviscid_simd_f32` 挂接 typed 一阶内面，与 f64 `simd_batch4` 对称。
- 非结构 f32 面几何预打包缓存：`UnstructuredSolverMeshCache::face_topology_f32`（法向、面积、体积、rhs_scale、`dr_*`、`lsq_dr`/`lsq_w` 等）+ `lsq_geometry_f32`；f32 无粘/粘性/谱半径/IDWLS 梯度热路径不再逐面读取 f64 `Vector3`。
- ADR 0018：非结构可压缩 f64/f32 统一 typed 驱动、`UnstructuredComputeBackend` 聚合 trait；f64 一阶内面复用 `simd-fvm` batch4 路径。`exec::scatter` 新增 `f32` 原子累加（`AtomicU32` CAS）、`scatter_inviscid_pairs_f32` / `scatter_viscous_valid_slots_f32`；非结构 typed 无粘内面在 `parallel-fvm` 下走着色桶 exec scatter；`unstructured_freestream` benchmark 文档补充 `f32` vs `f64` 对比说明。
- Restart I/O：单/多块 TOML 支持可选 `compute_precision = "f32"`；`load_*_checked` 与 case `[numerics]` 校验一致，跨精度 restart 报错；新增 `write_conserved_fields_typed` / `load_conserved_fields_typed`。结构化 3D `compute_incompressible_divergence_3d`、`compute_incompressible_velocity_laplacian_3d`、`compute_incompressible_rhie_chow_divergence_3d`、含一阶迎风对流、动量边界面贡献与 `velocity_under_relaxation` 的伪瞬态动量预测 CSR、不可压缩 cell-centered 边界应用、使用面插值 \(d_P\) 与压力出口 \(p'=0\) 的压力校正 CSR、显式 `[incompressible.reference]` 无量纲化；不可压缩 runner 通过 `solver::run_incompressible_simplec` 接入 SIMPLEC 外层迭代、`pressure_under_relaxation` 压力欠松弛、连续性/动量残差历史、最终修正场输出、`max|div(u)|`、`max|div(u*)|`、动量预测三分量 GMRES 求解诊断、由 Rhie-Chow 面通量连续性残差驱动的压力校正 GMRES 求解与 \(p,\mathbf{u}\) 修正诊断，并支持 `[incompressible.linear.momentum]` / `[incompressible.linear.pressure]` 配置 GMRES 参数，补充理论映射。
- 不可压缩 V&V 骨架算例：`tests/benchmarks/channel_poiseuille/` 与 `tests/benchmarks/lid_driven_cavity_re100/`，包含 case、expected 与 case runner smoke 测试；lid cavity 指标可导出中心线剖面，并记录 Ghia et al. (1982) Re=100 参考点。
- 不可压缩 SIMPLEC 收敛语义：未配置 `time.tolerance` 时仅执行固定外层步数，不再把 `simplec_converged` 误标为 true。
- 不可压缩压力校正 GMRES 默认迭代预算提高到 `restart=64`、`max_iters=500`、`tolerance=1.0e-10`，Poiseuille 与 lid cavity smoke benchmark 现在要求压力校正收敛。
- 不可压缩 SIMPLEC 修正后连续性残差改为压力校正方程质量残差 `max|b_p - A_p p'|`，Poiseuille 与 lid cavity smoke benchmark 收紧到 `1.0e-8`。
- 不可压缩 Poiseuille 与 lid cavity smoke benchmark 增加 `time.tolerance = 1.0e-8`，现在要求 SIMPLEC 外层收敛标记为 true。
- 不可压缩动量预测支持 `[incompressible].body_force = [fx, fy, fz]` 每单位质量体力源项，并按 \(f^*=fL_{\mathrm{ref}}/U_{\mathrm{ref}}^2\) 无量纲化。
- Poiseuille 不可压缩 benchmark 改为体力驱动，两端压力出口，并输出中心线 \(u(y)\) 与解析 Poiseuille 剖面误差诊断。
- 不可压缩 SIMPLEC 外层收敛判据加入速度更新量 \(\max|\Delta\mathbf{u}|\)，避免体力驱动或方腔算例在线性残差小但尚未稳态时被误标为收敛。
- 不可压缩 SIMPLEC 外层增加发散保护；GMRES 对兼容 Hessenberg 退化回代不再报错，由重算残差决定是否继续迭代或标记未收敛。
- 结构化不可压缩 SIMPLEC 支持 `i_min/i_max` 成对周期边界的动量、Rhie-Chow、压力校正与速度修正路径；Poiseuille benchmark 升级为周期体力驱动并启用解析剖面误差阈值。
- Lid cavity Re=100 benchmark 增加基于 Ghia et al. (1982) 中心线表格的误差诊断；长迭代封闭腔体收敛仍作为后续压力-速度耦合改进目标。
- 不可压缩 SIMPLEC 封闭腔体稳定性改进：修正场后重施加 owner-cell 边界约束，速度约束边界 owner 行在压力校正中使用 \(p'=0\)，闭域压力校正 RHS 移除非参考行均值，并对速度修正应用 `pressure_under_relaxation`；lid cavity Re=100 benchmark 从 2 步 smoke 升级为 100 步长迭代诊断。
- 不可压缩 SIMPLEC 增加收敛排查诊断：记录修正场边界重施加前/后的真实 cell-centered 散度，以及跳过 \(p'=0\) 约束行后的压力校正 RHS 总和，用于区分压力方程残差小与速度场真实连续性未收敛。
- 不可压缩 SIMPLEC 收敛残差改为按 `pressure_under_relaxation` 缩放后的压力校正连续性残差 `max|b_p - alpha_p A_p p'|`，避免全量压力校正方程残差很小但实际欠松弛速度修正尚未满足连续性时误判收敛。
- 不可压缩 SIMPLEC 速度更新量诊断拆分为总量、非速度约束 owner 与速度约束边界 owner，用于定位 lid cavity 剩余 `max|Delta u|` 是否由边界 owner 重施加主导。
- 不可压缩 SIMPLEC 修正场散度诊断改为边界感知 face-flux 净通量，墙面/对称面使用无穿透面通量，速度入口/动壁使用给定面速度，避免用零梯度 cell-centered 差分误读边界面连续性。
- 不可压缩 `wall no_slip` owner-cell 边界应用改为只约束法向速度，切向无滑移通过动量边界面源项驱动，避免 owner-cell 层与面源项双重施加切向壁速。
- 不可压缩 `moving_wall` owner-cell 边界应用改为只约束法向速度，切向壁速通过动量边界面源项驱动，避免 lid cavity 顶盖 owner 单元被强制为切向壁速后在侧壁附近产生虚假的 cell-centered 水平通量。
- 不可压缩收敛与 V&V 改进：`time.min_steps` 防止早停假收敛，lid cavity Re=100 在 PISO/transient smoke 路径下达到粗网格定量 Ghia 中心线阈值，并新增 12x12 refined-grid smoke 验证入口。
- 不可压缩动量预测增加 `convection_scheme = "upwind" | "central"` 配置，默认 upwind；`central` 使用内部面中心对流矩阵分支作为二阶格式入口。
- 不可压缩 pressure-velocity 求解器拆分 SIMPLEC/PISO summary 语义，新增结构化边界 face state API，并记录 PISO 每个 pressure corrector 的连续性残差与最大压力修正历史。
- `time.scheme = "simplec"` 现在可解析为不可压缩稳态 pressure-velocity 路径，并在 case 层强制单 pressure corrector。
- ADR 0015：三维不可压 NS（collocated FVM + **SIMPLEC** + **PISO**，结构化六面体首版，I0–I6）；补充通量格式、边界条件、时间积分（BDF1/伪瞬态）；理论页 [docs/theory/incompressible_simplec_piso.md](docs/theory/incompressible_simplec_piso.md)
- ADR 0014：可压 RANS **Menter k-ω SST**（壁距场、分裂 LU-SGS、T0–T5）；理论页 [docs/theory/turbulence_k_omega_sst.md](docs/theory/turbulence_k_omega_sst.md)
- ADR 0013 **E5**：dual_ellipsoid benchmark 说明（`tests/benchmarks/dual_ellipsoid/`）；scatter 每色桶 1 次契约测试；Chrome trace 桶级 span 改 `trace` 级 + `include_args(false)`
- ADR 0013 **E3**：IDWLS RHS 缓冲迁入 `ExecScratch::idwls`；`ExecutionContext::idwls_accumulate_*` / `csr_spmv`；`CsrMatrix::apply_with_context`
- ADR 0013 **E2**：`discretization` / `solver` 移除直接 `rayon` 依赖；P8′ 桶级 flat buffer 迁入 `ExecScratch::colored_viscous`；`ExecFaceBatchStatic4` 为 exec 自有 batch 静态几何；新增 `exec::parallel` 并行调度 API
- ADR 0012：非结构二阶线性重构与梯度限制器（Barth–Jespersen / Venkatakrishnan）；与结构化 `SlopeLimiter` 分离，case 校验禁止混用
- 非结构 M4 二阶无粘：`reconstruction = muscl`（实现为**二阶线性重构**，非 MUSCL 宽模板）+ `[euler].unstructured_limiter`；IDWLS \(\nabla\rho,\nabla p\) + BJ/V 限制器 + `assemble_inviscid_residual_unstructured` 二阶面循环；benchmark `unstructured_freestream`
- 非结构内面 **graph coloring**（`InteriorFaceColoring`）：粘性/无粘内面共用着色桶；feature `parallel-fvm`（rayon 桶内 flux 并行 + scatter 串行，**默认启用**）；golden 测试覆盖着色顺序、缓存 vs mesh 循环、并行 vs 串行（见 ADR 0011）
- 非结构混合单元网格 M1：`UnstructuredMesh3d` 支持 tet / hex / pyramid / prism（VTK 10/12/13/14）面拓扑、owner/neighbor、体积与面度量；新增 `load_vtu`、`load_cgns_unstructured_zone` 与 `check_unstructured_mesh3d`，`mesh_check` 可检查 `.vtu` 与 CGNS unstructured zone，并支持 CGNS FaceCenter ZoneBC 边界 patch 读入与覆盖检查
- 非结构 CGNS case 求解首版：`CaseMesh::Unstructured3d` 支持单域混合网格一阶无粘 Euler 面循环、IDWLS 粘性梯度与 Navier-Stokes 粘性通量、含粘性抛物项的 local time step、显式 Euler/RK4、对角 LU-SGS 与非结构 LU-SGS sweep，并将非结构流场写出为 VTU
- 非结构网格梯度：新增 `compute_unstructured_gradients_idw_lsq`，使用逆距离加权最小二乘法计算 `UnstructuredMesh3d` 单元中心速度与温度梯度
- 多块 3D 可压缩 case 支持 `[restart]` 初场：version=2 TOML 按 block 名称加载守恒量，单 block restart（version=1）仍可用于仅含 1 个 block 的多块网格

### Changed

- **f32 能力矩阵扩展**（ADR 0016）：结构化 3D typed MUSCL（`muscl_stencil_3d_typed`）；多块 1-to-1 共享接口通量（`compressible_multiblock_driver_typed` + `apply_interface_residuals_typed`）
- **非结构 f64 MUSCL typed 原生路径**（ADR 0018 U3）：新增 `assembly_unstructured_inviscid_f64`，删除 `muscl_f64_params` / `assemble_boundary_faces_muscl_typed` 桥接；`InviscidMusclAssembly for f64` 与 f32 对称直连重构与 Riemann
- **Cargo Feature 矩阵与 CI 覆盖**：ARCHITECTURE §8.7 文档化 `parallel-fvm` / `simd-fvm` / I/O features 组合、Makefile/CI 矩阵与已知缺口；同步 `docs/en/ARCHITECTURE.md` 摘要
- **`parallel-fvm` 默认启用**：`Cargo.toml` `default = ["parallel-fvm"]`；`make check` / CI / pre-commit 含 `io-vtk,parallel-fvm`（dual_ellipsoid trace：475 万内面 / 9 色桶；见 ADR 0011 修订）
- **IDWLS RHS 单元并行累加**（`parallel-fvm`）：`LsqRhsCellIncidence` + 单元 `rayon` 路径；粘性梯度与二阶线性重构 \(\nabla\rho,\nabla p\) 共用；golden `parallel_idw_lsq_accumulate_matches_face_serial`
- **谱半径单元并行**（P2）：`cell_spectral_radius_unstructured` 复用 `mesh_cache` + `LsqRhsCellIncidence`；`parallel-fvm` 下单元 `rayon` 累加 \(\sigma_i\)
- **粘性 transport 单元/面并行**（P3）：`fill_cell_transport_coefficients` / `fill_face_transport_coefficients` 在 `parallel-fvm` 下 `rayon` 并行（Sutherland 等非恒定 \(\mu\) 路径）；dual_ellipsoid A/B benchmark 显示 `par_try_for_each_bucket` 相对 `par_map_buckets` 回归约 26%，已回退 P4、保留 `par_map_buckets`（桶内 `with_min_len=1024`）
- **`simd-fvm` + `parallel-fvm` 桶内并行**：无粘/粘性 SIMD batch 路径改为与 `par_map_buckets` 一致（各色 bucket 串行、`full_batches`/`remainder` 桶内 `rayon`）；修复此前仅 9 路 bucket 间并行导致 CPU 利用不足
- **残差监控语义统一**：所有时间积分路径（结构化/非结构、显式 Euler/RK4、LU-SGS、GMRES）的 `log10_residual` 均取步初 \(\|R(U^0)\|\)（`storage.k1` 或 GMRES `base_residual`），不再步末 `post_rhs` 重算
- **粘性内面面心预平均 SoA**（P7）：IDWLS 后 `fill_face_averaged_viscous_soa` 预写 `ViscousFaceAveragedSoA`；flux 阶段顺序读面数组（非 `simd-fvm`）
- **粘性 SIMD full_batch 直通 flux**（P7b）：`simd-fvm` 下 `full_batches` 用 `gather_viscous_face_batch4` cell 直 gather + batch4 flux，跳过全量 `face_avg` 填充；remainder 仍 cell 直读
- **非结构内面 compute+scatter 融合**（P8）：`parallel-fvm` 下各色 bucket 桶内并行 compute 后立即 scatter（`unstructured_*_interior_flux_fused`），取消整桶 `Vec<(geom, flux)>` 缓冲与二次遍历；scatter 仍串行（ADR 0011）
- **粘性 flux 桶级 flat buffer**（P8′）：`parallel-fvm`+`simd-fvm` 每桶一次预分配固定槽（batch×4 + remainder），`rayon` 按 batch 索引写入、桶末串行 scatter；取消 ~119 万/步 per-batch `Vec` 分配
- **粘性 batch4 SoA 融合内核**（P9）：`fused_interior_viscous_face_flux_batch4_from_soa` 在 f64x4 寄存器内完成 cell gather + 面平均 + τ·n，跳过 `ViscousFaceGather4` 物化
- **CPU SIMD 热算子**（P5/P6，`simd-fvm` feature）：新增 `exec::cpu` 模块（`wide` f64x4）；LU-SGS 对角更新、粘性内面四路批处理 flux、IDWLS 3×3 四单元求解、Roe / **Hanel–Van Leer** 一阶内面四路批处理；`InteriorFaceBucketBatchLayout` init-time 静态几何 SoA；标量回退始终可用
- **非结构无粘内面 fused scatter**（P6-1）：`scatter_fused_interior_inviscid_face` 直接写残差 SoA 切片，消费面 cache 预存 `owner_rhs_scale` / `neighbor_rhs_scale`，避免热路径 `-A/V` 除法与 `Result` 分支；SIMD / 并行 / 串行缓存路径共用
- **非结构无粘一阶 SoA flux**（P6-2）：`face_inviscid_flux_first_order_interior_soa` / `_boundary_soa` 从 `PrimitiveFields` 直读；FVS 格式跳过 ghost 原始变量解码
- **非结构无粘一阶边界面 cache**（P6-4）：`UnstructuredBoundaryFace::owner_rhs_scale` + `assemble_boundary_faces_first_order_cached`，与二阶共用 `face_topology.boundary`
- 结构/非结构可压缩路径共用化：LU-SGS 稳定化（`lu_sgs_common`）、粘性边界面通量（`viscous_assembly`）、BC/原始变量刷新与时间步策略（`compressible_helpers`）；结构化粘性内面改走 `accumulate_fused_interior_viscous_face`；谱半径双曲项共用 `accumulate_hyperbolic_face_sigma`
- 非结构求解性能：新增 `UnstructuredSolverMeshCache` 预计算面拓扑与 IDWLS 几何矩阵 \(A\)；`compute_unstructured_gradients_idw_lsq` 与粘性通量装配每步仅累加 RHS 并复用缓存面列表，数值与逐步枚举 `mesh` 等价
- 3D 可压缩读入层统一：`structured_3d` 与单 zone CGNS 解析为 1-block `MultiBlockStructured3d`，求解入口不再 runtime 包装；移除 `CaseMesh::Structured3d` 变体
- 有 1-to-1 接口的多块可压缩 case 在解析阶段校验 `time.scheme = lu_sgs` 且 `lusgs_sweep = false`；`mesh.zone` 废弃并告警
- 3D 可压缩求解统一为 block 编排路径：输出/间隔快照/初场与多块共用一套逻辑；无接口时跳过共享通量装配，单块仍可使用 GMRES/RK4/LU-SGS sweep

- 可压缩算例仅保留无量纲求解：移除 `[nondimensional] enabled` 开关，可压缩算例解析后必定缩放为 \(*\) 变量
- 无量纲热传导 \(\lambda^*\)：修正 \(c_p^*=1/(\gamma-1)\)（此前误用 \(\gamma R^*\) 导致壁面热通量偏大）
- CFL 爬升：`max_steps < cfl_ramp_steps` 时仍按 `cfl_ramp_steps` 线性增 CFL，不再压缩到 `max_steps` 内达 `cfl_max`
- 3D 可压缩局部时间步：RK4 与 LU-SGS 统一使用 Blazek 结构网格 face-sum 谱半径 \(\Delta t_i=\mathrm{CFL}/\sigma_i\)，Navier-Stokes 叠加粘性面贡献
- 3D 网格度量：`FaceMetric` / `FaceGeometry3d` 新增几何面心 `center`；边界面 `spacing` 改为单元中心→面心沿法向投影（替代逻辑 Δ/2）
- 粘性通量梯度：结构网格上改用有限差分 + 局部物理梯度求解，不再使用 Green-Gauss 梯度
- TOML 出口边界：`supersonic = true` 现在正确启用零梯度超声速出口，且不再要求 `static_pressure`
- LU-SGS 固定步长：`[time].dt > 0` 现在会覆盖 LU-SGS 的局部 CFL 步长，同时仍保留谱半径用于隐式分母
- LU-SGS 默认值：`lusgs_sweep` 默认改为 `false`，对角隐式作为稳健默认路径；双扫需显式设为 `true`
- SLAU2 无粘通量：修正右侧压力分裂、压力跳跃项符号，以及质量通量低马赫开关的速度幅值定义，保证均匀态退化为物理通量
- Van Leer 无粘通量：修正亚音速 FVS 质量/动量/能量分裂公式；MUSCL `van_leer` 限制器在线性区恢复完整斜率
- 可压缩 farfield/inlet/outlet 边界：改用法向特征关系生成边界外侧状态，减少简单 ghost 外推的反射与过约束
- 可压缩 3D 稳态推进：方向分裂隐式残差光顺新增逐单元正性回退，避免光顺后更新方向导致内能非正
- LU-SGS 双扫：逐单元正性限制、后扫耦合阻尼与全场线搜索；失败时回退对角隐式更新
- GMRES 隐式伪时间：有限差分扰动按守恒量量级自适应缩放，扰动与最终增量写回均加入逐单元正性限制，并记录裁剪诊断
- GMRES 单元块预条件器：改为局部无粘 Jacobian 块近似，不再为每个分量触发全场 RHS 差分
- GMRES profiling：`GMRES 隐式步诊断` 日志新增时间步、预条件器构造、线性求解、线搜索与更新后残差评估耗时
- 可压缩正性下限：恢复来流静压 1% 的 `positivity_pressure_floor`，通量/输出 primitive 恢复钳制最低压力，避免极低温度伪解
- CGNS case 读入：`mesh.kind = "cgns"` 现在自动读取全部 structured zone；多 zone 文件组装为 `MultiBlockStructured3d`，并按 `IN` / `OUT` / `WALL` family 名修正边界类型
- 多块 3D 可压缩 case：新增同步 block 推进路径，可运行 `case_dualcone` 这类多 zone CGNS；1-to-1 接口按 CGNS transform 映射并通过共享无粘通量守恒装配，支持最终流场与间隔快照写出为单个多 Zone CGNS 文件

### Added

- 多块结构化 3D 网格首版：`MultiBlockStructuredMesh3d`、`mesh.kind = "multi_block_structured_3d"` 与 `mesh_check` 诊断；跨 block 求解仍未启用
- 可压缩 3D 边界面无粘通量接口：`BoundaryInviscidFluxInput` / `inviscid_boundary_face_flux`
- 可压缩 3D 时间推进：`time.scheme = "gmres"` 现在启用 matrix-free GMRES 隐式伪时间步，支持 LU-SGS 标量对角与单元 5×5 块对角预条件器
- 线性代数：矩阵无关 restarted GMRES、CSR 矩阵、ILU(0) 预条件器与 LU-SGS 对角预条件器
- 可压缩流无量纲化：`[nondimensional]`、`FreestreamContext` 单一来流入口、理论页 [docs/theory/nondimensional.md](docs/theory/nondimensional.md)
- CGNS 结构化 zone 读入 + VTS/VTM 导出：`io::load_cgns_zone` / `export_cgns_to_vtm`（features `io-cgns-vts`）；ADR 0008；链接系统 `libcgns-dev`
- VTK VTS **二进制 appended** 读入/写出：`io::load_vts` / `write_vts`（feature `io-vtk`）；支持 zlib + 3D；ADR 0007
- v0.2 启动准备：`docs/CASE_FORMAT.md`；`docs/theory/fvm_diffusion.md`
- v0.2 模块骨架：`field`、`discretization`、`linalg`、`solver/time`；`core::Real` 与 ID newtype
- 首个 V&V 算例目录 `tests/benchmarks/1d_diffusion_analytical/`（case + expected + README）
- AGENTS「数值理论与参考文献」约束；`docs/theory/` 索引
- 运行产物 / V&V / 可观测性：`docs/BENCHMARKS.md`、`docs/OBSERVABILITY.md`、`docs/en/CROSS_CUTTING.md`；**四大横向能力**写入 ARCHITECTURE §1.4、§4.3、§8.5–§8.6
- ADR 0005（时间推进）、ADR 0006（FFI/Python）
- `SECURITY.md` 不可信输入限制；`config/default.toml` 预留 `[output]`/`[time]`/`[study]`
- MCP 集成规划：`docs/MCP.md`、ADR 0004、`.cursor/mcp.json.example`
- 架构设计文档 `docs/ARCHITECTURE.md`（含多精度/GPU §8.4、MCP §4.3）
- `src/app/` 应用编排层；库 API 与 `prelude` 分离
- 数据模型文档 `docs/DATA_MODEL.md`
- ADR 0002：CFD 分层架构与 v0.2 数值基线
- ADR 0003：多精度与 CPU/GPU 执行后端规划
- AGENTS.md 编程风格约束
- 项目骨架：Rust binary + library 结构
- 模块化占位实现：`core`、`mesh`、`solver`、`io`、`config`
- CLI 入口与 TOML 配置加载
- 单元测试与集成测试目录
- CI、pre-commit、Makefile 统一命令入口
- AGENTS.md 与协作模板
