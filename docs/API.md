# asimu 公开 API

本文描述 **library crate** 的公开模块与契约。CLI 参数见 `asimu --help`。

> **库 vs 应用**：数值与数据结构 API 在 crate 根模块；CLI 编排在 [`asimu::app`](#asimuapp-应用层)（应用层，演进节奏可快于数值 API）。

---

## 稳定库 API（semver 保护）

以下模块构成面向集成方的 **数值库公开面**：

| 模块 | 说明 |
|------|------|
| [`asimu::prelude`](#asimu-prelude) | 常用类型 re-export |
| [`asimu::config`](#asimuconfig) | 配置结构（非 CLI 解析） |
| [`asimu::error`](#asimuerror) | 统一错误 |
| [`asimu::core`](#asimucore) | 基础数值类型 |
| [`asimu::mesh`](#asimumesh) | 网格 |
| [`asimu::solver`](#asimusolver) | 求解器 |
| [`asimu::io`](#asimuio) | 输入/输出 |

### `asimu::prelude`

| 项 | 说明 |
|----|------|
| `AppConfig`, `SolverConfig` | 配置 |
| `AsimuError`, `Result` | 错误 |
| `Mesh` | 网格 |
| `Solver`, `SolveResult` | 求解器 |

库集成示例：

```rust
use asimu::prelude::*;
use asimu::config::SolverConfig;

let mesh = Mesh::new("channel", 128)?;
let solver = Solver::new(SolverConfig {
    max_steps: 100,
});
let result = solver.run(&mesh)?;
```

### `asimu::config`

| 类型 | 说明 |
|------|------|
| `AppConfig` | 全局配置（solver + logging） |
| `SolverConfig` | `max_steps` |
| `LoggingConfig` | `level` |
| `Cli` | clap 命令行解析（CLI / 应用层使用） |
| `init_tracing(&str, Option<&Path>) -> Result<TracingGuard>` | 初始化 tracing；可选 Chrome trace 输出路径 |

配置加载：`Cli::load_config()` — 优先级见 [ARCHITECTURE.md](ARCHITECTURE.md)。

### `asimu::error`

| 类型 | 说明 |
|------|------|
| `AsimuError` | 统一错误枚举 |
| `Result<T>` | `std::result::Result<T, AsimuError>` |

### `asimu::core`

| 类型 | 说明 |
|------|------|
| `Vector3` | 三维向量占位类型 |

### `asimu::mesh`

| 类型 | 说明 |
|------|------|
| `Mesh` | 网格元数据（`name`, `cell_count`） |
| `StructuredMesh1d` | 1D 均匀网格 + `BoundaryMesh` |
| `MultiBlockStructuredMesh3d` | 多块 3D 结构化网格容器；支持读入、缩放、metric 批量设置、接口元数据与诊断；`from_single_mesh` 将单块包装为 1-block 容器 |
| `StructuredBlock3d` | 单个 block + 全局 `cell_offset` |
| `BoundaryMesh` | 逻辑边界名 → 面 ID |
| `MeshDiagnostics` | 坐标范围、间距统计、简单警告 |
| `structured_mesh_diagnostics(&StructuredMesh) -> MeshDiagnostics` | 2D/3D 结构化网格诊断 |
| `mesh1d_diagnostics` / `mesh3d_diagnostics` / `multiblock_mesh3d_diagnostics` | 1D / 3D / 多块 3D 专用诊断 |
| `Mesh::new(name, cell_count) -> Result<Mesh>` | 构造；`cell_count == 0` 返回错误 |

### `asimu::solver`

| 类型 | 说明 |
|------|------|
| `Solver` | 占位求解器 |
| `SolveResult` | `iterations`, `residual`, `converged` |
| `Solver::run(&Mesh) -> Result<SolveResult>` | 执行占位求解 |

### `asimu::io`

| 函数 | 说明 |
|------|------|
| `load_case(&Path) -> Result<CaseSpec>` | 解析 TOML 算例（网格 + BC + 物性） |
| `CaseSpec` | `mesh`, `boundary`, `initial`, `diffusivity`, `solver` |
| `CaseSpec::build_initial_fields()` | 构建 `Fields` |
| `CaseSpec::initial_scalar(name)` | 单标量；未声明则全零 |
| `CaseSpec::build_multiblock_conserved_fields(blocks)` | 按 block 顺序构建多块守恒初场 |
| `Incompressible3dRunMetrics` | I1 runner 指标：SIMPLEC 外层迭代/收敛/连续性与动量残差历史、初始/预测/修正速度散度、压力校正 CSR 行数/非零数、GMRES 收敛与最大 \(p'\)、动量预测 CSR、三分量 GMRES 收敛、最大 \(d_P\)、速度变化与不可压缩边界应用统计 |
| `run_incompressible_simplec(config)` | 不可压缩 SIMPLEC solver 层编排：动量预测、Rhie-Chow 连续性 RHS、压力校正、\(p,\mathbf{u}\) 修正与收敛历史 |
| `IncompressibleLinearSolverConfig` | 不可压缩动量/压力线性求解配置；当前映射 `[incompressible.linear.momentum]` 与 `[incompressible.linear.pressure]` 的 GMRES 参数 |
| `load_conserved_fields(path)` / `write_conserved_fields(path, fields)` | 单 block restart TOML（version=1） |
| `load_multiblock_conserved_fields(path, block_names)` / `write_multiblock_conserved_fields(path, blocks)` | 多块 restart TOML（version=2） |
| `load_mesh_from_case(&Path) -> Result<Mesh>` | 从占位 case 文件加载网格 |

#### 非结构 FVM 内面并行（feature `parallel-fvm`，**默认启用**）

完整 **Feature 矩阵与 CI 覆盖**见 [ARCHITECTURE.md §8.7](ARCHITECTURE.md#87-cargo-feature-矩阵与-ci-覆盖)。摘要：

| 项 | 说明 |
|----|------|
| `parallel-fvm` | 依赖 `rayon`；着色桶内 flux compute 并行、scatter 串行（[ADR 0011](adr/0011-parallel-fvm-face-coloring.md)） |
| `simd-fvm` | 与 `parallel-fvm` 正交；`make test-simd-fvm` = `io-vtk,parallel-fvm,simd-fvm` |
| 关闭并行 | `cargo build --no-default-features --features io-vtk` |
| CI 默认 | `io-vtk,parallel-fvm`（**不含** `simd-fvm`，合并前建议本地跑 `make test-simd-fvm`） |

#### VTK VTS / VTU 读入（feature `io-vtk`）

启用：`cargo build --features io-vtk`（`make check` 默认已启用）。

| 函数 / 类型 | 说明 |
|-------------|------|
| `load_vts(&Path) -> Result<VtsLoadResult>` | 读取 **二进制 appended** `.vts` |
| `VtsLoadResult` | `{ mesh: StructuredMesh }`（`D2` / `D3`） |
| `load_vtu(&Path) -> Result<VtuLoadResult>` | 读取 `.vtu` UnstructuredGrid（ASCII / inline binary Points + Cells） |
| `VtuLoadResult` | `{ mesh: UnstructuredMesh3d }` |
| `write_flow_vtu_unstructured(path, mesh, fields, eos, p_floor)` | 写出非结构混合单元流场 VTU（CellData） |
| `StructuredMesh2d` / `StructuredMesh3d` | 2D/3D 结构化网格（`nx`, `ny`[, `nz`]，节点坐标） |
| `CellKind` | 非结构 3D 线性单元：Tet / Hex / Pyramid / Prism（VTK 10/12/13/14） |
| `UnstructuredCell` | 单个非结构单元（`kind` + 全局节点索引） |
| `UnstructuredMesh3d` | 混合单元非结构 3D 网格；构造期完成面拓扑（owner/neighbor）、体积与面度量 |

**`UnstructuredMesh3d::new(name, points, cells)`**：节点顺序遵循 VTK；面合并按排序后的节点键（三角↔三角、四边↔四边）；非流形（≥3 单元共面）返回 `Mesh` 错误。Tier 1 读入：CGNS unstructured zone、VTU。非结构 case 支持单域无粘 Euler（一阶或 `reconstruction = muscl` 对应的**二阶线性重构** + `unstructured_limiter = barth_jespersen | venkatakrishnan`）、IDWLS 粘性梯度与 Navier-Stokes 粘性通量、含粘性抛物项的 local time step、显式 Euler/RK4、对角 LU-SGS 与按 CellId 拓扑邻接的 LU-SGS sweep；GMRES 与非结构 CGNS 流场写出暂未实现。术语与算法见 [adr/0012-unstructured-gradient-limiters.md](adr/0012-unstructured-gradient-limiters.md)。

| `write_vts(&StructuredMesh, &Path) -> Result<()>` | 写出 appended 二进制 VTS |

**不支持**：ASCII VTS、非 zlib 压缩器、VTU appended / compressed DataArray。结构化 VTS 见 [adr/0007-vts-binary-io.md](adr/0007-vts-binary-io.md)。

#### CGNS 读入与 VTS 导出（feature `io-cgns-vts`）

需系统安装 `libcgns-dev`。

| 函数 / 类型 | 说明 |
|-------------|------|
| `list_cgns_zones(&Path) -> Result<Vec<CgnsZoneInfo>>` | 列出全部 structured / unstructured zone |
| `load_cgns_zone(&Path, zone_index) -> Result<CgnsLoadResult>` | 读取单 zone |
| `load_cgns_unstructured_zone(&Path, zone_index) -> Result<CgnsUnstructuredLoadResult>` | 读取 unstructured zone → `UnstructuredMesh3d` + `BoundarySet`（tet/hex/pyramid/prism；固定类型或 MIXED sections；FaceCenter ZoneBC） |
| `load_cgns_all_zones(&Path) -> Result<CgnsMultiLoadResult>` | 读取全部 structured zone；case 解析会将多 zone CGNS 组装为 `MultiBlockStructured3d` |
| `write_multiblock_flow_cgns(path, mesh, fields, eos, time, p_floor)` | 将多块结构化可压缩流场写为单个多 Zone CGNS 文件 |
| `write_structured_vertex_solution_cgns(path, mesh, solution)` | 将单 Zone 结构化网格与任意 Vertex 标量字段写为 CGNS |
| `export_cgns_zone_to_vts(input, zone, output) -> Result<CgnsLoadResult>` | CGNS zone → VTS |
| `export_cgns_to_vtm(input, output) -> Result<CgnsMultiLoadResult>` | CGNS 全部 zone → VTM + 多个 VTS |

多 zone CGNS case 可进入 3D 可压缩求解路径；当前按 block 同步推进，1-to-1 接口按 CGNS transform 映射并用共享无粘通量守恒装配（一次计算、两侧等量反号），最终 `solution_cgns` 与 `solution_every` 快照写为单个多 Zone CGNS 文件。严格守恒多块路径目前要求 `time.scheme = "lu_sgs"` 且 `lusgs_sweep = false`。不可压缩 I0 placeholder 使用通用 Vertex 字段写出单 Zone CGNS，默认字段为 `Pressure`、`VelocityX`、`VelocityY`、`VelocityZ`，输出还原 SI。

见 [adr/0008-cgns-io.md](adr/0008-cgns-io.md)。Structured zone 与 Unstructured zone（混合单元）均已支持；`mesh.kind = "cgns"` 会按 zone 类型进入多块结构化或单域非结构路径。非结构求解接入按 [adr/0010-unstructured-mixed-mesh.md](adr/0010-unstructured-mixed-mesh.md) 分阶段推进。

#### 网格诊断报告

| 函数 / 类型 | 说明 |
|-------------|------|
| `MeshReport` | 可读摘要（`Display`）：范围、间距、BC patch |
| `check_unstructured_mesh3d(&UnstructuredMesh3d, Option<&BoundarySet>, source)` | 非结构网格几何/拓扑/边界预检：体积、面面积/法向、owner/neighbor、单元类型统计、patch face 引用与覆盖率 |
| `report_structured_mesh` / `report_cgns_zone` / `report_vts` / `report_case_mesh` | 由网格或读入结果生成报告 |

CLI 示例：`cargo run --example mesh_probe --features io-cgns-vts -- mesh.cgns`

#### Case 文件格式（临时）

```
name=<mesh_name>;cells=<count>
```

示例见 `tests/fixtures/demo.case`（v0.1 遗留）。v0.2 算例 TOML 见 [CASE_FORMAT.md](CASE_FORMAT.md) 与 `tests/benchmarks/`。

---

## v0.2 新增模块（骨架，semver 0.2.0 前可能调整）

| 模块 | 说明 | 状态 |
|------|------|------|
| [`asimu::boundary`](#asimuboundary) | `BoundaryKind`、`BoundaryPatch`、`BoundaryRegistry` | v0.2 已实现 |
| [`asimu::core`](#asimucore) | `Real`、`CellId`、`approx_eq` | 骨架已实现 |
| [`asimu::field`](#asimufield) | `ScalarField` / `IncompressibleFields` SoA | 骨架已实现 |
| [`asimu::linalg`](#asimulinalg) | `LinearSystem` 三对角 + Thomas | v0.2 已实现 |
| [`asimu::discretization`](#asimudiscretization) | FVM 扩散装配 + BC 施加 | v0.2 1D 已实现 |
| [`asimu::exec`](#asimuexec) | `ExecutionContext`、scatter 调度（ADR 0013） | E0 已实现 |
| [`asimu::solver::time`](#asimusolver) | `TimeIntegrator`、`SteadyStateIntegrator` | 骨架已实现 |

### `asimu::boundary`

| 类型 | 说明 |
|------|------|
| `BoundaryKind` | `Dirichlet { value }` / `Neumann { flux }` |
| `BoundaryPatch` | 逻辑名 + `face_ids` + `kind` |
| `BoundarySet` | patch 有序列表 |
| `BoundaryRegistry` | 校验 + `handler_for` 调度 |

理论：[theory/boundary_conditions.md](theory/boundary_conditions.md)

### `asimu::core`

| 类型 | 说明 |
|------|------|
| `Real` | 默认 `f64` 数值标量别名 |
| `CellId`, `FaceId`, `NodeId` | 网格实体 newtype |
| `Vector3` | 三维向量 |
| `approx_eq(a, b, tol)` | 浮点容差比较 |

### `asimu::field`

| 类型 | 说明 |
|------|------|
| `ScalarField` | 标量场；`uniform` / `from_values` 构造 |
| `IncompressibleFields` | 不可压缩主变量场：`pressure`、`velocity_x/y/z` |
| `InitialKind` | `uniform` / `linear` / `values` |
| `InitialSet` | 命名初始条件集合 |
| `Fields` | 命名标量场 map；`from_initial_set` 构建 |

### `asimu::linalg`

| 类型 | 说明 |
|------|------|
| `LinearSystem` | 三对角 `rhs` / `diag` / `lower` / `upper` |
| `LinearSystem::zeros(n)` | 构造零系统 |
| `LinearSystem::solve_tridiagonal()` | Thomas 算法求解 |
| `LinearOperator` | 矩阵无关线性算子接口 `y = A x` |
| `Preconditioner` | 左预条件器接口 `z = M^{-1}r` |
| `GmresSolver` / `GmresConfig` | restarted GMRES Krylov 求解器 |
| `CsrMatrix` | CSR 显式稀疏矩阵，同时实现 `LinearOperator` |
| `Ilu0Preconditioner` | CSR 矩阵的 ILU(0) 预条件器 |
| `LusgsDiagonalPreconditioner` | 由 `dt` / `sigma` 构造的 LU-SGS 对角预条件器 |
| `CellBlockDiagonalPreconditioner` | 每单元固定大小局部块逆矩阵预条件器（GMRES 3D 可压缩块对角路径复用） |

### `asimu::discretization`

| 函数 | 说明 |
|------|------|
| `assemble_diffusion_1d` | 1D 内部面扩散装配 |
| `apply_boundary_conditions` | 按 patch 顺序施加 BC |
| `apply_dirichlet_face` / `apply_neumann` | 单面 BC |
| `UnstructuredSolverMeshCache` / `from_mesh` | 非结构求解器网格缓存：面拓扑（`UnstructuredFaceTopology`）+ 内面着色（`InteriorFaceColoring`）+ IDWLS 单元–面关联（`LsqRhsCellIncidence`）+ 每单元 IDWLS 正规方程矩阵 \(A\) |
| `UnstructuredGradientLsqInput` / `compute_unstructured_gradients_idw_lsq` | `UnstructuredMesh3d` 上的逆距离加权最小二乘梯度；**必须**提供 `mesh_cache`；内部面用相邻单元中心，边界面用 ghost 镜像样本 |
| `compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq` | 二阶线性重构用 IDWLS 梯度（\(\nabla\rho,\nabla u,\nabla p\) 等）；装配前由 `EvaluateRhsUnstructured` 调用 |
| `UnstructuredGradientLimiter` | 非结构梯度限制器（`barth_jespersen` / `venkatakrishnan`）；与结构化 `SlopeLimiter` 独立 |
| `UnstructuredLinearReconstructionCtx` / `reconstruct_unstructured_interior_face` | IDWLS 梯度外推 + 限制器面重构；下游接 `face_inviscid_flux_from_interface` |
| `face_inviscid_flux_from_interface` | 由左右原始变量界面态计算无粘数值通量 |
| `UnstructuredGradientScratch` | IDWLS 每步 RHS 缓冲（`bu`/`bv`/`bw`/`bt`）与温度 scratch；`compute_unstructured_gradients_idw_lsq_with_scratch` 复用 |
| `ViscousAssemblyUnstructuredInput` / `compute_gradients_and_assemble_viscous_unstructured` | `UnstructuredMesh3d` 上计算 IDWLS 梯度并叠加 Newtonian/Fourier 粘性通量残差；面循环走 `mesh_cache.face_topology` |
| `InteriorFaceColoring` | 非结构内面贪心着色桶；`for_each_face_index` 按桶遍历；默认启用 `parallel-fvm` 时 `par_map_buckets` 桶内 rayon 并行 compute + 串行 scatter |
| `viscous_assembly` | 结构/非结构共用粘性边界面通量（`viscous_flux_at_boundary`）、scatter（`accumulate_viscous_*`）与壁面梯度外推 |
| `compute_incompressible_divergence_3d` | 结构化 3D 不可压缩 I1 连续性残差 \(\nabla\cdot\mathbf{u}\) |
| `compute_incompressible_velocity_laplacian_3d` | 结构化 3D 不可压缩 I1 速度三分量 Laplacian skeleton |
| `apply_incompressible_boundary_conditions_3d` | 结构化 3D 不可压缩 cell-centered 边界应用，支持 wall / moving_wall / velocity_inlet / pressure_outlet / symmetry |
| `compute_incompressible_rhie_chow_divergence_3d` | 结构化 3D 不可压缩 Rhie-Chow 面质量通量连续性残差 |
| `assemble_incompressible_pressure_correction_3d` | 结构化 3D 不可压缩压力校正 CSR，使用面插值 \(d_P\)、压力出口 \(p'=0\) 与参考压力策略 |
| `assemble_incompressible_pressure_poisson_3d` | 结构化 3D 不可压缩 I1 压力校正 Poisson CSR 兼容骨架 |
| `IncompressiblePressureCorrectionConfig` / `IncompressiblePressureCorrectionSystem` | 压力校正装配配置与 `CsrMatrix + rhs` 输出 |
| `assemble_incompressible_momentum_predictor_3d` / `assemble_incompressible_momentum_predictor_with_boundary_3d` | 结构化 3D 不可压缩伪瞬态动量预测 CSR，含内部扩散、一阶迎风对流、动量边界面贡献、压力梯度、欠松弛与 \(d_P\) |
| `IncompressibleMomentumPredictorConfig` / `IncompressibleMomentumPredictorSystem` | 动量预测装配配置与三分量共用 `CsrMatrix`、`rhs_x/y/z`、`d_coefficient` |
| `assemble_diffusion_placeholder` | 尺寸校验 + RHS 清零占位 |

### `asimu::exec`

ADR 0013：CPU/GPU 执行后端与 scatter 调度。E0 串行 scatter + `Auto` 解析；E1 并行 atomic scatter；E2 `exec::parallel` 统一 rayon、`ExecScratch` 着色桶缓冲。

| 类型 / 函数 | 说明 |
|-------------|------|
| `ExecutionContext` | 算例级 exec 上下文（backend、已解析 scatter 模式、步间 `ExecScratch`） |
| `ExecConfig` | `backend`、`scatter_mode`、`parallel_min_len`、`scatter_parallel_min_faces` |
| `MeshExecMetrics` | `num_cells`、`interior_faces`、`max_bucket_faces`（init-time 一次） |
| `ExecScratch::with_metrics` | 按网格规模预分配 `IdwlsRhsBuffer` 与 `ColoredViscousFaceBuffer`（`parallel-fvm`） |
| `ExecFaceBatchStatic4` | 四内面静态几何 SoA（init-time；discretization 类型别名 `InteriorFaceBatchStatic4`） |
| `ColoredViscousFaceGeom` / `ColoredViscousFaceFlux` | exec 着色桶 compute/scatter 面槽（粘性） |
| `IdwlsRhsBuffer` | IDWLS 正规方程 RHS \(b_u,b_v,b_w,b_T,b_\rho,b_p\)（E3；步间复用） |
| `ExecutionContext::idwls_prepare_viscous` / `idwls_prepare_inviscid` | 清零 IDWLS RHS 槽 |
| `ExecutionContext::idwls_accumulate_viscous_cells` / `idwls_accumulate_inviscid_cells` | 单元并行 IDWLS RHS 累加 |
| `ExecutionContext::csr_spmv` / `CsrSpmvView` | CSR SpMV（`CpuParallel` 行并行） |
| `CsrMatrix::apply_with_context` | `linalg` → exec SpMV 入口 |
| `exec::parallel::*` | 桶内/单元并行 map、zip、IDWLS RHS 等（`parallel-fvm`；rayon 仅经此模块） |
| `exec::scatter::run_bucket_scatter` | 单着色桶串行 scatter 回退 span |
| `exec::scatter::scatter_viscous_valid_slots` / `scatter_viscous_bucket_range` | 粘性内面 scatter（`Serial` 或 `ParallelUnsafeAtomics` + rayon） |
| `exec::scatter::scatter_inviscid_pairs` | 无粘内面 scatter（同上） |
| `ViscousScatterOp` / `InviscidScatterOp` | 单面 scatter 贡献（discretization → exec 映射） |

### `asimu::solver`（扩展）

| 类型 | 说明 |
|------|------|
| `SolverState` | 显式求解状态（伪步、物理时间、迭代） |
| `TimeMode` | `Steady` / `Transient` |
| `TimeStepInfo` | 单步推进摘要 |
| `TimeIntegrator` | 时间推进 trait |
| `SteadyStateIntegrator` | v0.2 稳态伪时间 |
| `GmresImplicitConfig` / `GmresImplicitDelta` | 3D 可压缩 matrix-free 隐式 GMRES 更新入口 |
| `GmresPreconditionerKind` | `ScalarDiagonal` / `CellBlockDiagonal`，对应 `[time] gmres_preconditioner` |
| `EvaluateRhsUnstructured` | 非结构 3D RHS 求值（镜像 `EvaluateRhs3d`）；`run` 含 BC 刷新，`assemble_from_current_state` 供 LU-SGS 内层复用 |
| `RefreshCompressibleStateInput` / `refresh_compressible_ghosts_and_primitives` | 结构/非结构共用的 BC ghost + 原始变量刷新 |
| `finalize_cell_dts_from_sigma` | 由谱半径计算局部 \(\Delta t_i\) 并应用固定 dt / 全局 dt 策略 |
| `lu_sgs_common` | LU-SGS 双扫共用稳定化（线搜索、对角回退、正性限制） |

理论参考：[docs/theory/fvm_diffusion.md](theory/fvm_diffusion.md)。

### `asimu::physics`（可压缩 v1.x）

理论：[docs/theory/nondimensional.md](theory/nondimensional.md)（无量纲）；[adr/0009-compressible-navier-stokes.md](adr/0009-compressible-navier-stokes.md)

| 类型 / 函数 | 说明 |
|-------------|------|
| `IdealGasEoS` | \(\gamma, R^*\)（可压缩求解为 \(*\) 变量） |
| `FreestreamParams` | `[freestream]` 对应 Mach、\(p,T\)、方向（TOML 写 SI，解析后缩放） |
| `ReferenceScales` | 参考量与 Re；`from_freestream` |
| `IncompressibleReferenceScales` | 不可压缩显式参考量：\(L_{\mathrm{ref}}\)、\(U_{\mathrm{ref}}\)、\(\rho\)、\(Re\)、\(p_{\mathrm{ref}}\) |
| `FreestreamContext` | **来流单一入口**（\(*\) primitive / conserved / \(\rho^*\) 由 \(p^*,T^*\)） |
| `ViscousPhysicsConfig` | Sutherland/常数粘度；`static_temperature`（式 (2)） |
| `ConservedFields::from_freestream` | 由 SI 来流参数构造无量纲均匀初场 |
| `ConservedFields::from_freestream_context` | 初场（经 `CaseSpec::build_conserved_fields`） |
| `ConservedFields::to_dimensional` | 输出还原 SI |
| `CaseSpec::is_nondimensional` | 可压缩算例恒为 `true` |
| `CaseSpec::dimensional_eos` | 输出用有量纲 EOS |

BC：`apply_compressible_boundary_conditions(..., &FreestreamContext, ...)` — 见 [theory/nondimensional.md §4](theory/nondimensional.md#4-边界条件)。

不可压缩 case 要求 `[incompressible.reference] length/velocity`；解析后内部为星号量，CGNS 输出还原 SI。理论见 [theory/incompressible_simplec_piso.md §1.1](theory/incompressible_simplec_piso.md#11-不可压缩无量纲化)。

---

## `asimu::app`（应用层）

面向 **binary / CLI** 的编排 API，**不**承诺与数值模块相同的 semver 稳定性。

| 项 | 说明 |
|----|------|
| `run(cli: Cli) -> Result<()>` | CLI 完整流程（配置 → 日志 → 占位算例） |
| `demo_config() -> AppConfig` | 测试用默认配置 |

CLI 入口（`main.rs`）仅调用 `app::run`。库集成方应直接使用 `mesh` / `solver` 等模块，而非 `app::run`。

v0.3+ 计划将本模块演进为 `case`（见 [ARCHITECTURE.md](ARCHITECTURE.md)）。

---

## 破坏性变更政策

**库 API**（`prelude`、`mesh`、`solver` 等）变更必须：

1. 更新本文档及 `docs/en/API.md`（英文版待同步）
2. 写入 [CHANGELOG.md](../CHANGELOG.md)
3. 按 semver 递增版本

**应用层**（`app`）变更须在 CHANGELOG 记录，但允许在 minor 版本调整编排逻辑。

---

## CLI

```
asimu [OPTIONS]

Options:
      --config <PATH>           配置文件（TOML）
      --max-steps <N>           最大时间步数（占位求解器）
      --log-level <LEVEL>       日志级别
  -h, --help                    帮助
```

环境变量：`ASIMU_CONFIG`、`ASIMU_MAX_STEPS`、`ASIMU_LOG_LEVEL`。
