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
| `load_case(&Path) -> Result<CaseSpec>` | 解析 TOML 算例（网格 + BC + 物性 + `[numerics]`） |
| `CaseNumericsConfig` | `[numerics]` 段；`compute_precision: ComputePrecision`（默认 `F64`） |
| `CaseSpec` | `mesh`, `boundary`, `initial`, `numerics`, `diffusivity`, `solver` |
| `CaseSpec::build_initial_fields()` | 构建 `Fields` |
| `CaseSpec::initial_scalar(name)` | 单标量；未声明则全零 |
| `CaseSpec::build_multiblock_conserved_fields(blocks)` | 按 block 顺序构建多块守恒初场 |
| `CaseSpec::resolved_max_steps()` | 时间推进步数上限：`[time].max_steps`，其次 `[euler].max_steps`，默认 100（可压/不可压共用） |
| `Incompressible3dRunMetrics` | I1 runner 指标：pressure-velocity algorithm（`simplec`/`piso`）、pressure corrector 数量、外层迭代/收敛/连续性与动量残差历史、PISO corrector 连续性残差与最大 \(p'\) 历史、初始边界面通量散度、预测 Rhie-Chow 散度、显式 `phi` 修正后散度、全量压力校正方程质量残差、修正场重施加边界前/后的边界感知 face-flux 散度（开域出口 \(p'=0\) owner 不参与 SIMPLEC 连续性判据）、压力校正 active RHS 总和、压力校正 CSR 行数/非零数、PCG/GMRES 收敛与最大 \(p'\)、动量预测 CSR、三分量 GMRES 收敛、最大 \(d_P\)、总速度变化及非速度约束 owner / 速度约束边界 owner 的速度变化拆分、不可压缩边界应用统计、I4 `mass_flux_net` / `mass_flux_inlet_magnitude` / `mass_flux_imbalance_ratio`（`compute_incompressible_boundary_mass_balance_3d`）、Poiseuille 解析剖面误差（周期体 force 与 `channel_re100_3d` 充分发展段 \(u/U_m=6(y/H)(1-y/H)\)）、lid cavity Ghia 中心线误差，以及 lid cavity / Poiseuille / `channel_re100_3d` 的中心线剖面诊断 |
| `run_incompressible_pressure_velocity(config)` / `run_incompressible_pressure_velocity_with_observer(config, observer)` / `run_incompressible_simplec(config)` | 不可压缩 pressure-velocity solver 层编排；`time.scheme = "simplec"` 在 case 层强制单 pressure corrector，`time.scheme = "piso"` 使用 `[incompressible].piso_correctors`。`IncompressiblePressureVelocityConfig` 可选 `initial_face_flux`（Rhie-Chow IC 播种）；每个外层步动量预测后对固定 \(\mathbf{u}^*\) 调用 `reconcile_rhie_chow_pressure_with_fixed_velocity_3d` 再构造 \(\phi^{H/A}\)。流程包括动量预测、Rhie-Chow 初始化显式 `phi`、一次或多次压力校正、显式 `phi` 更新、\(p,\mathbf{u}\) 修正、修正后边界重施加，以及按 `time.mode` 区分的收敛判据：`steady` 要求连续性/动量/非速度约束 owner 速度更新量同时收敛，`transient` 只用连续性和动量判断 pressure-velocity coupling，速度更新量作为物理瞬态诊断；结构化路径支持 `i_min/i_max` 成对周期边界，`time.min_steps` 可防止早停假收敛。`with_observer` 在每个外层步提供当前 step、历史残差与修正场只读视图，case 层用它按 `solution_every` 即时刷新不可压缩残差与间隔流场输出 |
| `IncompressibleLinearSolverConfig` | 不可压缩动量/压力线性求解配置；当前映射 `[incompressible.linear.momentum]` 的 GMRES 参数与 `[incompressible.linear.pressure]` 的 `pcg` / `gmres` 参数，压力校正默认使用 Jacobi-preconditioned PCG |
| `IncompressibleConvectionScheme` | 不可压缩动量预测对流格式：`upwind` 默认；`central` 为内部面中心对流入口 |
| `load_conserved_fields(path)` / `write_conserved_fields(path, fields)` | 单 block restart TOML（version=1；默认 `f64`） |
| `load_conserved_fields_checked(path, expected)` / `write_conserved_fields_with_precision(path, fields, precision)` | 单 block restart，校验/写入 `compute_precision` |
| `load_conserved_fields_from_flow_cgns(path, expected_num_cells, eos, reference)` | 从非结构 `flow.cgns`（`FlowSolution` 的 `Density`/`VelocityX/Y/Z`/`Pressure`）读取 SI 原始量并按 `reference` 转无量纲守恒场 |
| `load_conserved_fields_typed::<T>(path)` / `write_conserved_fields_typed(path, fields)` | typed 单 block restart（ADR 0016 §6） |
| `read_restart_precision(path) -> RestartPrecision` | 读取 restart 文件标注精度（缺省 `f64`） |
| `load_multiblock_conserved_fields(path, block_names)` / `write_multiblock_conserved_fields(path, blocks)` | 多块 restart TOML（version=2） |
| `load_multiblock_conserved_fields_checked` / `write_multiblock_conserved_fields_with_precision` | 多块 restart，校验/写入 `compute_precision` |
| `load_mesh_from_case(&Path) -> Result<Mesh>` | 从占位 case 文件加载网格 |

#### 非结构 FVM 内面并行（feature `parallel-fvm`，**默认启用**）

完整 **Feature 矩阵与 CI 覆盖**见 [ARCHITECTURE.md §8.7](ARCHITECTURE.md#87-cargo-feature-矩阵与-ci-覆盖)。摘要：

| 项 | 说明 |
|----|------|
| `parallel-fvm` | 依赖 `rayon`；着色桶内 flux compute 并行、scatter 串行或 atomic（[ADR 0011](adr/0011-parallel-fvm-face-coloring.md)）；非结构 f32 一阶/MUSCL 无粘与粘性内面均支持着色桶并行 |
| `io-vtk` / `io-cgns` | **默认启用** `io-cgns` + `io-vtk` + `parallel-fvm` + `simd-fvm`（需系统 `libcgns-dev`） |
| `simd-fvm` | 与 `parallel-fvm` 正交；**默认启用**；`make test-no-simd-fvm` 测无 SIMD 路径 |
| 关闭并行 | `cargo build --no-default-features --features io-cgns,io-vtk` |
| 串行 FVM | `cargo build --no-default-features --features io-cgns,io-vtk`（见 `Makefile` `CARGO_SCALAR_FLAGS`） |
| CI 默认 | 默认 features（含 `io-cgns`、`io-vtk`、`parallel-fvm`、`simd-fvm`） |

#### VTK VTS / VTU 读入（feature `io-vtk`）

启用：`cargo build`（默认已含 `io-vtk`）。

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

#### CGNS 读入与 VTS 导出（features `io-cgns` + `io-vtk`）

需系统安装 `libcgns-dev`。

| 函数 / 类型 | 说明 |
|-------------|------|
| `list_cgns_zones(&Path) -> Result<Vec<CgnsZoneInfo>>` | 列出全部 structured / unstructured zone |
| `load_cgns_zone(&Path, zone_index) -> Result<CgnsLoadResult>` | 读取单 zone |
| `load_cgns_unstructured_zone(&Path, zone_index) -> Result<CgnsUnstructuredLoadResult>` | 读取 unstructured zone → `UnstructuredMesh3d` + `BoundarySet`（tet/hex/pyramid/prism；固定类型或 MIXED sections；FaceCenter ZoneBC） |
| `load_cgns_all_zones(&Path) -> Result<CgnsMultiLoadResult>` | 读取全部 structured zone；case 解析会将多 zone CGNS 组装为 `MultiBlockStructured3d` |
| `write_multiblock_flow_cgns(path, mesh, fields, eos, time, p_floor)` | 将多块结构化可压缩流场写为单个多 Zone CGNS 文件 |
| `write_flow_cgns_unstructured(path, mesh, fields, eos, time, p_floor)` | 将非结构可压缩流场写为单 Zone CGNS（按单元类型分 section；场 @ CellCenter） |
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

CLI 示例：`cargo run --example mesh_probe -- mesh.cgns`（默认 features 已含 `io-cgns`、`io-vtk`）

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
| `ComputePrecision` | 核心计算精度：`F64`（默认）\| `F32`（ADR 0016；非结构 3D 可压缩路径 Validate 能力矩阵见 `case::validate`） |
| `ComputeFloat` | 核心计算标量 trait；实现 `f32` / `f64` |
| `parse_compute_precision(&str) -> Result<ComputePrecision>` | 解析 `[numerics].compute_precision` |
| `CellId`, `FaceId`, `NodeId` | 网格实体 newtype |
| `Vector3` | 三维向量 |
| `approx_eq(a, b, tol)` | 浮点容差比较 |

### `asimu::field`

| 类型 | 说明 |
|------|------|
| `ScalarField` | 标量场；`uniform` / `from_values` 构造（默认 `Real`/`f64`） |
| `ScalarFieldT<T>` | 泛型标量场（`T: ComputeFloat`） |
| `ConservedFieldsT<T>` | 可压缩守恒变量 SoA；`ConservedFields = ConservedFieldsT<Real>` |
| `ConservedResidualT<T>` | 守恒残差 SoA；`ConservedResidual = ConservedResidualT<Real>` |
| `PrimitiveFieldsT<T>` | 原始变量 cache；`PrimitiveFields = PrimitiveFieldsT<Real>` |
| `assign_lusgs_diagonal_update_f32` | f32 对角 LU-SGS：`out ← base + ω·Δt_i·R/(1+Δt_i·σ_i)`（原生 f32 \(\sigma,\Delta t_i\)） |
| `assign_axpy_dt_f32`（`ConservedFieldsT<f32>`） | 逐单元 \(\Delta t_i\) 的 axpy 更新（RK4/Euler 当地时间步） |
| `state_after_increment_f32` / `is_physical_conserved_f32` / `max_physical_increment_scale_f32` | f32 守恒态正性限制（LU-SGS 扫掠热路径） |
| `IncompressibleFields` | 不可压缩主变量场：`pressure`、`velocity_x/y/z` |
| `InitialKind` | `uniform` / `linear` / `values` |
| `InitialSet` | 命名初始条件集合 |
| `Fields` | 命名标量场 map；`from_initial_set` 构建 |

### `asimu::core`

| 函数 | 说明 |
|------|------|
| `elapsed_ms` | 自 `Instant` 起经过的 wall time（毫秒），solver/case 诊断日志复用 |

### `asimu::linalg`

| 类型 | 说明 |
|------|------|
| `LinearSystem` | 三对角 `rhs` / `diag` / `lower` / `upper` |
| `LinearSystem::zeros(n)` | 构造零系统 |
| `LinearSystem::solve_tridiagonal()` | Thomas 算法求解 |
| `LinearOperator` | 矩阵无关线性算子接口 `y = A x` |
| `Preconditioner` | 左预条件器接口 `z = M^{-1}r` |
| `GmresSolver` / `GmresConfig` | restarted GMRES Krylov 求解器；`GmresConfig::validate` 校验 restart/max_iters/tolerance |
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
| `UnstructuredSolverMeshCache` / `from_mesh` | 非结构求解器网格缓存：面拓扑（`UnstructuredFaceTopology`）+ f32 预打包面几何（`face_topology_f32`）+ f32 IDWLS 矩阵（`lsq_geometry_f32`）+ f32 MUSCL 限制器样本（`cell_gradient_samples_f32`）+ f32 LU-SGS 面耦合（`lusgs_couplings_f32`）+ 内面着色（`InteriorFaceColoring`）+ IDWLS 单元–面关联（`LsqRhsCellIncidence`）+ 每单元 IDWLS 正规方程矩阵 \(A\)（`lsq_geometry`） |
| `UnstructuredGradientLsqInput` / `compute_unstructured_gradients_idw_lsq` | `UnstructuredMesh3d` 上的逆距离平方加权最小二乘梯度（\(w=1/|\Delta\mathbf x|^2\)，对标 SU2 WLS）；**必须**提供 `mesh_cache`；内部面用相邻单元中心，边界面用面心 ghost 场值 |
| `compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq` | 二阶线性重构用 IDWLS 梯度（\(\nabla\rho,\nabla u,\nabla p\) 等）；装配前由 `EvaluateRhsUnstructured` 调用 |
| `UnstructuredGradientLimiter` | 非结构梯度限制器（`barth_jespersen` / `venkatakrishnan`）；与结构化 `SlopeLimiter` 独立 |
| `UnstructuredLinearReconstructionCtx` / `reconstruct_unstructured_interior_face` | IDWLS 梯度外推 + 限制器面重构；下游接 `face_inviscid_flux_from_interface` |
| `face_inviscid_flux_from_interface` | 由左右原始变量界面态计算无粘数值通量 |
| `UnstructuredGradientScratch` | IDWLS 每步 RHS 缓冲（`bu`/`bv`/`bw`/`bt`）与温度 scratch；`compute_unstructured_gradients_idw_lsq_with_scratch` 复用 |
| `ViscousAssemblyUnstructuredInput` / `compute_gradients_and_assemble_viscous_unstructured` | `UnstructuredMesh3d` 上计算 IDWLS 梯度并叠加 Newtonian/Fourier 粘性通量残差；面循环走 `mesh_cache.face_topology` |
| `InteriorFaceColoring` | 非结构内面贪心着色桶；`for_each_face_index` 按桶遍历；默认启用 `parallel-fvm` 时 `par_map_buckets` 桶内 rayon 并行 compute + 串行 scatter |
| `viscous_assembly` | 结构/非结构共用粘性边界面通量（`viscous_flux_at_boundary`）、scatter（`accumulate_viscous_*`）与壁面梯度外推 |
| `compute_incompressible_divergence_3d` | 结构化 3D 不可压缩 I1 连续性残差 \(\nabla\cdot\mathbf{u}\) |
| `compute_incompressible_face_flux_divergence_3d` | 结构化 3D 不可压缩边界感知 face-flux 散度诊断，内部面使用局部 \(\mathbf{u}_f\cdot\mathbf{S}_f\)，墙面/对称面无穿透、速度入口/动壁使用边界面速度 |
| `incompressible_boundary_face_state` | 不可压缩边界 face state 统一入口，集中给出 face 速度、可选边界压力、压力校正约束语义与质量通量类型，供 face-flux、Rhie-Chow 与压力校正路径复用 |
| `compute_incompressible_velocity_laplacian_3d` | 结构化 3D 不可压缩 I1 速度三分量 Laplacian skeleton |
| `apply_incompressible_boundary_conditions_3d` | 结构化 3D 不可压缩 cell-centered 边界应用，支持 wall / moving_wall / velocity_inlet / pressure_outlet / symmetry |
| `compute_incompressible_rhie_chow_divergence_3d` / `IncompressibleFaceFluxField` | 结构化 3D 不可压缩 Rhie-Chow 面质量通量连续性残差；显式 `phi` 字段可由 Rhie-Chow 初始化、由 pressure correction 直接更新，并供动量对流项复用；Cartesian 与结构化贴体网格均使用局部面面积、法向和单元体积 |
| `assemble_incompressible_pressure_correction_3d` | 结构化 3D 不可压缩压力校正 CSR，使用面插值 \(d_P\)、局部 \(A_f/\Delta n_f\)、压力出口 \(p'=0\)、wall/moving wall/symmetry Neumann-like 通量语义与参考压力策略 |
| `assemble_incompressible_pressure_poisson_3d` | 结构化 3D 不可压缩 I1 压力校正 Poisson CSR 兼容骨架 |
| `IncompressiblePressureCorrectionConfig` / `IncompressiblePressureCorrectionSystem` | 压力校正装配配置与 `CsrMatrix + rhs` 输出 |
| `assemble_incompressible_momentum_predictor_3d` / `assemble_incompressible_momentum_predictor_with_boundary_3d` / `assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d` | 结构化 3D 不可压缩伪瞬态动量预测 CSR，含基于局部 metric 的内部扩散、非正交交叉扩散 deferred correction、一阶迎风对流、可选显式 `phi` 对流通量、动量边界面贡献、共享结构化标量梯度驱动的 Green-Gauss 压力梯度、每单位质量体力源项、欠松弛与 \(d_P\) |
| `IncompressibleMomentumPredictorConfig` / `IncompressibleMomentumPredictorSystem` | 动量预测装配配置（含 `body_force`）与三分量共用 `CsrMatrix`、`rhs_x/y/z`、`d_coefficient` |
| `assemble_diffusion_placeholder` | 尺寸校验 + RHS 清零占位 |

### `asimu::exec`

ADR 0013：CPU/GPU 执行后端与 scatter 调度。E0 串行 scatter + `Auto` 解析；E1 并行 atomic scatter；E2 `exec::parallel` 统一 rayon、`ExecScratch` 着色桶缓冲。ADR 0017 G0：`ExecDevice` / `ExecCpuPolicy`、可选 feature `cuda`（`cudarc` 占位初始化）。

| 类型 / 函数 | 说明 |
|-------------|------|
| `ExecutionContext` | 算例级 exec 上下文（设备、scatter 模式、步间 `ExecScratch`；`new` 返回 `Result`） |
| `ExecConfig` | `device`、`cpu_policy`、`compute_precision`、`scatter_mode`、…；`from_numerics` / `for_test_backend` |
| `ExecDevice` / `ExecCpuPolicy` / `ExecBackend` | 多 backend 配置（ADR 0017）；`parse_exec_backend` 解析 TOML |
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
| `exec::scatter::scatter_inviscid_pairs` | 无粘内面 scatter（`Serial` 或 `ParallelUnsafeAtomics` + rayon） |
| `exec::scatter::scatter_inviscid_pairs_f32` | 无粘内面 f32 scatter（`InviscidScatterOpF32`，无 Real 桥接） |
| `ViscousScatterOp` / `InviscidScatterOp` | 单面 scatter 贡献（discretization → exec 映射） |
| `exec::scatter::scatter_viscous_valid_slots_f32` | 粘性内面 f32 scatter（`ViscousScatterOpF32`，无 Real 桥接）；非结构 f32 粘性内面 `parallel-fvm` 着色桶路径 |
| `InviscidFluxF32` / `FaceNormalF32` | f32 无粘面通量与法向类型；Roe/HLLC/Van Leer/SLAU2 f32 核与 `face_inviscid_flux_*_soa_f32` 直接接受 `[f32; 3]` |
| `ViscousPhysicsConfig::face_transport_coefficients_f32` | f32 面输运系数（Sutherland/常数；含无量纲 \(1/\mathrm{Re}\) 缩放） |
| `viscous_flux_at_boundary_f32` / `ViscousBoundaryFluxParamsF32` | f32 粘性边界面通量；法向 `FaceNormalF32`（`[f32; 3]`） |

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
| `finalize_cell_dts_from_sigma_f32` | f32 版 `finalize_cell_dts_from_sigma`（`volumes` / `sigma` / CFL 均为 f32） |
| `cell_spectral_radius_unstructured_f32` | 非结构 f32 单元谱半径 \(\sigma_i\)，返回 `Vec<f32>`（面循环 f32；单元 f64 累加后落盘）；`backend=cuda` 时 prepare 步经 `ExecutionContext` CUDA 单元并行 kernel 优先，不可用时 CPU 串行 |
| `UnstructuredSpectralRadiusTyped` | typed 谱半径分发；`f32` 关联 `Sigma = Vec<f32>`，`f64` 为 `Vec<Real>` |
| `min_positive_dt_f32` | 从 f32 逐单元 \(\Delta t_i\) 缓冲取正最小值 |
| `euler_step_local_f32` / `rk4_step_local_f32` | 逐单元 \(\Delta t_i\) 的 Euler / RK4 显式推进（f32 场与残差） |
| `lu_sgs_common` | LU-SGS 双扫共用稳定化（线搜索、对角回退、正性限制）；f32 非结构扫掠热路径含 `residual_cell_vector_f32` / `conserved_vector_f32` / `stabilize_sweep_update_f32` |
| `LuSgsSweepUnstructuredF32Input` | 非结构 f32 LU-SGS sweep 输入（`dt` / `sigma` / `volumes` / `couplings` / `omega` / `gamma`） |
| `lu_sgs_sweep_unstructured_f32` | 非结构 f32 LU-SGS 双扫（`lusgs_couplings_f32`；source/耦合差分/正性限制全 f32）；CUDA：`lusgs_sweep_forward_color_f32` / `lusgs_sweep_backward_color_f32` 图着色 wavefront + host stabilize（串行 `lusgs_sweep_unstructured_serial_f32` 保留对照） |
| `compressible_unstructured_explicit_typed` | 非结构 typed 显式时间推进精度分发（`UnstructuredExplicitTimeAdvance`） |
| `StructuredComputeBackend` | 结构化 3D 可压缩 typed 热路径精度聚合（ADR 0019 S0；密封于 `f32`/`f64`） |
| `StructuredSpectralRadiusTyped` | 结构化 3D typed 谱半径分发（ADR 0019 S1-b） |
| `StructuredFaceCacheF32` | 结构化 3D 内面/单元体积 f32 预打包（ADR 0019 S1-a） |
| `StructuredTimestepBuffers` | 结构化 typed 谱半径 / 单元 \(\Delta t_i\) 缓冲（`sigma_f32` / `cell_dts_f32` + Real 镜像；ADR 0019 S1-c） |
| `StructuredSpectralTimestepPrepare` | 结构化 typed 谱半径与 Δt 准备（f32 原生缓冲；GMRES 仍读 Real 镜像） |
| `StructuredExplicitTimeAdvance` | 结构化 typed 显式 RK4/Euler（f32 LTS 走 `euler/rk4_step_local_f32`） |
| `StructuredLusgsDiagonalUpdate` | 结构化 typed LU-SGS 非扫掠对角更新（f32 用原生 \(\sigma,\Delta t_i\)） |
| `StructuredMultiblockInterfaceTyped` | 结构化 typed 多块 1-to-1 共享接口通量（f32 原生装配/scatter；ADR 0019 S2） |
| `InterfaceInviscidFlux` | 多块接口贡献通量存储（`Real` / `InviscidFluxF32` 枚举） |
| `run_multiblock_structured_typed_with_observer` | 多块结构化 typed 时间推进入口（`compute_precision` 分发） |
| `structured_prepare_timestep_typed` / `structured_explicit_typed` / `structured_lusgs_typed` | 结构化 typed 谱半径/Δt 准备、显式 RK4/Euler、LU-SGS 对角步（`pub(crate)` 子模块） |

理论：[adr/0019-structured-compute-backend.md](adr/0019-structured-compute-backend.md)（结构化 f32）；[adr/0018-unstructured-compute-backend.md](adr/0018-unstructured-compute-backend.md)（非结构 typed）。

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
