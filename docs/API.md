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
| `BoundaryMesh` | 逻辑边界名 → 面 ID |
| `MeshDiagnostics` | 坐标范围、间距统计、简单警告 |
| `structured_mesh_diagnostics(&StructuredMesh) -> MeshDiagnostics` | 2D/3D 结构化网格诊断 |
| `mesh1d_diagnostics` / `mesh3d_diagnostics` | 1D / 3D 专用诊断 |
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
| `load_mesh_from_case(&Path) -> Result<Mesh>` | 从占位 case 文件加载网格 |

#### VTK VTS 读入（feature `io-vtk`）

启用：`cargo build --features io-vtk`（`make check` 默认已启用）。

| 函数 / 类型 | 说明 |
|-------------|------|
| `load_vts(&Path) -> Result<VtsLoadResult>` | 读取 **二进制 appended** `.vts` |
| `VtsLoadResult` | `{ mesh: StructuredMesh }`（`D2` / `D3`） |
| `StructuredMesh2d` / `StructuredMesh3d` | 2D/3D 结构化网格（`nx`, `ny`[, `nz`]，节点坐标） |

| `write_vts(&StructuredMesh, &Path) -> Result<()>` | 写出 appended 二进制 VTS |

**不支持**：ASCII VTS、非 zlib 压缩器、`.vtu`。见 [adr/0007-vts-binary-io.md](adr/0007-vts-binary-io.md)。

#### CGNS 读入与 VTS 导出（feature `io-cgns-vts`）

需系统安装 `libcgns-dev`。

| 函数 / 类型 | 说明 |
|-------------|------|
| `list_cgns_zones(&Path) -> Result<Vec<CgnsZoneInfo>>` | 列出全部 structured zone |
| `load_cgns_zone(&Path, zone_index) -> Result<CgnsLoadResult>` | 读取单 zone |
| `export_cgns_zone_to_vts(input, zone, output) -> Result<CgnsLoadResult>` | CGNS zone → VTS |

见 [adr/0008-cgns-io.md](adr/0008-cgns-io.md)。

#### 网格诊断报告

| 函数 / 类型 | 说明 |
|-------------|------|
| `MeshReport` | 可读摘要（`Display`）：范围、间距、BC patch |
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
| [`asimu::field`](#asimufield) | `ScalarField` SoA | 骨架已实现 |
| [`asimu::linalg`](#asimulinalg) | `LinearSystem` 三对角 + Thomas | v0.2 已实现 |
| [`asimu::discretization`](#asimudiscretization) | FVM 扩散装配 + BC 施加 | v0.2 1D 已实现 |
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

### `asimu::discretization`

| 函数 | 说明 |
|------|------|
| `assemble_diffusion_1d` | 1D 内部面扩散装配 |
| `apply_boundary_conditions` | 按 patch 顺序施加 BC |
| `apply_dirichlet_face` / `apply_neumann` | 单面 BC |
| `assemble_diffusion_placeholder` | 尺寸校验 + RHS 清零占位 |

### `asimu::solver`（扩展）

| 类型 | 说明 |
|------|------|
| `SolverState` | 显式求解状态（伪步、物理时间、迭代） |
| `TimeMode` | `Steady` / `Transient` |
| `TimeStepInfo` | 单步推进摘要 |
| `TimeIntegrator` | 时间推进 trait |
| `SteadyStateIntegrator` | v0.2 稳态伪时间 |
| `GmresImplicitConfig` / `GmresImplicitDelta` | 3D 可压缩 matrix-free 隐式 GMRES 更新入口 |

理论参考：[docs/theory/fvm_diffusion.md](theory/fvm_diffusion.md)。

### `asimu::physics`（可压缩 v1.x）

理论：[docs/theory/nondimensional.md](theory/nondimensional.md)（无量纲）；[adr/0009-compressible-navier-stokes.md](adr/0009-compressible-navier-stokes.md)

| 类型 / 函数 | 说明 |
|-------------|------|
| `IdealGasEoS` | \(\gamma, R\)；`freestream_primitive`（有量纲来流） |
| `FreestreamParams` | `[freestream]` 对应 Mach、\(p,T\)、方向 |
| `ReferenceScales` | 参考量与 Re；`from_freestream` |
| `FreestreamContext` | **来流单一入口**：`primitive` / `conserved` / `density_from_pressure_temperature` |
| `FreestreamMode` | `Dimensional` / `Nondimensional` |
| `ViscousPhysicsConfig` | Sutherland/常数粘度；`static_temperature`（式 (1)(2)） |
| `ConservedFields::from_freestream_context` | 初场（经 `CaseSpec::build_conserved_fields`） |
| `ConservedFields::to_dimensional` | 输出还原 SI |
| `CaseSpec::is_nondimensional` | `reference.is_some()` |
| `CaseSpec::dimensional_eos` | 输出用有量纲 EOS |

BC：`apply_compressible_boundary_conditions(..., &FreestreamContext, ...)` — 见 [theory/nondimensional.md §4](theory/nondimensional.md#4-边界条件)。

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
