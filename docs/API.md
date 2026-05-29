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
    max_iterations: 100,
    tolerance: 1.0e-6,
});
let result = solver.run(&mesh)?;
```

### `asimu::config`

| 类型 | 说明 |
|------|------|
| `AppConfig` | 全局配置（solver + logging） |
| `SolverConfig` | `max_iterations`, `tolerance` |
| `LoggingConfig` | `level` |
| `Cli` | clap 命令行解析（CLI / 应用层使用） |
| `init_tracing(&str) -> Result<()>` | 初始化 tracing（应用层调用） |

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

#### Case 文件格式（临时）

```
name=<mesh_name>;cells=<count>
```

示例见 `tests/fixtures/demo.case`（v0.1 遗留）。v0.2 算例 TOML 见 [CASE_FORMAT.md](CASE_FORMAT.md) 与 `tests/benchmarks/`。

---

## v0.2 新增模块（骨架，semver 0.2.0 前可能调整）

| 模块 | 说明 | 状态 |
|------|------|------|
| [`asimu::core`](#asimucore) | `Real`、`CellId`、`approx_eq` | 骨架已实现 |
| [`asimu::field`](#asimufield) | `ScalarField` SoA | 骨架已实现 |
| [`asimu::linalg`](#asimulinalg) | `LinearSystem` | 骨架已实现 |
| [`asimu::discretization`](#asimudiscretization) | FVM 装配 | 占位函数 |
| [`asimu::solver::time`](#asimusolver) | `TimeIntegrator`、`SteadyStateIntegrator` | 骨架已实现 |

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

### `asimu::linalg`

| 类型 | 说明 |
|------|------|
| `LinearSystem` | 线性系统 RHS 占位（稀疏矩阵后续 PR） |

### `asimu::discretization`

| 函数 | 说明 |
|------|------|
| `assemble_diffusion_placeholder` | 尺寸校验 + RHS 清零占位 |

### `asimu::solver`（扩展）

| 类型 | 说明 |
|------|------|
| `SolverState` | 显式求解状态（伪步、物理时间、迭代） |
| `TimeMode` | `Steady` / `Transient` |
| `TimeStepInfo` | 单步推进摘要 |
| `TimeIntegrator` | 时间推进 trait |
| `SteadyStateIntegrator` | v0.2 稳态伪时间 |

理论参考：[docs/theory/fvm_diffusion.md](theory/fvm_diffusion.md)。

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
      --max-iterations <N>      最大迭代步数
      --tolerance <FLOAT>       收敛容差
      --log-level <LEVEL>       日志级别
  -h, --help                    帮助
```

环境变量：`ASIMU_CONFIG`、`ASIMU_MAX_ITERATIONS`、`ASIMU_TOLERANCE`、`ASIMU_LOG_LEVEL`。
