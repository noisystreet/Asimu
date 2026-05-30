# asimu Case 文件格式（v0.2）

> 数据模型背景：[DATA_MODEL.md](DATA_MODEL.md) §8、§9 · I/O 实现：`src/io/`（v0.2 起逐步落地）

## 1. 概述

| 项 | v0.2 约定 |
|----|-----------|
| 格式 | **TOML**（`.toml` 或 `case.toml`） |
| 编码 | UTF-8 |
| v0.1 遗留 | `name=...;cells=...` 单行格式仍可读，**新算例请用 TOML** |

解析流程：**Parse → Validate → Trust**（见 AGENTS.md）。校验在 `io` 层完成；数值热路径信任已验证结构。

---

## 2. 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 算例名称 |
| `benchmark_id` | string | 否 | 对应 `tests/benchmarks/{id}/`；写入 Run Manifest（v0.3+） |
| `mesh` | table | 是 | 网格描述（§3） |
| `physics` | table | 是 | 物性（§4） |
| `boundary` | table | 是 | 边界条件（§5） |
| `initial` | table | 否 | 初始条件（§5.5）；缺省为全零 |
| `time` | table | 否 | 时间推进（§6）；默认 `mode = "steady"` |

---

## 3. `[mesh]`

v0.2 首版支持 **1D 结构化均匀网格**；2D 在 v0.2.x 后续 PR 扩展，字段预留如下。

### 3.1 1D（v0.2 必实现）

```toml
[mesh]
kind = "structured_1d"
cells = 32
length = 1.0          # 域长度 [m]，均匀划分
origin = 0.0          # 可选，默认 0.0
```

| 字段 | 类型 | 约束 |
|------|------|------|
| `kind` | string | 必须为 `structured_1d` |
| `cells` | integer | ≥ 1，≤ `io.limits.max_cells`（规划） |
| `length` | float | > 0 |
| `origin` | float | 可选 |

### 3.2 2D（v0.2.x 规划）

```toml
[mesh]
kind = "structured_2d"
nx = 32
ny = 32
lx = 1.0
ly = 1.0
```

### 3.3 外部 VTK VTS（v0.3.x，二进制 appended）

```toml
[mesh]
kind = "vts"
path = "mesh/cavity.vts"
```

| 项 | 约定 |
|----|------|
| 格式 | VTK XML StructuredGrid，**仅** `format="appended"` + base64 `AppendedData` |
| 不支持 | ASCII、inline binary、多 Piece |
| Feature | 库集成须启用 `io-vtk` |
| API | `asimu::io::load_vts(&path)` → `StructuredMesh`（2D/3D） |
| 写出 | `asimu::io::write_vts(&mesh, &path)`（Float64 appended，未压缩） |

详见 [adr/0007-vts-binary-io.md](adr/0007-vts-binary-io.md)。

### 3.4 外部 CGNS（feature `io-cgns-vts`）

```toml
[mesh]
kind = "cgns"
path = "mesh/wing.cgns"
zone = 1
```

| 项 | 约定 |
|----|------|
| 依赖 | 系统 `libcgns-dev`（`build.rs` 链接 `-lcgns`） |
| 支持 | Structured zone；ADF / HDF5 由 libcgns 处理 |
| 导出 | `export_cgns_zone_to_vts` 或 `make cgns-to-vts IN=... OUT=...` |

详见 [adr/0008-cgns-io.md](adr/0008-cgns-io.md)。

---

## 4. `[physics]`

v0.2 稳态对流-扩散 / 纯扩散：

```toml
[physics]
diffusivity = 1.0     # 分子扩散系数 D
# velocity = [1.0, 0.0]   # v0.2.x 对流项启用后
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `diffusivity` | float | ≥ 0 |
| `velocity` | float 数组 | 可选；2D 为 `[ux, uy]` |

---

## 5. `[boundary]`

v0.2 支持 **Dirichlet** 与 **Neumann**。键名为逻辑边界名（`left` / `right` / `bottom` / `top`）。

```toml
[boundary.left]
kind = "dirichlet"
value = 0.0

[boundary.right]
kind = "dirichlet"
value = 1.0
```

| `kind` | 字段 | 说明 |
|--------|------|------|
| `dirichlet` | `value` | 固定值 |
| `neumann` | `flux` | 法向通量（扩散问题为 `-D ∂φ/∂n`） |

1D 默认映射：`left` → 首端面，`right` → 末端面。

含 `[euler]` 段时，所有 `wall` patch 自动改为**无粘滑移壁**（`no_slip = false`），即使 CGNS 映射为有滑移壁。可用同名 `[boundary.<patch>]` 覆盖 CGNS patch 的其他参数，但 `[euler]` 仍会在解析末将壁面设为滑移。

| `wall` 字段 | 说明 |
|-------------|------|
| `no_slip` | `true` 无滑移（粘性）；`false` 滑移（无粘 Euler 默认） |
| `heat` | `adiabatic` / `isothermal` / `heat_flux` |

---

## 5.5 `[initial]`（可选）

v0.2 支持标量场初始条件。键名为场名（如 `phi`）。未声明时，求解器以**全零**场作为初值。

```toml
[initial.phi]
kind = "uniform"
value = 0.0

[initial.phi]
kind = "linear"
left = 0.0
right = 1.0

[initial.phi]
kind = "values"
data = [0.0, 0.25, 0.5, 0.75]
```

| `kind` | 字段 | 说明 |
|--------|------|------|
| `uniform` | `value` | 常值 |
| `linear` | `left`, `right` | 沿域长线性插值（单元中心） |
| `values` | `data` | 逐单元数组，长度 = `mesh.cells` |

API：`CaseSpec::build_initial_fields()` / `initial_scalar("phi")`。

---

## 6. `[time]`

见 ADR 0005、[DATA_MODEL.md](DATA_MODEL.md) §11。

```toml
[time]
mode = "steady"       # steady | transient
# scheme = "rk4"      # rk4（默认）| euler（一阶前向 Euler，排错对照）
# dt = 1.0e-3
# cfl = 0.4
# final_time = 0.2
max_steps = 1000      # 时间推进步数上限（稳态伪时间 / 瞬态物理时间共用）
# tolerance = -6.0    # 可选；log₁₀(RMS(ρ̇)) 阈值，满足则早停
```

| 字段 | 说明 |
|------|------|
| `mode` | `steady` 稳态伪时间推进；`transient` 瞬态物理时间推进 |
| `scheme` | 可选；`rk4`（默认）或 `euler`（一阶前向 Euler，用于排除 RK 多阶段 bug） |
| `local_time_step` | 可选；`true` 时逐单元 CFL 时间步（稳态加速，默认 `false` 为全局最小间距） |
| `max_steps` | 最大推进步数（稳态与瞬态共用，不再使用 `max_iterations`） |
| `cfl` | CFL 初值（3D Euler 时间步控制；默认 0.4） |
| `cfl_max` | 可选；CFL 终值，从 `cfl` 线性增至 `cfl_max` |
| `cfl_ramp_steps` | 可选；线性爬升步数（第 1 步…`cfl_ramp_steps`）；未设则在 `max_steps` 全程爬升；爬升结束后保持 `cfl_max` |
| `tolerance` | 可选；log₁₀(RMS(ρ̇)) 阈值，与 `max_steps` 成对用于残差早停 |
| `dt` | 固定时间步（设正数时覆盖 CFL 估算） |
| `final_time` | 可选物理终止时刻（Sod 等算例亦可在 `[sod]` 指定） |

含 `[sod]` 段时若省略 `[time]`，默认 `mode = "transient"`。

### 6.1 `[sod]`（Sod 激波管 benchmark）

```toml
[sod]
diaphragm = 0.5      # 间断位置（域坐标）
final_time = 0.2     # 物理终止时刻
cfl = 0.4            # CFL 数（固定 dt 时可在 [time] 指定 dt）

# 可选：无粘离散（省略时为一阶 Roe）
flux = "roe"              # roe | hllc
reconstruction = "muscl"  # first_order | muscl
limiter = "van_albada"    # minmod | van_leer | van_albada
```

| 字段 | 默认 | 说明 |
|------|------|------|
| `flux` | `roe`（一阶） | 指定 `roe`/`hllc` 且未写 `reconstruction` 时默认 **MUSCL** 重构 |
| `reconstruction` | 随 `flux` | `first_order` 强制一阶分段常数 |
| `limiter` | `minmod` | 仅 `reconstruction = "muscl"` 时生效 |

须配合 `structured_1d` 网格与 `[physics] gamma/gas_constant`。CLI：`asimu --case tests/benchmarks/sod_1d/case.toml`。

HLLC 变体示例：`tests/benchmarks/sod_1d/case_muscl_hllc.toml`。

---

## 7. 全局 `[solver]`（`config/default.toml`，非算例）

算例时间推进与收敛见 `[time].max_steps` / `[time].tolerance`。全局 `config/default.toml` 的 `[solver]` 仅保留 CLI 占位求解器步数：

```toml
[solver]
max_steps = 100
```

CLI：`--max-steps` / `ASIMU_MAX_STEPS`。

---

## 8. 完整示例（1D 扩散）

见 `tests/benchmarks/1d_diffusion_analytical/case.toml`。

---

## 9. 与 v0.1 占位格式迁移

| v0.1 | v0.2 TOML 等价 |
|------|----------------|
| `name=demo;cells=256` | `name = "demo"` + `[mesh] kind = "structured_1d" cells = 256 length = 1.0` |

`io::load_mesh_from_case` 在 v0.2 将检测扩展名 / 内容：`.toml` 走新解析器，遗留单行格式保持兼容至 v0.3。

---

## 10. 相关文档

- [BENCHMARKS.md](BENCHMARKS.md) — V&V 算例与 `expected.json`
- [theory/fvm_diffusion.md](theory/fvm_diffusion.md) — 扩散方程离散
- [SECURITY.md](../SECURITY.md) — 文件大小与路径限制
