# 双时间步长（可压缩非结构）

> 模块：`src/solver/time/`、`src/solver/compressible/unstructured_*` · 版本：v1.x · 状态：**P0–P3b 已实现（CPU f32/f64 + CUDA f32）；P4 GMRES+DTS 规划**

本文描述 **3D 可压缩非结构 FVM** 上双时间步长（Dual Time Stepping, DTS）的数学形式、与现有伪时间/LU-SGS 的关系，以及分阶段落地计划。稳态/瞬态时间积分基线见 [time_integration.md](time_integration.md)；非结构残差与 LU-SGS 见 [unstructured_fvm.md](unstructured_fvm.md)。选型背景见 ADR [0005](../adr/0005-time-integration.md)、[0009](../adr/0009-compressible-navier-stokes.md)。

---

## 1. 背景与目标

### 1.1 已有能力（非结构可压缩）

| 能力 | 状态 | 说明 |
|------|------|------|
| 局部时间步 LTS | **已实现** | \(\Delta\tau_i=\mathrm{CFL}/\sigma_i\)，\(\sigma_i\) 为面求和谱半径（[unstructured_fvm.md](unstructured_fvm.md) 式 (8)） |
| 稳态伪时间 LU-SGS | **已实现** | 对角隐式（式 (9)）+ 可选 `CellId` 双扫（式 (10)） |
| 显式瞬态 RK4/Euler | **已实现** | 单层物理时间步，无内层迭代 |
| GMRES 隐式伪时间 | **未实现（非结构）** | 仅结构化 3D；`case::validate` 拒绝非结构 GMRES |
| 方向分裂残差光顺 | **未实现（非结构）** | 依赖结构化 i-j-k 线；非结构 case 忽略 `residual_smoothing` |
| f32 / f64 typed 非结构 LU-SGS | **已实现** | `run_unstructured_typed_with_observer::<T>`；`UnstructuredTimestepBuffers` 双缓冲 |
| CUDA f32 非结构 LU-SGS | **已实现（首版）** | ADR 0017；`backend = cuda` + `compute_precision = "f32"` |
| CUDA f32 非结构双时间步 | **已实现（P3b）** | `backend = cuda` + `scheme = dual_time` + `compute_precision = "f32"` |
| CUDA f32 非结构 LU-SGS 双扫 | **已实现** | `lusgs_sweep = true`；device 串行前/后扫 + host 线搜索 stabilize |

### 1.2 DTS 要补什么

**双时间步**引入两层时间尺度：

| 尺度 | 符号 | 用途 |
|------|------|------|
| 物理时间 | \(t,\ \Delta t_{\mathrm{phys}}\) | 真实瞬态 \(U^n\to U^{n+1}\) |
| 伪时间 | \(\tau,\ \Delta\tau_i\) | 每个物理步内迭代至 \(\mathbf{R}_{\mathrm{eff}}\) 收敛 |

asimu 非结构路径已实现 P0–P3b：物理时间**存储项**、**内外双层循环**、LU-SGS 分母 \(1/\Delta t_{\mathrm{phys}}\) 扩展，以及 CPU f32/f64 与 CUDA f32 typed 路径。仍待：

1. （可选）非结构 matrix-free GMRES + DTS（**P4**）；

**主要应用场景**：细网格/高马赫瞬态，外层可用较大 \(\Delta t_{\mathrm{phys}}\)（BDF1），内层伪时间 + LTS 收敛每个物理时刻。稳态加速已由 `scheme = "lu_sgs"` + `local_time_step = true` 覆盖，DTS 稳态扩展优先级较低。

---

## 2. 控制方程与离散

### 2.1 半离散形式

与 [time_integration.md](time_integration.md) 式 (1) 一致：

\[
\frac{\mathrm{d}\mathbf{U}_i}{\mathrm{d}t}
= \mathbf{R}_i(\mathbf{U}),
\qquad
\mathbf{R}_i = -\frac{1}{V_i}\sum_{f\in\partial\Omega_i}\hat{\mathbf{F}}_f\cdot\mathbf{S}_f.
\tag{1}
\]

`ConservedResidual` 存 \(\mathrm{d}\mathbf{U}/\mathrm{d}t\)。

### 2.2 物理时间 BDF1 + 伪时间迭代

对物理步 \(U^n\to U^{n+1}\)，Jameson / Blazek §6.2 一类双时间离散：

\[
\frac{U^{k+1}-U^n}{\Delta t_{\mathrm{phys}}}
+ \frac{U^{k+1}-U^k}{\Delta\tau}
+ \mathbf{R}(U^{k+1}) = 0.
\tag{2}
\]

等价伪时间右端（内层迭代用）：

\[
\frac{\partial U}{\partial\tau}
= -\mathbf{R}_{\mathrm{eff}}(U),
\qquad
\mathbf{R}_{\mathrm{eff}}(U)
= \mathbf{R}(U) + \frac{U-U^n}{\Delta t_{\mathrm{phys}}}.
\tag{3}
\]

**稳态**时去掉存储项，式 (3) 退化为 \(\partial U/\partial\tau=-\mathbf{R}(U)\)，与当前 LU-SGS 伪时间一致。

### 2.3 与空间残差 \(R_i=\mathrm dU_i/\mathrm dt\) 的量纲一致

式 (1) 中 \(\mathbf{R}_i\) 为单元平均守恒量的时间变化率（FVM 已除以 \(V_i\)）。BDF1 存储项写入残差为：

\[
\mathbf{R}_{\mathrm{eff},i}
= \mathbf{R}_i(\mathbf{U})
+ \frac{\mathbf{U}_i - \mathbf{U}^n_i}{\Delta t_{\mathrm{phys}}}.
\tag{4}
\]

**不再** 对存储项除 \(V_i\)：`ConservedFields` 存 \(\rho,\rho\mathbf u,E\) 等**单元平均**量，与 `ConservedResidual` 中 \(\mathrm dU/\mathrm dt\) 同量纲。实现：`solver/time/dual_time.rs::add_physical_storage_residual`。

### 2.4 隐式伪时间更新（LU-SGS 扩展）

现有对角更新（[unstructured_fvm.md](unstructured_fvm.md) 式 (9)）：

\[
\Delta\mathbf{U}_i
= \frac{\omega\,\Delta\tau_i}{1 + \Delta\tau_i\sigma_i}\mathbf{R}_i.
\tag{5}
\]

加入物理时间项后，线性化对角为：

\[
\Delta\mathbf{U}_i
= \frac{\omega\,\Delta\tau_i}{1 + \Delta\tau_i\sigma_i + \Delta\tau_i/\Delta t_{\mathrm{phys}}}\mathbf{R}_{\mathrm{eff},i}.
\tag{6}
\]

非结构双扫（式 (10)）中邻居耦合仍用标量 \(\lambda_{ij}\) 近似；存储项只改单元对角，与结构化路径一致。

### 2.5 时间步策略

| 参数 | 建议 | 代码入口（规划） |
|------|------|------------------|
| \(\Delta\tau_i\) | 伪时间 LTS：`CFL_{\mathrm{pseudo}}/\sigma_i` | `prepare_unstructured_timestep_typed` |
| \(\Delta t_{\mathrm{phys}}\) | `[time].dt` 或物理 CFL 估计（全场统一） | `DualTimeConfig::dt_phys` |
| 内层收敛 | \(\|\mathbf{R}_{\mathrm{eff}}\|_{\mathrm{rms}}\) 或 \(\log_{10}\) 阈值 | `inner_log10_tolerance` |

`local_time_step` **仅**作用于 \(\Delta\tau_i\)，不与 \(\Delta t_{\mathrm{phys}}\) 混用。

### 2.6 GMRES 扩展（可选，Phase 4）

结构化已有（[linear_gmres.md](linear_gmres.md) 式 (11)）：

\[
(D_{\Delta\tau}-J_R)\,\Delta U = R(U).
\tag{7}
\]

DTS 扩展为：

\[
(D_{\Delta\tau}+D_{\mathrm{phys}}-J_R)\,\Delta U = R_{\mathrm{eff}}(U),
\qquad
D_{\mathrm{phys},i}=\frac{1}{\Delta t_{\mathrm{phys}}}I.
\tag{8}
\]

BDF1 下存储项 Jacobian 对向量 \(v\) 的贡献为 \(v/\Delta t_{\mathrm{phys}}\)。预条件器分母（LU-SGS 对角）须同步加 \(1/\Delta t_{\mathrm{phys}}\)。

---

## 3. 架构与数据流

### 3.1 分层（ADR 0005）

```text
discretization   →  R(U)     （无粘 + 粘性，不变）
solver/time      →  R_eff, 内外循环, Δτ, Δt_phys
solver/compressible/unstructured_driver_typed → 编排 BC、梯度、正性、manifest
```

**禁止**在 `discretization` 热路径引入全局可变状态；\(U^n\) 冻结在 `DualTimeState` 中，每物理步初 `copy_from`。

### 3.2 内外循环

```text
for each physical_step (n → n+1):
    dual_state.snapshot_u_n(fields)          # 冻结 U^n
    for pseudo_k in 0 .. max_inner_steps:
        σ, Δτ ← prepare_unstructured_timestep (pseudo CFL)
        R ← assemble_rhs(U^k)
        R_eff ← R + storage(U^k, U^n, dt_phys)   # 式 (4)
        U^{k+1} ← lu_sgs_step_local(R_eff, Δτ, σ, inv_dt_phys)  # 式 (6)
        if inner_converged(R_eff): break
    if not inner_converged: 记录/可选减半 dt_phys
    t += dt_phys; n++
```

### 3.3 规划类型（`solver/time/dual_time.rs`）

```rust
/// 双时间步配置（Parse → Validate；无隐式全局状态）。
pub struct DualTimeConfig {
    pub dt_phys: Real,
    pub pseudo_cfl: Real,
    pub max_inner_steps: u32,
    pub inner_log10_tolerance: Option<Real>,
    pub local_pseudo_time_step: bool, // 默认 true
}

/// 每个物理步内可变；封装于 UnstructuredStepWorkTyped 扩展字段。
pub struct DualTimeState<T> {
    pub physical_step: u64,
    pub pseudo_step: u32,
    pub u_at_physical_level: ConservedFieldsT<T>,
}
```

纯函数示例（热路径零分配）：

```rust
fn add_physical_storage_residual<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    fields: &ConservedFieldsT<T>,
    u_at_level_n: &ConservedFieldsT<T>,
    volumes: &[T],
    dt_phys: T,
) -> Result<()>;
```

### 3.4 收敛判据

| 层级 | 监控量 | 用途 |
|------|--------|------|
| 内层 | \(\|R_{\mathrm{eff}}\|_{\mathrm{rms}}\) | 当前物理时刻是否解完 |
| 外层（瞬态） | 内层是否收敛 | 未收敛则不推进 \(t\) 或减小 \(\Delta t_{\mathrm{phys}}\) |
| Manifest | `inner_iterations`, `pseudo_residual_log10`, `physical_time` | V&V / 调试 |

扩展 `TransientStepControl` 或新增 `DualTimeStepControl`；**不**用 `tracing` 作为唯一状态存储。

### 3.5 f32 / f64 双精度与 GPU（硬约束）

DTS **不得**实现为仅 `f64` 的专用路径。自 **P0** 起所有新增热路径须泛型于 `T: ComputeFloat`，经 `run_unstructured_typed_with_observer::<T>` 单态化，与现有非结构可压缩 typed 岛一致（ADR [0016](../adr/0016-runtime-compute-precision.md) §编排边界、ADR [0018](../adr/0018-unstructured-compute-backend.md) `UnstructuredComputeBackend`）。

#### 3.5.1 精度分工（沿用 ADR 0016）

| 数据 / 运算 | `f64` | `f32` |
|-------------|-------|-------|
| 守恒场 \(U\)、\(U^n\)、残差 \(R\)、LU-SGS 增量 | `ConservedFieldsT<f64>` 等 | `ConservedFieldsT<f32>` |
| 几何（体积、法向、面心） | `f64` | `f64`（kernel 入口按需转换或预打包） |
| \(\Delta\tau_i\)、\(\sigma_i\) 缓冲 | `timestep.sigma` / `cell_dts` | `timestep.sigma_f32` / `cell_dts_f32` |
| \(\Delta t_{\mathrm{phys}}\)、CFL、收敛判据、`inner_tolerance` | `Real`（`f64`）编排 | 同左；写回 `T` 时显式转换 |
| 残差 RMS / manifest 归约 | `f64` 累加 | **`f64` 累加**，输入来自 `f32` 场 |
| 输出 / restart | `f64` 写出 | 计算 `f32` → 写出转 `f64`；**禁止跨精度 restart** |

存储项式 (4) 与 LU-SGS 分母式 (6) 在 `T` 上逐单元计算；**热路径内禁止** `if precision == F32` 分支（仅 case 编排边界允许 dispatch）。

#### 3.5.2 与 GPU 管线的衔接

非结构 CUDA 首版（`unstructured_cuda_prepare_f32`、`backend = cuda`）已在 LU-SGS 步内驻留 \(\sigma\)/守恒场/device RHS。DTS 扩展须：

1. **`U^n` 快照**：物理步初与 \(U^k\) 同精度 device 缓冲；内层迭代中 device 只读，避免每伪时间步 D2H/H2D 全量场；
2. **存储项**：优先在 device 上对 `residual` 做 \(\mathbf{R}_{\mathrm{eff}} \mathrel{+}= -(\mathbf{U}-\mathbf{U}^n)/\Delta t_{\mathrm{phys}}\)（**不除** \(V\)）；若步内暂缺 kernel，须在 validate 中明确降级（例如该组合报错），**不得**静默用 f64 算存储项再 cast 回 f32；
3. **`inv_dt_phys`**：标量 host 常数，传入 LU-SGS / 存储项 kernel 参数；
4. **步间同步**：复用 `cuda_reset_between_timesteps` / `mark_cuda_primitives_stale` 契约；内外循环边界（物理步末）再全量同步。

GPU 为 **f32 优先**后端；**f64 DTS 首版目标为 CPU 参考实现**，f64 CUDA 不在 P0–P3 范围。

#### 3.5.3 Validate 能力矩阵（已实现）

`case::validate` 在 `scheme = "dual_time"` 时校验：

| 组合 | 首版预期 |
|------|----------|
| `f64` + CPU | **必须**（P2–P3 基准） |
| `f32` + CPU | **必须**（与 f64 对照 smoke） |
| `f32` + CUDA | **已实现**（LU-SGS 对角或双扫内层；须 `local_time_step = true`） |
| `f64` + CUDA | 拒绝（与 ADR 0017 一致） |
| `dual_time` + 非结构 GMRES | P4 前拒绝 |

未实现组合在 Validate 阶段 **`Err`**，禁止静默回退 `f64`（ADR 0016 §迁移规则）。

#### 3.5.4 测试与容差

- **P0/P1**：`add_physical_storage_residual` / 对角更新对 `f32`、`f64` 各一组单元测试；
- **P2/P3**：复用 `f32_unstructured_lusgs_sweep_matches_f64_on_single_tet` 模式——同一网格上 f32 DTS 内层残差与 f64 相对误差在约定阈值内；
- **benchmark**：`expected.json` 以 **f64 为 reference**；f32 用相对误差/守恒阈值，**不得**复用 f64 的 `inner_tolerance` 默认值（f32 默认不低于 \(\sim 10^{-5}\) 量级，见 ADR 0016 §容差）；
- manifest 记录 `compute_precision` 与 `exec_device`，便于 GPU 回归对比。

### 3.6 边界条件

每伪时间步：`refresh_compressible_ghosts_and_primitives_typed` 随 \(U^k\) 更新。\(U^n\) **仅**进入式 (4) 存储项，不替代 BC ghost。

---

## 4. 非结构特有问题

### 4.1 计算成本

每伪时间步 = 1 次完整 RHS（无粘 + IDWLS 梯度 + 粘性）。缓解：

- 内层优先 LU-SGS sweep，而非 GMRES；
- 伪时间 CFL 可大于物理 CFL（常取 5–20）；
- `max_inner_steps` + `inner_tolerance` 早停；
- 内层未收敛时减半 `dt_phys` 或报错退出。

### 4.2 CellId 扫掠顺序

`lu_sgs_sweep_unstructured` 按 `CellId` 全序前/后扫，收敛率可能弱于结构化 i-j-k。**首版**复用现有顺序；**后续**可评估流方向重编号或图着色 Gauss-Seidel。

### 4.3 正性保持

复用 `assign_lusgs_diagonal_increment` 正性限制、sweep 失败回退对角、GMRES 路径 `max_physical_increment_scale` 线搜索。`f32` 正性下限须按 ADR 0016 放宽，与现有 LU-SGS typed 路径一致。

---

## 5. 实施阶段

| 阶段 | 内容 | 主要改动 | 精度 / GPU | 验证 |
|------|------|----------|------------|------|
| **P0** | 存储项 `ComputeFloat` 泛型 + 单元测试 | `solver/time/dual_time.rs`：`add_physical_storage_residual<T>` | **f32 + f64** 同 PR | 无网格手工向量，两精度各测 |
| **P1** | LU-SGS 分母加 `inv_dt_phys` | `assign_lusgs_diagonal_increment`、`lu_sgs_sweep_unstructured_*`（typed f32/f64） | **f32 + f64** | 对角/扫掠更新单测 |
| **P2** | 内外双层循环 | `unstructured_driver_typed.rs`、`DualTimeState<T>`、`scheme = "dual_time"` | **f32 + f64** CPU | 1-tet f64 + f32 smoke |
| **P3** | V&V 算例 | benchmark：涡对流、Sod 非结构 | **f64** reference + **f32** 相对阈值 | `expected.json` + README |
| **P3b** | CUDA f32 DTS | device `U^n` 快照 + 存储项 kernel | **f32 + CUDA** | `#[ignore=gpu]` smoke；`unstructured_dual_time_freestream/case_cuda_f32.toml` |
| **P4** | 非结构 GMRES + DTS | matrix-free RHS → unstructured；`Gmres` typed | **f64** 先；**f32** 随 GMRES typed 化 | 高马赫稳态（可选） |
| **P5** | 配置与 manifest | `io/case.rs`、`validate` 能力矩阵、`CompressibleStepInfo` | 记录 `compute_precision` / `exec_device` | `make check` |

### 5.1 模块衔接

| 模块 | 改动 |
|------|------|
| `discretization/.../residual` | **不改**通量；存储项在 solver 层 |
| `unstructured_prepare_timestep_typed` | 伪时间 CFL 复用（已有 f32/f64 缓冲） |
| `unstructured_lusgs_typed` | 传入 `inv_dt_phys`；f32/f64 精度分发 |
| `unstructured_cuda_prepare_f32` | P3b：**已实现** device 存储项 / `U^n` 缓冲 |
| `exec` / CUDA pipeline | P3b：**已实现** 存储项 kernel；步间 device 驻留契约 |
| `gmres_implicit_3d.rs` | P4：mesh-agnostic RHS + typed GMRES |
| `case/validate.rs` | `dual_time` × `compute_precision` × `backend` 矩阵 |
| `SolverState` / manifest | `physical_step`、`pseudo_step`、`inner_iterations`、`compute_precision` |

### 5.2 配置示例（`CASE_FORMAT`）

```toml
[numerics]
compute_precision = "f64"  # f64 | f32；DTS 须与 typed 路径一致（ADR 0016）
# backend = "cuda"       # cpu | cuda；cuda 仅 f32 + dual_time（P3b）

[time]
mode = "transient"
scheme = "dual_time"
dt = 1.0e-4                # Δt_phys
cfl = 5.0                  # 伪时间 CFL（内层）
local_time_step = true     # 伪时间 LTS（须 true）
max_steps = 500            # 物理步上限
max_inner_steps = 30
inner_tolerance = -3.0     # log₁₀(R_eff,rms) 阈值；f32 默认须放宽（ADR 0016）
# backend = "cpu"          # cpu | cuda；cuda 仅 f32 + dual_time（P3b）
# lusgs_sweep = false
# lusgs_omega = 1.0
```

与现有字段关系：

- `max_steps`：物理步计数（非伪时间步）；
- `tolerance`：可保留为外层稳态语义；瞬态 DTS 建议用 `inner_tolerance` 控制内层；
- `gmres`：P4 前非结构仍不可用。

### 5.3 验收标准（P3）

1. 均匀来流：**f64** 内层 1–3 步 \(\|R_{\mathrm{eff}}\|\) 降至阈值以下；**f32** CPU 与 f64 相对误差在 benchmark 阈值内；
2. 周期涡对流：DTS 与显式 RK4 在相同 \(\Delta t_{\mathrm{phys}}\) 下 L2 误差同级（f64 reference）；
3. Sod（非结构单层）：激波位置与 RK4 参考一致（f64；f32 用相对容差）；
4. manifest 含 `inner_iterations`、`pseudo_residual_log10`、`compute_precision`；
5. **P3b**（若合并发布）：`backend=cuda` + `compute_precision=f32` + `dual_time` GPU smoke 通过；
6. `make check` 全绿，无生产路径 `unwrap`；热路径无逐面精度分支。

---

## 6. 与不可压 DTS 的区分

ADR [0015](../adr/0015-incompressible-navier-stokes-simplec-piso.md) §7.5 规划了不可压双时间步（**不含声速项**，与 SIMPLEC 外层合并）。可压非结构 DTS：

- 主变量为守恒量 \(\mathbf{U}\)，含 \(|u_n|+a\) 谱半径；
- 不与 pressure-velocity 耦合共用 RK4 stage 缓冲；
- 共享 `TimeMode`、`TimeStepInfo` 字段名与 `solver::time` 框架（`CflSchedule`、`min_positive_dt`）。

---

## 7. 实现映射

| 式 / 步骤 | 代码位置 | 状态 |
|-----------|----------|------|
| (4) 存储项 | `solver/time/dual_time.rs::add_physical_storage_residual<T>`；CUDA：`cuda_add_physical_storage_residual_f32` | **已实现** |
| (6) 扩展 LU-SGS | `field` LUSGS 更新 + `lu_sgs_sweep_unstructured_*`（f32/f64）；CUDA 对角 `lusgs_diagonal_f32`、双扫 wavefront `lusgs_sweep_forward_color_f32` / `lusgs_sweep_backward_color_f32`（串行 `lusgs_sweep_unstructured_serial_f32` 对照） | **已实现** |
| 内外循环 | `unstructured_dual_time_typed.rs` | **已实现** |
| CUDA 存储项 / \(U^n\) | `unstructured_cuda_prepare_f32`、exec CUDA kernel | **已实现**（P3b） |
| (8) GMRES+DTS | `gmres_implicit_3d.rs` 抽象 + unstructured RHS | 规划（P4） |
| 配置解析 | `io/case.rs`、`DualTimeConfig::parse`、`validate` 精度矩阵 | **已实现** |

---

## 8. 参考文献

1. Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications* (3rd ed.). Elsevier. **§6.2** 隐式方法与双时间步；**§6.1.4**、**§9.1** 局部时间步。
2. Jameson, A. (1991). Time dependent calculations using multigrid, with applications to unsteady flows past airfoils and wings. *AIAA Paper 91-1596*.
3. Ferziger, J. H., Perić, M., & Street, R. L. (2020). *Computational Methods for Fluid Dynamics*. Springer. Ch. 6、Ch. 11.
4. asimu [time_integration.md](time_integration.md)、[unstructured_fvm.md](unstructured_fvm.md)、[linear_gmres.md](linear_gmres.md)；ADR [0005](../adr/0005-time-integration.md)、[0009](../adr/0009-compressible-navier-stokes.md)、[0016](../adr/0016-runtime-compute-precision.md)、[0018](../adr/0018-unstructured-compute-backend.md)。

---

## 9. 相关算例

| 算例 | 用途 |
|------|------|
| `tests/benchmarks/unstructured_dual_time_freestream/` | P3 CPU f64/f32 + P3b CUDA f32 内层收敛 smoke |
| `tests/benchmarks/unstructured_freestream/` | P2 单层 RHS 近零 |
| `tests/benchmarks/sod_1d/`（非结构 `nz=1` 扩展） | P3 激波对比 RK4（规划） |
| 新建 `tests/benchmarks/vortex_convection_unstructured/`（规划） | P3 周期涡对流 |
