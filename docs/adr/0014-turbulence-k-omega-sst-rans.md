# ADR 0014: 可压 RANS 湍流闭包（Menter k-ω SST）

- **状态**: 已接受（规划基线，实现分阶段 T0–T5）
- **日期**: 2026-06-08
- **关联**: [ADR 0009](0009-compressible-navier-stokes.md)、[ADR 0010](0010-unstructured-mixed-mesh.md)、[ADR 0011](0011-parallel-fvm-face-coloring.md)、[nondimensional.md](../theory/nondimensional.md)、[boundary_conditions.md](../theory/boundary_conditions.md)、[turbulence_k_omega_sst.md](../theory/turbulence_k_omega_sst.md)

## 背景

[ADR 0009](0009-compressible-navier-stokes.md) 定案可压 NS 以 **层流 Sutherland 粘性** 首版；湍流标注为「SA / k-ω 单独 ADR」。asimu 已具备：

- 3D FVM 可压 NS（结构 / 非结构）、无粘 + 层流粘性、LU-SGS / RK4；
- `ViscousPhysicsConfig::face_transport_coefficients(μ, λ)` — 粘性通量仅依赖面心 **标量** \(\mu,\lambda\)；
- Case / BC：`turbulent_inlet` 已解析 `turbulent_k` / `turbulent_omega`，但 ghost 与 `inlet` 相同，**未** 施加于湍流场。

工程算例（圆柱 Mach 8、dual_ellipsoid 等）以 **外气动 RANS** 为目标。纯 Wilcox k-ω 在逆压梯度 / 分离区偏弱；**Menter k-ω SST** 为工业界外气动常用闭包，且与现有 `turbulent_k` / `turbulent_omega` schema 兼容。

SST 相对纯 k-ω **额外依赖壁面距离** \(y\)（混合函数 \(F_1,F_2\)）与 **交叉扩散项**；非结构混合网格上 \(y\) 需单独预计算。本 ADR 定案模型、模块边界、Case 枚举与 **T0–T5** 交付顺序。

## 决策

### 1. 首版 RANS 闭包：Menter k-ω SST（2003 可压常用形式）

| 项 | 定案 |
|----|------|
| 对外模型名 | **`k_omega_sst`**（Case TOML）；**不** 单独暴露纯 `wilcox_k_omega` 枚举 |
| 涡粘 | Boussinesq：\(\mu_t = \rho a_1 k / \max(a_1\omega,\, S F_2)\)（\(S\) 为应变率模，见理论页） |
| NS 耦合 | \(\mu_{\mathrm{eff}} = \mu_{\mathrm{lam}}(T) + \mu_t\)，经现有 `face_transport_coefficients` 注入 |
| 备选模型 | Spalart–Allmaras 等经 `TurbulenceModel` trait **T5+** 扩展，非 v1.x 阻塞项 |

**不采纳**首版 k-ε 或纯 Wilcox k-ω 作为对外默认：SST 在 \(F_1\equiv 1\) 极限下退化为 k-ω，可用同一套代码路径分阶段验证。

### 2. 模块职责（遵守 ADR 0002 分层）

```
physics/turbulence/     # TurbulenceModel trait、SST 常数、μ_t、源项闭包（无网格遍历）
field/turbulence.rs     # TurbulenceFields SoA: k, omega
mesh/wall_distance.rs   # 壁距场 y（SST 专用，Parse 后预计算）
discretization/turbulence/  # k/ω 输运 FVM 装配、壁面 BC、P_k
solver/                 # 与 NS 同 LU-SGS 分裂扫；湍流隐式源项进对角
```

| 禁止 | 原因 |
|------|------|
| 在 `solver` 内写 SST 源项公式 | ADR 0009 §10 |
| `io` 解析时注册全局湍流状态 | AGENTS 隐式状态 |
| 首版 GMRES 7×7 全耦合块 | 实现成本；T2–T4 用分裂 LU-SGS |

**扩展点**：

```rust
/// 湍流闭包（physics 层，无 mesh 引用）。
pub trait TurbulenceModel {
    fn eddy_viscosity(&self, k: Real, omega: Real, rho: Real, s: Real, f2: Real) -> Real;
    fn source_k(&self, /* ... */) -> Real;
    fn source_omega(&self, /* ... */) -> Real;
    fn cross_diffusion(&self, /* ... */) -> Real;
}
```

首版具体类型：`MenterKOmegaSst`（`src/physics/turbulence/sst.rs`）。

### 3. Case 配置

```toml
[physics.turbulence]
enabled = true
model = "k_omega_sst"    # 校验：仅允许此值（v1.x）；未启用时省略整表

[physics.turbulence.k_omega_sst]
# 缺省 = Menter 2003 常数集；非缺省须文档化
sigma_k1 = 0.85
sigma_k2 = 1.0
sigma_omega1 = 0.5
sigma_omega2 = 0.856
beta1 = 0.075
beta2 = 0.0828
beta_star = 0.09
a1 = 0.31
k_floor = 1.0e-12
omega_floor = 1.0e-6

[physics.turbulence.wall_distance]
method = "brute_force"   # v1.x 首版：壁面 patch 节点/面心 BFS；结构化可选 "structured_normal"
```

**Parse → Validate**：

- `enabled = true` 且 `model = k_omega_sst` → 必须 `[navier_stokes]` / 粘性已开；
- `turbulent_inlet` patch 必须提供 `turbulent_k`、`turbulent_omega`；
- 无量纲算例：\(k^*,\omega^*\) 缩放见 [nondimensional.md](../theory/nondimensional.md)（实现 T3 时扩展 §）。

`BoundaryKind::TurbulentInlet` **保留**现有字段；数值施加在 T3 修复（Dirichlet \(k,\omega\) + 现有 `inlet_ghost` 求 \(\mathbf U\)）。

### 4. 壁面距离 \(y\)

SST 混合函数 \(F_1 = \tanh(\arg_1^4)\)，\(\arg_1\) 依赖 **到最近壁面的距离** \(y\)（Menter 1994；见理论页）。

| 网格 | v1.x 策略 |
|------|-----------|
| 结构化 3D | 沿壁面法向 / 索引距离（优先 T1.5） |
| 非结构混合 | ** brute-force BFS**：壁面 patch 面心种子，沿内面拓扑多源最短距离（T4 前必须完成） |

\(y\) 存入 `WallDistanceField`（`field` 或 `mesh` 缓存），**求解前一次计算**；随 restart 不必单独持久化（可由 mesh + BC 重建）。

**退化模式**（仅 golden / 调试）：`F1 = 1` 全域 → 纯 k-ω 极限；**生产路径默认关闭**。

### 5. 时间推进与 LU-SGS

| 项 | 定案 |
|----|------|
| 与 NS 关系 | **分裂 LU-SGS**：单步内先/后扫 NS 与 \(k,\omega\) 标量方程（顺序见理论页） |
| 隐式源项 | \(\omega\) 销毁项等 stiff 部分 **对角隐式** 加入 LU-SGS 分母 |
| 谱半径 | \(k,\omega\) 对流 + 扩散特征速度并入 `cell_spectral_radius_*`（Blazek §10 类比） |
| GMRES | T2–T4 **不**扩展 7×7 块；T5 评估 |

显式 RK 路径可保留用于湍流单元测试，**生产默认 LU-SGS**（与现有可压算例一致）。

### 6. 分阶段交付（T0–T5）

| 阶段 | 交付 | 网格 | 验证 |
|:----:|------|------|------|
| **T0** | 本 ADR + [turbulence_k_omega_sst.md](../theory/turbulence_k_omega_sst.md) + DATA_MODEL / CASE_FORMAT 草案 | — | — |
| **T1** | 常数 \(\mu_t\) case 钩子 → `mu_eff`（无输运方程） | 结构 3D | 层流 vs 固定 \(\mu_t\) 平板 |
| **T1.5** | `WallDistanceField` + 结构化壁距 | 结构 3D | 壁距场 golden |
| **T2** | \(k,\omega\) 输运 + \(F_1\equiv 1\) 退化 k-ω；\(\mu_t\) 耦合 | 结构 3D | 湍流平板 \(C_f\)、\(u^+\) |
| **T3** | 完整 SST（\(F_1,F_2\) + 交叉扩散）；壁面 / `turbulent_inlet` / farfield BC | 结构 3D | 圆柱低 Re、通道 \(Re_\tau\) |
| **T4** | 非结构 FVM + `parallel-fvm`；复用 IDWLS 梯度 | 非结构 | dual_ellipsoid 降 Ma/Re |
| **T5** | Restart 含 \(k,\omega\)；VTU 写出；可选 SA trait | 两者 | manifest 回归 |

**POC 不以加速比为合入条件**；每阶段须 golden / benchmark 通过方可进入下一阶段。

### 7. V&V 算例（`tests/benchmarks/`）

| ID | 阶段 | 验证量 |
|----|:----:|--------|
| `flat_plate_turbulent` | T2–T3 | \(C_f(x)\)、\(u^+(y^+)\) |
| `channel_re_tau_395` | T3 | 平均 \(u^+\) |
| `cylinder_low_re_turb` | T3 | \(C_D\)、分离点 |
| `dual_ellipsoid_rans` | T4 | 工程回归（降 Re） |

登记见 [BENCHMARKS.md](../BENCHMARKS.md)（实现时追加）。

### 8. 与 ADR 0009 修订

[ADR 0009](0009-compressible-navier-stokes.md) §7 数值基线「湍流 | 层流首版；SA / k-ω 单独 ADR」→ **本 ADR 定案 SST**；SA 降为 T5 可选。

## 后果

### 正面

- 外气动 RANS 与工业实践对齐；分离 / 再附着优于纯 k-ω
- \(\mu_t\) 经现有粘性链注入，NS 装配改动面小
- `turbulent_inlet` schema 无需新字段
- \(F_1\equiv 1\) 退化路径降低 T2 调试风险

### 负面

- 壁距场 + 交叉扩散增加 T1.5 / T3 工作量
- 非结构 \(y\) 在复杂几何上可能昂贵（brute-force 可接受 v1.x，远期 ADR 优化）
- 湍流 + LU-SGS 隐式源项增加谱半径与对角装配复杂度
- SST 常数与可压修正文献版本多，须在理论页锁定引用表

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| 首版纯 Wilcox k-ω | 远场 / 分离区弱；后期必迁 SST，重复 V&V |
| 首版 Spalart–Allmaras | Case 已押 k-ω；SA 作 T5 trait 扩展更合适 |
| 首版 k-ε | 近壁 stiff；外气动少选 |
| 湍流与 NS 全耦合 Newton–GMRES | 实现成本高；显式/分裂 LU-SGS 先验证 |
| 无壁距 SST（\(F_1\) 常数） | 失去 SST 核心优势；仅作 T2 中间态 |

## 修订记录

| 日期 | 内容 |
|------|------|
| 2026-06-08 | 初版：定案 Menter k-ω SST、模块边界、T0–T5 |

修订时 **不删除** 已有条目；变更模型族或时间耦合策略须新开 ADR 或修订段落。
