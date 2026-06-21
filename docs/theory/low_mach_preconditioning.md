# 低马赫预处理（可压缩非结构）

> 模块：`src/solver/compressible/unstructured_*` · 版本：v1.x · 状态：**P1–P2 已实现（CPU 非结构）；CUDA 待实现**

本文给出 asimu 在非结构可压缩路径上引入**低马赫预处理**（Low-Mach Preconditioning）的设计目标、数学形式与落地计划。背景见 [unstructured_fvm.md](unstructured_fvm.md)、[time_integration.md](time_integration.md)、[dual_time_stepping.md](dual_time_stepping.md)。

---

## 1. 问题与目标

### 1.1 为什么低马赫收敛慢

在密度基可压缩离散中，特征速度通常包含声速尺度 \(a\)。当 \(M=|u|/a\ll 1\) 时，声学特征远大于对流特征，导致：

1. 谱半径 \(\sigma\) 被声速主导，伪时间刚性增大；
2. 压力-速度耦合过“硬”，LU-SGS 更新方向对低马赫稳态问题不够友好；
3. `restart` 初场稍差时更容易出现内层/外层残差平台或反弹。

### 1.2 本功能目标

在不切换到压力基求解器的前提下，通过低马赫预处理：

- 改善 `Ma<=0.3` 区间稳态收敛效率；
- 保持 `Ma~O(1)` 以上时与现有可压缩离散一致（平滑退化）；
- 与现有 `lu_sgs`、`dual_time`、CPU/CUDA typed 路径兼容。

---

## 2. 数学形式（设计基线）

### 2.1 预处理思想

采用 Weiss-Smith / Turkel 一类“时间导数预处理”思路：在伪时间（或隐式对角近似）中缩放声学刚性，而非直接改动物理方程稳态解。

可用一个低马赫因子 \(\beta\in(0,1]\) 表示声学缩放：

\[
\beta^2 = \max\!\left(M_\text{loc}^2, M_\text{cut}^2\right),\quad
M_\text{loc}=\frac{|u|}{a}.
\tag{1}
\]

其中 \(M_\text{cut}\) 为下限（防止 \(\beta\to0\) 造成病态）。

### 2.2 对谱半径与伪时间步的影响

当前非结构 LTS 核心量可写为（示意）：

\[
\sigma_i \sim \frac{1}{V_i}\sum_{f\in\partial\Omega_i}
\left(|u_n| + a + C_v \Lambda_v\right)_f A_f.
\tag{2}
\]

低马赫预处理第一阶段建议改为“声速缩放”：

\[
a \;\Rightarrow\; a_p=\beta a,\qquad
\sigma_i^\text{LM} \sim \frac{1}{V_i}\sum_f
\left(|u_n| + a_p + C_v \Lambda_v\right)_f A_f.
\tag{3}
\]

随后伪时间步保持现有形式：

\[
\Delta\tau_i = \frac{\mathrm{CFL}}{\sigma_i^\text{LM}}.
\tag{4}
\]

> 备注：式 (3) 是工程上最小侵入的 P1 版本。更严格版本会在残差 Jacobian/预条件矩阵层面保持一致性（见 §5 分阶段计划）。

### 2.3 与 dual-time 的关系

对 dual-time 内层：

\[
R_\text{eff}(U)=R(U)+\text{storage}(U;\Delta t_\text{phys}),
\tag{5}
\]

低马赫预处理主要作用于伪时间隐式推进（\(\sigma,\Delta\tau\)、LU-SGS 对角项），不改动物理时间存储项定义。即：

- 物理时间精度（BDF1/BDF2）保持原语义；
- 内层收敛速度通过预处理改善。

---

## 3. 配置

在 `[time]` 增加：

```toml
[time]
low_mach_preconditioning = false
# low_mach_mach_cutoff = 0.1    # M_cut
# low_mach_max_mach = 0.3       # M_max：超过该值逐步退化到常规可压缩形式
# low_mach_jacobian = false   # true：块双扫 + 预处理 Roe 面 Jacobian（须 f64、first_order、lusgs_sweep 或 block/gmres 预条件）
```

字段语义：

| 字段 | 默认 | 说明 |
|------|------|------|
| `low_mach_preconditioning` | `false` | 是否启用低马赫预处理 |
| `low_mach_mach_cutoff` | `0.1` | 式 (1) 的 \(M_\text{cut}\) |
| `low_mach_max_mach` | `0.3` | 从低马赫模型向常规模型退化的上界 |
| `low_mach_blend` | `smooth` | 平滑退化（推荐）或硬阈值 |
| `low_mach_jacobian` | `false` | 启用预处理 Roe 面 Jacobian；`lu_sgs`+`lusgs_sweep` 时走块双扫 |

约束：

- 仅对可压缩 3D 非结构路径生效；
- `scheme = "lu_sgs"` 与 `scheme = "dual_time"` 优先支持；
- CPU f64/f32 已支持；CUDA 待实现。

### 3.1 退化策略（P2）

记 \(M_\text{loc}=|u|/a\)，预处理声速乘子 \(\beta_\text{lm}=\max(M_\text{loc},M_\text{cut})\)。

**`hard_cut`**：若 \(M_\text{loc}\ge M_\text{max}\)，取 \(\beta_\text{eff}=1\)（与常规可压缩谱半径一致）；否则 \(\beta_\text{eff}=\beta_\text{lm}\)。

**`smooth`**：若 \(M_\text{loc}\ge M_\text{max}\)，\(\beta_\text{eff}=1\)；若 \(M_\text{loc}\le M_\text{cut}\)，\(\beta_\text{eff}=\beta_\text{lm}\)；否则

\[
\beta_\text{eff}
= w\,\beta_\text{lm} + (1-w),\qquad
w=\frac{M_\text{max}-M_\text{loc}}{M_\text{max}-M_\text{cut}}.
\tag{6}
\]

面谱半径与 LU-SGS 扫掠共用 `LowMachPreconditioningConfig::sound_speed_multiplier`（`low_mach_face_spectral.rs` / `spectral_radius_f32.rs`）。

---

## 4. 代码映射（计划）

| 目标 | 主要位置 | 说明 |
|------|----------|------|
| 配置解析与校验 | `src/io/case.rs` / `src/case/validate.rs` | 新增 `[time]` 字段与约束 |
| 低马赫因子计算 | `src/solver/compressible/unstructured_prepare_timestep_typed.rs` | 基于局部 primitive 计算 \(\beta\) |
| 谱半径修改 | `UnstructuredSpectralRadiusAtPrepare` 实现 | 用 \(a_p=\beta a\) 替换声速项 |
| LU-SGS 扫掠/对角一致性 | `lu_sgs_sweep_unstructured_typed.rs` / `spectral_radius*.rs` | P2：扫掠 \(\lambda_{ij}\) 与 \(\sigma^\text{LM}\) 共用 \(\beta\) 缩放；对角分母已用 \(\sigma^\text{LM}\) |
| CUDA 同步 | `unstructured_cuda_prepare_f32.rs` + CUDA kernel 参数 | 将 \(\beta\)/等效声速下发 device |
| 日志与诊断 | 现有 `dual_time 伪时间步诊断` | 增加 `mach_min/max`、`beta_min/max` |

---

## 5. 分阶段落地

### P0（当前）

- 完成本文档；
- 明确配置、公式与验证口径。

### P1（最小可用）

- 仅改 \(\sigma,\Delta\tau\)（式 (3)(4)）；
- 支持 `lu_sgs` 稳态 CPU f64；
- 给出低马赫算例收敛对比（迭代数、残差降幅）。

### P2（一致性增强）— **已实现**

- LU-SGS 扫掠面耦合 \(\lambda_{ij}\) 与预处理谱半径一致（`face_spectral_radius_with_low_mach`）；
- 对角隐式分母已通过 \(\sigma^\text{LM}\) 与 \(\Delta\tau^\text{LM}\) 一致（P1）；
- **`low_mach_max_mach` / `low_mach_blend` 平滑或硬阈值退化**（`LowMachPreconditioningConfig::sound_speed_multiplier`）。

### P3（预处理特征速度）— **已实现**

- 面/单元 \(\lambda\) 由预处理 Riemann 声学特征速度 \(\lambda_\pm=\tfrac12(u_n\pm\sqrt{u_n^2+\beta^2 a^2})\) 导出（`side_preconditioned_hyperbolic_lambda`）；
- \(M\ge M_\text{max}\) 或 \(\beta\to1\) 时退化为常规 \(|u_n|+a\)。

### P4（块双扫 + 预处理 Jacobian）— **已实现（CPU f64 非结构）**

- `[time].low_mach_jacobian = true` 时 `block_lusgs` / GMRES 块预条件使用预处理 Roe \(|A|\)（`first_order_interior_flux_jacobian_with_low_mach`）；
- `scheme = "lu_sgs"` 且 `lusgs_sweep = true` 时以 block LU-SGS 双扫替代标量扫掠（`apply_lusgs_block_jacobian_sweep_f64`）。

### P5（dual-time / CUDA）

- 在 `dual_time` 内层启用同一预处理；
- 与 BDF1/BDF2、inner 诊断日志联动；
- 与 CPU 路径对齐，补齐 device 侧预处理参数与单测/回归。

---

## 6. 验证计划

### 6.1 数值验证

1. `Ma=0.1` 稳态圆柱/涡街预热 case：
   - 固定容差下迭代步数下降；
   - 同步比较 `inner/outer log10 residual` 曲线。
2. `Ma=0.3~0.5` 过渡区：
   - 检查预处理退化是否平滑。
3. `Ma>=0.8`：
   - 结果应近似回到原始可压缩路径。

### 6.2 物理一致性

- 稳态积分量（阻力、压力分布）与基线差异应在可接受范围；
- 非定常主频不应因预处理引入非物理漂移。

### 6.3 回归门槛（建议）

- 加入低马赫专用 probe matrix（可复用 `dual_time_inner_regression` 框架）；
- 比较启用/关闭预处理下的固定步数残差降幅。

---

## 7. 风险与边界

- 仅改谱半径但不改隐式线性化时，可能出现“加速有限或局部不一致”；
- 过小 `M_cut` 可能导致数值病态，过大则收益不足；
- 与湍流模型（后续 SST）耦合时需重新评估黏性尺度与预处理尺度。

---

## 8. 参考文献

1. Weiss, J. M., & Smith, W. A. (1995). *Preconditioning applied to variable and constant density flows*. AIAA Journal, 33(11), 2050–2057.
2. Turkel, E. (1999). *Preconditioning techniques in computational fluid dynamics*. Annual Review of Fluid Mechanics, 31, 385–416.
3. Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications* (3rd ed.). Elsevier.
4. Guillard, H., & Viozat, C. (1999). *On the behaviour of upwind schemes in the low Mach number limit*. Computers & Fluids, 28(1), 63–86.
