# ADR 0012: 非结构二阶线性重构与梯度限制器（Barth–Jespersen / Venkatakrishnan）

- **状态**: 已接受（规划基线，M4 实现分阶段）
- **日期**: 2026-06-07
- **关联**: [ADR 0010](0010-unstructured-mixed-mesh.md)、[ADR 0009](0009-compressible-navier-stokes.md)、[unstructured_fvm.md](../theory/unstructured_fvm.md)、[interface_reconstruction.md](../theory/interface_reconstruction.md)

## 背景

[ADR 0010](0010-unstructured-mixed-mesh.md) M4 要求非结构路径接入二阶无粘面重构。结构化网格已实现 **宽模板 MUSCL** + **TVD 斜率限制器**（`SlopeLimiter`: minmod / van Leer / van Albada），作用对象为一维差分 \(d_\pm\)（见 `reconstruction.rs` / `flux_common::limited_slope`）。

非结构网格无 \(i\pm1\) 逻辑链，二阶格式通常采用 **IDWLS 单元梯度 + 面心线性外推 + 梯度限制器**。文献标准限制器为 **Barth–Jespersen (1989)** 与 **Venkatakrishnan (1993)**，与结构化 TVD MUSCL 限制器 **不是同一套函数**，不可将 TOML 中 `limiter = van_albada` 直接映射到非结构路径而不改语义。

本 ADR 定案：算法选型、配置边界、类型 API 与交付顺序。

## 决策

### 1. 术语与配置枚举

| 对外（case TOML） | 含义 |
|-------------------|------|
| `reconstruction = "first_order"` | 分段常数（式 (1)），`limiter` 忽略 |
| `reconstruction = "muscl"` | **非结构二阶线性重构**（梯度外推），**不**调用 `muscl_stencil_3d` |

非结构二阶路径使用 **独立限制器枚举**（规划名 `UnstructuredGradientLimiter`），**不**扩展 `SlopeLimiter`：

| 枚举值 | TOML 字符串 | 说明 |
|--------|-------------|------|
| `BarthJespersen` | `barth_jespersen` | M4 **首版默认**；非光滑、单调性保证强 |
| `Venkatakrishnan` | `venkatakrishnan` | M4 **第二子阶段**；光滑、平坦区少触发 |

**Parse → Validate**：`CaseMesh::Unstructured3d` 且 `reconstruction = muscl` 时：

- 接受 `unstructured_limiter`（或经 case 层映射的专用字段）为上述二者之一；
- 若用户仅填结构化字段 `limiter = minmod|van_leer|van_albada` 而未指定非结构限制器 → **校验错误**，错误信息指向本 ADR 与理论页（禁止静默 fallback）。

结构化网格继续使用 `InviscidFluxConfig.limiter: SlopeLimiter`，行为不变。

### 2. 重构公式（原始变量）

对单元 \(i\)、面心 \(\mathbf x_f\)、IDWLS 梯度 \(\nabla\phi_i\)（\(\phi\in\{\rho,u,v,w,p\}\)）：

**无限制外增量**（内部面 owner 侧示例）：

\[
\tilde\phi_i = \nabla\phi_i \cdot (\mathbf x_f - \mathbf x_i)
\tag{A}
\]

**限制后界面值**：

\[
\phi_f^- = \phi_i + \psi_i\,\tilde\phi_i
\tag{B}
\]

neighbor 侧同理得 \(\phi_f^+\)。边界面：owner 侧 (B)；ghost 侧 \(\phi_f^+ = \phi_\mathrm{ghost}\)（BC 给定，不外推）。

梯度由式 (4)–(6) 扩展至 \(\rho,p\)（与 \(u,v,w\) 同一 IDWLS 框架）；\(T\) 仍由 EOS 导出，**不**单独外推到面心。详见 [unstructured_fvm.md](../theory/unstructured_fvm.md)。

**极值样本集** \(\mathcal N_i\)：与单元 \(i\) 共享面的所有邻接单元中心值；边界面额外含 ghost 样本（与 IDWLS 边界样本一致）。

### 3. Barth–Jespersen 限制器

对每个原始变量分量独立计算 \(\psi_i\in[0,1]\)（**分量限制**，首版；标量限制为可选优化）。

对邻接样本 \(m\in\mathcal N_i\)，\(\tilde\phi_{i\to m}=\nabla\phi_i\cdot(\mathbf x_m-\mathbf x_i)\)：

\[
\psi_{i,m} =
\begin{cases}
\min\!\left(1,\dfrac{\phi_\mathrm{max}-\phi_i}{\tilde\phi_{i\to m}}\right) & \tilde\phi_{i\to m} > 0 \\[6pt]
\min\!\left(1,\dfrac{\phi_\mathrm{min}-\phi_i}{\tilde\phi_{i\to m}}\right) & \tilde\phi_{i\to m} < 0 \\[6pt]
1 & \tilde\phi_{i\to m} = 0
\end{cases}
\tag{BJ}
\]

\[
\psi_i = \min_{m\in\mathcal N_i} \psi_{i,m}
\tag{BJ2}
\]

\(\phi_\mathrm{max/min}\) 在 \(\{\phi_i\}\cup\{\phi_m\}_{m\in\mathcal N_i}\) 上取。实现映射：`limit_barth_jespersen_scalar` → `reconstruct_unstructured_face_primitives`。

**性质**：保证邻单元中心不出现新极值；在激波附近可能较耗散（限制频繁为 0）。

### 4. Venkatakrishnan 限制器

在 Barth–Jespersen 的邻接比基础上使用 **光滑限制函数**（Venkatakrishnan 1993, Eq. 4 常用形式）。对每个邻接样本 \(m\)：

\[
\xi_{i,m} = \frac{\phi_m - \phi_i}{2\,\tilde\phi_{i\to m}}
\quad (\tilde\phi_{i\to m}\neq 0;\ \text{否则}\ \psi_{i,m}=1)
\tag{V1}
\]

\[
\varphi(\xi) = \frac{\xi^2 + 2\xi}{\xi^2 + \xi + 2}
\tag{V2}
\]

\[
\psi_i = \min_{m\in\mathcal N_i} \varphi(\xi_{i,m})
\tag{V3}
\]

**性质**：\(\varphi\in[0,1]\)，在 \(\xi\to 0\) 时光滑趋于 0；平坦区比 BJ 更少「开关式」降为 0，工程上常用于减振。实现映射：`limit_venkatakrishnan_scalar`。

**不采纳**首版即实现 NIS / modified Venkatakrishnan；若后续 V&V 显示耗散过大，以 **ADR 修订段落** 追加。

### 5. 与结构化 `SlopeLimiter` 对照

| 项 | 结构化 MUSCL | 非结构 M4 |
|----|--------------|-----------|
| 限制对象 | 法向差分 \(d_\pm\) | 梯度外推 \(\nabla\phi\cdot\Delta\mathbf x\) |
| 类型 | `SlopeLimiter` | `UnstructuredGradientLimiter` |
| minmod / van Leer / van Albada | **已实现** | **不映射** |
| Barth–Jespersen | 不适用 | **M4.1** |
| Venkatakrishnan | 不适用 | **M4.2** |
| 共享下游 | `face_inviscid_flux` → Riemann | 同上 |

### 6. 模块与 API 边界

| 符号 | 模块 | 职责 |
|------|------|------|
| `UnstructuredGradientLimiter` | `discretization/flux_config` 或 `reconstruction_unstructured.rs` | BJ / V 枚举 + TOML 解析 |
| `limit_barth_jespersen` / `limit_venkatakrishnan` | `discretization/reconstruction`（或子模块） | 纯函数，无 mesh 隐式状态 |
| `reconstruct_unstructured_face_primitives` | `discretization/reconstruction` | 式 (A)(B) + 限制器 dispatch |
| `compute_unstructured_primitive_gradients_idw_lsq` | `discretization/gradient_unstructured` | 扩展 \(\nabla\rho,\nabla p\) |
| `assemble_inviscid_residual_unstructured` | `discretization/residual` | 面循环 + `face_inviscid_flux` |

`discretization` **不得**依赖 `solver`；梯度与限制器输入均由 `EvaluateRhsUnstructured` 显式传入。

### 7. 分阶段交付

| 阶段 | 内容 | 出口 |
|------|------|------|
| **M4.0** | 一阶 benchmark `unstructured_freestream` | case 可跑，\(\|\mathrm{RHS}\|\) 近零 |
| **M4.1** | IDWLS \(\nabla\rho,\nabla p\) + **Barth–Jespersen** + 二阶均匀来流 golden | `make check` |
| **M4.2** | **Venkatakrishnan** + 单元测试（光滑区 \(\psi\to 1\)、极值区 \(\psi<1\)） | `make check` |
| **M4.3** | case TOML `unstructured_limiter` 文档 + 与结构化 limiter 校验分离 | `docs/API.md`、理论页 |

### 8. 测试与 V&V

| 测试 | 说明 |
|------|------|
| 均匀来流 RHS | 一阶 / BJ / V 均应近零 |
| 限制器纯函数 |  manufactured \(\phi\)、梯度，验证 \(\psi\in[0,1]\) 与 BJ 极值条件 |
| 与缓存/着色路径 | 同 ADR 0011 golden |
| benchmark | `tests/benchmarks/unstructured_freestream/` README 引用本 ADR |

数值变更须同步 [unstructured_fvm.md](../theory/unstructured_fvm.md) 与 [CHANGELOG.md](../CHANGELOG.md)。

## 后果

### 正面

- 非结构二阶与文献及工业实践一致；
- 结构化 / 非结构 limiter 类型分离，避免错误复用 `van_albada` 语义；
- BJ 与 V 分阶段，先保证单调性再优化光滑区行为。

### 负面

- case 配置比结构化多一个字段（或更严格的校验）；
- 分量限制 BJ 在向量变量上可能略耗散（可接受的首版权衡）；
- Venkatakrishnan 推迟到 M4.2，首版仅有 BJ 可选。

## 未采纳

| 方案 | 原因 |
|------|------|
| 非结构复用 `SlopeLimiter` + 弦向 \(d_\pm\) | 非标准、不规则网格精度差；与 ADR 0009 结构化 MUSCL 语义混淆 |
| TOML `van_albada` 静默映射到 Venkatakrishnan | 名称不等价，V&V 不可追溯 |
| 首版 WENO / k-exact | 超出 M4 范围 |
| 标量 BJ（所有分量共用一个 \(\psi\)） | 首版优先分量限制；标量版可作为性能优化后续评估 |

## 参考文献

1. Barth, T. J., & Jespersen, D. C. (1989). The design and application of upwind schemes on unstructured meshes. *AIAA Paper 89-0366*.
2. Venkatakrishnan, V. (1993). On the accuracy of limiters and convergence to steady state solutions. *AIAA Paper 93-0880*; see also *Journal of Computational Physics* 107, 1–18.
3. Blazek, J. (2015). *Computational Fluid Dynamics*, 3rd ed. Elsevier. §8（非结构梯度与限制）。
4. Mavriplis, D. J. (1997). Unstructured grid techniques. *Annu. Rev. Fluid Mech.* 29, 473–514.

## 实现追踪

| 项 | 状态 |
|----|------|
| 理论页公式 (2a)–(2c) | 已写入 `unstructured_fvm.md` |
| `UnstructuredGradientLimiter` | 规划 |
| Barth–Jespersen | 规划（M4.1） |
| Venkatakrishnan | 规划（M4.2） |
| case 校验（禁止结构化 limiter 混用） | 规划 |

修订时 **不删除** 已有决策；新增限制器种类（如 NIS）须显式修订段落或新 ADR。
