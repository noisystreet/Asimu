# 非结构有限体积面循环

本文记录 `UnstructuredMesh3d` 上首版可压缩 Euler 求解的离散假设。

## 控制方程

无粘可压缩 Euler 方程写为

\[
\frac{\partial \mathbf U}{\partial t} + \nabla\cdot \mathbf F(\mathbf U)=0 .
\]

对非结构控制体 \(\Omega_i\) 积分并使用面求和：

\[
\frac{d\mathbf U_i}{dt}
= -\frac{1}{|\Omega_i|}\sum_{f\in\partial\Omega_i}
\hat{\mathbf F}_{f}\, A_f .
\tag{1}
\]

其中 \(A_f\) 为面面积，\(\hat{\mathbf F}_f\) 是沿 owner 单元外法向的数值通量。
内部面同时给 owner 与 neighbor 累加等量反号贡献；边界面只给 owner 累加。

## 几何与拓扑假设

- 网格为 3D 线性 tet / hex / pyramid / prism 混合单元。
- `mesh` 构造期按排序后的全局节点集合合并面；一个面最多允许两个相邻单元。
- `FaceMetric.normal` 对 owner 单元外向；面循环不再依赖结构化 \(i/j/k\)。
- 当前只支持节点集合完全一致的同型面合并；quad-tri conformal 接口仍需网格预处理或后续拓扑扩展。

## 空间离散

### 一阶（已实现）

首版非结构求解使用一阶分段常数重构：

\[
\mathbf U_f^- = \mathbf U_\mathrm{owner}, \qquad
\mathbf U_f^+ = \mathbf U_\mathrm{neighbor}
\tag{2}
\]

边界面以边界条件生成 ghost / exterior 状态：

\[
\mathbf U_f^+ = \mathbf U_\mathrm{ghost}(\mathbf U_\mathrm{owner}, \mathbf n_f, \mathrm{BC}) .
\tag{3}
\]

式 (2) 与式 (3) 之后复用结构化路径已有 Riemann / FVS 通量，包括 Roe、HLLC、Van Leer、Hanel-Van Leer 与 SLAU2。

### 二阶线性重构（M4 规划，**未实现**）

> **术语**：文献中通常称 **gradient-based linear reconstruction** 或 **slope-limited linear reconstruction**，而非独立的「Unstructured MUSCL scheme」。asimu case 配置仍用 `reconstruction = "muscl"` 与结构化路径对齐；实现上走 **IDWLS 梯度外推 + 斜率限制**，**不**复用结构化宽模板 `muscl_stencil_3d`（无 \(i\pm1\) 逻辑链）。

#### 与结构化 MUSCL 的区别

| 项 | 结构化 `muscl_stencil_3d` | 非结构 M4 |
|----|---------------------------|-----------|
| 模板 | 沿面法向 4 点宽模板 | 单元中心 + IDWLS 梯度 |
| 梯度 | 模板差分隐含 | 式 (4)–(6) 显式 \(\nabla\phi_i\) |
| 限制器 | 对法向差分 \(d_\pm\) 做 minmod / van Albada 等（`SlopeLimiter`） | **Barth–Jespersen**（M4.1）/ **Venkatakrishnan**（M4.2）；见 [ADR 0012](../adr/0012-unstructured-gradient-limiters.md) |
| 共享 | `FaceFluxInput` → `face_inviscid_flux` → Riemann | 同上 |

#### 面心原始变量外推

记面心 \(\mathbf x_f\)、单元中心 \(\mathbf x_i\)、IDWLS 原始变量梯度 \(\nabla\phi_i\)（\(\phi\in\{\rho,u,v,w,p\}\)）。内部面 owner \(i\) / neighbor \(j\)：

\[
\tilde\phi_i = \nabla\phi_i \cdot (\mathbf x_f - \mathbf x_i), \qquad
\tilde\phi_j = \nabla\phi_j \cdot (\mathbf x_f - \mathbf x_j)
\tag{2a}
\]

Barth–Jespersen 标量限制因子 \(\psi_i\in[0,1]\)（**首版按原始变量分量**独立限制；对单元 \(i\) 所有邻接样本 \(m\) 取最小）。公式与 Venkatakrishnan 光滑限制器见 [ADR 0012](../adr/0012-unstructured-gradient-limiters.md) 式 (BJ)–(V3)：

\[
\psi_i = \min_m \begin{cases}
\dfrac{\phi_\mathrm{max}-\phi_i}{\tilde\phi_{i\to m}} & \tilde\phi_{i\to m} > 0 \\
\dfrac{\phi_\mathrm{min}-\phi_i}{\tilde\phi_{i\to m}} & \tilde\phi_{i\to m} < 0 \\
1 & \text{otherwise}
\end{cases}
\tag{2b}
\]

其中 \(\phi_\mathrm{max/min}\) 为单元 \(i\) 与邻接单元（及边界面 ghost 样本）上的极值；\(\tilde\phi_{i\to m}=\nabla\phi_i\cdot(\mathbf x_m-\mathbf x_i)\)。限制后面状态：

\[
\phi_f^- = \phi_i + \psi_i\,\tilde\phi_i, \qquad
\phi_f^+ = \phi_j + \psi_j\,\tilde\phi_j .
\tag{2c}
\]

边界面：owner 侧按式 (2c) 左态外推；ghost 侧取 BC 给出的 \(\phi_\mathrm{ghost}\)（**不**对 ghost 做外推）。再经 `interface_conserved_pair` 转守恒态并调用现有 Riemann。

**退化**：\(|\Omega_i|\to 0\)、LSQ 奇异、或 \(\psi=0\) 时退化为式 (2) 一阶。

#### 梯度场扩展

当前 IDWLS 仅输出 \(\nabla u,\nabla v,\nabla w,\nabla T\)（粘性路径）。M4 需对 **\(\rho,p\)** 同样累加 RHS 并求解，建议扩展为 `PrimitiveGradientFields`（或在 `GradientFields` 上增列，实现时二选一）。温度仍由 EOS 从 \((\rho,p)\) 导出，**不**单独外推 \(T\) 到面心。

#### 几何缓存

`UnstructuredInteriorFace` / `UnstructuredBoundaryFace` 需缓存面心 \(\mathbf x_f\) 与 owner/neighbor 单元中心 \(\mathbf x_i,\mathbf x_j\)（或 \(\Delta\mathbf x_{i\to f}=\mathbf x_f-\mathbf x_i\)），避免面循环重复查询 `mesh.face_metric`。数值与逐步读 mesh 等价。

#### 验收标准

| 测试 | 判据 |
|------|------|
| 均匀来流（一阶 / 二阶） | \(\|\mathrm{RHS}\|\) 近零 |
| 着色 / 并行 / 缓存路径 | 与线性面序 golden 一致（见 ADR 0011） |
| `tests/benchmarks/unstructured_freestream/` | case 可跑 + manifest `benchmark_id` |

## 逆距离平方加权最小二乘梯度

`UnstructuredMesh3d` 上的单元中心梯度可用逆距离平方加权最小二乘（WLS，对标 SU2 `WEIGHTED_LEAST_SQUARES`）估计。对单元 \(i\) 与样本点 \(m\)：

\[
\Delta \mathbf x_m = \mathbf x_m - \mathbf x_i,\qquad
\Delta \phi_m = \phi_m - \phi_i .
\tag{4}
\]

梯度 \(\nabla\phi_i\) 由下式确定：

\[
\nabla\phi_i =
\arg\min_{\mathbf g}
\sum_m w_m(\mathbf g\cdot\Delta \mathbf x_m-\Delta\phi_m)^2,
\qquad
w_m = \frac{1}{|\Delta \mathbf x_m|^2}.
\tag{5}
\]

内部面样本取相邻单元中心。边界面样本取 **面心** \(\mathbf x_f\) 与 ghost/边界状态 \(\phi_f\)（由 BC 在面心给出，对标 SU2 边界邻点）：

\[
\Delta \mathbf x_m = \mathbf x_f - \mathbf x_i,\qquad
\Delta \phi_m = \phi_f - \phi_i .
\tag{6}
\]

实现中对 \(u,v,w,T\) 分别累加同一个 \(3\times3\) 对称正规方程；若局部样本退化导致矩阵奇异，则返回网格错误而不静默给出梯度。

将式 (5) 写成对称正规方程 \(A_i\mathbf g=\mathbf b_i\)：

\[
A_i = \sum_m w_m\,\Delta\mathbf x_m\,\Delta\mathbf x_m^{\mathsf T},
\qquad
\mathbf b_i = \sum_m w_m\,\Delta\phi_m\,\Delta\mathbf x_m .
\tag{5a}
\]

其中 \(A_i\) 仅依赖网格几何与样本位置（内部面为相邻单元中心，边界面为面心），\(\mathbf b_i\) 随场变量每步变化。

## 面拓扑与 IDWLS 几何预计算

非结构粘性梯度与粘性通量面循环在每步重复遍历全部面。网格与边界 patch 在求解器 work 区初始化一次后不变，因此将几何固定部分预计算并缓存：

1. **面拓扑** `UnstructuredFaceTopology`：内部面记录 owner/neighbor、面积、法向、两侧体积及 IDWLS 样本 \((\Delta\mathbf x_m, w_m)\)；边界面记录 `FaceId`、owner、度量、壁面粘性类别与边界样本权重。
2. **IDWLS 几何矩阵** `LsqPrecomputedCell`：对每个单元累加式 (5a) 中的 \(A_i\)（6 个独立对称分量），存入 `UnstructuredSolverMeshCache::lsq_geometry`。

每步梯度计算仅：

- 由当前原始变量与 ghost 状态累加 \(\mathbf b_u,\mathbf b_v,\mathbf b_w,\mathbf b_T\)；
- 用预计算 \(A_i\) 求解 \(\nabla u,\nabla v,\nabla w,\nabla T\)。

粘性通量装配复用同一 `face_topology`，避免重复查询 `mesh.face_owner` / `face_neighbor` 与面度量。数值结果与逐步从 `mesh` 枚举面的旧路径等价；差异仅为热路径上的分配与索引开销。

## 粘性通量

非结构 Navier-Stokes 首版复用结构化路径的 Newtonian 应力与 Fourier 热传导通量：

\[
\mathbf F_v\cdot\mathbf n =
\begin{bmatrix}
0 \\
\boldsymbol\tau\cdot\mathbf n \\
-(\lambda\nabla T\cdot\mathbf n + \mathbf u\cdot\boldsymbol\tau\cdot\mathbf n)
\end{bmatrix}.
\tag{7}
\]

内部面使用 owner / neighbor 两侧的原始变量与 IDWLS 梯度算术平均。边界面使用 ghost 原始变量；壁面会用 owner 到 ghost 的法向差分修正速度与温度梯度，并支持绝热、等温与给定热通量壁面。残差装配仍遵循式 (1) 的面循环符号约定，粘性动量项在装配前转换为 \(+\nabla\cdot\boldsymbol\tau\) 的右端贡献。

## 本地时间步与 LU-SGS 更新

非结构局部谱半径使用面求和；Navier-Stokes 会叠加粘性/热传导抛物型项：

\[
\sigma_i = \frac{1}{|\Omega_i|}
\sum_{f\in\partial\Omega_i} (|u_n| + a)_f A_f
+ C_v\sum_{f\in\partial\Omega_i}
\max(\nu_i,\alpha_i)\frac{A_f^2}{|\Omega_i|^2},
\qquad
\Delta t_i = \frac{\mathrm{CFL}}{\sigma_i}.
\tag{8}
\]

其中 \(C_v=6\)，\(\nu=\mu/\rho\)，\(\alpha=\mu/(\rho Pr)\)。该形式与结构化路径使用同一个单面粘性谱半径贡献函数。`parallel-fvm` 下复用 `LsqRhsCellIncidence` 做单元并行累加 \(\sigma_i\)（每单元只写自身分量）；未启用时仍按单元串行遍历缓存面拓扑，不再逐步枚举裸 `mesh` 面索引。

对角 LU-SGS 复用已有伪时间更新：

\[
\Delta \mathbf U_i =
\frac{\omega\,\Delta t_i}{1+\Delta t_i\sigma_i}\mathbf R_i .
\tag{9}
\]

**残差监控（`log10_residual`）**：所有时间积分路径（显式 Euler/RK4、LU-SGS、GMRES）统一取步初 \(\|R(U^0)\|\)（`storage.k1` 的 RMS），不再步末重算 RHS。显式 RK4/Euler 的 stage1 与 LU-SGS 的 `lusgs_rhs` 已写入 `k1`；GMRES 复用隐式线性化阶段的 `base_residual`。避免重复 IDWLS/装配，且相邻步因 \(U^0\) 已更新而监控曲线仍反映收敛趋势。

当 `lusgs_sweep = true` 时，非结构路径按 `CellId` 顺序定义下/上三角邻接并执行前/后扫：

\[
\Delta\mathbf U_i^{F} =
\frac{\omega\Delta t_i}{1+\Delta t_i\sigma_i}
\left(\mathbf R_i-\sum_{j<i}\frac{A_{ij}\lambda_{ij}}{|\Omega_i|}\Delta\mathbf U_j\right),
\tag{10}
\]

后扫对 \(j>i\) 的邻接项做同类修正，并使用 `lusgs_sweep_backward_damping` 阻尼。扫掠候选会经过正性检查；若全场线搜索仍失败，则回退到式 (9) 的对角更新。

**双时间步（规划）**：瞬态隐式路径在式 (9)(10) 分母加 \(\Delta\tau_i/\Delta t_{\mathrm{phys}}\)，并在 RHS 叠加存储项 \((\mathbf{U}-\mathbf{U}^n)/\Delta t_{\mathrm{phys}}\)（**不除** \(V_i\)，与 \(\mathrm dU/\mathrm dt\) 同量纲）；内外循环与配置见 [dual_time_stepping.md](dual_time_stepping.md)。

## 内面并行 scatter（面着色）

粘性/无粘内面装配对每个面执行 \(\mathbf R_i \mathrel{+}= s_i\,\mathbf f_f\)。若两线程同时更新共享单元，会产生数据竞争。标准做法是 **面着色（graph coloring）**：

- 将内面划分为颜色桶 \(C_0,\ldots,C_{K-1}\)；
- 同一桶内任意两面不共享 owner/neighbor 单元；
- 桶内可并行 compute；**scatter 调度粒度 = 着色桶**（每桶一次 `exec::scatter`，ADR 0013）。

`UnstructuredSolverMeshCache` 在网格初始化时对 `face_topology.interior` 做贪心着色，结果存于 `InteriorFaceColoring::buckets`。

启用 Cargo feature `parallel-fvm`（**默认开启**）时：

| 阶段 | 模式 | 说明 |
|------|------|------|
| flux compute | 桶内 `rayon` | 各色 bucket **串行**，桶内 `par_iter`（勿 bucket 间并行，见 CHANGELOG P4 回归） |
| scatter | `ExecutionContext` + `ScatterMode::Auto` | 大网格（内面 ≥ 65536）→ `ParallelUnsafeAtomics` 桶内 atomic；小网格 / 小桶 → 串行 |
| 契约 | 每色桶 **1 次** scatter | SIMD 粘性路径禁止 per-batch 外层 scatter（E5 回归） |

**P8** 起各色 bucket 内 compute 后立即 scatter（span `unstructured_*_interior_flux_fused`）。桶级 scatter 细节 span 为 **`exec_colored_bucket_scatter`（trace 级）**；Chrome trace 默认 `{log_level},asimu::exec::scatter=trace`（见 [OBSERVABILITY.md](../OBSERVABILITY.md)）。

### 粘性 transport 系数并行（P3）

非恒定粘度（如 Sutherland）路径在装配内面粘性通量前，需逐单元计算 \(\mu(T)\)、\(\lambda(T)\) 并平均到面心。`parallel-fvm` 下：

1. `fill_cell_transport_coefficients`：`cell_mu` / `cell_lambda` 上 `par_iter_mut` + 温度切片；
2. `fill_face_transport_coefficients`：内面 `par_iter_mut` 与 `interior.par_iter()` zip，只读单元系数、写面系数（无写冲突）。

### 面心预平均 SoA（P7）

IDWLS 梯度完成后、粘性 flux 阶段前，对非 `simd-fvm` 路径预计算面心 \(0.5(\mathbf u_O+\mathbf u_N)\) 与 12 个速度/温度梯度分量，写入 `ViscousFaceAveragedSoA`（`fill_face_averaged_viscous_soa`）：

| 阶段 | span | 说明 |
|------|------|------|
| 填充 | `unstructured_viscous_face_avg` | 仅 **非** `simd-fvm`；`parallel-fvm` 下 `lanes.par_iter_mut().zip(interior.par_iter())` |
| flux | `unstructured_viscous_interior_flux` | 非 SIMD：顺序读 `face_averaged.lane(i)`；**P7b** SIMD full_batch：cell SoA gather → batch4 flux，跳过面缓冲 |

`fused_interior_viscous_face_flux_averaged` 与 owner/neighbor cell gather 路径在面心预平均假设下数值一致；**P9** 起 `simd-fvm` full_batch 用 `fused_interior_viscous_face_flux_batch4_from_soa` 在 f64x4 内完成 gather + 平均 + τ·n（不物化 `ViscousFaceGather4`）；remainder 仍 cell 直读。

### IDWLS RHS 单元并行（P0）

IDWLS 梯度每步需对每个单元累加最小二乘右端项 \(b_i\)。与 flux scatter 不同，**每单元只写自身** `scratch.bu[i]` 等分量，无需面着色即可安全并行：

1. 网格初始化时构建 `LsqRhsCellIncidence`（每单元作为 owner/neighbor 的内面列表 + 边界面）。
2. `parallel-fvm` 下对 `bu/bv/bw/bt`（粘性）或 `br/bp/bu/bv/bw`（二阶无粘）做 `par_iter_mut().enumerate()`，每单元遍历关联面并累加。
3. 未启用 `parallel-fvm` 时仍用面循环串行路径；golden 测试 `parallel_idw_lsq_accumulate_matches_face_serial` 对齐两路径。

### CPU SIMD（P5，`simd-fvm`）

启用可选 feature `simd-fvm`（依赖 `wide`，**默认关闭**）时，`exec::cpu` 提供标量等价 SIMD 路径：

| 算子 | 实现 | 说明 |
|------|------|------|
| LU-SGS 对角更新 | `assign_lusgs_diagonal_update` | 5 个 SoA 场 f64x4 AXPY |
| 粘性内面 flux | `fused_interior_viscous_face_flux_batch4` | 着色桶内四路 gather + f64x4 τ·n |
| IDWLS 梯度求解 | `solve_symmetric_3x3_batch4` | 四单元一批 Cramer 求解 |
| Roe 一阶通量 | `face_inviscid_flux_first_order_roe_batch4` | 着色桶内四路 gather + f64x4 特征值修正；`assemble_inviscid_residual_unstructured` 一阶 Roe 路径 |
| Hanel–Van Leer 一阶通量 | `face_inviscid_flux_first_order_hanel_batch4` | 着色桶内四路 gather；面坐标系标量变换 + 亚音速 \(F_E^+=F_m^+\cdot h\) 四路 f64x4；`assemble_inviscid_residual_unstructured` 一阶 HVL 路径 |

与 `parallel-fvm` 叠加：桶间 `rayon`、桶内 SIMD。验证：`make test-simd-fvm`。

### 着色桶面批 SoA 预处理（init-time）

`UnstructuredFaceTopology` 着色完成后，对每个桶预建 `InteriorFaceBucketBatchLayout`：

| 字段 | 内容 |
|------|------|
| `full_batches` | 四路对齐的 `InteriorFaceBatchStatic4`（owner/neighbor、法向、面积、体积、RHS scale） |
| `remainder` | 桶尾不足 4 面的面索引（标量回退） |

μ/λ 与原始变量/梯度仍每步从 SoA 场 gather；静态几何在 mesh cache 初始化时写入，热路径不再重复读 `UnstructuredInteriorFace` 做 lane 填充。

无粘内面 scatter（P6-1）：`InteriorInviscidScatterGeom` 携带预存 `owner_scale` / `neighbor_scale`（init-time 写入 `UnstructuredInteriorFace`）；`scatter_fused_interior_inviscid_face` 直接写残差 SoA 切片，与粘性 `scatter_fused_interior_viscous_face` 同模式。

一阶无粘 flux（P6-2）：`face_inviscid_flux_first_order_interior_soa` 从 `PrimitiveFields` SoA 直读；FVS（Van Leer / HVL / SLAU2）仅组装守恒态，跳过 `FaceFluxInput` 与界面 struct 拷贝。

一阶边界面（P6-4）：有 `face_topology` 时走 `face_topology.boundary` 缓存（法向、面积、体积、`owner_rhs_scale`）；`scatter_fused_boundary_inviscid_face` 写 owner 残差；ghost 侧 FVS 直接用守恒态。

## 实现映射

| 公式 | 实现 | 状态 |
|------|------|------|
| (1) | `assemble_inviscid_residual_unstructured` | 已实现 |
| (2) | `FaceFluxInput::first_order` | 已实现 |
| (2a)–(2c) | `reconstruct_unstructured_face_primitives` + `UnstructuredGradientLimiter`（规划） | **M4**；限制器 ADR [0012](../adr/0012-unstructured-gradient-limiters.md) |
| (3) | `apply_compressible_boundary_conditions` + 边界面 ghost | 已实现 |
| (4)-(6) | `compute_unstructured_gradients_idw_lsq` | 已实现（\(u,v,w,T\)） |
| \(\nabla\rho,\nabla p\) | 扩展 IDWLS RHS（`gradient_unstructured`） | **M4** |
| (5a) 几何预计算 | `UnstructuredSolverMeshCache::from_mesh` | 已实现 |
| 面心 / 单元中心偏移 | `UnstructuredInteriorFace` 增字段（规划） | **M4** |
| 面拓扑缓存 | `UnstructuredFaceTopology`（`unstructured_face_cache`） | 已实现 |
| 内面着色 | `InteriorFaceColoring` / `color_interior_faces` | 已实现 |
| 桶内面批静态 SoA | `InteriorFaceBucketBatchLayout` / `InteriorFaceBatchStatic4` | 已实现 |
| (7) | `compute_gradients_and_assemble_viscous_unstructured` | 已实现 |
| (8) | `cell_spectral_radius_unstructured` + `cell_local_dt_spectral` | 已实现 |
| 谱半径单元并行 | `LsqRhsCellIncidence` + `parallel-fvm` 单元 `rayon` | **P2** |
| 粘性 transport 并行 | `fill_cell/face_transport_coefficients` | **P3** |
| 面心预平均 SoA | `ViscousFaceAveragedSoA` / `fill_face_averaged_viscous_soa`（非 SIMD） | **P7** |
| SIMD 直通 flux | `gather_viscous_face_batch4` + `fused_interior_viscous_face_flux_batch4`（跳过 face_avg） | **P7b** |
| 桶内 compute+scatter 融合 | `unstructured_*_interior_flux_fused`（P8，无整桶 flux Vec） | **P8** |
| 粘性桶级 flat buffer | 每桶 batch×4 固定槽 + `mem::take` 并行写入（P8′，无 per-batch Vec） | **P8′** |
| 粘性 SoA 融合 batch4 | `fused_interior_viscous_face_flux_batch4_from_soa`（P9，无 `ViscousFaceGather4`） | **P9** |
| 无粘/粘性桶内并行 | 各色 bucket 串行、桶内 `rayon`（`with_min_len=1024`） | **P0 POC** |
| CPU SIMD 热算子 | `exec::cpu` + feature `simd-fvm` | **P5** |
| (9) | `ConservedFields::assign_lusgs_diagonal_update` | 已实现 |
| (10) | `lu_sgs_sweep_unstructured` | 已实现 |
| DTS 存储项 + 内外循环 | [dual_time_stepping.md](dual_time_stepping.md) §5 | 规划 |

### M4 实现分步（建议 PR 顺序）

1. **benchmark 骨架**：`tests/benchmarks/unstructured_freestream/`（一阶均匀来流，先验收盘）。
2. **梯度扩展**：IDWLS 增 \(\rho,p\)；`EvaluateRhsUnstructured` 在 `Muscl` 时先算梯度。
3. **面重构**：`reconstruct_unstructured_face_primitives` + 内部/边界面调用；移除 case 层 `first_order` 硬拒绝。
4. **golden**：二阶均匀来流近零；缓存/着色/并行路径与一阶相同约束。
5. **文档 / API**：`docs/API.md`、`CHANGELOG.md`、本页状态改为「二阶已实现」。

调用链（规划）：

```text
EvaluateRhsUnstructured
  → [Muscl] compute_unstructured_primitive_gradients_idw_lsq
  → assemble_inviscid_residual_unstructured(face_topology, gradients)
       → reconstruct_unstructured_face_primitives (式 2a–2c)
       → face_inviscid_flux → Riemann
```

## 参考文献

- Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications*, 3rd ed. Elsevier. ISBN 978-0-08-099995-1.（非结构梯度重构 §8）
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed. Springer. ISBN 978-3-540-25202-3.
- Mavriplis, D. J. (1997). Unstructured grid techniques. *Annual Review of Fluid Mechanics*, 29, 473-514. DOI: 10.1146/annurev.fluid.29.1.473.
- Barth, T. J., & Jespersen, D. C. (1989). AIAA 89-0366 — Barth–Jespersen 限制器；Venkatakrishnan, V. (1993). AIAA 93-0880 — 光滑限制器；选型见 [ADR 0012](../adr/0012-unstructured-gradient-limiters.md).
