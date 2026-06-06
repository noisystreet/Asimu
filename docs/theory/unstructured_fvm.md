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
MUSCL 面值重构尚未接入非结构路径。

## 逆距离加权最小二乘梯度

`UnstructuredMesh3d` 上的单元中心梯度可用逆距离加权最小二乘估计。对单元 \(i\) 与样本点 \(m\)：

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
w_m = \frac{1}{|\Delta \mathbf x_m|}.
\tag{5}
\]

内部面样本取相邻单元中心。边界面样本取 ghost 状态，并放在面心关于 owner 单元中心的镜像点：

\[
\mathbf x_g = 2\mathbf x_f - \mathbf x_i .
\tag{6}
\]

实现中对 \(u,v,w,T\) 分别累加同一个 \(3\times3\) 对称正规方程；若局部样本退化导致矩阵奇异，则返回网格错误而不静默给出梯度。

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

其中 \(C_v=6\)，\(\nu=\mu/\rho\)，\(\alpha=\mu/(\rho Pr)\)。该形式与结构化路径使用同一个单面粘性谱半径贡献函数，差异只在 face 枚举方式。

对角 LU-SGS 复用已有伪时间更新：

\[
\Delta \mathbf U_i =
\frac{\omega\,\Delta t_i}{1+\Delta t_i\sigma_i}\mathbf R_i .
\tag{9}
\]

当 `lusgs_sweep = true` 时，非结构路径按 `CellId` 顺序定义下/上三角邻接并执行前/后扫：

\[
\Delta\mathbf U_i^{F} =
\frac{\omega\Delta t_i}{1+\Delta t_i\sigma_i}
\left(\mathbf R_i-\sum_{j<i}\frac{A_{ij}\lambda_{ij}}{|\Omega_i|}\Delta\mathbf U_j\right),
\tag{10}
\]

后扫对 \(j>i\) 的邻接项做同类修正，并使用 `lusgs_sweep_backward_damping` 阻尼。扫掠候选会经过正性检查；若全场线搜索仍失败，则回退到式 (9) 的对角更新。

## 实现映射

| 公式 | 实现 |
|------|------|
| (1) | `assemble_inviscid_residual_unstructured` |
| (2) | `FaceFluxInput::first_order` |
| (3) | `apply_compressible_boundary_conditions` + `UnstructuredMesh3d::face_geometry_3d` |
| (4)-(6) | `compute_unstructured_gradients_idw_lsq` |
| (7) | `compute_gradients_and_assemble_viscous_unstructured` |
| (8) | `cell_spectral_radius_unstructured` + `cell_local_dt_spectral` |
| (9) | `ConservedFields::assign_lusgs_diagonal_update` |
| (10) | `lu_sgs_sweep_unstructured` |

## 参考文献

- Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications*, 3rd ed. Elsevier. ISBN 978-0-08-099995-1.
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed. Springer. ISBN 978-3-540-25202-3.
- Mavriplis, D. J. (1997). Unstructured grid techniques. *Annual Review of Fluid Mechanics*, 29, 473-514. DOI: 10.1146/annurev.fluid.29.1.473.
