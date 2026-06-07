# GMRES 与预条件器

> 模块：`src/linalg/` · 版本：v1.x · 状态：**已实现（矩阵无关 GMRES + CSR/ILU(0) + LU-SGS 对角预条件）**

## 1. 线性系统

隐式时间推进或 Newton-Krylov 线性化后得到

$$
A\,\delta x = b. \tag{1}
$$

`LinearOperator` 表示矩阵无关算子 \(y=Ax\)，可由 CSR 矩阵或未来的残差差分 \(Jv\) 实现。`Preconditioner` 表示左预条件器

$$
z = M^{-1}r. \tag{2}
$$

当前 GMRES 求解左预条件系统

$$
M^{-1}A\,\delta x = M^{-1}b. \tag{3}
$$

## 2. Restarted GMRES

给定初值 \(x_0\)，残差

$$
r_0 = b - A x_0,\qquad z_0=M^{-1}r_0,\qquad \beta=\lVert z_0\rVert_2. \tag{4}
$$

Arnoldi 过程构造 Krylov 子空间

$$
\mathcal{K}_m(M^{-1}A,z_0)
= \mathrm{span}\{z_0,(M^{-1}A)z_0,\dots,(M^{-1}A)^{m-1}z_0\}. \tag{5}
$$

每轮 restart 在该空间内最小化

$$
\min_y \left\lVert \beta e_1 - \bar{H}_m y \right\rVert_2,\qquad
x_m=x_0+V_m y. \tag{6}
$$

实现使用 Givens 旋转递推维护小型最小二乘问题，避免显式求正规方程。

## 3. LU-SGS 对角预条件器

对可压缩伪时间推进，现有对角 LU-SGS 更新可作为预条件器：

$$
M_i^{-1} r_i
= \omega\,\frac{\Delta t_i}{1+\Delta t_i\sigma_i}\,r_i. \tag{7}
$$

`LusgsDiagonalPreconditioner::from_lusgs_diagonal` 将每个单元的式 (7) 按守恒分量重复。该版本是对角预条件器；完整 sweep 预条件器后续应作为 `Preconditioner` 的 matrix-free 实现接入，而不是写入 GMRES 核心。

## 4. 单元块对角预条件器

对 matrix-free 3D 可压缩伪时间，可选构造每个单元的局部 \(5\times5\) 块：

$$
B_i[:,k] \approx D_{\Delta t,i} e_k
- \frac{R_i(U+\epsilon_k e_{i,k})-R_i(U)}{\epsilon_k},
\qquad k=1,\dots,5. \tag{8}
$$

其中 \(e_{i,k}\) 只扰动单元 \(i\) 的第 \(k\) 个守恒分量，\(R_i^{\mathrm{local}}\) 只重算该单元相邻内部面与边界面的无粘通量贡献，不调用全场 RHS。`CellBlockDiagonalPreconditioner` 存储 \(B_i^{-1}\)，应用时逐单元求 \(z_i=B_i^{-1}r_i\)。该预条件器比式 (7) 的标量对角更能捕捉密度、动量、能量耦合，构造成本为 \(O(5N)\) 个局部面通量差分；粘性与远邻 MUSCL 耦合仍由外层 matrix-free GMRES 算子处理。

## 5. ILU(0)

CSR 路径提供 ILU(0) 分解，保持原矩阵非零结构：

$$
A \approx LU,\qquad \mathrm{pattern}(L+U)=\mathrm{pattern}(A). \tag{9}
$$

预条件应用为两次三角求解：

$$
Ly=r,\qquad Uz=y. \tag{10}
$$

ILU(0) 适合已显式装配的稀疏 Jacobian 或扩散类线性系统。对可压缩 NS 的 matrix-free Jacobian，推荐先使用 LU-SGS 对角预条件器，等 Jacobian/块 CSR 装配稳定后再切换到 ILU。

## 6. 实现映射

| 公式 / 步骤 | 代码 |
|-------------|------|
| (1) 算子接口 | `LinearOperator` |
| (2)(3) 左预条件 | `Preconditioner`, `GmresSolver::solve` |
| (4) 初始残差 | `compute_preconditioned_residual` |
| (5)(6) Arnoldi + Givens | `GmresSolver::restart_cycle` |
| (7) LU-SGS 对角预条件 | `LusgsDiagonalPreconditioner` |
| (8) 单元块对角预条件 | `CellBlockDiagonalPreconditioner`, `GmresPreconditionerKind::CellBlockDiagonal` |
| (9)(10) ILU(0) | `Ilu0Preconditioner` |
| CSR 显式矩阵 | `CsrMatrix` |
| 可压缩 matrix-free 入口 | `CompressibleEulerSolver::solve_gmres_implicit_delta_3d` |

## 7. 可压缩残差的 Matrix-Free 线性化

3D 可压缩稳态伪时间入口求解

$$
\left(D_{\Delta t}-J_R\right)\Delta U = R(U),
\qquad
D_{\Delta t,i}=\frac{1}{\Delta t_i}I. \tag{11}
$$

其中 \(R(U)\) 是 `ConservedResidual`，\(J_R=\partial R/\partial U\)。算子不显式装配 Jacobian，而用有限差分近似

$$
J_R v \approx \frac{R(U+\epsilon v)-R(U)}{\epsilon}. \tag{12}
$$

因此 `LinearOperator::apply` 返回

$$
A v = D_{\Delta t} v - J_R v. \tag{13}
$$

左预条件器默认使用式 (7) 的 `LusgsDiagonalPreconditioner`；`[time] gmres_preconditioner = "cell_block_diagonal"` 时使用式 (8) 的局部无粘 Jacobian 块近似。`time.scheme = "gmres"` 时，3D 可压缩求解器会调用该入口；有限差分扰动 \(U+\epsilon v\) 与最终更新 \(\Delta U\) 都会按单元限制到正密度、正压力可行范围，并在线搜索确认后接受。显式 CSR 的 `Ilu0Preconditioner` 仍用于已装配矩阵问题；当前可压缩 matrix-free 路径不装配 CSR Jacobian，因此不使用 ILU(0)。

实现会把基础残差、预条件器构造、GMRES 线性求解等阶段耗时写入 `GmresImplicitDiagnostics::timing`，外层 `advance_gmres_step_3d` 再补充局部时间步、线搜索与整步总耗时日志；`log10_residual` 复用步初 `base_residual` 的 RMS。便于比较标量对角与块对角预条件器成本。

## 8. 参考文献

1. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM. Ch. 6（GMRES）、Ch. 10（预条件）。
2. Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications* (3rd ed.). Elsevier. §6.2（LU-SGS 隐式近似）。
3. Kelley, C. T. (1995). *Iterative Methods for Linear and Nonlinear Equations*. SIAM. Ch. 3–4.
