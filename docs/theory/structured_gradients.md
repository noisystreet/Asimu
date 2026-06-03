# 结构化网格梯度

> 模块：`src/discretization/gradient.rs` · 版本：v1.x · 状态：**已实现**

## 1. 算法

粘性通量需要速度分量与温度的笛卡尔梯度：

$$
\nabla \phi = \left(\frac{\partial\phi}{\partial x},
\frac{\partial\phi}{\partial y},
\frac{\partial\phi}{\partial z}\right)^T,\quad
\phi \in \{u,v,w,T\}.
\tag{1}
$$

对结构化单元 \((i,j,k)\)，沿逻辑方向 \(m\in\{i,j,k\}\) 构造物理空间差分：

$$
\Delta \phi_m = \phi_m^+ - \phi_m^-,
\quad
\Delta \mathbf{x}_m = \mathbf{x}_m^+ - \mathbf{x}_m^-.
\tag{2}
$$

假设单元局部梯度常值，则有：

$$
\Delta \phi_m = \Delta \mathbf{x}_m \cdot \nabla\phi.
\tag{3}
$$

三个逻辑方向组成 \(3\times3\) 线性系统：

$$
\begin{bmatrix}
\Delta \mathbf{x}_i^T\\
\Delta \mathbf{x}_j^T\\
\Delta \mathbf{x}_k^T
\end{bmatrix}
\nabla\phi =
\begin{bmatrix}
\Delta\phi_i\\
\Delta\phi_j\\
\Delta\phi_k
\end{bmatrix}.
\tag{4}
$$

实现中用三重积显式求解式 (4)。若三方向退化导致行列式接近零，返回网格错误。

## 2. 离散化

- 网格假设：`StructuredMesh3d`，物理坐标来自单元中心 `cell_metric(...).center`。
- 内部单元：两侧都有邻居时使用中心差分。
- 边界单元：若边界 ghost 存在，用 ghost 镜像点与内部邻居构造差分；否则用单边内部邻居差分。
- 精度：均匀笛卡尔网格上线性场梯度精确；非正交曲线网格上为局部结构化差分近似。

## 3. 实现映射

| 式 / 步骤 | 代码位置 |
|-----------|----------|
| (2) 逻辑方向样本 | `difference_along_axis` |
| (3)(4) 物理梯度求解 | `solve_physical_gradient` |
| 速度 / 温度梯度场 | `compute_structured_gradients_3d` |
| 粘性通量使用 | `compute_gradients_and_assemble_viscous_3d` |

## 4. 参考文献

1. Vinokur, M. (1989). *An analysis of finite-difference and finite-volume formulations of conservation laws*. NASA CR-177512.
2. Ferziger, J. H., Peric, M., & Street, R. L. (2020). *Computational Methods for Fluid Dynamics* (4th ed.). Springer. Ch. 8.
3. Rumsey, C. L., Biedron, R. T., & Thomas, J. L. (2010). *CFL3D: Its history and some recent applications*. NASA/TM-2010-216758.
