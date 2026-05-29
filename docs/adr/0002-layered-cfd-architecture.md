# ADR 0002: CFD 分层架构与 v0.2 数值基线

- **状态**: 已接受
- **日期**: 2026-05-29
- **关联**: [ARCHITECTURE.md](../ARCHITECTURE.md)、[DATA_MODEL.md](../DATA_MODEL.md)

## 背景

v0.1 骨架采用 `core → mesh → solver → io` 四层，足以启动项目，但不足以支撑真实 PDE 实现。若将所有数值逻辑堆入 `solver`，会导致：

- 离散格式无法独立单测
- 模块间循环依赖风险
- 单文件迅速超过复杂度门禁（800 行）

## 决策

### 1. 扩展模块分层

在 v0.2 引入以下模块，并调整依赖方向：

```
core ← mesh ← field ← discretization
core ← physics
core ← linalg
mesh + field + discretization + physics + linalg ← solver
```

新增 `case` 作为应用编排层；`solver` 仅负责时间/非线性迭代编排。

### 2. v0.2 数值基线

| 项 | 选择 |
|----|------|
| 空间离散 | 有限体积法（FVM） |
| 网格 | 2D 结构化矩形 |
| 方程 | 稳态对流-扩散 |
| 线性求解 | 手写稀疏矩阵 + 共轭梯度（CG） |
| 并行 | 单线程 |
| 场存储 | SoA（Structure of Arrays） |

### 3. 扩展点

以下边界使用 trait，其余热路径保持具体类型：

- `FluxScheme` — 对流通量
- `GradientScheme` — 梯度重构
- `LinearSolver` — 迭代求解器
- `BoundaryCondition` — 边界应用

### 4. 单 crate 延续

v0.2 仍保持单 crate 目录结构；workspace 拆分推迟至编译时间或 API 稳定需求出现。

## 后果

### 正面

- 离散、线性代数、求解编排可独立测试与替换
- 演进路线与模块交付一一对应
- 符合 CFD 社区惯用的数据/算法分离

### 负面

- v0.1 → v0.2 需重构现有 `solver` 占位代码
- 模块数量增加，新贡献者学习曲线略升（靠 ARCHITECTURE + DATA_MODEL 文档缓解）

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| 全部保留在 `solver` 单模块 | 不可测试、不可维护 |
| 第一版即非结构化网格 + GMRES | 复杂度过高，拖慢验证 |
| 第一版引入 nalgebra | 额外依赖；v0.2 矩阵规模小，手写足够 |
