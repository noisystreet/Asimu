# ADR 0011: 非结构 FVM 内面并行（面着色 POC）

- **状态**: 已接受（POC 已落地，默认路径仍串行）
- **日期**: 2026-06-07
- **关联**: [ADR 0010](0010-unstructured-mixed-mesh.md)、[ADR 0003](0003-multi-precision-and-gpu.md)、[unstructured_fvm.md](../theory/unstructured_fvm.md)

## 背景

非结构 FVM 内面装配对每个面执行 \(\mathbf R_i \mathrel{+}= s_i\,\mathbf f_f\)。并行写同一单元会产生数据竞争。项目架构（ARCHITECTURE §3）约定 **v0.x 先正确后并行**，v1.x 再引入 `rayon` 面循环并行。

粘性/无粘共用化与 `UnstructuredSolverMeshCache` 已就绪；需在不改变默认数值路径的前提下验证 **graph coloring + compute/scatter 分离** 是否可行。

## 决策

### 1. 面着色在网格初始化时预计算

- `UnstructuredFaceTopology::interior_coloring`（`InteriorFaceColoring`）对 `interior` 面列表做 **贪心着色**；
- 同色桶内任意两面不共享 owner/neighbor 单元 → 桶内 flux **compute** 可并行；
- 着色结果随 mesh cache 一次分配，热路径零额外分配。

### 2. compute / scatter 分离

| 路径 | compute | scatter |
|------|---------|---------|
| 粘性内面 | `fused_interior_viscous_face_flux` | `scatter_fused_interior_viscous_face` |
| 无粘内面 | `compute_interior_inviscid_face_contribution` | `scatter_interior_inviscid_face` |

**scatter 仍串行**：主 crate `unsafe_code = forbid`，禁止并行写 `&mut [f64]`；桶内并行 scatter 留待 v1.x `exec` 层（atomics 或 reduction，需单独 ADR 批准 unsafe）。

### 3. 可选 Cargo feature `parallel-fvm`

```toml
parallel-fvm = ["dep:rayon"]
```

- **不**加入 `default` features；
- 启用时：`InteriorFaceColoring::par_map_buckets` 桶内 rayon 并行 compute，桶间 + scatter 串行；
- 未启用时：按着色桶顺序串行（与线性面索引顺序在浮点非结合性下可能有末位差异）。

### 4. 验证与 CI

| 测试 | 说明 |
|------|------|
| `interior_face_coloring_has_no_same_color_cell_conflicts` | 着色正确性 |
| `colored_*_matches_linear_face_order` | 着色 vs 线性顺序（粘性/无粘） |
| `cached_interior_inviscid_matches_mesh_face_loop` | 缓存拓扑 vs 裸 mesh 循环 |
| `parallel_interior_*_matches_colored_serial` | `parallel-fvm` 并行 vs 串行 |

Makefile：`make check-parallel-fvm` = `clippy`（`io-vtk,parallel-fvm`）+ 全量测试（含上述并行 golden）。

### 5. 性能评估（POC 后）

POC **不**以加速比为合入条件。若需决策是否默认启用 `rayon`：

1. 在代表性非结构网格（≥10⁵ 面）上对比串行 vs `parallel-fvm`；
2. 记录着色桶数、桶大小分布、RHS 装配耗时；
3. 小网格可能无收益 — 以 profiling 数据驱动 ADR 修订或默认 feature 变更。

## 后果

### 正面

- 粘性/无粘共用着色基础设施；
- 并行路径与串行路径数值 golden 对齐；
- 为 v1.x `exec` CPU 并行 scatter 预留接口形态。

### 负面 / 限制

- 额外内存：`InteriorFaceColoring::buckets`；
- 贪心着色桶数非最优，极大网格可能限制并行度；
- `parallel-fvm` 增加 CI 矩阵（`check-parallel-fvm`）；
- scatter 串行时，flux compute 并行收益受 Amdahl 限制。

## 未采纳

| 方案 | 原因 |
|------|------|
| `parallel-fvm` 默认启用 | 需 benchmark + 依赖体积评估；违反「先正确后并行」节奏 |
| `discretization` 内 `Mutex`/`RefCell` 并行 scatter | AGENTS 禁止热路径隐式锁 |
| 桶内 scatter 用 `unsafe` 原子加 | 主 crate forbid unsafe；应经 `exec` |
| 无着色的 face 分区（metis） | POC 范围过大；着色足够验证 scatter 分离模式 |

## 实现追踪

| 项 | 状态 |
|----|------|
| `InteriorFaceColoring` / `color_interior_faces` | 已实现 |
| 粘性 `parallel-fvm` compute 并行 | 已实现 |
| 无粘 `parallel-fvm` compute 并行 | 已实现 |
| golden 测试（粘性 + 无粘） | 已实现 |
| `exec` 并行 scatter | v1.x 规划 |
| GPU 面循环 | v1.2+ 经 `exec`（ADR 0003） |

修订时 **不删除** 已有条目；默认启用 `rayon` 或并行 scatter 须新开修订段落或 ADR。
