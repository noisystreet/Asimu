# ADR 0010: 非结构混合单元网格（面拓扑路线）

- **状态**: 已接受（规划基线；M1 部分实现）
- **日期**: 2026-06-06
- **关联**: [ADR 0002](0002-layered-cfd-architecture.md)、[ADR 0008](0008-cgns-io.md)、[ADR 0009](0009-compressible-navier-stokes.md)、[ARCHITECTURE.md](../ARCHITECTURE.md)、[DATA_MODEL.md](../DATA_MODEL.md)、[API.md](../API.md)

## 背景

asimu 当前 3D 可压缩路径以 **多块结构化网格**（`MultiBlockStructuredMesh3d`）为主，几何与通量装配均依赖逻辑 `(i,j,k)` 索引与 I/J/K 面缓存（见 [curvilinear_metrics.md](../theory/curvilinear_metrics.md)）。工程网格（Gmsh、Pointwise、Fluent 导出 VTU 等）常见 **同一网格内混合 tet / hex / pyramid / prism**，无法用单一结构化块表达。

[ADR 0002](0002-layered-cfd-architecture.md) 将「第一版即非结构化网格 + GMRES」列为未采纳方案，理由是复杂度过高、拖慢验证——该判断针对 **v0.2 数值基线**，并不禁止在结构化路径稳定后 **分阶段** 引入非结构能力。本 ADR 定案非结构混合网格的目标形态、模块边界与交付节奏，避免：

- 将 tet 塞进 `StructuredMesh3d` 或 `MultiBlockStructuredMesh3d` 的运行时包装；
- 在 `discretization` 内复制一套与结构化平行的隐式网格假设；
- 未定义 conformal 接口规则即接入 Tier 1 读入（CGNS / VTU），导致 silent 拓扑错误。

## 决策

### 1. 核心路线：统一面列表 + owner/neighbor 面循环

非结构网格在 `mesh` 层表示为 **显式面拓扑**，求解与离散通过 **面循环** 访问几何与连通，而非逻辑 I/J/K：

```
构造期（mesh）:
  points + cells → 面模板 → 面合并 → face_owner / face_neighbor / FaceMetric / CellMetric

运行期（discretization / solver）:
  for each interior face: flux(owner, neighbor)
  for each boundary face: BC ghost + flux
```

**禁止**反向依赖：`discretization` 不得依赖 `solver`；非结构通量装配所需 `mesh`、`field`、`boundary` 均由调用方显式传入（与 AGENTS.md「减少隐式状态」一致）。

**禁止**将非结构网格伪装为 `StructuredMesh3d` 或在其上叠加 runtime 分支；`CaseMesh` 应新增独立变体（规划名 `Unstructured3d`），与 `Structured1d` / `MultiBlockStructured3d` 并列。

### 2. M1 单元类型与节点约定

首版支持 **VTK 线性 3D 单元**（与 VTU `types` 对齐）：

| `CellKind` | VTK `types` | 节点数 | 说明 |
|------------|-------------|--------|------|
| `Tet` | 10 | 4 | 四面体 |
| `Hex` | 12 | 8 | 六面体 |
| `Pyramid` | 14 | 5 | 四角锥 |
| `Prism` | 13 | 6 | 三棱柱（wedge） |

- 节点顺序遵循 **VTK/ParaView** 局部编号；面模板见 `src/mesh/unstructured_templates.rs`。
- **M1 已实现**：`UnstructuredMesh3d::new` 在构造期完成面拓扑、体积、面度量；见 [API.md](../API.md) §`asimu::mesh`。
- **M1 未实现**：`CaseMesh` 接入、求解器面循环、Tier 1 网格读入（CGNS / VTU）。

### 3. 面合并与法向约定

| 规则 | 说明 |
|------|------|
| 面键 | 面节点全局索引 **排序** 后作为 HashMap 键 |
| 边界 | 键仅命中 1 个单元 → `face_neighbor = None` |
| 内部 | 键命中 2 个单元 → owner = 较小 `cell_index`，neighbor = 较大 |
| 非流形 | 键命中 ≥3 个单元 → 构造失败（`AsimuError::Mesh`） |
| 同型匹配 | **仅**节点集完全相同的面可合并（3 节点↔3 节点，4 节点↔4 节点） |

**法向**：VTK 模板部分面 winding 指向单元内侧（如四面体底面法向指向顶点）。构造拓扑时按 **单元中心 → 面心** 方向翻转面积向量与节点顺序，使 owner 侧法向 **指向单元外侧**；与结构化贴体网格 `orient_internal_face_area_vector` 一致。

**体积**（凸单元）：散度公式 \(V = \frac{1}{3}\sum_f \mathbf{S}_f\cdot\mathbf{x}_f\)，\(\mathbf{S}_f\) 为已定向外向面积向量。

### 4. 分阶段交付（M1–M4）

| 阶段 | 范围 | 出口标准 |
|------|------|----------|
| **M1** | `UnstructuredMesh3d` 拓扑 + 几何度量；tet/hex/pyramid/prism | 单单元/两单元共面/非流形单测；`make check` |
| **M2** | `discretization` 非结构面循环；一阶 Euler 无粘通量 + 现有 Riemann 求解器 | **已实现首版**：均匀来流闭合 tet \(\|\mathrm{RHS}\|\) 近零 |
| **M3** | **Tier 1** 读入：CGNS unstructured zone + VTU；`CaseMesh::Unstructured3d` + case 解析 | **已实现首版**：CGNS/VTU 读入、CGNS FaceCenter ZoneBC、单域非结构 case smoke |
| **M4** | 二阶线性重构（IDWLS + 梯度限制器）、粘性通量、边界 patch、网格检查与 V&V 算例 | 限制器见 [ADR 0012](0012-unstructured-gradient-limiters.md)；与结构化路径共享 Riemann/BC |

**M4 之后**（单独评估，不在本 ADR 承诺）：

- 四边形面 ↔ 两三角形 conformal 接口（hex–tet 混合顶面）；
- `polyhedral` / 悬挂节点；
- 非结构 + 多块接口（与 `MultiBlockStructuredMesh3d` 正交，不强行统一）。

### 5. 模块与类型边界

```
core ← mesh::unstructured
mesh ← field          # 非结构场长度 = num_cells
mesh ← discretization # 面循环装配，参数显式传入 mesh + fields + patches
case  → io, mesh, solver, config
```

| 类型 | 模块 | 职责 |
|------|------|------|
| `CellKind` / `UnstructuredCell` | `mesh` | 单元类型与全局节点索引 |
| `UnstructuredMesh3d` | `mesh` | 点、单元、面列表、owner/neighbor、`CellMetric`/`FaceMetric` |
| `assemble_inviscid_residual_unstructured` | `discretization` | 遍历面、调用 `FluxScheme`、累加 owner/neighbor RHS |
| `load_vtu` | `io` | Parse → Validate → `UnstructuredMesh3d`（Tier 1） |
| `load_cgns_unstructured_zone` | `io` | CGNS `ZoneType_t=Unstructured` → `UnstructuredMesh3d` + `BoundarySet`（Tier 1） |
| `CaseMesh::Unstructured3d` | `case` / `io` | 单域非结构算例编排入口 |

**不**在 M1–M3 引入非结构专用 `MetricCache`  trait 体系；热路径保持 `UnstructuredMesh3d` 具体类型 + 预计算 `Vec<FaceMetric>`（构造期一次分配，面循环零分配）。

### 6. 与现有结构化路径的关系

| 项 | 结构化（现状） | 非结构（本 ADR） |
|----|----------------|------------------|
| 几何缓存 | `MetricCache3d`（I/J/K + 边界） | 构造期 `face_metrics` / `cell_metrics` |
| 通量装配 | `assembly_3d` 逻辑索引 | 面 ID 循环 + owner/neighbor |
| 读入 | CGNS structured、VTS | CGNS unstructured（Tier 1）、VTU（Tier 1） |
| 可压求解 | `MultiBlockStructuredMesh3d` block 推进 | 单域 `UnstructuredMesh3d` 推进（M2+） |

两套 mesh 类型 **共享** `discretization` 内 Riemann 求解器、EOS、BC 语义；**不共享** 面遍历实现（YAGNI：规则 of Three 后再抽象 `MeshFaceIterator` trait）。

### 7. hex–tet 混合接口（延后）

Conformal hex 顶面（四边形）与相邻 tet 底面（三角形）在 M1 **不会**自动合并为内部面；hex 该面保持边界面，tet 三角面亦为边界或 tet–tet 内部面。真实 conformal 网格需在 **M4+** 之一：

1. **读入预处理**：VTU 中 hex 面已拆为两三角（网格生成器导出）；
2. **面细分注册**：构造拓扑时将 quad 与两 tri 建立 master/slave 关系（需新 ADR 或本 ADR 修订）；
3. **约束**：case 仅允许同型面 conformal（文档化，短期成本最低）。

M1 测试以 **两 hex 共四边形面**、**两 tet 共三角面** 验证拓扑；hex+tet 堆叠用例仅验证体积为正与非流形拒绝，不假设 quad–tri 自动缝合。

### 8. 测试与 V&V

| 层级 | 内容 |
|------|------|
| 单元 | 单 tet/hex/pyramid/prism 体积；共面；非流形拒绝；VTK type 映射 |
| 集成 | Tier 1 读入 smoke（CGNS / VTU）；均匀流 RHS（M2） |
| Benchmark | M4 新增 `tests/benchmarks/unstructured_*`（如单位球 tet 网格均匀来流） |

数值变更须同步 `docs/theory/`（非结构 FVM 面通量可扩展 [curvilinear_metrics.md](../theory/curvilinear_metrics.md) 或新建 `unstructured_fvm.md`）。

### 9. Feature 与依赖

- **Tier 1 — VTU**：扩展现有 **`io-vtk`**（ADR 0007）；`quick-xml` / appended binary 已在 feature 内。
- **Tier 1 — CGNS unstructured**：扩展现有 **`io-cgns`**（ADR 0008）；复用系统 `libcgns`，`ElementType_t` / 单元连接 → `CellKind` + `UnstructuredCell`；BC 映射与 structured 路径共用 family / `ZoneBC` 语义（M3 最小集）。
- GPU（ADR 0003）：非结构面循环热算子 v1.2+ 经 `exec` 模块；M1–M4 仅 CPU。

### 10. 网格读入优先级（Tier）

非结构混合网格 **I/O 分 Tier 排期**；Tier 1 为 M3 必达，与工程主流程（case TOML → 读网格 → 求解）绑定。

| Tier | 格式 | 网格 / 产出类型 | 阶段 | 说明 |
|------|------|-----------------|------|------|
| **1** | **CGNS** | Structured zone → `MultiBlockStructuredMesh3d` | **已实现** | ADR 0008；多块可压主路径 |
| **1** | **CGNS** | Unstructured zone → `UnstructuredMesh3d` | M3 | 混合 tet/hex/pyramid/prism；`mesh.kind = "cgns"` 非结构 case |
| **1** | **VTU** | VTK unstructured → `UnstructuredMesh3d` | M3 | Gmsh / ParaView 常见导出；`mesh.kind = "vtu"` |
| 2 | OpenFOAM `polyMesh` | — | 未定 | 不在 M1–M4 |
| 2 | Fluent / NASTRAN 原生网格 | — | 未定 | 需单独 ADR |

**Tier 1 约束**：

- Parse → Validate → `UnstructuredMesh3d::new`（不在 `io` 内绕过 mesh 构造期校验）；
- 首版 unstructured CGNS 仅 **线性** 单元（与 §2 `CellKind` 一致）；高阶单元拒绝或降级策略在 M3 实现时于 API 文档写明；
- Structured / Unstructured CGNS **同一 feature `io-cgns`**，不新增 crate 依赖。

## 后果

### 正面

- 与 OpenFOAM/Fluent 类 **面中心 FVM** 一致，扩展 tet/hex 混合网格自然；
- `mesh` 构造期 Parse→Validate→Trust，求解热路径无拓扑分配；
- 与 ADR 0002/0009 分层兼容，不推翻结构化已交付能力。

### 负面

- 短期内维护 **两套** 面遍历（结构化 I/J/K + 非结构面列表）；
- M3 前无法从 case TOML 直接跑非结构可压算例；
- hex–tet conformal 延迟，部分工程网格需预处理或等 M4+。

### 迁移

- 现有 `CaseMesh` / 结构化求解 **无破坏性变更**；
- 新增变体与 API 按 [API.md](../API.md) 与 CHANGELOG 记录。

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| 首版即 polyhedral + 任意面 | 拓扑与 VTU 复杂度远超 M1 验证需求 |
| 非结构塞进 `StructuredMesh3d` | 破坏类型不变量，逻辑索引语义失真 |
| 全局 `FaceRegistry` 静态缓存 | 违反隐式状态约束，不可测试 |
| 首版 CGNS unstructured 单独排期 | **已纳入 Tier 1**（M3）；与 VTU 并列，复用 ADR 0008 链接方式 |
| 首版即非结构 GMRES 隐式 | ADR 0002 教训；M2 从一阶显式面循环验证拓扑 |
| 抽象 `UnstructuredMesh` trait 覆盖 2D/3D | YAGNI；v0.x 仅 3D 可压需求 |

## 实现追踪

| 项 | 状态 |
|----|------|
| `UnstructuredMesh3d` / `CellKind` / 面模板 | M1 已合入工作区 |
| `CaseMesh::Unstructured3d` | 已实现（单域） |
| 非结构面循环 / Euler 装配 | 已实现（一阶无粘） |
| `load_vtu` | 已实现 |
| `load_cgns_unstructured_zone` | 已实现（含 FaceCenter ZoneBC） |
| hex–tri conformal | M4+ 评估 |

修订本 ADR 时 **不删除** 已有决策条目；重大变更（如引入 polyhedral）应新开 ADR 或显式「修订」段落。
