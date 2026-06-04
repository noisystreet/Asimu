# asimu 滑移网格规划

> 相关：`docs/theory/curvilinear_metrics.md`、`docs/theory/time_integration.md`、`docs/adr/0009-compressible-navier-stokes.md`

**状态**：规划。当前代码只支持静止结构化网格与滑移壁面（slip wall），尚不支持 sliding mesh、overset mesh 或 ALE 动网格。

---

## 1. 目标与非目标

### 1.1 目标

| 目标 | 说明 |
|------|------|
| **分阶段交付** | 先支持 MRF / frozen rotor，再推进共形 sliding interface、非共形 sliding mesh 与 ALE |
| **守恒接口** | 跨滑移界面的质量、动量、能量通量必须以面积权重守恒方式交换 |
| **显式状态** | 网格运动、接口配对、插值权重和网格速度作为参数或状态结构传入，不依赖全局缓存 |
| **可验证** | 每阶段提供小网格单元测试、均匀流保持测试和旋转机械 smoke case |
| **兼容现有层次** | `mesh` 管拓扑与几何，`discretization` 管接口通量，`solver` 编排时间推进 |

### 1.2 非目标

- 不在首版实现任意拓扑非结构网格。
- 不把滑移接口伪装成普通 `BoundaryKind::Periodic`；它需要独立的接口拓扑、相对运动和守恒插值。
- 不在 `discretization` 热路径中引入全局 registry、动态查找或隐式缓存。
- 不在 MRF 阶段移动网格；MRF 只提供旋转参考系近似。

---

## 2. 术语与范围

| 术语 | 含义 | 首选阶段 |
|------|------|----------|
| **滑移壁面** | 固壁无穿透、切向自由的边界条件；当前已有 `Wall(no_slip=false)` | 已有 |
| **MRF / frozen rotor** | 静网格中用旋转参考系处理转子区域，接口几何不随时间变化 | M1 |
| **共形 sliding interface** | 两侧接口面一一匹配，但相对角度随时间变化，需要重配对 | M2 |
| **非共形 sliding mesh** | 两侧接口面不一一匹配，需要面积交叠搜索和守恒插值 | M3 |
| **ALE 动网格** | 控制体随时间变形或运动，通量使用相对网格速度并满足 GCL | M4 |
| **overset mesh** | 重叠网格通过 donor cell 插值交换场量 | 远期 |

---

## 3. 当前差距

当前 3D 装配以 `StructuredMesh3d` 的 `i/j/k` 固定邻接为核心：

- 内部面由 `(i,j,k)` 与相邻单元直接配对。
- 边界 patch 是固定 `FaceId` 列表，边界状态来自 ghost cell。
- 面通量只接收流体状态与面法向，没有网格速度或相对界面速度。
- 单一 `StructuredMesh3d` 承载一个网格块；没有多 zone 运动、接口配对表或 donor/receiver 权重表。

因此 sliding mesh 不能通过新增一个 TOML 边界类型完成，必须先引入接口数据模型和独立装配路径。

---

## 4. 分阶段路线

### M1：MRF / frozen rotor

**目标**：在静止网格上支持旋转区域稳态近似，作为旋转机械能力的第一步。

| 项 | 规划 |
|----|------|
| 配置 | `[motion.zone.<name>] kind = "mrf"`，包含 `axis_origin`、`axis_direction`、`angular_velocity`、`cell_zone` |
| 数据 | `mesh::CellZone` 或等价 cell id 集合；`physics::RotatingFrame` 描述角速度 |
| 离散 | 在旋转区域使用相对速度计算对流通量，并加入离心 / 科氏源项 |
| 求解 | 首版仅稳态伪时间；GMRES / LU-SGS 先按显式源项处理 |
| 验收 | 均匀刚体旋转小网格源项单测；静止区结果与无 MRF 基线一致 |

**约束**：MRF 不改变网格坐标，不需要接口面重配对；这是最小侵入的第一阶段。

### M2：共形 sliding interface

**目标**：支持两侧接口面几何共形、一一匹配的相对旋转。

| 项 | 规划 |
|----|------|
| 配置 | `[interface.<name>] kind = "sliding_conformal"`，指定 `side_a`、`side_b`、旋转轴与角速度 |
| 数据 | `mesh::SlidingInterface` 保存两侧 face id、当前相位和配对表 |
| 装配 | 新增接口面通量装配，替代两侧 ghost 边界；每个接口通量同时累加到 owner 与 receiver |
| 时间 | 每步由 `solver` 更新相位并重建配对；稳态伪时间需冻结相位或使用时间平均模型 |
| 验收 | 圆环二维/薄三维共形接口均匀流保持；接口质量通量两侧守恒到舍入误差 |

**关键点**：即使几何共形，也不能把两侧都当成普通边界 ghost 独立处理，否则界面通量不严格守恒。

### M3：非共形 sliding mesh

**目标**：支持转静子接口面数量不同或角向位置不同的网格。

| 项 | 规划 |
|----|------|
| 搜索 | 在接口局部坐标中做面投影、区间/多边形交叠搜索 |
| 权重 | `InterfaceOverlap { face_a, face_b, area, centroid, normal }` 显式保存面积交叠 |
| 通量 | 对每个 overlap 计算一次 Riemann 通量，并按交叠面积分别累加 |
| 插值 | 首版一阶面积加权；MUSCL 需限制器避免跨非共形接口产生振荡 |
| 性能 | 配对/交叠表按相位缓存，但缓存属于 `SlidingInterfaceState`，由 `solver` 显式持有 |
| 验收 | 非共形均匀流保持；接口总质量残差小于网格尺度误差阈值；转静子 smoke case 残差有限 |

**关键风险**：面积交叠搜索和法向方向一致性是主要缺陷来源，应先用二维圆环截面建立黄金测试。

### M4：ALE 动网格

**目标**：支持控制体随时间运动或变形，提供完整瞬态滑移网格基础。

| 项 | 规划 |
|----|------|
| 数据 | `mesh::MotionState` 保存当前/上一时刻节点坐标、面速度、单元体积变化 |
| 方程 | 无粘通量使用相对速度 \((\mathbf{u}-\mathbf{w})\cdot\mathbf{n}\)，其中 \(\mathbf{w}\) 是面网格速度 |
| GCL | 几何守恒律测试：均匀流在运动网格上保持均匀 |
| 时间 | `TimeIntegrator` 需要显式区分物理时间步、网格更新时间和伪时间迭代 |
| 输出 | restart / manifest 记录运动配置、相位、角速度和网格时间 |
| 验收 | 均匀流 GCL 测试；刚体平移/旋转网格下守恒量误差可控 |

---

## 5. 架构落点

| 模块 | 规划职责 |
|------|----------|
| `mesh` | 多 zone / cell zone / face zone、滑移接口几何、接口配对和运动状态 |
| `field` | 保持按 cell 存储；接口插值不应改变 field 布局 |
| `discretization` | 提供接口通量装配、ALE 相对通量和 MRF 源项离散 |
| `physics` | 旋转参考系、惯性力源项、参考系速度变换 |
| `solver` | 持有 `SlidingInterfaceState` / `MotionState`，在每步显式更新接口相位与权重 |
| `io` | 解析 CGNS zone、1-to-1 / grid connectivity 和 TOML 运动配置 |
| `case` | 校验 motion/interface 配置与网格 patch / zone 的一致性 |

依赖方向仍遵循 `core ← mesh ← field ← discretization`，`solver` 只编排下层能力；接口状态不得通过全局单例共享。

---

## 6. Case 配置草案

MRF 首版示例：

```toml
[motion.zone.rotor]
kind = "mrf"
cell_zone = "rotor"
axis_origin = [0.0, 0.0, 0.0]
axis_direction = [0.0, 0.0, 1.0]
angular_velocity = 314.1592653589793
```

共形滑移接口草案：

```toml
[interface.rotor_stator]
kind = "sliding_conformal"
side_a = "rotor_interface"
side_b = "stator_interface"
axis_origin = [0.0, 0.0, 0.0]
axis_direction = [0.0, 0.0, 1.0]
angular_velocity = 314.1592653589793
initial_phase = 0.0
```

非共形接口后续扩展：

```toml
[interface.rotor_stator]
kind = "sliding_nonconformal"
search_tolerance = 1.0e-10
conservative_interpolation = "area_weighted"
```

---

## 7. 验证计划

| 阶段 | 必测项 |
|------|--------|
| M1 | 旋转参考系速度变换单测；MRF 源项符号测试；静止区回归测试 |
| M2 | 共形接口配对测试；均匀流保持；接口两侧通量反对称 |
| M3 | 面交叠面积守恒；非共形均匀流保持；相位变化后权重重建稳定 |
| M4 | GCL 均匀流测试；动网格体积变化测试；restart 相位恢复测试 |

V&V 算例建议：

- 同心圆环 Couette / inviscid uniform flow：验证接口守恒与均匀流保持。
- 转静子二维叶栅 smoke case：验证相位推进与残差有限性。
- 刚体旋转盒子 ALE：验证 GCL 和相对通量。

---

## 8. 风险与决策点

| 风险 | 处理 |
|------|------|
| 非共形接口不守恒 | 以 overlap 面积通量为唯一数据交换路径，禁止双侧 ghost 独立通量 |
| GCL 破坏均匀流 | ALE 阶段先实现几何守恒测试，再接入复杂物理 |
| 接口搜索成本高 | 首版单线程显式重建；后续按相位缓存和局部区间搜索优化 |
| 与 GMRES 线性化不一致 | 首版接口权重在单个非线性/伪时间步内冻结；后续再评估运动项线性化 |
| CGNS 多 zone 复杂 | 先用 TOML 显式配置 + 简单 fixture，CGNS GridConnectivity 后接入 |

需要 ADR 的决策点：

- 是否引入通用多 zone 网格数据模型。
- 非共形接口交叠搜索算法与容差策略。
- ALE 是否作为 `time` 的一部分，还是独立 `motion` 层。

---

## 9. 推荐实现顺序

1. 为静网格添加 `CellZone` / `FaceZone` 数据模型和 TOML 校验。
2. 实现 MRF 速度变换与源项，建立旋转参考系单元测试。
3. 抽象接口通量装配入口，先支持共形 face 一一配对。
4. 增加非共形 overlap 表和守恒面积权重。
5. 引入 ALE 面速度和 GCL 测试，再扩展到真正动网格。

---

## 10. 参考文献

1. Ferziger, J. H., Perić, M., & Street, R. L. (2020). *Computational Methods for Fluid Dynamics* (6th ed.). Springer. Ch. 8, Ch. 10.
2. Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications* (3rd ed.). Butterworth-Heinemann. Ch. 6, Ch. 10.
3. Thomas, J. L., & Lombard, C. K. (1979). Geometric conservation law and its application to flow computations on moving grids. *AIAA Journal*, 17(10), 1030-1037. DOI: 10.2514/3.61273.
4. Rai, M. M. (1986). A conservative treatment of zonal boundaries for Euler equation calculations. *Journal of Computational Physics*, 62(2), 472-503. DOI: 10.1016/0021-9991(86)90141-8.
