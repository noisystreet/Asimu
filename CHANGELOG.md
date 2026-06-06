# 变更日志

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- 非结构混合单元网格 M1：`UnstructuredMesh3d` 支持 tet / hex / pyramid / prism（VTK 10/12/13/14）面拓扑、owner/neighbor、体积与面度量；新增 `load_vtu`、`load_cgns_unstructured_zone` 与 `check_unstructured_mesh3d`，`mesh_check` 可检查 `.vtu` 与 CGNS unstructured zone，并支持 CGNS FaceCenter ZoneBC 边界 patch 读入与覆盖检查
- 非结构 CGNS case 求解首版：`CaseMesh::Unstructured3d` 支持单域混合网格一阶无粘 Euler 面循环、IDWLS 粘性梯度与 Navier-Stokes 粘性通量、含粘性抛物项的 local time step、显式 Euler/RK4、对角 LU-SGS 与非结构 LU-SGS sweep，并将非结构流场写出为 VTU
- 非结构网格梯度：新增 `compute_unstructured_gradients_idw_lsq`，使用逆距离加权最小二乘法计算 `UnstructuredMesh3d` 单元中心速度与温度梯度
- 多块 3D 可压缩 case 支持 `[restart]` 初场：version=2 TOML 按 block 名称加载守恒量，单 block restart（version=1）仍可用于仅含 1 个 block 的多块网格

### Changed

- 3D 可压缩读入层统一：`structured_3d` 与单 zone CGNS 解析为 1-block `MultiBlockStructured3d`，求解入口不再 runtime 包装；移除 `CaseMesh::Structured3d` 变体
- 有 1-to-1 接口的多块可压缩 case 在解析阶段校验 `time.scheme = lu_sgs` 且 `lusgs_sweep = false`；`mesh.zone` 废弃并告警
- 3D 可压缩求解统一为 block 编排路径：输出/间隔快照/初场与多块共用一套逻辑；无接口时跳过共享通量装配，单块仍可使用 GMRES/RK4/LU-SGS sweep

- 可压缩算例仅保留无量纲求解：移除 `[nondimensional] enabled` 开关，可压缩算例解析后必定缩放为 \(*\) 变量
- 无量纲热传导 \(\lambda^*\)：修正 \(c_p^*=1/(\gamma-1)\)（此前误用 \(\gamma R^*\) 导致壁面热通量偏大）
- CFL 爬升：`max_steps < cfl_ramp_steps` 时仍按 `cfl_ramp_steps` 线性增 CFL，不再压缩到 `max_steps` 内达 `cfl_max`
- 3D 可压缩局部时间步：RK4 与 LU-SGS 统一使用 Blazek 结构网格 face-sum 谱半径 \(\Delta t_i=\mathrm{CFL}/\sigma_i\)，Navier-Stokes 叠加粘性面贡献
- 3D 网格度量：`FaceMetric` / `FaceGeometry3d` 新增几何面心 `center`；边界面 `spacing` 改为单元中心→面心沿法向投影（替代逻辑 Δ/2）
- 粘性通量梯度：结构网格上改用有限差分 + 局部物理梯度求解，不再使用 Green-Gauss 梯度
- TOML 出口边界：`supersonic = true` 现在正确启用零梯度超声速出口，且不再要求 `static_pressure`
- LU-SGS 固定步长：`[time].dt > 0` 现在会覆盖 LU-SGS 的局部 CFL 步长，同时仍保留谱半径用于隐式分母
- LU-SGS 默认值：`lusgs_sweep` 默认改为 `false`，对角隐式作为稳健默认路径；双扫需显式设为 `true`
- SLAU2 无粘通量：修正右侧压力分裂、压力跳跃项符号，以及质量通量低马赫开关的速度幅值定义，保证均匀态退化为物理通量
- Van Leer 无粘通量：修正亚音速 FVS 质量/动量/能量分裂公式；MUSCL `van_leer` 限制器在线性区恢复完整斜率
- 可压缩 farfield/inlet/outlet 边界：改用法向特征关系生成边界外侧状态，减少简单 ghost 外推的反射与过约束
- 可压缩 3D 稳态推进：方向分裂隐式残差光顺新增逐单元正性回退，避免光顺后更新方向导致内能非正
- LU-SGS 双扫：逐单元正性限制、后扫耦合阻尼与全场线搜索；失败时回退对角隐式更新
- GMRES 隐式伪时间：有限差分扰动按守恒量量级自适应缩放，扰动与最终增量写回均加入逐单元正性限制，并记录裁剪诊断
- GMRES 单元块预条件器：改为局部无粘 Jacobian 块近似，不再为每个分量触发全场 RHS 差分
- GMRES profiling：`GMRES 隐式步诊断` 日志新增时间步、预条件器构造、线性求解、线搜索与更新后残差评估耗时
- 可压缩正性下限：恢复来流静压 1% 的 `positivity_pressure_floor`，通量/输出 primitive 恢复钳制最低压力，避免极低温度伪解
- CGNS case 读入：`mesh.kind = "cgns"` 现在自动读取全部 structured zone；多 zone 文件组装为 `MultiBlockStructured3d`，并按 `IN` / `OUT` / `WALL` family 名修正边界类型
- 多块 3D 可压缩 case：新增同步 block 推进路径，可运行 `case_dualcone` 这类多 zone CGNS；1-to-1 接口按 CGNS transform 映射并通过共享无粘通量守恒装配，支持最终流场与间隔快照写出为单个多 Zone CGNS 文件

### Added

- 多块结构化 3D 网格首版：`MultiBlockStructuredMesh3d`、`mesh.kind = "multi_block_structured_3d"` 与 `mesh_check` 诊断；跨 block 求解仍未启用
- 可压缩 3D 边界面无粘通量接口：`BoundaryInviscidFluxInput` / `inviscid_boundary_face_flux`
- 可压缩 3D 时间推进：`time.scheme = "gmres"` 现在启用 matrix-free GMRES 隐式伪时间步，支持 LU-SGS 标量对角与单元 5×5 块对角预条件器
- 线性代数：矩阵无关 restarted GMRES、CSR 矩阵、ILU(0) 预条件器与 LU-SGS 对角预条件器
- 可压缩流无量纲化：`[nondimensional]`、`FreestreamContext` 单一来流入口、理论页 [docs/theory/nondimensional.md](docs/theory/nondimensional.md)
- CGNS 结构化 zone 读入 + VTS/VTM 导出：`io::load_cgns_zone` / `export_cgns_to_vtm`（features `io-cgns-vts`）；ADR 0008；链接系统 `libcgns-dev`
- VTK VTS **二进制 appended** 读入/写出：`io::load_vts` / `write_vts`（feature `io-vtk`）；支持 zlib + 3D；ADR 0007
- v0.2 启动准备：`docs/CASE_FORMAT.md`；`docs/theory/fvm_diffusion.md`
- v0.2 模块骨架：`field`、`discretization`、`linalg`、`solver/time`；`core::Real` 与 ID newtype
- 首个 V&V 算例目录 `tests/benchmarks/1d_diffusion_analytical/`（case + expected + README）
- AGENTS「数值理论与参考文献」约束；`docs/theory/` 索引
- 运行产物 / V&V / 可观测性：`docs/BENCHMARKS.md`、`docs/OBSERVABILITY.md`、`docs/en/CROSS_CUTTING.md`；**四大横向能力**写入 ARCHITECTURE §1.4、§4.3、§8.5–§8.6
- ADR 0005（时间推进）、ADR 0006（FFI/Python）
- `SECURITY.md` 不可信输入限制；`config/default.toml` 预留 `[output]`/`[time]`/`[study]`
- MCP 集成规划：`docs/MCP.md`、ADR 0004、`.cursor/mcp.json.example`
- 架构设计文档 `docs/ARCHITECTURE.md`（含多精度/GPU §8.4、MCP §4.3）
- `src/app/` 应用编排层；库 API 与 `prelude` 分离
- 数据模型文档 `docs/DATA_MODEL.md`
- ADR 0002：CFD 分层架构与 v0.2 数值基线
- ADR 0003：多精度与 CPU/GPU 执行后端规划
- AGENTS.md 编程风格约束
- 项目骨架：Rust binary + library 结构
- 模块化占位实现：`core`、`mesh`、`solver`、`io`、`config`
- CLI 入口与 TOML 配置加载
- 单元测试与集成测试目录
- CI、pre-commit、Makefile 统一命令入口
- AGENTS.md 与协作模板
