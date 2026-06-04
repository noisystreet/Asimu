# 变更日志

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Changed

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
- 可压缩正性下限：恢复来流静压 1% 的 `positivity_pressure_floor`，通量/输出 primitive 恢复钳制最低压力，避免极低温度伪解

### Added

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
