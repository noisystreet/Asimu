# 工程 TODO

## 3D 可压缩多块求解

### 阶段 3：完全单入口 + 扩展时间推进（未开始）

- [x] 读入层将单块 `structured_3d` / 单 zone CGNS 统一为 `MultiBlockStructured3d`（消除 runtime 包装）
- [x] 有接口多块 case 在 case 解析阶段校验 LU-SGS 对角隐式约束，并更新 mesh 诊断文案
- [ ] 多块路径支持 GMRES / RK4 / LU-SGS sweep（含跨 block 隐式 Jacobian 或等效耦合）

## 非结构可压缩双时间步（DTS）

理论路线图：[docs/theory/dual_time_stepping.md](theory/dual_time_stepping.md)

- [ ] **P0** — `add_physical_storage_residual<T: ComputeFloat>` + f32/f64 单元测试
- [ ] **P1** — LU-SGS 分母 `inv_dt_phys`（typed f32/f64 对角 + sweep）
- [ ] **P2** — `unstructured_driver_typed` 内外循环 + `DualTimeState<T>` + `scheme = "dual_time"`
- [ ] **P3** — V&V：f64 reference + f32 相对阈值；freestream / Sod / 涡对流
- [ ] **P3b** — CUDA f32：device `U^n`、存储项 kernel、`validate` 能力矩阵
- [ ] **P4** — （可选）非结构 typed GMRES + DTS
- [ ] **P5** — manifest（`compute_precision`、`exec_device`、`inner_iterations`）
