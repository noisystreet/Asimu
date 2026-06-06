# 工程 TODO

## 3D 可压缩多块求解

### 阶段 3：完全单入口 + 扩展时间推进（未开始）

- [x] 读入层将单块 `structured_3d` / 单 zone CGNS 统一为 `MultiBlockStructured3d`（消除 runtime 包装）
- [x] 有接口多块 case 在 case 解析阶段校验 LU-SGS 对角隐式约束，并更新 mesh 诊断文案
- [ ] 多块路径支持 GMRES / RK4 / LU-SGS sweep（含跨 block 隐式 Jacobian 或等效耦合）
