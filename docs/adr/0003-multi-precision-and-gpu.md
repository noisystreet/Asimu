# ADR 0003: 多精度与执行后端（CPU/GPU）规划

- **状态**: 已接受（规划基线，实现分阶段）
- **日期**: 2026-05-29
- **关联**: [ARCHITECTURE.md](../ARCHITECTURE.md) §8.4、[DATA_MODEL.md](../DATA_MODEL.md) §10

## 背景

asimu 需在架构上预留：

1. **多精度求解** — 开发调试与生产性能对精度需求不同（`f64` 参考解 vs `f32` 加速）
2. **GPU 加速** — 大规模面循环、稀疏矩阵-向量乘等占 CFD 运行时主要部分

若在 v0.2 全部写死 `f64` + CPU 裸循环，后期引入多精度/GPU 将触发全库重写。

## 决策

### 1. 精度模型：类型别名 + 渐进泛型

| 阶段 | 策略 |
|------|------|
| v0.2–v0.4 | 全局 `pub type Real = f64`；API 签名优先使用 `Real` 而非字面 `f64` |
| v0.5 | Cargo feature `precision-f32`；编译期选定单一 `Real` |
| v0.6+ | 可选 **混合精度**：场变量 `f32` + 残差/归约 `f64`（`mixed` 模式） |

**不在 v0.2 引入**全库 `T: Float` 泛型（避免编译膨胀与 Agent 复杂度）；仅在 `core::math`、`linalg` 接口层预留 `Real`。

配置预留（TOML）：

```toml
[numerics]
precision = "f64"   # f64 | f32 | mixed（mixed 为 v0.6+）
```

### 2. 执行后端：独立 `exec` 适配层

新增 **`exec`** 模块（v1.2+ 实现），位于 `discretization` / `linalg` 与硬件之间：

```
discretization / linalg
        ↓  （trait 调用）
      exec  ←  ExecutionContext { backend: Cpu | Gpu }
        ↓
   cpu / gpu-wgpu / gpu-cuda（可选 feature）
```

| 组件 | CPU（默认） | GPU（可选） |
|------|-------------|-------------|
| 场存储 | `Vec<Real>` | 设备 buffer（经 `exec` 封装） |
| 通量装配 | 面循环 | compute shader / CUDA kernel |
| SpMV | CSR 手写 | cuSPARSE / 自研 kernel |
| BC 应用 | CPU（不规则访问） | 暂留 CPU，避免过早 GPU 化 |

**原则**：GPU 只加速**规则、数据并行**热点；I/O、BC、收敛判断留在 CPU。

### 3. Feature 与 unsafe 边界

| Cargo feature | 含义 |
|---------------|------|
| （默认） | 仅 CPU + `f64` |
| `precision-f32` | 编译期 `Real = f32` |
| `gpu-wgpu` | wgpu compute 后端（跨平台优先评估） |
| `gpu-cuda` | NVIDIA CUDA 后端（Linux 服务器，ADR 单独评估依赖） |

- 主 crate 保持 `unsafe_code = forbid`
- GPU 底层封装在 **`asimu-exec-gpu`**（规划 crate）或 `src/exec/gpu/` 中，经 ADR 批准可在隔离模块使用 `unsafe`
- 对外仅暴露 safe 的 `ExecutionContext` API

### 4. 依赖方向

```
core ← exec ← discretization
core ← exec ← linalg
exec 不得依赖 solver / io / case
gpu 实现不得反向依赖 discretization 具体格式
```

### 5. 验证策略

| 能力 | 验证 |
|------|------|
| f32 vs f64 | 同一算例相对误差在预期阶内；golden test 分精度存储 |
| mixed | 残差 `f64` 与纯 `f64` 参考解偏差 < 配置阈值 |
| GPU vs CPU | 同一网格结果 `approx_eq`；CI 无 GPU 时跳过 GPU 测试 |

## 后果

### 正面

- v0.2 起用 `Real` 别名，后期改精度不全文替换 `f64`
- `exec` 层隔离 GPU，CPU 路径保持简单可测
- feature 可选，默认构建无 GPU 依赖

### 负面

- 早期 `Real` 别名增加一层间接（几乎零成本）
- GPU 路径维护成本高，需双路径测试
- mixed 精度调试难度高于单精度

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| v0.2 全库 `T: Float` 泛型 | 复杂度高、编译慢、Agent 易出错 |
| 运行时动态精度（enum 分支每运算） | 性能差，热路径无法 monomorphize |
| 直接在 discretization 内写 CUDA | 与 CPU 逻辑耦合，不可测试 |
| 第一版 GPU 覆盖整个 solver | 边界/耦合逻辑不适合 GPU，风险大 |

## 实现里程碑（摘要）

见 [ARCHITECTURE.md](../ARCHITECTURE.md) §10 演进路线扩展项。
