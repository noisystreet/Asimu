# ADR 0006: 外部互操作（FFI / Python）原则

- **状态**: 已接受（规划基线，实现远期）
- **日期**: 2026-05-29
- **关联**: [ARCHITECTURE.md](../ARCHITECTURE.md) §8.5.8

## 背景

CFD 工作流常见「Python 前处理 / 后处理 + 高性能求解器」。asimu 首版聚焦 Rust CLI/库，但 API 设计应 **可绑定**，避免 v1.0 后 FFI 大改。

## 决策

### 1. 非目标（v0.x–v1.0）

- 不交付 PyPI 包 / `pip install asimu`
- 不在主 crate 引入 `pyo3` 默认依赖

### 2. 设计原则（自 v0.2 起遵守）

| 原则 | 说明 |
|------|------|
| 稳定 C ABI 可选层 | v1.x 评估 `asimu-ffi` crate，`extern "C"` 仅暴露窄接口 |
| 所有权清晰 | 跨边界用 opaque handle，禁止传递 Rust 内部引用 |
| 错误码 | C API 返回 `AsimuStatus` + 线程局部错误消息 buffer |
| Python | v2.x 评估 `asimu-py`（PyO3），包装 C ABI 而非直接暴露 Rust 类型 |

### 3. 优先暴露的 FFI 能力（规划）

1. 加载 case / mesh
2. 单步或完整 `run`
3. 读取 `RunManifest` 与场数据指针（只读）

### 4. 依赖与许可证

- PyO3 / cxx 等引入需 ADR 修订或补充记录
- 与 GPL 传染条款相同约束：未经批准不引入

## 后果

- 库 API 文档需标注「FFI 稳定」子集（未来 `docs/FFI.md`）
- 部分 Rust 便利 API（如迭代器）可能不进入 FFI 面

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| v0.2 即 PyO3 | 分散数值验证精力 |
| 仅 CLI 无库 | 与项目 binary+library 目标矛盾 |
