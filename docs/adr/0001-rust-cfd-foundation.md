# ADR 0001: 以 Rust 构建 CFD 求解器基础

- **状态**: 已接受
- **日期**: 2026-05-29

## 背景

需要在 `asimu` 目录启动新的 CFD 项目，要求可复现构建、模块化扩展，并便于 AI Agent 协作。

## 决策

1. 使用 **Rust**（edition 2024，MSRV 1.85）实现 binary + library 结构
2. 分层模块：`core` / `mesh` / `solver` / `io` / `config`
3. 错误：`thiserror`；日志：`tracing`；CLI：`clap`；配置：TOML + 环境变量
4. 可执行项目提交 `Cargo.lock`；CI 运行 fmt、clippy（warnings deny）、test
5. 双许可 **Apache-2.0 OR MIT**

## 后果

### 正面

- 内存安全与类型系统有利于数值代码维护
- 工具链（cargo、clippy、rustfmt）与 CI 集成成熟

### 负面

- CFD 生态（网格库、线性代数）相对 C++/Fortran 较弱，部分能力需自研或 FFI
- 编译时间随依赖增长可能变长

## 备选方案

- **C++**：生态丰富，但 Agent 协作与安全边界较难统一
- **Fortran**：传统 CFD 首选，但与现有 Rust 工具链目标不一致
