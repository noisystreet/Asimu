# ADR 0005: 时间推进与稳态/瞬态统一模型

- **状态**: 已接受（规划基线）
- **日期**: 2026-05-29
- **关联**: [ARCHITECTURE.md](../ARCHITECTURE.md) §8.5.4、[DATA_MODEL.md](../DATA_MODEL.md) §11

## 背景

asimu 路线从 **稳态对流-扩散**（v0.2）演进到 **瞬态不可压 NS**（v0.3+）。若 `solver` 内硬编码「单步稳态迭代」，后续加入时间步进、CFL 限制、多阶段 Runge-Kutta 将难以扩展。

## 决策

### 1. 独立时间推进抽象

引入 **`TimeIntegrator`**（`src/solver/time/`），与空间离散 `discretization` 分离：

| 组件 | 职责 |
|------|------|
| `TimeIntegrator` | Δt、阶段系数、稳态/瞬态模式、物理时间 `t` |
| `discretization` | 空间残差 / 通量 / Jacobian |
| `solver` | 编排：循环调用 integrator + disc + linalg |

### 2. Trait 与实现路线

```rust
pub trait TimeIntegrator {
    fn mode(&self) -> TimeMode;
    fn advance(&mut self, state: &mut SolverState) -> Result<TimeStepInfo>;
    fn suggested_dt(&self, mesh: &dyn Mesh, fields: &Fields) -> Result<Real>;
}
```

| 实现 | 版本 | 说明 |
|------|------|------|
| `SteadyStateIntegrator` | v0.2 | 伪时间步或单次线性 solve |
| `ExplicitEulerIntegrator` | v0.4 | 瞬态 + CFL |
| `RungeKutta4Integrator` | v0.5+ | 评估 |
| `Bdf2Integrator` | v1.x+ | 开放 |

v0.2 优先 **enum dispatch**（`enum TimeScheme { Steady, ... }`），避免 trait object 热路径开销；trait 用于测试 mock 与 v0.5 扩展。

### 3. 配置

```toml
[time]
mode = "steady"       # steady | transient
dt = 1.0e-3
cfl_max = 0.5
max_steps = 1_000_000
```

### 4. 与 Run Manifest / Restart / 可观测性

| 关联 | 字段 |
|------|------|
| Run Manifest | `time.mode`、`solve`、`observability` |
| RestartSnapshot | `SolverState.time`、`step` |
| metrics.jsonl | 每时间步/迭代 `step`、`cfl`、`residual` |

### 5. 测试要求

每种新 Integrator 必须：

1. manufactured solution 或解析解单测
2. 在 [BENCHMARKS.md](../BENCHMARKS.md) 中至少一个对应用例（可与其他算例共用）

## 后果

- `solver` 保持编排角色，不膨胀为「时间公式大全」
- 稳态路径零额外抽象成本（默认 `SteadyStateIntegrator`）
- 新增 integrator 需配套验证算例与 manifest 字段

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| 稳态/瞬态两套 solver | 重复 BC、io、收敛、manifest 逻辑 |
| 全 trait 泛型时间推进 | v0.2 过度设计；enum 优先 |
| 时间在 discretization 内 | 违反空间/时间分离，难测 |
