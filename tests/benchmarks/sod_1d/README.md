# Sod 激波管（1D）

经典 Sod (1978) 激波管：\(\gamma=1.4\)，域 \([0,1]\)，\(t=0\) 时在 \(x=0.5\) 间断。

| 侧 | \(\rho\) | \(p\) | \(u\) |
|----|----------|-------|-------|
| 左 | 1.0 | 1.0 | 0 |
| 右 | 0.125 | 0.1 | 0 |

## 参考解

一维 Euler 方程精确 Riemann 解（Toro 2009 §4），实现见 `physics::riemann_exact`。

## 运行

```bash
# CLI（与集成测试同一路径）
asimu --case tests/benchmarks/sod_1d/case.toml

# 或
make run-case CASE=tests/benchmarks/sod_1d/case.toml
```

集成测试 `tests/sod_benchmark.rs` / `tests/case_run.rs` 调用 `case::run_case_path` 或 `solver::run_sod_benchmark`。

精确解采样使用 **相对隔膜坐标** \(x' = x - x_{\mathrm{diaphragm}}\)（Riemann 求解器默认间断位于 \(x'=0\)）。

## 导出与绘图

```bash
# 1. 运行 benchmark 并写出文本剖面
cargo run --example sod_benchmark_export -- sod_profile.txt

# 2. matplotlib 对比曲线（需 pip install -r scripts/requirements-plot.txt）
python3 scripts/plot_sod_benchmark.py sod_profile.txt -o sod_compare.png
```

文本格式：`#` 元数据行 + 列 `x rho_numeric rho_exact rho_error`。

## 数值方法

| 环节 | 理论 |
|------|------|
| 一阶界面重构 | [interface_reconstruction.md](../../../docs/theory/interface_reconstruction.md) |
| Roe + Harten 熵修正 | [inviscid_flux.md](../../../docs/theory/inviscid_flux.md) |
| RK4 + CFL=0.4 | [time_integration.md](../../../docs/theory/time_integration.md) |

## 参考文献

- Sod, G. A. (1978). *A Survey of Several Finite Difference Methods for Systems of Nonlinear Hyperbolic Conservation Laws.*
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*, §4.
