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
# 默认算例：MUSCL + van Albada + Roe
asimu --case tests/benchmarks/sod_1d/case.toml

# MUSCL + van Albada + HLLC
asimu --case tests/benchmarks/sod_1d/case_muscl_hllc.toml

# 或
make run-case CASE=tests/benchmarks/sod_1d/case.toml
```

集成测试 `tests/sod_benchmark.rs` / `tests/case_run.rs` 调用 `case::run_case_path` 或 `solver::run_sod_benchmark`。

精确解采样使用 **相对隔膜坐标** \(x' = x - x_{\mathrm{diaphragm}}\)（Riemann 求解器默认间断位于 \(x'=0\)）。

## case.toml 离散选项

```toml
[sod]
diaphragm = 0.5
final_time = 0.2
cfl = 0.4
flux = "roe"              # roe | hllc
reconstruction = "muscl"  # first_order | muscl
limiter = "van_albada"    # minmod | van_leer | van_albada
```

详见 [CASE_FORMAT.md](../../../docs/CASE_FORMAT.md) §6.1。

## 导出与绘图

```bash
# 1. MUSCL+Roe vs MUSCL+HLLC 对比（默认 van Albada）
cargo run --example sod_benchmark_export -- sod_compare.txt

# 2. matplotlib 对比曲线（需 pip install -r scripts/requirements-plot.txt）
python3 scripts/plot_sod_benchmark.py sod_compare.txt -o sod_compare.png
```

对比格式：`# format=compare` + 列 `x rho_roe rho_muscl_hllc rho_exact ...`。

## 数值方法

| 环节 | 默认配置 | 理论 |
|------|----------|------|
| 界面重构 | MUSCL + van Albada | [interface_reconstruction.md](../../../docs/theory/interface_reconstruction.md) |
| 无粘通量 | Roe（熵修正）或 HLLC | [inviscid_flux.md](../../../docs/theory/inviscid_flux.md) |
| 时间积分 | RK4 + CFL=0.4 | [time_integration.md](../../../docs/theory/time_integration.md) |

100 单元、\(t=0.2\) 参考：L1(ρ) ≈ 0.012（MUSCL+van Albada+Roe），见 `expected.json`。

## 参考文献

- Sod, G. A. (1978). *A Survey of Several Finite Difference Methods for Systems of Nonlinear Hyperbolic Conservation Laws.*
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*, §4.
