# Taylor–Green 3D 涡衰减（I3）

**benchmark_id**: `taylor_green_3d`

## 物理

周期域 \([0,2\pi]^2\times[0,L_z]\)（\(n_z=1\) 准 2D）上的 Taylor–Green 涡，Reynolds 数由 \(\nu\) 与 \(U_{\mathrm{ref}},L_{\mathrm{ref}}\) 决定。

初场（SI 输入，求解器内部无量纲化）：

\[
u=\sin x\cos y\cos z,\quad v=-\cos x\sin y\cos z,\quad w=0
\]

层流动能衰减（Brachet et al. 1983；见 ADR 0015 I3）：

\[
\frac{E(t)}{E(0)}=\exp(-4\,\nu^* t^*),\quad \nu^*=1/Re,\ t^*=t\,U_{\mathrm{ref}}/L_{\mathrm{ref}}
\]

## V&V 基线（CI 默认）

| 项 | 值 |
|----|-----|
| 网格 | 16×16×1，双周期 + z 对称 |
| 时间推进 | BDF1 动量 + **PISO-2** |
| 物理时间 | \(t^*=2.0\) |
| **`dt` / `max_steps`** | **`0.005` / `400`**（自 `dt=0.05/40` 下调，减轻首步耦合冲击） |
| 对流 | 中心格式 |
| 初场 | 解析 \(u,p\) + Rhie-Chow 压力投影（与动量装配同算子 \(d_P\)） |

运行产物写入 `out/`（`residual.csv`、`run-manifest.json`），**不入库**；本地复现后自行生成。

## 数值

- `time.mode = transient`，`time.scheme = bdf1`
- 初场 Rhie-Chow 投影目标：`max|div phi| < 10^{-6}`

## 验证（I3 V&V）

CI（`tests/case_run.rs`）判据：

| 量 | 判据 |
|----|------|
| 动能单调衰减 | \(E_{\mathrm{final}} < E_{\mathrm{initial}}\)，\(E/E_0 < 1\) |
| \(E/E_0\) vs 解析 | \(\ge 0.42 \times \exp(-4\nu^* t^*)\) |
| spin-up 衰减率 | \(-\mathrm{d}\ln E/\mathrm{d}t\) 在 \(0.5\times\)–\(40\times\) 的 \(4\nu^*\) 内 |
| 连续性 | `max_abs_corrected_field_divergence_after_boundary` \(< 10^{-6}\) |
| 压力 Poisson 残差 | `max_abs_corrected_divergence` \(< 10^{-6}\) |

机器可读参考值见 `expected.json`（`status = i3_piso_bdf1_kinetic_decay_vv`）。

## 参数敏感性（16×16，\(t^*=2\)，本地 `#[ignore]` 对照）

集成测试 `taylor_green_3d_parameter_sensitivity_baseline` 输出（2026-06 标定）：

| `dt` | steps | PISO | \(E/E_0\) | 解析 | \(\|E/E_0-\)解析\(\|\) | 末步 `max\|div(u*)\|` |
|------|-------|------|-----------|------|------------------------|----------------------|
| 0.05 | 40 | 2 | 0.433 | 0.980 | 0.547 | 5.5e-3 |
| 0.02 | 100 | 2 | 0.444 | 0.980 | 0.536 | 1.0e-3 |
| 0.01 | 200 | 2 | 0.449 | 0.980 | 0.531 | 3.0e-4 |
| **0.005** | **400** | **2** | **0.451** | **0.980** | **0.528** | **8.0e-5** |
| 0.005 | 400 | 1 | 0.451 | 0.980 | 0.528 | 8.0e-5 |
| 0.005 | 400 | 3 | 0.451 | 0.980 | 0.528 | 8.0e-5 |

结论：`dt` 减小显著改善 \(E/E_0\) 与末步散度；在 `dt=0.005` 下 PISO 校正步数 1–3 对动能衰减影响可忽略（当前粗网格瓶颈在 time-step / 首步耦合，而非 PISO 迭代数）。

## 精度路线（后续）

- `#[ignore]` 细网格对照：16×16 与 32×32 同物理时间比较 \(E/E_0\) 偏差。
- 目标：改进时间推进/压力-速度耦合后，使 \(\|E/E_0-\exp(-4\nu^*t^*)\|\) 随网格加密下降，并继续收紧容差。

```bash
asimu --case tests/benchmarks/taylor_green_3d/case.toml
cargo test --test case_run taylor_green_3d
cargo test --test case_run taylor_green_3d_parameter_sensitivity_baseline -- --ignored --nocapture
```

## 参考文献

1. Brachet, M. E., et al. (1983). Small-scale structure of the Taylor–Green vortex. *Journal of Fluid Mechanics*, 130, 411–452.
2. Ghia et al. (1982) — 方腔对照；本算例为周期 TG 衰减。
