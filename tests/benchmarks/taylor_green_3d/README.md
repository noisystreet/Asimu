# Taylor–Green 3D 涡衰减（I3）

**benchmark_id**: `taylor_green_3d` · **状态**: **I3 完成**（ADR 0015）

## 物理

周期域 \([0,2\pi]^2\times[0,L_z]\)（\(n_z=1\) 准 2D）上的 Taylor–Green 涡，Reynolds 数由 \(\nu\) 与 \(U_{\mathrm{ref}},L_{\mathrm{ref}}\) 决定。

初场（SI 输入，求解器内部无量纲化）：

\[
u=\sin x\cos y\cos z,\quad v=-\cos x\sin y\cos z,\quad w=0
\]

层流动能衰减（Brachet et al. 1983；见 ADR 0015 I3）：

\[
\frac{E(t)}{E(0)}=\exp(-4\,\nu\, t)
\]

其中 \(\nu,t\) 为有量纲运动粘度与物理时间。Case 内部在网格缩至 \([0,1]^d\) 后以 \(\nu^*=1/Re\)、\(t^*=t\,U_{\mathrm{ref}}/L_{\mathrm{ref}}\) 推进，等价形式为 \(\exp(-4\,\nu^*\,L_{\mathrm{ref}}^2\,t^*)\)（**不是** \(\exp(-4\nu^* t^*)\)）。

## V&V 基线（CI 默认）

| 项 | 值 |
|----|-----|
| 网格 | 16×16×1，双周期 + z 对称 |
| 时间推进 | BDF1 动量 + **PISO-2** |
| 物理时间 | \(t=2.0\)（`400 × 0.005` s） |
| **`dt` / `max_steps`** | **`0.005` / `400`** |
| 对流 | 中心格式 |
| 初场 | 解析 \(u,p\) + Rhie-Chow 压力投影 + `initial_face_flux` 播种 |

运行产物写入 `out/`（`residual.csv`、`run-manifest.json`），**不入库**；本地复现后自行生成。

## 数值

- `time.mode = transient`，`time.scheme = bdf1`
- 初场 Rhie-Chow 投影目标：`max|div phi| < 10^{-6}`

## 验证（I3 V&V）

CI（`tests/case_run.rs`）判据：

| 量 | 判据 |
|----|------|
| 动能单调衰减 | \(E_{\mathrm{final}} < E_{\mathrm{initial}}\)，\(E/E_0 < 1\) |
| \(E/E_0\) vs 解析 | \(\|E/E_0-\exp(-4\nu t)\| < 0.01\) |
| spin-up 衰减率 | \(-\mathrm{d}\ln E/\mathrm{d}t^*\) 在 \(0.5\times\)–\(2\times\) 的 \(4\nu^* L_{\mathrm{ref}}^2\) 内 |
| 连续性 | `max_abs_corrected_field_divergence_after_boundary` \(< 10^{-6}\) |
| 压力 Poisson 残差 | `max_abs_corrected_divergence` \(< 10^{-6}\) |

机器可读参考值见 `expected.json`（`status = i3_piso_bdf1_kinetic_decay_vv`）。
metrics 与文献对不上时，先查 [docs/DEBUG_CHECKLIST.md](../../../docs/DEBUG_CHECKLIST.md) §2–§3。

## 参数敏感性（16×16，\(t=2\)，本地 `#[ignore]` 对照）

集成测试 `taylor_green_3d_parameter_sensitivity_baseline` 输出（2026-06 标定；各行断言 \(\|E/E_0-\)解析\(\| < 0.02\)）：

| `dt` | steps | PISO | \(E/E_0\) | 解析 | \(\|E/E_0-\)解析\(\|\) | 末步 `max\|div(u*)\|` |
|------|-------|------|-----------|------|------------------------|----------------------|
| 0.05 | 40 | 2 | 0.433 | 0.449 | 0.017 | 5.4e-7 |
| 0.02 | 100 | 2 | 0.444 | 0.449 | 0.005 | 4.7e-7 |
| 0.01 | 200 | 2 | 0.449 | 0.449 | 0.0003 | 3.0e-7 |
| **0.005** | **400** | **2** | **0.451** | **0.449** | **0.002** | **2.4e-7** |
| 0.005 | 400 | 1 | 0.451 | 0.449 | 0.002 | 2.4e-7 |
| 0.005 | 400 | 3 | 0.451 | 0.449 | 0.002 | 2.4e-7 |

结论：`dt` 减小改善 \(E/E_0\) 对齐；PISO corrector 数（1–3）对基线动能比无可见影响；末步 `max|div(u*)|` 均为 \(10^{-7}\) 量级。

## 网格收敛（`#[ignore]`）

`taylor_green_3d_refined_grid_reduces_energy_ratio_error`（同 \(t=2\)、PISO-2）：

| 网格 | \(E/E_0\) | \(\|E/E_0-\)解析\(\|\) |
|------|-----------|------------------------|
| 16×16 | 0.451 | 0.002 |
| 32×32 | 0.448 | 0.001 |

32×32 相对解析误差优于 16×16；CI 仍用 16×16 以控制耗时。

## 运行

```bash
asimu --case tests/benchmarks/taylor_green_3d/case.toml
cargo test --test case_run taylor_green_3d
cargo test --test case_run taylor_green_3d_parameter_sensitivity_baseline -- --ignored --nocapture
cargo test --test case_run taylor_green_3d_refined_grid_reduces_energy_ratio_error -- --ignored --nocapture
```

## 参考文献

1. Brachet, M. E., et al. (1983). Small-scale structure of the Taylor–Green vortex. *Journal of Fluid Mechanics*, 130, 411–452.
2. Ghia et al. (1982) — 方腔对照；本算例为周期 TG 衰减。
