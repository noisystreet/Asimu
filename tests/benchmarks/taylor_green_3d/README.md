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

## 数值

- `time.mode = transient`，`time.scheme = bdf1`（BDF1 动量 + PISO-2，`dt=0.005`，`max_steps=400`）
- 16×16×1，双周期 + z 对称
- 中心对流格式
- **初场**：解析 \(u,p\) 后做 Rhie-Chow **压力投影**（固定速度、1 步 Poisson），使 `max|div phi|` \(<10^{-6}\)

## 验证（I3 V&V）

CI（`tests/case_run.rs`）在 smoke 量级约束之上收紧为：

| 量 | 判据 |
|----|------|
| 动能单调衰减 | \(E_{\mathrm{final}} < E_{\mathrm{initial}}\)，\(E/E_0 < 1\) |
| \(E/E_0\) vs 解析 | \(\ge 0.35 \times \exp(-4\nu^* t^*)\)（粗网格允许超耗散，较 `dt=0.05` 已减轻） |
| spin-up 衰减率 | \(-\mathrm{d}\ln E/\mathrm{d}t\) 在 \(0.5\times\)–\(45\times\) 的 \(4\nu^*\) 内 |
| 连续性 | `max_abs_corrected_field_divergence_after_boundary` \(< 10^{-5}\)（当前基线约 \(2.5\times10^{-7}\)） |
| 压力 Poisson 残差 | `max_abs_corrected_divergence` \(< 10^{-5}\) |

机器可读参考值见 `expected.json`（`status = i3_piso_bdf1_kinetic_decay_vv`）。

## 精度路线（后续）

- 集成测试已提供 `#[ignore]` 细网格对照：16×16 与 32×32 在相同物理时间比较 \(E/E_0\) 偏差。
- 当前 32×32 仍可能不优于 16×16（以 CI 日志实测为准）；该测试用于建立可复现实验基线。
- 目标是在改进离散/时间参数后实现 \(|E/E_0-\exp(-4\nu^*t^*)|\) 随网格加密下降，并逐步收紧 I3 必跑容差。

```bash
asimu --case tests/benchmarks/taylor_green_3d/case.toml
cargo test --test case_run taylor_green_3d
```

## 参考文献

1. Brachet, M. E., et al. (1983). Small-scale structure of the Taylor–Green vortex. *Journal of Fluid Mechanics*, 130, 411–452.
2. Ghia et al. (1982) — 方腔对照；本算例为周期 TG 衰减。
