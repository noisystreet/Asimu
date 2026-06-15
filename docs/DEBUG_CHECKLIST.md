# V&V / 无量纲排查清单

> **用途**：metrics 与文献/解析解对不上时，按顺序排除；**不是** golden test 回归调试手册。
> 理论推导见 [theory/nondimensional.md](theory/nondimensional.md)、[theory/incompressible_simplec_piso.md](theory/incompressible_simplec_piso.md)。
> 算例参考值见各 `tests/benchmarks/*/README.md` 与 [BENCHMARKS.md](BENCHMARKS.md)。

---

## 1. 先分清三类问题

勾选当前怀疑的方向（可多选，但**分开查**）：

| 类型 | 典型指标 | 常见根因 | 进一步阅读 |
|------|----------|----------|--------------|
| **A. 连续性 / PISO** | `max\|div(u*)\|`、`max_abs_corrected_divergence` | 首步面通量未播种、Rhie–Chow 与 Poisson RHS 不一致、PISO 迭代不足 | [incompressible_simplec_piso.md §8](theory/incompressible_simplec_piso.md) |
| **B. 时间推进 / coupling** | 首步残差尖峰、spin-up 后仍发散 | `dt` 过大、BDF1 与 pressure corrector 不匹配 | 算例 README 参数敏感性表 |
| **C. 物理量 V&V** | \(E/E_0\)、剖面误差、\(Re\) 相关量 | **解析对照式错误**、无量纲 \(t\) vs 物理 \(t\)、网格过粗 | 本文 §2–§3 |

**规则**：A/B 修好后，C 仍不对 → 优先查 **解析式与无量纲**，不要先改离散系数。

---

## 2. 不可压无量纲对照（必查）

Case TOML 为 **SI**；`CaseSpec` 解析后网格缩至 \([0,1]^d\)，求解器用 \(*\) 变量。

### 2.1 时间

- [ ] **物理时间** \(t_{\mathrm{phys}} = t^* \cdot (L_{\mathrm{ref}}/U_{\mathrm{ref}})\)，不是 TOML 里 `dt × steps` 直接当 \(t^*\)（若未做无量纲化则另论）。
- [ ] **`[time].dt` 在解析后已除以 `time_scale`**（见 `io::nondimensional::scale_incompressible_time`）。
- [ ] README 写「\(t=2\) s」时，代码 metrics 里的 `nondimensional_time` 可能是 \(t^* \approx 0.32\)（例：\(L=2\pi\)、\(U=1\)）。

### 2.2 粘度与 Reynolds

- [ ] 动量扩散系数 **`kinematic_viscosity`（\*）= `1/Re = ν/(U L)`**，不是文献里的 SI \(\nu\) 直接代入。
- [ ] 解析 **`inv_re` / `ν*` 来自 `IncompressibleReferenceScales::inv_reynolds()`**，不要与 FVM 装配里的 `config.kinematic_viscosity` 混用（当前二者相同，但语义不同）。

### 2.3 解析公式（Taylor–Green 等周期衰减）

Brachet 有量纲形式：

\[
\frac{E(t)}{E(0)} = \exp(-4\,\nu\, t_{\mathrm{phys}})
\]

网格缩至 \([0,1]^d\)、初场 \(\sin(2\pi x^*)\)、扩散仍用 \(\nu^*=1/Re\) 时，**代码中等价**为：

\[
\frac{E}{E_0} = \exp(-4\,\nu^*\, L_{\mathrm{ref}}^2\, t^*)
\]

- [ ] **不是** \(\exp(-4\,\nu^*\, t^*)\)（漏 \(L_{\mathrm{ref}}^2\) 会把 0.45 误判成「距 0.98 很远」）。
- [ ] 实现：`case::taylor_green::analytical_kinetic_energy_ratio(inv_re, L, t*)`。
- [ ] spin-up 衰减率基准：**\(4\,\nu^*\, L_{\mathrm{ref}}^2\)**（不是 \(4\nu^*\)）。

### 2.4 手算核对（TG 默认基线）

| 量 | 典型值（`taylor_green_3d`） |
|----|----------------------------|
| \(L_{\mathrm{ref}}\) | \(2\pi\) |
| \(\nu\)（SI 输入） | 0.1 |
| \(t_{\mathrm{phys}}\) | \(400 \times 0.005 = 2\) |
| \(\exp(-4\nu t)\) | \(\approx 0.449\) |
| 16×16 数值 \(E/E_0\) | \(\approx 0.451\)（相对误差 \(<1\%\) 即正常） |

---

## 3. Taylor–Green（I3）专用顺序

按序执行，**上一步未通过不要改离散/粘性系数**：

1. [ ] **解析对照**：手算或 `analytical_kinetic_energy_ratio`，确认目标约为 **0.45**（非 0.98），见 §2.4。
2. [ ] **CI 基线**：`cargo test --test case_run taylor_green_3d_incompressible_benchmark_runs`
3. [ ] **连续性**：`max_abs_corrected_field_divergence_after_boundary` \(< 10^{-6}\)；首步 `max|div(u*)|` 应为 \(10^{-7}\) 量级（Rhie–Chow IC + `initial_face_flux`）。
4. [ ] **细网格**（本地）：
   `cargo test --test case_run taylor_green_3d_refined_grid_reduces_energy_ratio_error -- --ignored --nocapture`
   期望：32×32 相对解析误差 **≤** 16×16。
5. [ ] **参数敏感性**（可选）：`taylor_green_3d_parameter_sensitivity_baseline -- --ignored`

算例细节：[tests/benchmarks/taylor_green_3d/README.md](../tests/benchmarks/taylor_green_3d/README.md)

---

## 4. 其他不可压 benchmark 快查

| `benchmark_id` | 先查什么 | 文档 |
|----------------|----------|------|
| `lid_driven_cavity_re100` | 稳态 SIMPLEC 收敛、Ghia 剖面；**不要**用 TG 衰减公式 | `tests/benchmarks/lid_driven_cavity_re100/` |
| `channel_poiseuille` | 体 force + \(\nu\)、剖面 \(L_2\) 误差 | `tests/benchmarks/channel_poiseuille/` |

---

## 5. 常见误判（反例库）

新增踩坑条目时在此追加一行（现象 → 易误判 → 实际原因）。

| 现象 | 易误判为 | 实际原因 | 修复/参考 |
|------|----------|----------|-----------|
| TG \(E/E_0 \approx 0.45\)，对照「解析 0.98」 | 粘性离散过强、需改 `ν/L²` | 解析用 \(\exp(-4\nu^* t^*)\) **漏 \(L^2\)**；数值符合 Brachet \(\exp(-4\nu t)\) | 修正 `analytical_kinetic_energy_ratio`；commit `c9c751c` |
| 首步 `max\|div(u*)\| \sim 10^{-2}` | Poisson 求解器坏了 | 首步对流用 cell 插值通量，与 Rhie–Chow 投影不一致 | `initial_face_flux` + 动量预测后 pressure-only 对齐 |
| 加密网格 \(E/E_0\) 不变 | 空间离散阶数不够 | 解析对照式本身错误，误差被常数偏移掩盖 | 先 §2.3 再谈网格收敛 |

---

## 6. 常用命令

```bash
make check
cargo test --test case_run taylor_green_3d
cargo test --test case_run taylor_green_3d_refined_grid_reduces_energy_ratio_error -- --ignored --nocapture
cargo run --bin asimu -- --case tests/benchmarks/taylor_green_3d/case.toml
```

产物（本地，不入库）：`tests/benchmarks/taylor_green_3d/out/residual.csv`、`run-manifest.json`。

---

## 7. 相关链接

| 文档 | 内容 |
|------|------|
| [BENCHMARKS.md](BENCHMARKS.md) | V&V 算例库、`expected.json` |
| [CASE_FORMAT.md](CASE_FORMAT.md) | TOML、`[incompressible.reference]` |
| [theory/nondimensional.md](theory/nondimensional.md) | 可压/通用无量纲 |
| [MCP.md](MCP.md) | 规划中的 `debug_divergence`（偏残差/连续性） |
