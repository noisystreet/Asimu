# 低马赫 `M_cut` 扫参（hex_vortex_street 定常）

> 日期：2026-06-19
> 前置：[low_mach_ab_hex_vortex.md](low_mach_ab_hex_vortex.md)（A/B，`cfl_max=100`）

## 1. 目的

在更保守 CFL 上限下，扫描 `low_mach_mach_cutoff ∈ {0.05, 0.1, 0.2}`，观察 P1 预处理（仅 \(\sigma,\Delta\tau\) 声速缩放）对定常残差趋势的影响。

## 2. 运行配置

与 A/B 探针相同，除下列差异外保持一致：

| 项 | A/B（已记录） | 本次扫参 |
|----|---------------|----------|
| `cfl_max` | 100 | **20** |
| `low_mach_preconditioning` | OFF / ON | **全部 ON** |
| `low_mach_mach_cutoff` | 0.1（ON 组） | **0.05 / 0.1 / 0.2** |
| 其余 | `Ma=0.1`，`lu_sgs`，`f64+cpu`，800 步 | 同左 |

Case 文件（本地探针，不入库）：

- `output/case_hex_votexstreet/case_steady_probe_lm_cutoff_0p05.toml`
- `output/case_hex_votexstreet/case_steady_probe_lm_cutoff_0p1.toml`
- `output/case_hex_votexstreet/case_steady_probe_lm_cutoff_0p2.toml`

## 3. 指标（`residual.csv`，800 步）

| Case | steps | first log10 | last log10 | drop | slope/step | min | max | reach -3 |
|------|------:|------------:|-----------:|-----:|-----------:|----:|----:|:--------:|
| OFF baseline（A/B，`cfl_max=100`） | 800 | 2.0022 | 1.0673 | +0.935 | -1.17e-3 | -0.22 | 2.00 | 否 |
| ON `M_cut=0.05` | 800 | 2.0022 | 2.6588 | **-0.657** | +8.22e-4 | 1.76 | 3.46 | 否 |
| ON `M_cut=0.10` | 800 | 2.0022 | 2.6692 | **-0.667** | +8.35e-4 | 1.76 | 3.84 | 否 |
| ON `M_cut=0.20` | 800 | 2.0022 | 3.3573 | **-1.355** | +1.70e-3 | 1.64 | 4.32 | 否 |

> 注：OFF baseline 来自 A/B 实验（`cfl_max=100`），与扫参 CFL 不同，仅作方向参考；扫参组之间 CFL 一致，可直接比较 cutoff。

## 4. 观察

1. **三者均未收敛**：末步 `log10_residual` 均高于初值，斜率为正；中间段 `min` 约 1.6–1.8，说明曾短暂下降后反弹。
2. **cutoff 越小并未改善**：`M_cut=0.05` 与 `0.10` 末值接近；`0.20` 发散最快（末值 3.36，峰值 4.32）。
3. **与 A/B 对照**：在 `cfl_max=100` 时 OFF 缓慢下降而 ON 爆炸；降至 `cfl_max=20` 后 ON 不再爆炸至 10² 量级，但仍不稳定。
4. **P1 局限（P2 已部分修复）**：P1 扫掠 \(\lambda_{ij}\) 未与 \(\sigma^\text{LM}\) 一致；P2 验证见 [low_mach_p2_hex_vortex.md](low_mach_p2_hex_vortex.md)。

## 5. 结论与下一步

- 在本涡街定常探针上，**P1 单独启用 + 保守 CFL 仍不足以稳定收敛**；cutoff 在 `{0.05, 0.1, 0.2}` 内无明确最优。
- **P2 已完成**（见 [low_mach_p2_hex_vortex.md](low_mach_p2_hex_vortex.md)）：高 CFL 发散消除，但 ON 仍不如 OFF；可继续扫更低 `cfl_max` 或长跑主 case。
