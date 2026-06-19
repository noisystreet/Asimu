# 双时间步内层成败矩阵

**用途**：量化 `time.scheme = "dual_time"` 内层 \(\|R_{\mathrm{eff}}\|\) 是否随 pseudo-time 迭代下降，形成可回归的配置-成败表。

理论见 [dual_time_stepping.md](../../../docs/theory/dual_time_stepping.md)。

## 判定指标

每个探针从日志 `dual_time 内迭代残差`（info 级）提取：

| 字段 | 含义 |
|------|------|
| inner1 log10 | 首个内层 \(\log_{10}\|R_{\mathrm{eff,rms}}\|\) |
| innerN log10 | 末个内层（早停或 `max_inner_steps`） |
| drop | inner1 − innerN |
| verdict | PASS / FAIL / OBSERVE / SKIP |

默认 **PASS** 条件：`drop ≥ min_drop` 且至少 2 个内层采样。

## CI 内置探针（必跑）

单四面体 uniform freestream（网格由测试注入，与 `unstructured_dual_time_freestream` 一致）：

| probe | 变化 |
|-------|------|
| `freestream_f64_baseline` | 默认 cfl=0.4 |
| `freestream_f64_high_cfl` | cfl=100 |
| `freestream_f64_sweep` | lusgs_sweep=true |
| `freestream_f64_low_omega` | lusgs_omega=0.1 |
| `freestream_f32_cpu` | compute_precision=f32 |

```bash
cargo test --test dual_time_inner_regression dual_time_inner_matrix_builtin_probes -- --nocapture
```

`--nocapture` 会打印 Markdown 汇总表。

## 外部大算例

编辑 `probes_external.json` 或使用环境变量覆盖路径，然后：

```bash
ASIMU_DUAL_TIME_MATRIX=tests/benchmarks/dual_time_inner_matrix/probes_external.json \
cargo test --test dual_time_inner_regression dual_time_inner_matrix_from_file -- --nocapture
```

`skip_if_missing: true` 时本地无 CGNS / case 文件则跳过，不失败。

### 涡街探针建议

在 `output/case_hex_votexstreet/` 下维护专用探针 TOML（`max_steps=1` 加速），例如：

| 探针 id | 配置要点 | 期望 |
|---------|----------|------|
| `hex_no_restart` | freestream 初场、first_order | decrease |
| `hex_restart_s5000` | restart @ step5000 | decrease |
| `hex_restart_s7000` | restart @ step7000+ | observe（已知难例） |

## 单算例 ad-hoc

```bash
ASIMU_DUAL_TIME_CASE=/abs/path/case.toml \
ASIMU_DUAL_TIME_MIN_DROP=0.05 \
cargo test --test dual_time_inner_regression dual_time_inner_single_case -- --nocapture
```

## 相关文件

- 解析与判定：`tests/common/dual_time_inner.rs`
- 集成测试：`tests/dual_time_inner_regression.rs`
- smoke 基准：`tests/benchmarks/unstructured_dual_time_freestream/`
