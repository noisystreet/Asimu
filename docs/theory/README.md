# 数值理论手册

> 与代码实现一一对应的理论说明与参考文献。
> Agent 约束见 [AGENTS.md](../../AGENTS.md)「数值理论与参考文献」；算例级文献见 [BENCHMARKS.md](../BENCHMARKS.md)。

## 公式写法（Markdown 预览）

理论页使用 **LaTeX 标准定界符**（GitHub、Markdown Preview Enhanced 均支持）：

| 类型 | 写法 | 示例 |
|------|------|------|
| 行内 | `\(...\)` | `\(\phi = \phi_b\)` |
| 独立公式 | `\[...\]` | 块级公式；编号用 `\tag{n}` |

部分较早页面（如 `fvm_diffusion.md`）仍用 `$$...$$`，与上表等价；**新稿优先 `\(...\)` / `\[...\]`**，避免 `$` 与正文货币符号混淆。

### Cursor / VS Code 预览

内置预览（`Ctrl+Shift+V`）对 `\(...\)` 支持有限；推荐：

| 方式 | 操作 |
|------|------|
| **Markdown Preview Enhanced**（推荐） | 命令面板 → **`Markdown Preview Enhanced: Open Preview to the Side`** |
| 浏览器 | `./scripts/preview_theory_md.sh docs/theory/fvm_diffusion.md`（pandoc + MathJax） |

工作区见 [`.vscode/settings.json`](../../.vscode/settings.json) 与推荐扩展 [`.vscode/extensions.json`](../../.vscode/extensions.json)。

---

| 读者 | 用途 |
|------|------|
| 审查者 | 核对离散公式、假设与文献是否一致 |
| 维护者 | 改公式前先读对应理论页 |
| V&V | 与 `tests/benchmarks/` 算例 README 交叉引用 |

**不写在这里的内容**：架构分层、数据 schema → [ARCHITECTURE.md](../ARCHITECTURE.md)、[DATA_MODEL.md](../DATA_MODEL.md)；重大选型 → [adr/](../adr/)。

---

## 索引

| 文档 | 模块 | 版本 | 状态 | 主要参考 |
|------|------|------|------|----------|
| [fvm_diffusion.md](fvm_diffusion.md) | `discretization` | v0.2 | **骨架** | Patankar (1980) Ch. 5 |
| [heat_conduction_fvm.md](heat_conduction_fvm.md) | `discretization` / `solver` / `case` | v0.3+ | **设计** | Patankar (1980) Ch. 5–6；ADR 0016 |
| [interface_reconstruction.md](interface_reconstruction.md) | `discretization/reconstruction` | v1.x | **已实现（结构化 MUSCL）** | LeVeque (2002) Ch. 4；Toro (2009) |
| [inviscid_flux.md](inviscid_flux.md) | `discretization/roe` | v1.x | **已实现（Roe + 熵修正）** | Roe (1981)；Toro (2009) Ch. 10–11 |
| [unstructured_fvm.md](unstructured_fvm.md) | `mesh/unstructured` / `discretization/residual` | v1.x | **已实现（一阶 Euler）；M4 二阶设计 + ADR 0012** | Blazek (2015)；Barth & Jespersen (1989)；Venkatakrishnan (1993) |
| *(待建)* `fvm_convection_diffusion.md` | `discretization` | v0.2 | 规划 | Patankar (1980) Ch. 5–6 |
| [boundary_conditions.md](boundary_conditions.md) | `discretization` / BC | v0.2–v0.3 | **已实现（v0.2 Dirichlet/Neumann；§9 不可压规划）** | [boundary_conditions.md](boundary_conditions.md) |
| [time_integration.md](time_integration.md) | `solver/time` | v1.x | **已实现（RK4 + LU-SGS + CFL）** | Blazek (2015) §6.1.4/§9.1；ADR 0005 |
| [nondimensional.md](nondimensional.md) | `physics` / `io` / BC | v1.x | **已实现** | Toro (2009) Ch. 1；Anderson (1995) |
| [structured_gradients.md](structured_gradients.md) | `discretization/gradient` | v1.x | **已实现** | Vinokur (1989)；Ferziger et al. Ch. 8 |
| [curvilinear_metrics.md](curvilinear_metrics.md) | `mesh/metrics` | v1.x | **规划** | Vinokur (1989)；CFL3D TM-2010-216758 |
| [linear_gmres.md](linear_gmres.md) | `linalg` | v1.x | **已实现（GMRES + ILU(0)/LU-SGS 对角预条件）** | Saad (2003) Ch. 6、Ch. 10 |
| [turbulence_k_omega_sst.md](turbulence_k_omega_sst.md) | `physics/turbulence` / `discretization/turbulence` | v1.x | **规划（ADR 0014）** | Menter (1994, 2003)；Wilcox (2006) Ch. 4；Blazek (2015) §10 |
| [incompressible_simplec_piso.md](incompressible_simplec_piso.md) | `solver/incompressible` · `discretization/incompressible` | v0.3 | **已实现（I2 稳态 SIMPLEC + Ghia；I3 瞬态 PISO/BDF1 + TG smoke）** | Patankar (1980) Ch. 6–7；Issa (1986)；Ferziger et al. Ch. 8–9 |
| *(待建)* `compressible_ns.md` | `discretization` / `physics` | v1.x | 部分（Euler 无粘见上表） | [adr/0009](../adr/0009-compressible-navier-stokes.md)；Toro (2009) |

实现对应功能时：将「规划」改为链接，并从索引表移除 *(待建)* 前缀。

---

## 新建理论页模板

复制以下内容为新文件 `docs/theory/{topic}.md`，替换 `{...}` 占位符。

```markdown
# {标题}

> 模块：`src/{module}/` · 版本：v0.x · 状态：{草稿|已实现}

## 1. 控制方程 / 算法

{连续形式或算法步骤，式编号从 (1) 起；行内 `\(...\)`，块级 `\[...\]` + `\tag{n}`}

## 2. 离散化

- 网格假设：{结构化 FVM / …}
- 离散格式：{中心差分 / upwind / …}
- 稳定性 / 守恒性：{简要说明}

## 3. 边界条件（如适用）

| BC 类型 | 数学条件 | 离散处理 | 代码入口 |
|---------|----------|----------|----------|
| … | … | … | `apply_*` |

## 4. 实现映射

| 式 / 步骤 | 代码位置 |
|-----------|----------|
| (1) | `{module}::{fn}` |
| (2) | … |

## 5. 参考文献

1. {Author} ({Year}). *{Title}*. {Publisher/Journal}. {DOI or ISBN}
2. …

## 6. 相关算例

- `tests/benchmarks/{id}/` — {验证量}
```

---

## 维护规则

1. **新增**离散、BC、时间推进、本构或非平凡求解器 → 新增或扩展本目录页面。
2. **修改**公式、容差或参考值 → 同步更新理论页与 [CHANGELOG.md](../../CHANGELOG.md)。
3. 模块 rustdoc 顶部加一行：`/// 理论：docs/theory/{topic}.md`。
4. 算例专用文献仍写在 `tests/benchmarks/{id}/README.md`，理论页可链过去，避免重复粘贴大段表格。

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) — 何时必须写理论页
- [BENCHMARKS.md](../BENCHMARKS.md) — V&V 算例与 `expected.json`
- [adr/0002](../adr/0002-layered-cfd-architecture.md) — v0.2 数值基线（FVM + 结构化网格）
