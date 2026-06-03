# 可压缩流无量纲化

> 模块：`src/physics/`、`src/io/nondimensional.rs`、`src/discretization/bc_compressible.rs` · 版本：v1.x · 状态：**已实现**
> 算例开关：[CASE_FORMAT.md §6.5](../CASE_FORMAT.md#65-nondimensional可压缩算例可选)

## 1. 参考量与 \(*\) 变量

来流静参数 \((p_\infty, T_\infty, M_\infty)\) 与 SI 物性 \((\gamma, R, \mu)\) 确定参考量（`ReferenceScales::from_freestream`）：

| 符号 | 定义 | 代码字段 |
|------|------|----------|
| \(L_{\mathrm{ref}}\) | \(1\,\mathrm{m}\) | `ReferenceScales::length` |
| \(U_{\mathrm{ref}}\) | \(a_\infty=\sqrt{\gamma R T_\infty}\) | `velocity` |
| \(T_{\mathrm{ref}}\) | \(T_\infty\) | `temperature` |
| \(\mu_{\mathrm{ref}}\) | \(\mu(T_\infty)\) | `viscosity` |
| \(\rho_{\mathrm{ref}}\) | \(p_\infty/(R T_\infty)\) | `density` |
| \(p_{\mathrm{ref}}\) | \(\rho_{\mathrm{ref}} U_{\mathrm{ref}}^2=\gamma p_\infty\) | `pressure` |

任意量 \(q\) 的无量纲值 \(q^*=q/q_{\mathrm{ref}}\)（或按表 1 中对应参考量）。算例 TOML **仍写 SI**；含 `[freestream]` 的可压缩算例**默认**在 `CaseSpec` 解析完成后调用 `io::nondimensional::apply_nondimensionalization`（可用 `[nondimensional] enabled = false` 关闭）。

Reynolds 数与 NS 粘性缩放：

\[
\mathrm{Re}=\frac{\rho_{\mathrm{ref}} U_{\mathrm{ref}} L_{\mathrm{ref}}}{\mu_{\mathrm{ref}}},\qquad
\mu^*=\frac{1}{\mathrm{Re}}\frac{\mu(T)}{\mu_{\mathrm{ref}}}
\]

（`ViscousPhysicsConfig::inv_reynolds`、`face_transport_coefficients`）。

## 2. 来流 \(*\) 约定

启用无量纲后，`[freestream]` 被缩放为：

\[
p^*=\frac{1}{\gamma},\quad T^*=1,\quad \rho^*=1,\quad a^*=1,\quad u^*=M_\infty
\]

**注意**：存储于 EOS 的 \(R^*=U_{\mathrm{ref}}^2/T_{\mathrm{ref}}=\gamma R\) 与来流 \((\rho^*,p^*,T^*)\) **不**满足 \(p^*=\rho^* R^* T^*\)。来流原始变量由 `FreestreamContext::primitive` 显式构造，**禁止**在 BC/初场中单独使用 `p/(RT)`。

有量纲来流仍用理想气体：

\[
\rho=\frac{p}{RT},\quad a=\sqrt{\gamma p/\rho},\quad u=Ma
\]

（`IdealGasEoS::freestream_primitive`）。

## 3. 静温与粘性

### 3.1 静温

有量纲理想气体：

\[
T=\frac{p}{\rho R}
\tag{1}
\]

无量纲 NS（保证来流 \(T^*=1\)）：

\[
T^*=\frac{p^*}{\rho^*}\,\gamma
\tag{2}
\]

Sutherland 等模型所需的有量纲温度：

\[
T = T^*\,T_{\mathrm{ref}}
\tag{3}
\]

（`ViscousPhysicsConfig::dimensional_temperature_from_static`，`temperature_ref = T_{\mathrm{ref}}`）。

无量纲热传导使用 \(c_p^*=1/(\gamma-1)\)（非 \(\gamma R^*/(\gamma-1)\)），见 `ViscousPhysicsConfig::specific_heat_capacity`。

### 3.2 壁面 ghost 密度

等压壁面 ghost 密度：

\[
\rho=\frac{p}{RT}\ \text{（有量纲）},\qquad
\rho^*=\frac{p^*\gamma}{T^*}\ \text{（无量纲）}
\tag{4}
\]

（`FreestreamContext::density_from_pressure_temperature`）。

## 4. 边界条件

远场 / 超声速入口 ghost 与内场来流一致，均经 `FreestreamContext::primitive`（`farfield_ghost`、`inlet_ghost` 超声速分支）。

`apply_compressible_boundary_conditions` 接收 `&FreestreamContext`；模式判定：`CaseSpec.reference.is_some()` 优先，否则 `ViscousPhysicsConfig::is_nondimensional()`。

## 5. 输出还原

`ConservedFields::to_dimensional(reference)` 将 \(*\) 守恒量还原为 SI，供 CGNS/VTK 写出（`case/output_3d`）。

## 6. 实现映射

| 式 / 步骤 | 代码位置 |
|-----------|----------|
| 参考量表 | `physics::ReferenceScales::from_freestream` |
| 算例缩放 | `io::nondimensional::apply_nondimensionalization` |
| 来流单一入口 | `physics::FreestreamContext::{primitive, conserved}` |
| 初场 | `field::ConservedFields::from_freestream_context` ← `CaseSpec::build_conserved_fields` |
| 式 (1)(2) 静温 | `physics::ViscousPhysicsConfig::static_temperature` |
| 式 (3) Sutherland 输入 | `ViscousPhysicsConfig::dimensional_temperature_from_static` |
| 式 (4) ghost \(\rho\) | `FreestreamContext::density_from_pressure_temperature` |
| 粘性通量 \(1/\mathrm{Re}\) | `ViscousPhysicsConfig::face_transport_coefficients` |
| 粘性 CFL | `solver::spectral_radius::cell_viscous_diffusivity_max` |
| Green–Gauss 温度 | `discretization::gradient::cell_temperatures` |
| BC ghost | `discretization::bc_compressible::{farfield_ghost, wall_ghost}` |
| 成对单测 fixture | `discretization::freestream_pair`（`cfg(test)`） |

## 7. 参考文献

1. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics* (3rd ed.). Springer. ISBN 978-3-540-25202-3. Ch. 1（无量纲化与 Mach 数）。
2. Anderson, J. D. (1995). *Computational Fluid Dynamics: The Basics with Applications*. McGraw-Hill. ISBN 978-0-07-001685-9. Ch. 3–4（可压缩参数与边界层相似参数）。
3. White, F. M. (2006). *Viscous Fluid Flow* (3rd ed.). McGraw-Hill. ISBN 978-0-07-240231-5. Sutherland 粘度公式。

## 8. 相关算例与测试

- [CASE_FORMAT.md §6.5](../CASE_FORMAT.md#65-nondimensional可压缩算例可选) — TOML 开关
- `io::nondimensional` 单元测试 — 来流 \(p^*=1/\gamma\)
- `field::conserved::dimensionalize_reverses_reference_scaling` — 往返 SI
- `discretization::freestream_pair` + `for_each_*_side` — 有量纲 / 无量纲成对不变量测试
