# asimu Architecture Design

> Canonical detail: [../ARCHITECTURE.md](../ARCHITECTURE.md) (Chinese) · Data model: [../DATA_MODEL.md](../DATA_MODEL.md)

---

## Overview

**asimu** is a modular Rust CFD solver for developers and researchers. Architecture follows the **CFD pipeline**:

```
Read case → Build mesh → Init fields → Discretize → Solve → Output
```

## Layering

```
Application (CLI / app)
    ↓
config · io · app · solver (orchestration)
    ↓
mesh ← field ← discretization
physics · linalg
    ↓
exec (CPU / GPU backend, v1.2+)
    ↓
core · error
```

### Dependency rules

- `core` must not depend on domain modules
- `solver` orchestrates; it must not contain flux formulas or GPU kernels
- `discretization` / `linalg` call hot ops via `exec`, not raw GPU APIs
- `io` produces data structures only — no numerical assumptions at parse time

See [AGENTS.md](../../AGENTS.md) for hard constraints.

## Multi-precision (planned)

| Phase | Capability |
|-------|------------|
| v0.2–v0.4 | `pub type Real = f64`; public APIs use `Real` |
| v0.5 | Cargo feature `precision-f32` |
| v0.6+ | `mixed`: f32 fields + f64 residuals |

Geometry coordinates stay `f64`. See ADR [0003](../adr/0003-multi-precision-and-gpu.md).

## GPU / execution backend (planned)

- New **`exec`** module (v1.2+): `ExecutionContext`, `ExecBackend` trait
- GPU targets: flux assembly, SpMV — not BC / I/O / convergence control
- Features: `gpu-wgpu` (preferred), `gpu-cuda` (optional)
- Main crate stays `unsafe`-free; GPU drivers isolated in `exec/gpu/` or `asimu-exec` crate

## MCP integration (planned v1.1+)

- Independent **`asimu-mcp`** binary (stdio transport); not built by default
- Tools: `validate_config`, `run_case`, `list_fixtures`, `get_run_summary`
- Resources: docs + fixture URIs; v1.2+ `asimu://run/latest` (Run Manifest)
- Adapter layer only — see [MCP.md](../MCP.md) and ADR 0004

## Run artifacts & V&V (planned)

See [CROSS_CUTTING.md](CROSS_CUTTING.md) for the four cross-cutting capabilities summary.

- **Run Manifest** (`output/run-manifest.json`) — reproducibility metadata (v0.3+)
- **Time integration** — ADR 0005; `TimeIntegrator` in `solver/time/` (v0.2 SteadyState)
- **V&V benchmarks** — [BENCHMARKS.md](../BENCHMARKS.md); `tests/benchmarks/` (v0.2 1D)
- **Performance & observability** — [OBSERVABILITY.md](../OBSERVABILITY.md); tracing + metrics + manifest (v0.3+)
- **Checkpoint/restart** — `.asimu-restart` (v0.4+); see DATA_MODEL §12
- **BC framework** — apply phase after interior flux assembly
- **Study mode** — parameter sweeps at orchestration layer (v0.5+)
- **I/O security** — file size / cell count limits; [SECURITY.md](../../SECURITY.md)
- **FFI/Python** — ADR 0006; v1.x+ narrow C ABI

## Module responsibilities

| Module | Role |
|--------|------|
| `core` | Math types, `Real`, constants |
| `mesh` | Topology and geometry |
| `field` | SoA storage for DOFs (planned v0.2) |
| `discretization` | Gradients, fluxes, assembly (planned v0.2) |
| `physics` | Constitutive laws (planned v0.3) |
| `linalg` | Sparse matrices, iterative solvers (planned v0.2) |
| `exec` | CPU/GPU backend (planned v1.2) |
| `solver` | Time marching, coupling, convergence |
| `io` | Case / VTK adapters |
| `case` | End-to-end pipeline orchestration (planned; evolves from `app`) |
| `app` | CLI orchestration (`app::run`) — application layer, not core lib API |

## v0.2 numerical baseline (ADR 0002)

- FVM on 2D structured grids
- Steady convection-diffusion
- Hand-rolled sparse matrix + CG
- Single-threaded CPU; `Real = f64`

## Toolchain

- Rust ≥ 1.85, edition 2024
- Primary target: Linux x86_64
- Lockfile: committed
- `unsafe`: forbidden in main crate

## Configuration

CLI → `ASIMU_*` env → `config/default.toml` → code defaults.

Planned `[numerics] precision` / `backend` — commented in `config/default.toml`.

## Quality gates

- `cargo fmt` / `clippy -D warnings`
- `scripts/complexity_check.py`（lizard）：file ≤800 lines, function ≤150 lines, ≤8 params, CCN ≤15

## Roadmap (extended)

| Version | Multi-precision | Backend |
|---------|-----------------|---------|
| v0.2–v0.4 | `Real = f64` | CPU |
| v0.5 | `precision-f32` | CPU |
| v0.6 | mixed precision | CPU |
| v1.0 | stable API | CPU + rayon |
| v1.1 | MCP server prototype | `asimu-mcp` stdio tools |
| v1.2 | MCP resources + f32/f64 golden | `exec` + wgpu prototype |
| v1.3+ | MCP prompts; optional CUDA | GPU + mixed tuning |

## ADRs

- [0001](../adr/0001-rust-cfd-foundation.md) — Rust foundation
- [0002](../adr/0002-layered-cfd-architecture.md) — Layered CFD architecture
- [0003](../adr/0003-multi-precision-and-gpu.md) — Multi-precision and GPU
- [0004](../adr/0004-mcp-integration.md) — MCP integration
- [0005](../adr/0005-time-integration.md) — Time integration
- [0006](../adr/0006-ffi-interop.md) — FFI / Python interop
- [0007](../adr/0007-vts-binary-io.md) — VTS binary I/O
- [0008](../adr/0008-cgns-io.md) — CGNS read / VTS export
- [0009](../adr/0009-compressible-navier-stokes.md) — 3D compressible Navier-Stokes solver architecture
- [0010](../adr/0010-unstructured-mixed-mesh.md) — Unstructured mixed-cell mesh (face-topology roadmap M1–M4)

## Open decisions

- MPI parallelism (future ADR)
- Unstructured mesh: see [ADR 0010](../adr/0010-unstructured-mixed-mesh.md) (M1 topology in progress)
- Turbulence models (v1.x+)
