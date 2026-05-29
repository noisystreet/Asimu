# Four cross-cutting capabilities (planned)

See [ARCHITECTURE.md](../ARCHITECTURE.md) §4.3 · §8.5–§8.6 for full detail.

| # | Capability | Doc | First delivery |
|---|------------|-----|----------------|
| 1 | **Run Manifest** | [DATA_MODEL.md](../DATA_MODEL.md) §10 | v0.3 |
| 2 | **Time integration** | [ADR 0005](../adr/0005-time-integration.md) | v0.2 SteadyState |
| 3 | **V&V benchmark canon** | [BENCHMARKS.md](../BENCHMARKS.md) | v0.2 1D cases |
| 4 | **Performance & observability** | [OBSERVABILITY.md](../OBSERVABILITY.md) | v0.3 wall time in manifest |

## Run Manifest

- `output/run-manifest.json` after each run
- Fields: `run_id`, versions, `config_hash`, `solve`, `benchmark_id`, `observability`
- MCP: `asimu://run/latest` (v1.2+)

## Time integration

- `TimeIntegrator` in `solver/time/`
- `SteadyStateIntegrator` (v0.2) → `ExplicitEulerIntegrator` (v0.4)
- Config: `[time] mode`, `dt`, `cfl_max`

## V&V benchmarks

- Directory: `tests/benchmarks/{id}/` with README + case.toml + expected.json
- Distinct from golden tests in `tests/fixtures/`
- `make bench` (planned)

## Observability

- L1: tracing (done)
- L2: `metrics.jsonl` (v0.4+)
- L3: manifest + optional `--profile` flamegraph
- Micro: `benches/` + criterion (v0.4+)
