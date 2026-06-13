# asimu 统一命令入口 — 详见 README.md 与 AGENTS.md

CARGO := cargo
PYTHON := python3
# Cargo default features 已含 io-cgns-vts（io-cgns + io-vtk）与 parallel-fvm。
CARGO_FLAGS :=
CARGO_SIMD_FLAGS := --features simd-fvm
CARGO_SCALAR_FLAGS := --no-default-features --features io-vtk

.PHONY: help build run test test-parallel-fvm lint fmt complexity check check-parallel-fvm clean setup audit doc

help:
	@echo "Targets: build run test lint complexity fmt check clean setup audit doc"

build:
	$(CARGO) build $(CARGO_FLAGS)

run: build
	$(CARGO) run $(CARGO_FLAGS) --

run-case:
	@test -n "$(CASE)" || { echo "用法: make run-case CASE=tests/benchmarks/sod_1d/case.toml"; exit 1; }
	$(CARGO) run $(CARGO_FLAGS) -- --case $(CASE)

test:
	$(CARGO) test $(CARGO_FLAGS)

test-parallel-fvm:
	$(CARGO) test $(CARGO_FLAGS)

test-simd-fvm:
	$(CARGO) test $(CARGO_SIMD_FLAGS)

lint:
	$(CARGO) fmt --check
	$(CARGO) clippy --all-targets $(CARGO_FLAGS) -- -D warnings
	$(MAKE) complexity

complexity:
	$(PYTHON) scripts/complexity_check.py
	$(PYTHON) scripts/test_complexity_check.py

fmt:
	$(CARGO) fmt

check-exec-parallel-scatter:
	$(CARGO) test $(CARGO_FLAGS) exec::scatter::tests
	$(CARGO) test $(CARGO_FLAGS) \
		discretization::residual::assembly_unstructured_tests::exec_context_cpu_scalar_matches_legacy_path \
		discretization::residual::assembly_unstructured_viscous_tests::viscous_interior_one_scatter_invocation_per_color_bucket

check-exec-parallel-scatter-simd:
	$(CARGO) test $(CARGO_SIMD_FLAGS) exec::scatter::tests \
		discretization::residual::assembly_unstructured_viscous_tests::viscous_interior_one_scatter_invocation_per_color_bucket

check: lint test

check-parallel-fvm: check
	@echo "parallel-fvm 已包含在默认 CARGO_FLAGS / Cargo default features 中"

clean:
	$(CARGO) clean

setup:
	@command -v pre-commit >/dev/null 2>&1 || { echo "请先安装 pre-commit: pip install pre-commit"; exit 1; }
	pre-commit install --install-hooks -t pre-commit -t commit-msg

audit:
	@command -v cargo-audit >/dev/null 2>&1 || { echo "请先安装: cargo install cargo-audit"; exit 1; }
	cargo audit

doc:
	$(CARGO) doc --no-deps --open

probe-vts:
	@test -n "$(FILE)" || { echo "用法: make probe-vts FILE=/path/to/mesh.vts"; exit 1; }
	$(CARGO) run --example probe_vts $(CARGO_FLAGS) -- $(FILE)

cgns-to-vts:
	@test -n "$(IN)" && test -n "$(OUT)" || { echo "用法: make cgns-to-vts IN=mesh.cgns OUT=out.vts [ZONE=1]"; exit 1; }
	$(CARGO) run --example cgns_to_vts $(CARGO_FLAGS) -- $(IN) $(OUT) $(if $(ZONE),--zone $(ZONE),)

sod-export:
	@test -n "$(OUT)" || { echo "用法: make sod-export OUT=sod_profile.txt [CELLS=100]"; exit 1; }
	$(CARGO) run --example sod_benchmark_export -- $(OUT) $(if $(CELLS),--cells $(CELLS),)

sod-plot:
	@test -n "$(FILE)" || { echo "用法: make sod-plot FILE=sod_profile.txt [PNG=sod_compare.png]"; exit 1; }
	$(PYTHON) scripts/plot_sod_benchmark.py $(FILE) $(if $(PNG),-o $(PNG),)

test-cgns:
	$(CARGO) test $(CARGO_FLAGS)

check-cgns: lint
	$(CARGO) test $(CARGO_FLAGS)
