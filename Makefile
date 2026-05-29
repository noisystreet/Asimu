# asimu 统一命令入口 — 详见 README.md 与 AGENTS.md

CARGO := cargo
PYTHON := python3
CARGO_FLAGS := --features io-vtk

.PHONY: help build run test lint fmt complexity check clean setup audit doc

help:
	@echo "Targets: build run test lint complexity fmt check clean setup audit doc"

build:
	$(CARGO) build $(CARGO_FLAGS)

run: build
	$(CARGO) run $(CARGO_FLAGS) --

test:
	$(CARGO) test $(CARGO_FLAGS)

lint:
	$(CARGO) fmt --check
	$(CARGO) clippy --all-targets $(CARGO_FLAGS) -- -D warnings
	$(MAKE) complexity

complexity:
	$(PYTHON) scripts/complexity_check.py
	$(PYTHON) scripts/test_complexity_check.py

fmt:
	$(CARGO) fmt

check: lint test

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
