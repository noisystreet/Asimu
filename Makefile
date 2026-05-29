# asimu 统一命令入口 — 详见 README.md 与 AGENTS.md

CARGO := cargo
PYTHON := python3

.PHONY: help build run test lint fmt complexity check clean setup audit doc

help:
	@echo "Targets: build run test lint complexity fmt check clean setup audit doc"

build:
	$(CARGO) build

run: build
	$(CARGO) run --

test:
	$(CARGO) test

lint:
	$(CARGO) fmt --check
	$(CARGO) clippy --all-targets -- -D warnings
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
