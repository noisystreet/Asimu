#!/usr/bin/env python3
"""Rust 代码复杂度门禁（基于 lizard）。

固定阈值：
  文件行数 ≤ 800
  函数行数 ≤ 150
  函数参数 ≤ 8
  圈复杂度 ≤ 15
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import lizard
except ImportError as exc:
    print("缺少 lizard，请安装: pip install lizard", file=sys.stderr)
    raise SystemExit(2) from exc

MAX_FILE_LINES = 800
MAX_FUNCTION_LINES = 150
MAX_FUNCTION_PARAMS = 8
MAX_CYCLOMATIC_COMPLEXITY = 15

SKIP_DIRS = {"target", ".git"}


@dataclass(frozen=True)
class Violation:
    path: Path
    rule: str
    detail: str


def collect_rust_files(paths: list[str]) -> list[Path]:
    if paths:
        return sorted({Path(path).resolve() for path in paths if path.endswith(".rs")})

    root = Path.cwd()
    files: list[Path] = []
    for path in root.rglob("*.rs"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path.resolve())
    return sorted(files)


def count_physical_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def check_file(path: Path) -> list[Violation]:
    violations: list[Violation] = []
    line_count = count_physical_lines(path)
    if line_count > MAX_FILE_LINES:
        violations.append(
            Violation(
                path=path,
                rule="file-lines",
                detail=f"文件 {line_count} 行，上限 {MAX_FILE_LINES}",
            )
        )

    analysis = lizard.analyze_file(str(path))
    for function in analysis.function_list:
        name = function.name
        if function.length > MAX_FUNCTION_LINES:
            violations.append(
                Violation(
                    path=path,
                    rule="function-lines",
                    detail=(
                        f"函数 `{name}` 共 {function.length} 行"
                        f"（{function.start_line}-{function.end_line}），"
                        f"上限 {MAX_FUNCTION_LINES}"
                    ),
                )
            )
        if function.parameter_count > MAX_FUNCTION_PARAMS:
            violations.append(
                Violation(
                    path=path,
                    rule="function-params",
                    detail=(
                        f"函数 `{name}` 有 {function.parameter_count} 个参数，"
                        f"上限 {MAX_FUNCTION_PARAMS}"
                    ),
                )
            )
        if function.cyclomatic_complexity > MAX_CYCLOMATIC_COMPLEXITY:
            violations.append(
                Violation(
                    path=path,
                    rule="cyclomatic-complexity",
                    detail=(
                        f"函数 `{name}` 圈复杂度 {function.cyclomatic_complexity}，"
                        f"上限 {MAX_CYCLOMATIC_COMPLEXITY}"
                    ),
                )
            )

    return violations


def format_violation(violation: Violation) -> str:
    rel = violation.path
    try:
        rel = violation.path.relative_to(Path.cwd())
    except ValueError:
        pass
    return f"{rel}: [{violation.rule}] {violation.detail}"


def main() -> int:
    files = collect_rust_files(sys.argv[1:])
    if not files:
        return 0

    all_violations: list[Violation] = []
    for file_path in files:
        all_violations.extend(check_file(file_path))

    if not all_violations:
        return 0

    print("代码复杂度检查失败:", file=sys.stderr)
    for violation in all_violations:
        print(format_violation(violation), file=sys.stderr)
    print(
        "\n阈值: "
        f"文件<={MAX_FILE_LINES} 行, "
        f"函数<={MAX_FUNCTION_LINES} 行, "
        f"参数<={MAX_FUNCTION_PARAMS} 个, "
        f"圈复杂度<={MAX_CYCLOMATIC_COMPLEXITY}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
