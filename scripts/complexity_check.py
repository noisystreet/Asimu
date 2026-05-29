#!/usr/bin/env python3
"""Rust 代码复杂度门禁：文件行数、函数行数、函数参数个数。

默认阈值（可通过环境变量覆盖）：
  ASIMU_MAX_FILE_LINES=800
  ASIMU_MAX_FUNCTION_LINES=150
  ASIMU_MAX_FUNCTION_PARAMS=8
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_LINES = int(os.environ.get("ASIMU_MAX_FILE_LINES", "800"))
MAX_FUNCTION_LINES = int(os.environ.get("ASIMU_MAX_FUNCTION_LINES", "150"))
MAX_FUNCTION_PARAMS = int(os.environ.get("ASIMU_MAX_FUNCTION_PARAMS", "8"))

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SKIP_DIRS = {"target", ".git"}


@dataclass(frozen=True)
class FunctionSpan:
    name: str
    start_line: int
    end_line: int
    param_count: int


@dataclass(frozen=True)
class Violation:
    path: Path
    rule: str
    detail: str


def line_number_at(source: str, index: int) -> int:
    return source.count("\n", 0, index) + 1


def is_ident_start(ch: str) -> bool:
    return ch == "_" or ch.isalpha()


def is_ident_part(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


def skip_whitespace(source: str, index: int) -> int:
    n = len(source)
    while index < n and source[index] in " \t\r\n":
        index += 1
    return index


def read_ident(source: str, index: int) -> tuple[str, int]:
    start = index
    n = len(source)
    while index < n and is_ident_part(source[index]):
        index += 1
    return source[start:index], index


def skip_line_comment(source: str, index: int) -> int:
    n = len(source)
    while index < n and source[index] != "\n":
        index += 1
    return index


def skip_block_comment(source: str, index: int) -> int:
    n = len(source)
    depth = 1
    index += 2
    while index < n and depth > 0:
        if source[index : index + 2] == "/*":
            depth += 1
            index += 2
        elif source[index : index + 2] == "*/":
            depth -= 1
            index += 2
        else:
            index += 1
    return index


def skip_string_like(source: str, index: int) -> int:
    quote = source[index]
    index += 1
    n = len(source)
    while index < n:
        ch = source[index]
        if ch == "\\":
            index += 2
            continue
        if ch == quote:
            return index + 1
        index += 1
    return index


def skip_raw_string(source: str, index: int) -> int:
    n = len(source)
    if index + 1 >= n or source[index + 1] != "#":
        return skip_string_like(source, index)
    hash_count = 0
    pos = index + 1
    while pos < n and source[pos] == "#":
        hash_count += 1
        pos += 1
    if pos >= n or source[pos] != '"':
        return pos
    pos += 1
    end_marker = f'"{"#" * hash_count}'
    while pos < n:
        close = source.find(end_marker, pos)
        if close == -1:
            return n
        return close + len(end_marker)
    return n


def find_matching_paren(source: str, open_index: int) -> int:
    depth = 0
    index = open_index
    n = len(source)
    while index < n:
        ch = source[index]
        if ch == "/" and index + 1 < n:
            if source[index + 1] == "/":
                index = skip_line_comment(source, index)
                continue
            if source[index + 1] == "*":
                index = skip_block_comment(source, index)
                continue
        if ch == "r" and index + 1 < n and source[index + 1] in {'"', "#"}:
            index = skip_raw_string(source, index)
            continue
        if ch in "\"'":
            index = skip_string_like(source, index)
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def find_matching_brace(source: str, open_index: int) -> int:
    depth = 0
    index = open_index
    n = len(source)
    while index < n:
        ch = source[index]
        if ch == "/" and index + 1 < n:
            if source[index + 1] == "/":
                index = skip_line_comment(source, index)
                continue
            if source[index + 1] == "*":
                index = skip_block_comment(source, index)
                continue
        if ch == "r" and index + 1 < n and source[index + 1] in {'"', "#"}:
            index = skip_raw_string(source, index)
            continue
        if ch in "\"'":
            index = skip_string_like(source, index)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def count_parameters(params_source: str) -> int:
    params_source = params_source.strip()
    if not params_source:
        return 0

    count = 0
    depth_angle = 0
    depth_paren = 0
    depth_bracket = 0
    token: list[str] = []

    def flush_token() -> None:
        nonlocal count, token
        if "".join(token).strip():
            count += 1
        token = []

    for ch in params_source:
        if ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle = max(0, depth_angle - 1)
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)

        if (
            ch == ","
            and depth_angle == 0
            and depth_paren == 0
            and depth_bracket == 0
        ):
            flush_token()
            continue
        token.append(ch)

    flush_token()
    return count


def parse_functions(source: str) -> list[FunctionSpan]:
    functions: list[FunctionSpan] = []
    index = 0
    n = len(source)

    while index < n:
        ch = source[index]
        if ch == "/" and index + 1 < n:
            if source[index + 1] == "/":
                index = skip_line_comment(source, index)
                continue
            if source[index + 1] == "*":
                index = skip_block_comment(source, index)
                continue
        if ch == "r" and index + 1 < n and source[index + 1] in {'"', "#"}:
            index = skip_raw_string(source, index)
            continue
        if ch in "\"'":
            index = skip_string_like(source, index)
            continue

        if ch == "f" and source[index : index + 2] == "fn":
            before = source[index - 1] if index > 0 else " "
            if is_ident_part(before):
                index += 1
                continue

            fn_start = index
            fn_line = line_number_at(source, fn_start)
            index += 2
            index = skip_whitespace(source, index)
            name, index = read_ident(source, index)
            if not name:
                continue

            index = skip_whitespace(source, index)
            if index >= n or source[index] != "(":
                continue

            close_paren = find_matching_paren(source, index)
            if close_paren == -1:
                continue

            params = source[index + 1 : close_paren]
            param_count = count_parameters(params)
            index = close_paren + 1
            index = skip_whitespace(source, index)

            end_line = fn_line
            if index < n and source[index] == "{":
                close_brace = find_matching_brace(source, index)
                if close_brace == -1:
                    continue
                end_line = line_number_at(source, close_brace)
                index = close_brace + 1
            else:
                while index < n and source[index] != ";":
                    if source[index] == "/" and index + 1 < n:
                        if source[index + 1] == "/":
                            index = skip_line_comment(source, index)
                            continue
                        if source[index + 1] == "*":
                            index = skip_block_comment(source, index)
                            continue
                    if source[index] in "\"'":
                        index = skip_string_like(source, index)
                        continue
                    if (
                        source[index] == "r"
                        and index + 1 < n
                        and source[index + 1] in {'"', "#"}
                    ):
                        index = skip_raw_string(source, index)
                        continue
                    index += 1
                end_line = line_number_at(source, min(index, n - 1))

            functions.append(
                FunctionSpan(
                    name=name,
                    start_line=fn_line,
                    end_line=end_line,
                    param_count=param_count,
                )
            )
            continue

        index += 1

    return functions


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


def check_file(path: Path) -> list[Violation]:
    violations: list[Violation] = []
    source = path.read_text(encoding="utf-8")
    line_count = len(source.splitlines())

    if line_count > MAX_FILE_LINES:
        violations.append(
            Violation(
                path=path,
                rule="file-lines",
                detail=f"文件 {line_count} 行，上限 {MAX_FILE_LINES}",
            )
        )

    for function in parse_functions(source):
        fn_lines = function.end_line - function.start_line + 1
        if fn_lines > MAX_FUNCTION_LINES:
            violations.append(
                Violation(
                    path=path,
                    rule="function-lines",
                    detail=(
                        f"函数 `{function.name}` 共 {fn_lines} 行"
                        f"（{function.start_line}-{function.end_line}），"
                        f"上限 {MAX_FUNCTION_LINES}"
                    ),
                )
            )
        if function.param_count > MAX_FUNCTION_PARAMS:
            violations.append(
                Violation(
                    path=path,
                    rule="function-params",
                    detail=(
                        f"函数 `{function.name}` 有 {function.param_count} 个参数，"
                        f"上限 {MAX_FUNCTION_PARAMS}"
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
        f"参数<={MAX_FUNCTION_PARAMS} 个",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
