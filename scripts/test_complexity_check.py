#!/usr/bin/env python3
"""complexity_check.py 单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from complexity_check import (
    MAX_CYCLOMATIC_COMPLEXITY,
    MAX_FILE_LINES,
    MAX_FUNCTION_LINES,
    MAX_FUNCTION_PARAMS,
    check_file,
)


class ComplexityCheckTests(unittest.TestCase):
    def test_detects_too_many_parameters(self) -> None:
        params = ", ".join(f"a{i}: i32" for i in range(MAX_FUNCTION_PARAMS + 1))
        source = f"fn too_many({params}) {{}}\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.rs"
            path.write_text(source, encoding="utf-8")
            violations = check_file(path)
        rules = {item.rule for item in violations}
        self.assertIn("function-params", rules)

    def test_detects_long_function(self) -> None:
        body = "\n".join("let _ = 0;" for _ in range(MAX_FUNCTION_LINES))
        source = f"fn long_fn() {{\n{body}\n}}\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long.rs"
            path.write_text(source, encoding="utf-8")
            violations = check_file(path)
        rules = {item.rule for item in violations}
        self.assertIn("function-lines", rules)

    def test_detects_high_cyclomatic_complexity(self) -> None:
        branches = "\n".join(
            f"    if x > {i} {{ y += {i}; }}"
            for i in range(MAX_CYCLOMATIC_COMPLEXITY + 2)
        )
        source = f"fn high_ccn(x: i32) -> i32 {{\n    let mut y = 0;\n{branches}\n    y\n}}\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ccn.rs"
            path.write_text(source, encoding="utf-8")
            violations = check_file(path)
        rules = {item.rule for item in violations}
        self.assertIn("cyclomatic-complexity", rules)

    def test_detects_long_file(self) -> None:
        lines = "\n".join("// comment" for _ in range(MAX_FILE_LINES + 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.rs"
            path.write_text(lines + "\n", encoding="utf-8")
            violations = check_file(path)
        rules = {item.rule for item in violations}
        self.assertIn("file-lines", rules)

    def test_simple_function_passes(self) -> None:
        source = """
fn short(a: i32, b: i32) -> i32 {
    a + b
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.rs"
            path.write_text(source, encoding="utf-8")
            violations = check_file(path)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    raise SystemExit(unittest.main())
