#!/usr/bin/env python3
"""complexity_check.py 单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from complexity_check import (
    MAX_FILE_LINES,
    MAX_FUNCTION_LINES,
    MAX_FUNCTION_PARAMS,
    check_file,
    count_parameters,
    parse_functions,
)


class ComplexityCheckTests(unittest.TestCase):
    def test_count_parameters(self) -> None:
        self.assertEqual(count_parameters(""), 0)
        self.assertEqual(count_parameters("a: i32, b: i32"), 2)
        self.assertEqual(count_parameters("&self, x: i32"), 2)
        self.assertEqual(count_parameters("x: Vec<(i32, i32)>"), 1)

    def test_parse_function_span(self) -> None:
        source = """
fn short(a: i32, b: i32) -> i32 {
    a + b
}
"""
        functions = parse_functions(source)
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0].name, "short")
        self.assertEqual(functions[0].param_count, 2)
        self.assertEqual(functions[0].start_line, 2)
        self.assertEqual(functions[0].end_line, 4)

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

    def test_detects_long_file(self) -> None:
        lines = "\n".join("// comment" for _ in range(MAX_FILE_LINES + 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.rs"
            path.write_text(lines + "\n", encoding="utf-8")
            violations = check_file(path)
        rules = {item.rule for item in violations}
        self.assertIn("file-lines", rules)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
