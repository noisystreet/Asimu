#!/usr/bin/env python3
"""校验提交说明：第 1 行英文 Conventional Commits，第 2 行中文说明。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CONVENTIONAL = re.compile(
    r"^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(\([a-z0-9._/-]+\))?!?: .+"
)


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: commit_msg_check.py <commit-msg-file>", file=sys.stderr)
        return 1

    msg_path = Path(sys.argv[1])
    lines = [line.rstrip("\n") for line in msg_path.read_text(encoding="utf-8").splitlines()]

    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        print("提交说明不能为空", file=sys.stderr)
        return 1

    subject = non_empty[0]
    if not CONVENTIONAL.match(subject):
        print(
            "第 1 行须为英文 Conventional Commits，例如: feat(solver): add residual norm",
            file=sys.stderr,
        )
        return 1

    if len(non_empty) < 2:
        print("第 2 行须为中文说明（与第 1 行语义一致）", file=sys.stderr)
        return 1

    chinese_line = non_empty[1]
    if not re.search(r"[\u4e00-\u9fff]", chinese_line):
        print("第 2 行须包含中文字符", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
