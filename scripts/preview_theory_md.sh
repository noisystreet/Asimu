#!/usr/bin/env bash
# 用 pandoc + MathJax 在浏览器中预览理论手册（不依赖 Cursor Markdown 预览）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MD="${1:-$ROOT/docs/theory/fvm_diffusion.md}"

if [[ ! -f "$MD" ]]; then
  echo "file not found: $MD" >&2
  exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found; install pandoc or use Markdown Preview Enhanced in Cursor" >&2
  exit 1
fi

OUT="$(mktemp --suffix=-theory-preview.html)"
pandoc "$MD" -o "$OUT" --standalone \
  --metadata title="$(basename "$MD")" \
  --mathjax \
  --css=https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css

echo "$OUT"
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$OUT" >/dev/null 2>&1 &
elif command -v sensible-browser >/dev/null 2>&1; then
  sensible-browser "$OUT" >/dev/null 2>&1 &
else
  echo "open in browser: file://$OUT"
fi
