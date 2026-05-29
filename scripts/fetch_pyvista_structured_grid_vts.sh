#!/usr/bin/env bash
# 从 PyVista 官方数据仓库下载 StructuredGrid.vts（与 examples.download_structured_grid 同源）
#
# 上游: pyvista/examples/downloads.py → _dataset_structured_grid
# URL:  https://github.com/pyvista/data/raw/master/Data/StructuredGrid.vts
#
# 注意: 该文件为 zlib 压缩 + 3D，当前 asimu load_vts 会拒绝（可用于负例测试）。
# 用法:
#   ./scripts/fetch_pyvista_structured_grid_vts.sh
#   make probe-vts FILE=tests/fixtures/mesh/external/StructuredGrid.vts

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/tests/fixtures/mesh/external"
URL="https://github.com/pyvista/data/raw/master/Data/StructuredGrid.vts"
OUT="${OUT_DIR}/StructuredGrid.vts"

mkdir -p "${OUT_DIR}"

if command -v curl >/dev/null 2>&1; then
  curl -fsSL --retry 3 --connect-timeout 30 -o "${OUT}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -q -O "${OUT}" "${URL}"
else
  echo "需要 curl 或 wget" >&2
  exit 1
fi

echo "已下载: ${OUT} ($(wc -c < "${OUT}") bytes)"
echo
echo "探测 asimu 兼容性:"
if make -C "${ROOT}" probe-vts "FILE=${OUT}"; then
  echo "读取成功"
else
  echo "（预期可能失败: PyVista 样例含 compressor / 3D，见 ADR 0007）"
fi
