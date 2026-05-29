# ADR 0008: CGNS 读入与 VTS 导出（系统 libcgns）

- **状态**: 已接受
- **日期**: 2026-05-29
- **关联**: [ADR 0007](0007-vts-binary-io.md)、[CASE_FORMAT.md](../CASE_FORMAT.md)

## 背景

工程网格常以 **CGNS** 分发（如 DLR-F6 `dlr-f6.coar.cgns`）。asimu 需读取结构化 zone 几何，并可导出为 ParaView 可读的 **VTS**。

## 决策

### 1. 链接方式

- **不**引入 Rust CGNS crate；通过 `build.rs` 链接系统 **`libcgns-dev`**（`cargo:rustc-link-lib=cgns`）。
- Cargo feature：**`io-cgns`**（读入）；**`io-cgns-vts`** = `io-cgns` + `io-vtk`（含 `write_vts` 导出）。

### 2. 范围（首版）

| 支持 | 不支持 |
|------|--------|
| CGNS ADF / HDF5（由系统 libcgns 处理） | Unstructured zone |
| Structured zone（`ZoneType_t=Structured`） | 多 base |
| `CoordinateX/Y/Z` Float64 | 非标准坐标命名 |
| 单 zone 读入 → `StructuredMesh3d` | PointData / BC 写出 |
| `export_cgns_zone_to_vts`（单 zone → `.vts`） | |
| `export_cgns_to_vtm`（全部 zone → `.vtm` + 子 `.vts`） | 单文件多 `Piece` VTS（独立 block 不兼容 ParaView） |

### 3. 线程安全

CGNS MLL **非线程安全**；`io::cgns` 内所有 MLL 调用由全局 `Mutex` 串行化。

### 4. API

```rust
list_cgns_zones(path) -> Vec<CgnsZoneInfo>
load_cgns_zone(path, zone_index) -> CgnsLoadResult
export_cgns_zone_to_vts(input, zone_index, output) -> CgnsLoadResult  // io-cgns-vts
write_vts(mesh, output)  // io-vtk
```

CLI：`cargo run --example cgns_to_vts --features io-cgns-vts -- in.cgns out.vts --zone N`

### 5. 测试

- 不将 `dlr-f6.coar.cgns` 入库；可选路径：`../dlr-f6.coar.cgns` 或 `ASIMU_CGNS_PATH`。
- CI 需安装 `libcgns-dev`；`make check` 默认不启用 `io-cgns`（见 `make check-cgns`）。

## 后果

- 构建环境须已安装 CGNS 开发包；无系统库时 `io-cgns` 链接失败。
- 多 zone 文件（如 DLR-F6 26 blocks）按 zone 分别导出 VTS。
