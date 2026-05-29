# ADR 0007: VTK XML 结构化网格（VTS）二进制读入

- **状态**: 已接受
- **日期**: 2026-05-29
- **关联**: [ARCHITECTURE.md](../ARCHITECTURE.md) §9.2、[CASE_FORMAT.md](../CASE_FORMAT.md)、[SECURITY.md](../../SECURITY.md)

## 背景

外部工具（ParaView、PyVista、转换脚本等）常导出 **VTK XML StructuredGrid（`.vts`）**。asimu 需在 `io` 层读取网格几何，映射为内部 `StructuredMesh`（2D/3D），且不与 ASCII 遗留格式混在同一解析路径。

## 决策

### 1. 范围（首版）

| 支持 | 不支持 |
|------|--------|
| `.vts` StructuredGrid | `.vtu` / Legacy VTK |
| **二进制 appended**（`format="appended"` + base64 `AppendedData`） | `format="ascii"`、inline `format="binary"` |
| 单 `Piece` | 多 Piece / ghost / AMR |
| `Points` Float32 / Float64，3 组件 | 非 `vtkZLibDataCompressor` 压缩器 |
| 2D / 3D（`StructuredMesh2d` / `StructuredMesh3d`） | — |
| `vtkZLibDataCompressor` + 分段 base64 `AppendedData` | — |

### 2. 模块与 feature

- 实现：`src/io/vtk/vts.rs`
- Cargo feature：**`io-vtk`**（依赖 `quick-xml`、`base64`、`flate2`）
- CI / `make check` 启用 `--features io-vtk`

### 3. 依赖

| crate | 许可证 | 用途 |
|-------|--------|------|
| `quick-xml` | MIT | 流式解析 VTK XML |
| `base64` | MIT/Apache-2.0 | 解码 AppendedData（含 VTK 分段块） |
| `flate2` | MIT/Apache-2.0 | zlib 解压 appended 数据 |

### 4. 安全

在 `io` 层 Parse 阶段强制 [SECURITY.md](../../SECURITY.md) / DATA_MODEL §13 上限：文件大小、extent 推导单元数、禁止 `..` 路径。

### 5. 输出

```rust
pub struct VtsLoadResult {
    pub mesh: StructuredMesh, // D2 | D3
}
```

`CellData` / `PointData` 初场：后续 PR；首版仅几何。

## 后果

- 默认 `cargo build` 无 XML 依赖；集成方显式 `features = ["io-vtk"]`
- ASCII VTS 文件得到明确错误，便于用户转换格式

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| 同时支持 ASCII | 用户要求仅二进制；减少解析面 |
| 专用 `vtk-format` crate | 维护与许可证需额外审查 |
| 默认启用 feature | 增加无 VTK 场景编译依赖 |
