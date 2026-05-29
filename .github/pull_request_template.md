## Summary

<!-- 简要说明本 PR 的变更与动机 -->

## 变更类型

- [ ] feat — 新功能
- [ ] fix — 缺陷修复
- [ ] docs — 文档
- [ ] refactor — 重构
- [ ] test — 测试
- [ ] chore — 构建/工具

## 检查项

- [ ] `make check` 已通过（fmt + clippy + test）
- [ ] 新功能已添加测试（见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)）
- [ ] 公开 API 变更已更新 [docs/API.md](docs/API.md)
- [ ] [CHANGELOG.md](CHANGELOG.md) 已更新（若用户可见变更）
- [ ] 架构/分层变更已与维护者确认（若适用）

## Test plan

<!-- 如何验证本 PR -->

```bash
make check
cargo run --
```
