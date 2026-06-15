# 贡献指南

感谢考虑为 asimu 贡献代码或文档。

## 开始之前

1. 阅读 [README.md](README.md) 与 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
2. AI 协作者请先阅读 [AGENTS.md](AGENTS.md)
3. 安全漏洞请按 [SECURITY.md](SECURITY.md) 私下报告，**不要**在公开 Issue 中披露

## 开发环境

```bash
cd asimu
make setup    # 安装 pre-commit 钩子（可选）
make check    # fmt + clippy + test
```

## 提交规范

- 第 1 行：英文 Conventional Commits 摘要（如 `feat(solver): add residual norm`）
- 第 2 行：中文说明（与第 1 行语义一致）
- 空行后可追加详细说明

示例：

```
feat(mesh): add structured grid loader

添加结构化网格加载器的占位实现
```

本地 `commit-msg` 钩子会校验上述格式（见 `scripts/commit_msg_check.py`）。

## Pull Request

- 基于 `main` 创建功能分支（命名：`feat/xxx`、`fix/xxx`、`docs/xxx`）
- 确保 `make check` 通过
- 新功能需附带测试（见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 测试策略）
- V&V / 无量纲 metrics 与文献对不上时，先走 [docs/DEBUG_CHECKLIST.md](docs/DEBUG_CHECKLIST.md)
- 填写 PR 模板中的检查项

## 许可证

提交贡献即表示你同意在 [Apache-2.0 OR MIT](LICENSE) 双许可下授权你的贡献。
