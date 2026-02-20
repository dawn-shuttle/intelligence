# 贡献指南

感谢你对 `dawn_shuttle` 模板的关注！我们欢迎任何形式的贡献。

## 报告问题

请通过 [GitHub Issues](https://github.com/dawn-shuttle/template/issues) 提交问题报告，并尽量提供：

- 问题的详细描述
- 复现步骤
- 期望行为与实际行为
- Python 版本及操作系统信息

## 提交代码

1. Fork 本仓库
2. 基于 `main` 分支创建特性分支：

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. 提交变更并保持每个提交单一职责：

   ```bash
   git commit -m "feat: 简短描述"
   ```

4. 推送分支并创建 Pull Request

## 代码风格

本项目使用 [Ruff](https://docs.astral.sh/ruff/) 进行代码风格检查与格式化，使用 [mypy](https://mypy.readthedocs.io/) 进行静态类型检查。提交代码前请确保通过以下检查：

```bash
ruff check .
ruff format .
mypy dawn_shuttle
```

## 提交信息规范

提交信息使用以下前缀：

| 前缀       | 说明             |
| ---------- | ---------------- |
| `feat:`    | 新功能           |
| `fix:`     | 缺陷修复         |
| `docs:`    | 文档更新         |
| `refactor:`| 代码重构         |
| `test:`    | 测试相关         |
| `chore:`   | 构建或辅助工具   |

## 许可证

提交代码即表示你同意将贡献以 [GNU LGPL v2.1](LICENSE) 许可证发布。
