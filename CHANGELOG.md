# 版本变更说明

本文件记录所有重要的版本变更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

## [0.2.0] - 2026-02-22

### 新增

- **统一 AI 访问接口** - 类似 Vercel AI SDK 的统一 API
- **多供应商支持**
  - OpenAI (GPT-4o, GPT-4-turbo, GPT-3.5-turbo)
  - Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
  - Google (Gemini 2.0 Flash, Gemini 1.5 Pro)
  - DeepSeek (DeepSeek Chat, DeepSeek Coder)
  - Moonshot (Moonshot V1)
- **流式输出** - 异步生成器流式响应
- **多模态支持** - 图片输入（Vision）
- **工具调用** - 完整的 Function Calling 支持
  - `Tool` 基类
  - `ToolRegistry` 工具注册表
  - `run_with_tools` 工具循环执行器
  - 自动消息格式转换（OpenAI/Anthropic/Google）
- **错误处理** - 统一的错误类型
  - `AuthenticationError` - 认证错误
  - `RateLimitError` - 限流错误
  - `ModelNotFoundError` - 模型不存在
  - `QuotaExceededError` - 配额超限
  - `ContentFilterError` - 内容过滤
  - `ProviderNotAvailableError` - 服务不可用
- **核心类型**
  - `Message` - 统一消息格式
  - `GenerateConfig` - 生成配置
  - `GenerateResponse` - 响应对象
  - `StreamChunk` - 流式块

### 架构

- `core/` - 核心模块（类型、配置、错误、响应）
- `adapter/` - 供应商适配器
- `tools/` - 工具系统
  - `tool.py` - 工具基类
  - `registry.py` - 工具注册表
  - `executor.py` - 工具执行器
  - `loop.py` - 工具循环
  - `converter.py` - 格式转换
  - `schema.py` - Schema 推断
  - `mcp/` - MCP 协议支持

### 测试

- 测试覆盖率 90%
- 533 个测试用例

## [0.1.0] - 2026-02-20

### 新增

- 基于 PEP 420 隐式命名空间包的项目模板结构
- `pyproject.toml` 配置，包含 mypy、ruff 及 hatchling 构建后端设置
- `init_project.py` 项目初始化脚本，支持子包名输入、pyproject.toml 重命名及目录结构生成
- `CONTRIBUTING.md` 贡献指南
- GNU 宽松通用公共许可证 v2.1（LGPL-2.1）

[未发布]: https://github.com/dawn-shuttle/intelligence/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dawn-shuttle/intelligence/releases/tag/v0.2.0
[0.1.0]: https://github.com/dawn-shuttle/intelligence/releases/tag/v0.1.0