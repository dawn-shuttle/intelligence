# Dawn Shuttle Intelligence

统一的 AI 访问接口，类似 Vercel AI SDK。支持 OpenAI、Anthropic、Google、DeepSeek、Moonshot 等多供应商，提供一致的 API 体验。

## 特性

- 🔄 **统一接口** - 一套 API 访问多个 AI 供应商
- 📦 **开箱即用** - 支持 OpenAI、Anthropic、Google、DeepSeek、Moonshot
- 🛠️ **工具调用** - 完整的 Function Calling 支持
- 🌊 **流式输出** - 异步生成器流式响应
- 🖼️ **多模态** - 支持图片输入（Vision）
- ⚡ **异步优先** - 全异步设计，高性能
- 🔌 **易于扩展** - OpenAI 兼容基类，快速适配新供应商

## 安装

```bash
# 基础安装
pip install dawn_shuttle_intelligence

# 安装特定供应商
pip install dawn_shuttle_intelligence[openai]
pip install dawn_shuttle_intelligence[anthropic]
pip install dawn_shuttle_intelligence[google]

# 安装全部
pip install dawn_shuttle_intelligence[all]
```

## 快速开始

### 基础用法

```python
import asyncio
from dawn_shuttle.dawn_shuttle_intelligence import OpenAIProvider, Message, GenerateConfig

async def main():
    # 创建供应商
    provider = OpenAIProvider(api_key="your-api-key")
    
    # 构建消息
    messages = [
        Message.system("你是一个有用的助手"),
        Message.user("你好！"),
    ]
    
    # 配置参数
    config = GenerateConfig(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000,
    )
    
    # 生成响应
    response = await provider.generate(messages, config)
    print(response.text)

asyncio.run(main())
```

### 流式输出

```python
from dawn_shuttle.dawn_shuttle_intelligence import OpenAIProvider, Message, GenerateConfig

async def stream_example():
    provider = OpenAIProvider(api_key="your-api-key")
    messages = [Message.user("讲一个故事")]
    config = GenerateConfig(model="gpt-4o", stream=True)
    
    async for chunk in provider.generate_stream(messages, config):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
```

### 多模态（图片）

```python
from dawn_shuttle_intelligence import Message, TextContent, ImageContent

messages = [
    Message.user([
        TextContent(text="描述这张图片"),
        ImageContent(image="https://example.com/image.png"),
    ])
]
```

## 支持的供应商

| 供应商 | 类名 | 模型示例 |
|--------|------|----------|
| OpenAI | `OpenAIProvider` | gpt-4o, gpt-4-turbo, gpt-3.5-turbo |
| Anthropic | `AnthropicProvider` | claude-3-5-sonnet, claude-3-opus |
| Google | `GoogleProvider` | gemini-2.0-flash, gemini-1.5-pro |
| DeepSeek | `DeepSeekProvider` | deepseek-chat, deepseek-coder |
| Moonshot | `MoonshotProvider` | moonshot-v1-8k, moonshot-v1-32k |

### OpenAI 兼容供应商

DeepSeek 和 Moonshot 基于 `OpenAICompatibleProvider`，只需更改 `base_url`：

```python
from dawn_shuttle_intelligence import DeepSeekProvider

provider = DeepSeekProvider(
    api_key="your-deepseek-key",
    # 自动使用 DeepSeek API 端点
)
```

## 工具调用（Function Calling）

### 定义工具

```python
from dawn_shuttle_intelligence import Tool, ToolResult, ToolParameter

class WeatherTool(Tool):
    """天气查询工具。"""
    
    name = "get_weather"
    description = "获取指定城市的天气信息"
    
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="city",
                type="string",
                description="城市名称",
                required=True,
            ),
        ]
    
    async def execute(self, city: str) -> ToolResult:
        # 实现工具逻辑
        weather = f"{city}今天晴天，温度 25°C"
        return ToolResult(tool_call_id="", content=weather)
```

### 使用工具循环

```python
from dawn_shuttle_intelligence import (
    OpenAIProvider, Message, GenerateConfig,
    ToolRegistry, run_with_tools,
)

# 注册工具
registry = ToolRegistry()
registry.register(WeatherTool())

# 运行带工具的对话
result = await run_with_tools(
    messages=[Message.user("北京今天天气怎么样？")],
    provider=provider,
    tools=registry,
    config=GenerateConfig(model="gpt-4o"),
)

print(result.response.text)
```

## 错误处理

```python
from dawn_shuttle_intelligence import (
    AIError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    QuotaExceededError,
)

try:
    response = await provider.generate(messages, config)
except AuthenticationError as e:
    print(f"认证失败: {e}")
except RateLimitError as e:
    print(f"请求过快: {e}")
except ModelNotFoundError as e:
    print(f"模型不存在: {e}")
except AIError as e:
    print(f"AI 错误: {e}")
```

## API 参考

### GenerateConfig

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | str | 模型标识 |
| `temperature` | float | 采样温度 (0.0-2.0) |
| `top_p` | float | Top-p 采样 |
| `top_k` | int | Top-k 采样 |
| `max_tokens` | int | 最大输出 token |
| `stop` | str/list | 停止词 |
| `frequency_penalty` | float | 频率惩罚 |
| `presence_penalty` | float | 存在惩罚 |
| `seed` | int | 随机种子 |
| `tools` | list | 工具定义 |
| `tool_choice` | str/dict | 工具选择策略 |
| `response_format` | dict | 响应格式 |

### Message

```python
Message(role=Role.USER, content="Hello")
Message.user("Hello")           # 快捷方法
Message.system("You are...")    # 快捷方法
Message.assistant("Hi!")        # 快捷方法
```

### GenerateResponse

```python
response.text           # 响应文本
response.tool_calls     # 工具调用列表
response.finish_reason  # 结束原因
response.usage          # token 使用量
response.model          # 实际使用的模型
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ --cov=dawn_shuttle/dawn_shuttle_intelligence/src

# 代码检查
ruff check .
mypy .
```

## 许可证

[GNU 宽松通用公共许可证 v2.1](LICENSE)