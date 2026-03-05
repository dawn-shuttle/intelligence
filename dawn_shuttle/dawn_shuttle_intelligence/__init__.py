"""Dawn Shuttle Intelligence - 统一的 AI 访问接口。

提供一致的 API 访问多个 AI 供应商（OpenAI、Anthropic、Google 等）。

快速开始:
    from dawn_shuttle_intelligence import OpenAIProvider, Message, GenerateConfig
    
    provider = OpenAIProvider(api_key="your-key")
    messages = [Message.user("Hello!")]
    config = GenerateConfig(model="gpt-4o")
    response = await provider.generate(messages, config)
    print(response.text)
"""

from __future__ import annotations

# 供应商适配器
from .src.adapter.anthropic import AnthropicProvider
from .src.adapter.deepseek import DeepSeekProvider
from .src.adapter.google import GoogleProvider
from .src.adapter.moonshot import MoonshotProvider
from .src.adapter.openai import OpenAIProvider
from .src.adapter.openai_compatible import OpenAICompatibleProvider

# 核心类型和函数
from .src.core.config import GenerateConfig
from .src.core.generate import generate_text, stream_text
from .src.core.error import (
    AIError,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ContentFilterError,
    InternalServerError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderNotAvailableError,
    QuotaExceededError,
    RateLimitError,
    ResponseParseError,
    TimeoutError,
)
from .src.core.provider import BaseProvider
from .src.core.response import GenerateResponse, StreamChunk, Usage
from .src.core.types import (
    ImageContent,
    Message,
    Role,
    TextContent,
    ToolCall,
)

# 工具系统
from .src.tools.executor import ToolExecutor
from .src.tools.loop import LoopConfig, LoopResult, LoopStatus, run_with_tools
from .src.tools.registry import ToolRegistry
from .src.tools.tool import Tool
from .src.tools.types import ToolCall as ToolsToolCall
from .src.tools.types import ToolDefinition, ToolExecution, ToolParameter, ToolResult

__all__ = [
    # 供应商
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "DeepSeekProvider",
    "MoonshotProvider",
    "OpenAICompatibleProvider",
    "BaseProvider",
    # 核心函数
    "generate_text",
    "stream_text",
    # 核心类型
    "Message",
    "Role",
    "TextContent",
    "ImageContent",
    "ToolCall",
    "GenerateConfig",
    "GenerateResponse",
    "StreamChunk",
    "Usage",
    # 工具系统
    "Tool",
    "ToolResult",
    "ToolParameter",
    "ToolDefinition",
    "ToolExecution",
    "ToolRegistry",
    "ToolExecutor",
    "run_with_tools",
    "LoopConfig",
    "LoopResult",
    "LoopStatus",
    # 错误类型
    "AIError",
    "AuthenticationError",
    "ConfigurationError",
    "ConnectionError",
    "ContentFilterError",
    "InternalServerError",
    "InvalidRequestError",
    "ModelNotFoundError",
    "ProviderNotAvailableError",
    "QuotaExceededError",
    "RateLimitError",
    "ResponseParseError",
    "TimeoutError",
]

__version__ = "0.2.2"