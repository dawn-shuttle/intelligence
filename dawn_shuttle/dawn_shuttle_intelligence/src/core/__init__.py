"""Core 模块 - 核心抽象和类型定义。"""

from .config import GenerateConfig
from .error import (
    AIError,
    AuthenticationError,
    ConnectionError,
    ContentFilterError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderNotAvailableError,
    QuotaExceededError,
    RateLimitError,
    TimeoutError,
)
from .generate import generate_text, stream_text
from .provider import BaseProvider
from .response import GenerateResponse, StreamChunk, Usage
from .types import (
    ContentPart,
    ImageContent,
    Message,
    Role,
    TextContent,
    ToolCall,
    ToolResult,
)

__all__ = [
    # 类型定义
    "Message",
    "Role",
    "ContentPart",
    "TextContent",
    "ImageContent",
    "ToolCall",
    "ToolResult",
    # 配置
    "GenerateConfig",
    # 响应
    "GenerateResponse",
    "StreamChunk",
    "Usage",
    # Provider
    "BaseProvider",
    # 错误
    "AIError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "InvalidRequestError",
    "ContentFilterError",
    "QuotaExceededError",
    "TimeoutError",
    "ConnectionError",
    "ProviderNotAvailableError",
    # 入口函数
    "generate_text",
    "stream_text",
]
