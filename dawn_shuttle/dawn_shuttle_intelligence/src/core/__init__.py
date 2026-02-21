"""Core 模块 - 核心抽象和类型定义。"""

from .config import GenerateConfig, ResponseFormat, StopSequences, ToolChoice
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
    FinishReason,
    ImageContent,
    Message,
    MessageDict,
    Messages,
    Role,
    TextContent,
    ToolCall,
    ToolResult,
)

__all__ = [
    "AIError",
    "AuthenticationError",
    "BaseProvider",
    "ConnectionError",
    "ContentFilterError",
    "ContentPart",
    "FinishReason",
    "GenerateConfig",
    "GenerateResponse",
    "ImageContent",
    "InvalidRequestError",
    "Message",
    "MessageDict",
    "Messages",
    "ModelNotFoundError",
    "ProviderNotAvailableError",
    "QuotaExceededError",
    "RateLimitError",
    "ResponseFormat",
    "Role",
    "StopSequences",
    "StreamChunk",
    "TextContent",
    "TimeoutError",
    "ToolCall",
    "ToolChoice",
    "ToolResult",
    "Usage",
    "generate_text",
    "stream_text",
]
