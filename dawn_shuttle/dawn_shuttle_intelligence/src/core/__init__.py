"""Core 模块 - 核心抽象和类型定义。"""

from .config import GenerateConfig, ResponseFormat, StopSequences, ToolChoice
from .error import (
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
    "ConfigurationError",
    "ConnectionError",
    "ContentFilterError",
    "ContentPart",
    "FinishReason",
    "GenerateConfig",
    "GenerateResponse",
    "ImageContent",
    "InternalServerError",
    "InvalidRequestError",
    "Message",
    "MessageDict",
    "Messages",
    "ModelNotFoundError",
    "ProviderNotAvailableError",
    "QuotaExceededError",
    "RateLimitError",
    "ResponseFormat",
    "ResponseParseError",
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
