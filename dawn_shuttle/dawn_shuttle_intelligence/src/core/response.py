"""统一响应格式 - 标准化 AI 模型的返回结果。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    """Token 使用统计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GenerateResponse:
    """统一生成响应。"""

    # 生成的文本内容
    text: str = ""

    # 工具调用(如有)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # 结束原因: stop, length, tool_calls, content_filter 等
    finish_reason: str | None = None

    # 原始响应(提供商特定)
    raw: Any = None

    # Token 使用统计
    usage: Usage | None = None

    # 模型标识
    model: str | None = None

    # 请求 ID(用于追踪)
    request_id: str | None = None

    @property
    def is_tool_call(self) -> bool:
        """是否有工具调用。"""
        return len(self.tool_calls) > 0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        result: dict[str, Any] = {"text": self.text}

        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        if self.usage:
            result["usage"] = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            }
        if self.model:
            result["model"] = self.model
        if self.request_id:
            result["request_id"] = self.request_id

        return result


@dataclass
class StreamChunk:
    """流式响应的单个块。"""

    # 增量文本
    delta: str = ""

    # 工具调用增量
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # 是否结束
    is_finished: bool = False

    # 结束原因
    finish_reason: str | None = None

    # 使用统计(通常在最后一个 chunk)
    usage: Usage | None = None
