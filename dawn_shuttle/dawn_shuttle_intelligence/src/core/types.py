"""核心类型定义 - 消息、角色等基础数据结构。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Union


class Role(str, Enum):
    """消息角色枚举。"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class TextContent:
    """文本内容块。"""

    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageContent:
    """图片内容块。"""

    type: Literal["image"] = "image"
    image: str = ""  # URL 或 base64
    mime_type: str | None = None


ContentPart = Union[TextContent, ImageContent]


@dataclass
class ToolCall:
    """工具调用请求。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """工具调用结果。"""

    tool_call_id: str
    content: str


@dataclass
class Message:
    """统一消息格式。"""

    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # 用于 tool 角色的消息

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式，方便序列化。"""
        result: dict[str, Any] = {"role": self.role.value}

        if self.content is not None:
            if isinstance(self.content, str):
                result["content"] = self.content
            else:
                result["content"] = [
                    {"type": part.type, **({"text": part.text} if hasattr(part, "text") else {"image": part.image})}
                    for part in self.content
                ]

        if self.name:
            result["name"] = self.name

        if self.tool_calls:
            result["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ]

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        return result

    @classmethod
    def user(cls, content: str) -> "Message":
        """创建用户消息的便捷方法。"""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str | None = None, tool_calls: list[ToolCall] | None = None) -> "Message":
        """创建助手消息的便捷方法。"""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def system(cls, content: str) -> "Message":
        """创建系统消息的便捷方法。"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> "Message":
        """创建工具结果消息的便捷方法。"""
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)
