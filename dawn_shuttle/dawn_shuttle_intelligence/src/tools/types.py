"""工具模块核心类型定义。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# JSON Schema 类型
JSONSchemaType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
    "object",
    "array",
    "null",
]


@dataclass
class ToolParameter:
    """工具参数定义。

    Attributes:
        name: 参数名称。
        type: 参数类型(string, number, integer, boolean, object, array)。
        description: 参数描述。
        required: 是否必需。
        default: 默认值。
        enum: 枚举值列表。
        items: 数组元素类型(当 type 为 array 时)。
        properties: 对象属性(当 type 为 object 时)。
    """

    name: str
    type: JSONSchemaType | str
    description: str | None = None
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """转换为 JSON Schema 格式。"""
        schema: dict[str, Any] = {"type": self.type}

        if self.description:
            schema["description"] = self.description

        if self.enum:
            schema["enum"] = self.enum

        if self.type == "array" and self.items:
            schema["items"] = self.items

        if self.type == "object" and self.properties:
            schema["properties"] = self.properties

        return schema


@dataclass
class ToolDefinition:
    """统一的工具定义。

    Attributes:
        name: 工具名称(唯一标识)。
        description: 工具描述。
        parameters: 参数列表。
        return_type: 返回值类型描述。
        return_description: 返回值描述。
        examples: 使用示例。
        metadata: 元数据(权限、标签、超时等)。
    """

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    return_type: str | None = None
    return_description: str | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_schema(self) -> dict[str, Any]:
        """转换为 JSON Schema 格式(OpenAI tools 格式)。"""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }

        if required:
            schema["required"] = required

        return schema

    def to_openai_tool(self) -> dict[str, Any]:
        """转换为 OpenAI tools 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        """转换为 Anthropic tools 格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.to_json_schema(),
        }

    def to_google_tool(self) -> dict[str, Any]:
        """转换为 Google tools 格式。"""
        return {
            "function_declarations": [{
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            }]
        }


@dataclass
class ToolCall:
    """工具调用请求。

    Attributes:
        id: 调用 ID(由 AI 生成, 用于匹配结果)。
        name: 工具名称。
        arguments: 调用参数。
    """

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> ToolCall:
        """从 OpenAI 格式创建。"""
        import json

        args = data.get("function", {}).get("arguments", "{}")
        if isinstance(args, str):
            args = json.loads(args)

        return cls(
            id=data.get("id", ""),
            name=data.get("function", {}).get("name", ""),
            arguments=args,
        )

    @classmethod
    def from_anthropic(cls, block: Any) -> ToolCall:
        """从 Anthropic 格式创建。"""
        return cls(
            id=block.id,
            name=block.name,
            arguments=block.input if isinstance(block.input, dict) else {},
        )

    @classmethod
    def from_google(cls, part: Any) -> ToolCall:
        """从 Google 格式创建。"""
        fc = part.function_call
        return cls(
            id=f"call_{fc.name}",
            name=fc.name,
            arguments=dict(fc.args) if fc.args else {},
        )


@dataclass
class ToolResult:
    """工具执行结果。

    Attributes:
        tool_call_id: 对应的工具调用 ID。
        content: 返回内容(字符串、字典或二进制)。
        is_error: 是否为错误结果。
        error_message: 错误消息。
        metadata: 元数据(执行时间等)。
    """

    tool_call_id: str
    content: str | dict[str, Any] | bytes
    is_error: bool = False
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_message(self) -> dict[str, Any]:
        """转换为 OpenAI tool 消息格式。"""
        content = self.content
        if isinstance(content, dict):
            import json
            content = json.dumps(content, ensure_ascii=False)
        elif isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": str(content),
        }

    def to_anthropic_content(self) -> dict[str, Any]:
        """转换为 Anthropic tool_result 内容块。"""
        content = self.content
        if isinstance(content, dict):
            pass  # Anthropic 接受 dict
        elif isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        else:
            content = str(content)

        return {
            "type": "tool_result",
            "tool_use_id": self.tool_call_id,
            "content": content,
            "is_error": self.is_error,
        }

    def to_google_content(self, tool_name: str) -> dict[str, Any]:
        """转换为 Google function_response 内容块。"""
        content = self.content
        if isinstance(content, str):
            try:
                import json
                content = json.loads(content)
            except json.JSONDecodeError:
                content = {"result": content}
        elif isinstance(content, bytes):
            content = {"result": content.decode("utf-8", errors="replace")}

        if self.is_error:
            content = {"error": self.error_message or "Tool execution failed"}

        return {
            "function_response": {
                "name": tool_name,
                "response": content,
            }
        }


class ToolExecutionStatus(str, Enum):
    """工具执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ToolExecution:
    """工具执行记录。

    Attributes:
        tool_call: 工具调用请求。
        status: 执行状态。
        result: 执行结果。
        duration_ms: 执行耗时(毫秒)。
        retry_count: 重试次数。
    """

    tool_call: ToolCall
    status: ToolExecutionStatus = ToolExecutionStatus.PENDING
    result: ToolResult | None = None
    duration_ms: float | None = None
    retry_count: int = 0


# 类型别名
ToolExecuteFunc = Callable[..., Any]
"""工具执行函数类型。"""

ToolExecuteAsyncFunc = Callable[..., Any]
"""异步工具执行函数类型。"""
