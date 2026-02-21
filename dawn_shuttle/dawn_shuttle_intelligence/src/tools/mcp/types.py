"""MCP (Model Context Protocol) 协议类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MCPMessageType(str, Enum):
    """MCP 消息类型。"""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"


@dataclass
class MCPRequest:
    """MCP 请求。

    Attributes:
        jsonrpc: JSON-RPC 版本。
        id: 请求 ID。
        method: 方法名。
        params: 参数。
    """

    id: int | str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
            "params": self.params,
        }


@dataclass
class MCPResponse:
    """MCP 响应。

    Attributes:
        jsonrpc: JSON-RPC 版本。
        id: 对应的请求 ID。
        result: 结果。
        error: 错误。
    """

    id: int | str
    result: Any = None
    error: dict[str, Any] | None = None
    jsonrpc: str = "2.0"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPResponse:
        """从字典创建。"""
        return cls(
            id=data.get("id", 0),
            result=data.get("result"),
            error=data.get("error"),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )

    @property
    def is_error(self) -> bool:
        """是否为错误响应。"""
        return self.error is not None


@dataclass
class MCPToolDefinition:
    """MCP 工具定义。

    Attributes:
        name: 工具名称。
        description: 工具描述。
        input_schema: 输入 JSON Schema。
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPToolDefinition:
        """从字典创建。"""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {}),
        )


@dataclass
class MCPResource:
    """MCP 资源。

    Attributes:
        uri: 资源 URI。
        name: 资源名称。
        description: 资源描述。
        mime_type: MIME 类型。
    """

    uri: str
    name: str
    description: str | None = field(default=None)
    mime_type: str | None = field(default=None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPResource:
        """从字典创建。"""
        return cls(
            uri=data.get("uri", ""),
            name=data.get("name", ""),
            description=data.get("description"),
            mime_type=data.get("mimeType"),
        )


@dataclass
class MCPPrompt:
    """MCP 提示模板。

    Attributes:
        name: 提示名称。
        description: 提示描述。
        arguments: 参数列表。
    """

    name: str
    description: str | None = field(default=None)
    arguments: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPPrompt:
        """从字典创建。"""
        return cls(
            name=data.get("name", ""),
            description=data.get("description"),
            arguments=data.get("arguments", []),
        )


@dataclass
class MCPServerInfo:
    """MCP 服务器信息。

    Attributes:
        name: 服务器名称。
        version: 服务器版本。
        protocol_version: 协议版本。
        capabilities: 服务器能力。
    """

    name: str
    version: str
    protocol_version: str = "2024-11-05"
    capabilities: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPServerInfo:
        """从字典创建。"""
        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            protocol_version=data.get("protocolVersion", "2024-11-05"),
            capabilities=data.get("capabilities", {}),
        )


# MCP 方法常量
MCP_METHOD_INITIALIZE = "initialize"
MCP_METHOD_TOOLS_LIST = "tools/list"
MCP_METHOD_TOOLS_CALL = "tools/call"
MCP_METHOD_RESOURCES_LIST = "resources/list"
MCP_METHOD_RESOURCES_READ = "resources/read"
MCP_METHOD_PROMPTS_LIST = "prompts/list"
MCP_METHOD_PROMPTS_GET = "prompts/get"
