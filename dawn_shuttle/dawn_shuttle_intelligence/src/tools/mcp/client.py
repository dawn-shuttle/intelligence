"""MCP 客户端 - 连接外部 MCP 服务器并调用工具。"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..tool import Tool
from ..types import ToolParameter, ToolResult
from .types import (
    MCP_METHOD_INITIALIZE,
    MCP_METHOD_TOOLS_CALL,
    MCP_METHOD_TOOLS_LIST,
    MCPServerInfo,
    MCPToolDefinition,
)


class MCPError(Exception):
    """MCP 错误。"""
    pass


class MCPConnectionError(MCPError):
    """MCP 连接错误。"""
    pass


class MCPToolNotFoundError(MCPError):
    """MCP 工具未找到。"""
    pass


@dataclass
class MCPClient:
    """MCP 客户端。

    连接外部 MCP 服务器进程, 发现并调用工具。

    Attributes:
        command: 启动 MCP 服务器的命令。
        args: 命令参数。
        env: 环境变量。
        timeout: 通信超时(秒)。
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0

    _process: asyncio.subprocess.Process | None = field(
        default=None, init=False, repr=False
    )
    _reader: asyncio.StreamReader | None = field(
        default=None, init=False, repr=False
    )
    _writer: asyncio.StreamWriter | None = field(
        default=None, init=False, repr=False
    )
    _request_id: int = field(default=0, init=False)
    _server_info: MCPServerInfo | None = field(default=None, init=False)
    _tools: list[MCPToolDefinition] = field(
        default_factory=list, init=False, repr=False
    )

    async def connect(self) -> MCPServerInfo:
        """连接到 MCP 服务器。

        Returns:
            MCPServerInfo: 服务器信息。

        Raises:
            MCPConnectionError: 连接失败。
        """
        # 启动进程
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**dict.__class__.__bases__[0](), **self.env},
            )

            self._reader = self._process.stdout
            self._writer = self._process.stdin

        except Exception as e:
            raise MCPConnectionError(
                f"Failed to start MCP server: {e}"
            ) from e

        # 初始化连接
        try:
            response = await self._send_request(
                MCP_METHOD_INITIALIZE,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "dawn-shuttle-intelligence",
                        "version": "0.1.0",
                    },
                },
            )

            if response.is_error:
                raise MCPConnectionError(
                    f"Initialize failed: {response.error}"
                )

            self._server_info = MCPServerInfo.from_dict(
                response.result.get("serverInfo", {})
            )

            # 发现工具
            await self._discover_tools()

            return self._server_info

        except Exception as e:
            await self.disconnect()
            raise MCPConnectionError(
                f"Failed to initialize: {e}"
            ) from e

    async def disconnect(self) -> None:
        """断开连接。"""
        if self._writer:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except Exception:
                with contextlib.suppress(Exception):
                    self._process.kill()

        self._process = None
        self._reader = None
        self._writer = None
        self._server_info = None
        self._tools = []

    async def list_tools(self) -> list[MCPToolDefinition]:
        """获取服务器提供的工具列表。

        Returns:
            list[MCPToolDefinition]: 工具定义列表。
        """
        return self._tools.copy()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """调用工具。

        Args:
            name: 工具名称。
            arguments: 调用参数。

        Returns:
            ToolResult: 执行结果。

        Raises:
            MCPToolNotFoundError: 工具未找到。
            MCPError: 调用失败。
        """
        # 检查工具是否存在
        tool_names = [t.name for t in self._tools]
        if name not in tool_names:
            raise MCPToolNotFoundError(f"Tool '{name}' not found")

        # 发送调用请求
        response = await self._send_request(
            MCP_METHOD_TOOLS_CALL,
            {
                "name": name,
                "arguments": arguments,
            },
        )

        if response.is_error:
            return ToolResult(
                tool_call_id="",
                content="",
                is_error=True,
                error_message=str(response.error),
            )

        # 解析结果
        result = response.result or {}
        content = result.get("content", [])

        # 提取文本内容
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))

        return ToolResult(
            tool_call_id="",
            content="\n".join(text_parts) if text_parts else str(result),
        )

    def to_tools(self) -> list[Tool]:
        """将 MCP 工具转换为 Tool 对象。

        Returns:
            list[Tool]: Tool 对象列表。
        """
        tools = []

        for mcp_tool in self._tools:
            tool = MCPToolWrapper(client=self, definition=mcp_tool)
            tools.append(tool)

        return tools

    async def _discover_tools(self) -> None:
        """发现可用工具。"""
        response = await self._send_request(MCP_METHOD_TOOLS_LIST, {})

        if response.is_error:
            self._tools = []
            return

        tools_data = response.result.get("tools", [])
        self._tools = [
            MCPToolDefinition.from_dict(t)
            for t in tools_data
        ]

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> Any:
        """发送请求并等待响应。"""
        from .types import MCPResponse

        if self._writer is None or self._reader is None:
            raise MCPConnectionError("Not connected")

        # 构建请求
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        # 发送请求
        request_str = json.dumps(request) + "\n"
        self._writer.write(request_str.encode("utf-8"))
        await self._writer.drain()

        # 读取响应
        try:
            response_line = await asyncio.wait_for(
                self._reader.readline(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise MCPError(f"Request timeout: {method}") from None

        # 解析响应
        try:
            response_data = json.loads(response_line.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise MCPError(f"Invalid response: {e}") from e

        return MCPResponse.from_dict(response_data)


@dataclass
class MCPToolWrapper(Tool):
    """MCP 工具包装器 - 将 MCP 工具包装为 Tool 对象。"""

    client: MCPClient = field(default=None)
    definition: MCPToolDefinition = field(default=None)

    @property
    def name(self) -> str:
        return self.definition.name if self.definition else ""

    @name.setter
    def name(self, value: str) -> None:
        if self.definition:
            self.definition.name = value

    @property
    def description(self) -> str:
        return self.definition.description if self.definition else ""

    @description.setter
    def description(self, value: str) -> None:
        if self.definition:
            self.definition.description = value

    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行工具。"""
        if self.client is None:
            return ToolResult(
                tool_call_id="",
                content="",
                is_error=True,
                error_message="MCP client not connected",
            )

        return await self.client.call_tool(self.name, kwargs)

    def get_parameters(self) -> list[ToolParameter]:
        """获取参数列表。"""
        if self.definition is None:
            return []

        schema = self.definition.input_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        params: list[ToolParameter] = []
        for name, prop in properties.items():
            params.append(ToolParameter(
                name=name,
                type=prop.get("type", "string"),
                description=prop.get("description"),
                required=name in required,
                enum=prop.get("enum"),
            ))

        return params
