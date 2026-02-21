"""MCP 客户端测试 - 使用 mock 模拟 MCP 服务器。"""

from __future__ import annotations

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.tools.mcp.client import (
    MCPClient,
    MCPError,
    MCPToolNotFoundError,
    MCPToolWrapper,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.tools.mcp.types import (
    MCPResponse,
    MCPToolDefinition,
)


class MockProcess:
    """模拟进程。"""

    def __init__(self, responses: list[bytes] | None = None):
        self.responses = responses or []
        self.response_index = 0
        self.stdin_buffer = BytesIO()
        self.stdout_buffer = BytesIO()
        self.terminated = False
        self.killed = False

        # 预写入响应
        for resp in self.responses:
            self.stdout_buffer.write(resp)

        self.stdout_buffer.seek(0)

    @property
    def stdin(self):
        class Writer:
            def __init__(self, parent):
                self.parent = parent

            def write(self, data: bytes):
                self.parent.stdin_buffer.write(data)

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        return Writer(self)

    @property
    def stdout(self):
        class Reader:
            def __init__(self, parent):
                self.parent = parent

            async def readline(self):
                return self.parent.stdout_buffer.readline()

        return Reader(self)

    def terminate(self):
        self.terminated = True

    async def wait(self):
        return 0

    def kill(self):
        self.killed = True


class TestMCPClient:
    """MCPClient 测试。"""

    def test_create(self) -> None:
        """测试创建客户端。"""
        client = MCPClient(
            command="mcp-server",
            args=["--port", "8080"],
            env={"API_KEY": "test"},
            timeout=60.0,
        )

        assert client.command == "mcp-server"
        assert client.args == ["--port", "8080"]
        assert client.env == {"API_KEY": "test"}
        assert client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """测试断开连接。"""
        client = MCPClient(command="test")
        client._process = MockProcess()
        client._writer = client._process.stdin
        client._reader = client._process.stdout

        await client.disconnect()

        assert client._process is None
        assert client._writer is None
        assert client._reader is None

    @pytest.mark.asyncio
    async def test_list_tools(self) -> None:
        """测试列出工具。"""
        client = MCPClient(command="test")
        client._tools = [
            MCPToolDefinition(
                name="tool1",
                description="Tool 1",
                input_schema={"type": "object"},
            ),
            MCPToolDefinition(
                name="tool2",
                description="Tool 2",
                input_schema={"type": "object"},
            ),
        ]

        tools = await client.list_tools()

        assert len(tools) == 2
        assert tools[0].name == "tool1"

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self) -> None:
        """测试调用不存在的工具。"""
        client = MCPClient(command="test")
        client._tools = []

        with pytest.raises(MCPToolNotFoundError):
            await client.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_call_tool_success(self) -> None:
        """测试成功调用工具。"""
        client = MCPClient(command="test")
        client._tools = [
            MCPToolDefinition(
                name="echo",
                description="Echo tool",
                input_schema={"type": "object"},
            ),
        ]

        # Mock _send_request
        client._send_request = AsyncMock(
            return_value=MCPResponse(
                id=1,
                result={
                    "content": [
                        {"type": "text", "text": "Hello, world!"},
                    ]
                },
            )
        )

        result = await client.call_tool("echo", {"message": "Hello"})

        assert result.is_error is False
        assert "Hello, world!" in result.content

    @pytest.mark.asyncio
    async def test_call_tool_error(self) -> None:
        """测试工具调用返回错误。"""
        client = MCPClient(command="test")
        client._tools = [
            MCPToolDefinition(
                name="fail",
                description="Fail tool",
                input_schema={"type": "object"},
            ),
        ]

        client._send_request = AsyncMock(
            return_value=MCPResponse(
                id=1,
                error={"code": -1, "message": "Tool failed"},
            )
        )

        result = await client.call_tool("fail", {})

        assert result.is_error is True

    def test_to_tools(self) -> None:
        """测试转换为工具列表。"""
        client = MCPClient(command="test")
        client._tools = [
            MCPToolDefinition(
                name="tool1",
                description="Tool 1",
                input_schema={
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string", "description": "Arg 1"},
                    },
                    "required": ["arg1"],
                },
            ),
        ]

        tools = client.to_tools()

        assert len(tools) == 1
        assert tools[0].name == "tool1"
        assert tools[0].description == "Tool 1"

    @pytest.mark.asyncio
    async def test_send_request_timeout(self) -> None:
        """测试请求超时。"""
        client = MCPClient(command="test", timeout=0.1)
        client._writer = MagicMock()
        client._writer.write = MagicMock()
        client._writer.drain = AsyncMock()

        # Mock reader that never returns
        client._reader = MagicMock()
        client._reader.readline = AsyncMock(side_effect=asyncio.TimeoutError())

        client._request_id = 0

        with pytest.raises(MCPError) as exc_info:
            await client._send_request("test_method", {})

        assert "timeout" in str(exc_info.value).lower()


class TestMCPToolWrapper:
    """MCPToolWrapper 测试。"""

    def test_create(self) -> None:
        """测试创建。"""
        client = MCPClient(command="test")
        definition = MCPToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
        )

        wrapper = MCPToolWrapper(client=client, definition=definition)

        assert wrapper.name == "test_tool"
        assert wrapper.description == "A test tool"

    def test_get_parameters(self) -> None:
        """测试获取参数。"""
        client = MCPClient(command="test")
        definition = MCPToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "First arg"},
                    "arg2": {"type": "integer", "description": "Second arg"},
                },
                "required": ["arg1"],
            },
        )

        wrapper = MCPToolWrapper(client=client, definition=definition)
        params = wrapper.get_parameters()

        assert len(params) == 2
        assert params[0].name == "arg1"
        assert params[0].required is True
        assert params[1].name == "arg2"
        assert params[1].required is False

    @pytest.mark.asyncio
    async def test_execute_no_client(self) -> None:
        """测试无客户端执行。"""
        definition = MCPToolDefinition(
            name="test",
            description="Test",
            input_schema={},
        )
        wrapper = MCPToolWrapper(client=None, definition=definition)

        result = await wrapper.execute()

        assert result.is_error is True
        assert "not connected" in result.error_message.lower()


class TestMCPResponse:
    """MCPResponse 测试。"""

    def test_from_dict(self) -> None:
        """测试从字典创建。"""
        data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": []},
        }

        response = MCPResponse.from_dict(data)

        assert response.id == 1
        assert response.result == {"tools": []}
        assert response.is_error is False

    def test_from_dict_error(self) -> None:
        """测试从字典创建错误响应。"""
        data = {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32600, "message": "Invalid Request"},
        }

        response = MCPResponse.from_dict(data)

        assert response.is_error is True
        assert response.error is not None
