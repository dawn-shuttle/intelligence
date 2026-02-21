"""MCP 类型测试。"""

from __future__ import annotations

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.tools.mcp.types import (
    MCPPrompt,
    MCPRequest,
    MCPResource,
    MCPResponse,
    MCPServerInfo,
    MCPToolDefinition,
    MCP_METHOD_INITIALIZE,
    MCP_METHOD_TOOLS_LIST,
)


class TestMCPRequest:
    """MCPRequest 测试。"""

    def test_create(self) -> None:
        """测试创建。"""
        request = MCPRequest(
            id=1,
            method=MCP_METHOD_INITIALIZE,
            params={"protocolVersion": "2024-11-05"},
        )

        assert request.id == 1
        assert request.method == MCP_METHOD_INITIALIZE
        assert request.jsonrpc == "2.0"

    def test_to_dict(self) -> None:
        """测试转换为字典。"""
        request = MCPRequest(
            id="abc",
            method=MCP_METHOD_TOOLS_LIST,
            params={},
        )

        data = request.to_dict()

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "abc"
        assert data["method"] == MCP_METHOD_TOOLS_LIST


class TestMCPResponse:
    """MCPResponse 测试。"""

    def test_from_dict_success(self) -> None:
        """测试从字典创建成功响应。"""
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

        assert response.id == 2
        assert response.is_error is True
        assert response.error is not None


class TestMCPToolDefinition:
    """MCPToolDefinition 测试。"""

    def test_create(self) -> None:
        """测试创建。"""
        tool = MCPToolDefinition(
            name="get_weather",
            description="Get weather info",
            input_schema={"type": "object", "properties": {}},
        )

        assert tool.name == "get_weather"
        assert tool.description == "Get weather info"

    def test_from_dict(self) -> None:
        """测试从字典创建。"""
        data = {
            "name": "search",
            "description": "Search the web",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        }

        tool = MCPToolDefinition.from_dict(data)

        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert "query" in tool.input_schema.get("properties", {})


class TestMCPResource:
    """MCPResource 测试。"""

    def test_create(self) -> None:
        """测试创建。"""
        resource = MCPResource(
            uri="file:///path/to/file.txt",
            name="example.txt",
            description="An example file",
            mime_type="text/plain",
        )

        assert resource.uri == "file:///path/to/file.txt"
        assert resource.name == "example.txt"

    def test_from_dict(self) -> None:
        """测试从字典创建。"""
        data = {
            "uri": "file:///data.json",
            "name": "data.json",
            "mimeType": "application/json",
        }

        resource = MCPResource.from_dict(data)

        assert resource.uri == "file:///data.json"
        assert resource.mime_type == "application/json"


class TestMCPPrompt:
    """MCPPrompt 测试。"""

    def test_create(self) -> None:
        """测试创建。"""
        prompt = MCPPrompt(
            name="code_review",
            description="Review code",
            arguments=[{"name": "code", "description": "Code to review"}],
        )

        assert prompt.name == "code_review"
        assert len(prompt.arguments) == 1

    def test_from_dict(self) -> None:
        """测试从字典创建。"""
        data = {
            "name": "summarize",
            "description": "Summarize text",
            "arguments": [],
        }

        prompt = MCPPrompt.from_dict(data)

        assert prompt.name == "summarize"


class TestMCPServerInfo:
    """MCPServerInfo 测试。"""

    def test_create(self) -> None:
        """测试创建。"""
        info = MCPServerInfo(
            name="test-server",
            version="1.0.0",
            protocol_version="2024-11-05",
            capabilities={"tools": {}},
        )

        assert info.name == "test-server"
        assert info.version == "1.0.0"

    def test_from_dict(self) -> None:
        """测试从字典创建。"""
        data = {
            "name": "my-server",
            "version": "2.0.0",
            "protocolVersion": "2024-11-05",
            "capabilities": {"resources": {}, "tools": {}},
        }

        info = MCPServerInfo.from_dict(data)

        assert info.name == "my-server"
        assert info.version == "2.0.0"
        assert "tools" in info.capabilities
