"""格式转换器测试。"""

from __future__ import annotations

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.tools.converter import (
    ProviderType,
    ToolConverter,
    convert_tool_call,
    convert_tool_result,
    convert_tools,
)


class TestToolConverter:
    """ToolConverter 测试。"""

    def test_definition_to_openai(self) -> None:
        """测试转换为 OpenAI 格式。"""
        definition = ToolDefinition(
            name="get_weather",
            description="Get weather",
            parameters=[
                ToolParameter(name="city", type="string", description="City name"),
            ],
        )

        result = ToolConverter.definition_to_provider(definition, "openai")

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get weather"
        assert "parameters" in result["function"]

    def test_definition_to_anthropic(self) -> None:
        """测试转换为 Anthropic 格式。"""
        definition = ToolDefinition(
            name="search",
            description="Search web",
            parameters=[],
        )

        result = ToolConverter.definition_to_provider(definition, "anthropic")

        assert result["name"] == "search"
        assert result["description"] == "Search web"
        assert "input_schema" in result

    def test_definition_to_google(self) -> None:
        """测试转换为 Google 格式。"""
        definition = ToolDefinition(
            name="calculate",
            description="Calculate",
            parameters=[],
        )

        result = ToolConverter.definition_to_provider(definition, "google")

        assert "function_declarations" in result
        assert result["function_declarations"][0]["name"] == "calculate"

    def test_definitions_to_provider(self) -> None:
        """测试批量转换。"""
        definitions = [
            ToolDefinition(name="tool1", description="Tool 1", parameters=[]),
            ToolDefinition(name="tool2", description="Tool 2", parameters=[]),
        ]

        results = ToolConverter.definitions_to_provider(definitions, "openai")

        assert len(results) == 2
        assert results[0]["function"]["name"] == "tool1"
        assert results[1]["function"]["name"] == "tool2"

    def test_call_from_openai(self) -> None:
        """测试从 OpenAI 格式解析。"""
        data = {
            "id": "call_123",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Beijing"}',
            },
        }

        call = ToolConverter.call_from_provider(data, "openai")

        assert call.id == "call_123"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Beijing"}

    def test_call_from_openai_with_dict_args(self) -> None:
        """测试从 OpenAI 格式解析（参数已为字典）。"""
        data = {
            "id": "call_456",
            "function": {
                "name": "test",
                "arguments": {"key": "value"},
            },
        }

        call = ToolConverter.call_from_provider(data, "openai")

        assert call.arguments == {"key": "value"}

    def test_result_to_openai(self) -> None:
        """测试结果转换为 OpenAI 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content={"temp": 25},
        )

        msg = ToolConverter.result_to_provider(result, "openai")

        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert "temp" in msg["content"]

    def test_result_to_anthropic(self) -> None:
        """测试结果转换为 Anthropic 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content="Sunny day",
        )

        msg = ToolConverter.result_to_provider(result, "anthropic")

        assert msg["type"] == "tool_result"
        assert msg["tool_use_id"] == "call_123"
        assert msg["content"] == "Sunny day"

    def test_result_to_google(self) -> None:
        """测试结果转换为 Google 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content={"temp": 25},
        )

        msg = ToolConverter.result_to_provider(result, "google", tool_name="get_weather")

        assert "function_response" in msg
        assert msg["function_response"]["name"] == "get_weather"

    def test_result_to_google_with_error(self) -> None:
        """测试错误结果转换为 Google 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content="",
            is_error=True,
            error_message="City not found",
        )

        msg = ToolConverter.result_to_provider(result, "google", tool_name="get_weather")

        assert "error" in msg["function_response"]["response"]

    def test_unknown_provider(self) -> None:
        """测试未知提供商。"""
        definition = ToolDefinition(name="test", description="", parameters=[])

        with pytest.raises(ValueError):
            ToolConverter.definition_to_provider(definition, "unknown")  # type: ignore


class TestConvenienceFunctions:
    """便捷函数测试。"""

    def test_convert_tools(self) -> None:
        """测试 convert_tools。"""
        definitions = [
            ToolDefinition(name="tool1", description="", parameters=[]),
        ]

        results = convert_tools(definitions, "openai")

        assert len(results) == 1
        assert results[0]["function"]["name"] == "tool1"

    def test_convert_tool_call(self) -> None:
        """测试 convert_tool_call。"""
        data = {
            "id": "call_123",
            "function": {"name": "test", "arguments": "{}"},
        }

        call = convert_tool_call(data, "openai")

        assert call.id == "call_123"
        assert call.name == "test"

    def test_convert_tool_result(self) -> None:
        """测试 convert_tool_result。"""
        result = ToolResult(tool_call_id="1", content="done")

        msg = convert_tool_result(result, "openai")

        assert msg["role"] == "tool"
