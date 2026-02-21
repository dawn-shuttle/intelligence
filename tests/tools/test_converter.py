"""格式转换器测试。"""

from __future__ import annotations

import json

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.tools import (
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.tools.converter import (
    ToolConverter,
    convert_tool_call,
    convert_tool_result,
    convert_tools,
)


class MockAnthropicToolUse:
    """模拟 Anthropic tool_use 对象。"""

    def __init__(self, id: str, name: str, input: dict):
        self.id = id
        self.name = name
        self.input = input


class MockGoogleFunctionCall:
    """模拟 Google function_call 对象。"""

    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args


class MockGoogleResponse:
    """模拟 Google 响应对象。"""

    def __init__(self, function_call: MockGoogleFunctionCall | None):
        self.function_call = function_call


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
        """测试从 OpenAI 格式解析(参数已为字典)。"""
        data = {
            "id": "call_456",
            "function": {
                "name": "test",
                "arguments": {"key": "value"},
            },
        }

        call = ToolConverter.call_from_provider(data, "openai")

        assert call.arguments == {"key": "value"}

    def test_call_from_anthropic(self) -> None:
        """测试从 Anthropic 格式解析。"""
        data = MockAnthropicToolUse(
            id="toolu_123",
            name="search",
            input={"query": "hello"},
        )

        call = ToolConverter.call_from_provider(data, "anthropic")

        assert call.id == "toolu_123"
        assert call.name == "search"
        assert call.arguments == {"query": "hello"}

    def test_call_from_anthropic_with_string_input(self) -> None:
        """测试从 Anthropic 格式解析(字符串输入)。"""
        data = MockAnthropicToolUse(
            id="toolu_456",
            name="test",
            input='{"key": "value"}',
        )

        call = ToolConverter.call_from_provider(data, "anthropic")

        assert call.arguments == {"key": "value"}

    def test_call_from_anthropic_with_invalid_input(self) -> None:
        """测试从 Anthropic 格式解析(无效输入)。"""
        data = MockAnthropicToolUse(
            id="toolu_789",
            name="test",
            input=123,  # 非字典非字符串
        )

        call = ToolConverter.call_from_provider(data, "anthropic")

        # 应该返回空字典
        assert call.arguments == {}

    def test_call_from_google(self) -> None:
        """测试从 Google 格式解析。"""
        fc = MockGoogleFunctionCall(name="calculate", args={"x": 1, "y": 2})
        data = MockGoogleResponse(function_call=fc)

        call = ToolConverter.call_from_provider(data, "google")

        assert call.name == "calculate"
        assert call.arguments == {"x": 1, "y": 2}

    def test_call_from_google_with_dict_conversion(self) -> None:
        """测试从 Google 格式解析(需要转换参数)。"""
        # 模拟 Google 的 MapDict 类型
        class MapDict(dict):
            pass

        fc = MockGoogleFunctionCall(
            name="test",
            args=MapDict([("a", 1), ("b", 2)]),
        )
        data = MockGoogleResponse(function_call=fc)

        call = ToolConverter.call_from_provider(data, "google")

        assert call.arguments == {"a": 1, "b": 2}

    def test_call_from_google_no_function_call(self) -> None:
        """测试 Google 响应无 function_call。"""
        data = MockGoogleResponse(function_call=None)

        with pytest.raises(ValueError, match="No function_call"):
            ToolConverter.call_from_provider(data, "google")

    def test_calls_from_provider_batch(self) -> None:
        """测试批量解析工具调用。"""
        data_list = [
            {"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "tool2", "arguments": "{}"}},
        ]

        calls = ToolConverter.calls_from_provider(data_list, "openai")

        assert len(calls) == 2
        assert calls[0].name == "tool1"
        assert calls[1].name == "tool2"

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

    def test_result_to_openai_with_bytes(self) -> None:
        """测试 bytes 内容转换为 OpenAI 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content=b"binary data",
        )

        msg = ToolConverter.result_to_provider(result, "openai")

        assert msg["content"] == "binary data"

    def test_result_to_openai_with_string(self) -> None:
        """测试字符串内容转换为 OpenAI 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content="plain text",
        )

        msg = ToolConverter.result_to_provider(result, "openai")

        assert msg["content"] == "plain text"

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

    def test_result_to_anthropic_with_bytes(self) -> None:
        """测试 bytes 内容转换为 Anthropic 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content=b"binary",
        )

        msg = ToolConverter.result_to_provider(result, "anthropic")

        assert msg["content"] == "binary"

    def test_result_to_anthropic_with_dict(self) -> None:
        """测试 dict 内容转换为 Anthropic 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content={"key": "value"},
        )

        msg = ToolConverter.result_to_provider(result, "anthropic")

        assert msg["content"] == {"key": "value"}

    def test_result_to_anthropic_with_error(self) -> None:
        """测试错误结果转换为 Anthropic 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content="",
            is_error=True,
            error_message="Something went wrong",
        )

        msg = ToolConverter.result_to_provider(result, "anthropic")

        assert msg["is_error"] is True

    def test_result_to_google(self) -> None:
        """测试结果转换为 Google 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content={"temp": 25},
        )

        msg = ToolConverter.result_to_provider(
            result, "google", tool_name="get_weather"
        )

        assert "function_response" in msg
        assert msg["function_response"]["name"] == "get_weather"

    def test_result_to_google_with_string_content(self) -> None:
        """测试字符串内容转换为 Google 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content='{"parsed": true}',
        )

        msg = ToolConverter.result_to_provider(
            result, "google", tool_name="test"
        )

        assert msg["function_response"]["response"] == {"parsed": True}

    def test_result_to_google_with_invalid_json_string(self) -> None:
        """测试无效 JSON 字符串转换为 Google 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content="not a json",
        )

        msg = ToolConverter.result_to_provider(
            result, "google", tool_name="test"
        )

        # 应该包装为 {"result": "not a json"}
        assert msg["function_response"]["response"] == {"result": "not a json"}

    def test_result_to_google_with_bytes(self) -> None:
        """测试 bytes 内容转换为 Google 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content=b"binary data",
        )

        msg = ToolConverter.result_to_provider(
            result, "google", tool_name="test"
        )

        assert msg["function_response"]["response"] == {"result": "binary data"}

    def test_result_to_google_with_other_type(self) -> None:
        """测试其他类型内容转换为 Google 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content=12345,  # 整数
        )

        msg = ToolConverter.result_to_provider(
            result, "google", tool_name="test"
        )

        assert msg["function_response"]["response"] == {"result": "12345"}

    def test_result_to_google_with_error(self) -> None:
        """测试错误结果转换为 Google 格式。"""
        result = ToolResult(
            tool_call_id="call_123",
            content="",
            is_error=True,
            error_message="City not found",
        )

        msg = ToolConverter.result_to_provider(
            result, "google", tool_name="get_weather"
        )

        assert "error" in msg["function_response"]["response"]

    def test_unknown_provider_definition(self) -> None:
        """测试未知提供商定义转换。"""
        definition = ToolDefinition(name="test", description="", parameters=[])

        with pytest.raises(ValueError):
            ToolConverter.definition_to_provider(definition, "unknown")  # type: ignore

    def test_unknown_provider_result(self) -> None:
        """测试未知提供商结果转换。"""
        result = ToolResult(tool_call_id="1", content="test")

        with pytest.raises(ValueError):
            ToolConverter.result_to_provider(result, "unknown")  # type: ignore


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
