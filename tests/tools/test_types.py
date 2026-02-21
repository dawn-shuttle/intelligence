"""工具类型测试。"""

from __future__ import annotations

from dawn_shuttle.dawn_shuttle_intelligence.src.tools.types import (
    ToolCall,
    ToolDefinition,
    ToolExecution,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)


class TestToolParameter:
    """ToolParameter 测试。"""

    def test_create_basic(self) -> None:
        """测试基本创建。"""
        param = ToolParameter(
            name="city",
            type="string",
            description="城市名称",
        )

        assert param.name == "city"
        assert param.type == "string"
        assert param.description == "城市名称"
        assert param.required is True
        assert param.default is None
        assert param.enum is None

    def test_create_with_options(self) -> None:
        """测试带选项创建。"""
        param = ToolParameter(
            name="unit",
            type="string",
            description="温度单位",
            required=False,
            default="celsius",
            enum=["celsius", "fahrenheit"],
        )

        assert param.required is False
        assert param.default == "celsius"
        assert param.enum == ["celsius", "fahrenheit"]

    def test_to_json_schema_basic(self) -> None:
        """测试转换为 JSON Schema。"""
        param = ToolParameter(
            name="query",
            type="string",
            description="搜索关键词",
        )

        schema = param.to_json_schema()

        assert schema["type"] == "string"
        assert schema["description"] == "搜索关键词"

    def test_to_json_schema_with_enum(self) -> None:
        """测试带枚举的 JSON Schema。"""
        param = ToolParameter(
            name="status",
            type="string",
            enum=["active", "inactive"],
        )

        schema = param.to_json_schema()

        assert "enum" in schema
        assert schema["enum"] == ["active", "inactive"]


class TestToolDefinition:
    """ToolDefinition 测试。"""

    def test_create_basic(self) -> None:
        """测试基本创建。"""
        definition = ToolDefinition(
            name="get_weather",
            description="获取城市天气",
            parameters=[
                ToolParameter(name="city", type="string", description="城市"),
            ],
        )

        assert definition.name == "get_weather"
        assert definition.description == "获取城市天气"
        assert len(definition.parameters) == 1

    def test_to_json_schema(self) -> None:
        """测试转换为 JSON Schema。"""
        definition = ToolDefinition(
            name="search",
            description="搜索",
            parameters=[
                ToolParameter(name="query", type="string", required=True),
                ToolParameter(name="limit", type="integer", required=False),
            ],
        )

        schema = definition.to_json_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_to_openai_tool(self) -> None:
        """测试转换为 OpenAI 格式。"""
        definition = ToolDefinition(
            name="test",
            description="测试工具",
            parameters=[],
        )

        tool = definition.to_openai_tool()

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "test"
        assert tool["function"]["description"] == "测试工具"

    def test_to_anthropic_tool(self) -> None:
        """测试转换为 Anthropic 格式。"""
        definition = ToolDefinition(
            name="test",
            description="测试工具",
            parameters=[],
        )

        tool = definition.to_anthropic_tool()

        assert tool["name"] == "test"
        assert tool["description"] == "测试工具"
        assert "input_schema" in tool

    def test_to_google_tool(self) -> None:
        """测试转换为 Google 格式。"""
        definition = ToolDefinition(
            name="test",
            description="测试工具",
            parameters=[],
        )

        tool = definition.to_google_tool()

        assert "function_declarations" in tool
        assert tool["function_declarations"][0]["name"] == "test"


class TestToolCall:
    """ToolCall 测试。"""

    def test_create(self) -> None:
        """测试创建。"""
        call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"city": "Beijing"},
        )

        assert call.id == "call_123"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Beijing"}

    def test_from_openai(self) -> None:
        """测试从 OpenAI 格式创建。"""
        call = ToolCall.from_openai({
            "id": "call_456",
            "function": {
                "name": "search",
                "arguments": '{"query": "hello"}',
            },
        })

        assert call.id == "call_456"
        assert call.name == "search"
        assert call.arguments == {"query": "hello"}


class TestToolResult:
    """ToolResult 测试。"""

    def test_create_success(self) -> None:
        """测试成功结果。"""
        result = ToolResult(
            tool_call_id="call_123",
            content={"temp": 25, "city": "Beijing"},
        )

        assert result.tool_call_id == "call_123"
        assert result.content == {"temp": 25, "city": "Beijing"}
        assert result.is_error is False

    def test_create_error(self) -> None:
        """测试错误结果。"""
        result = ToolResult(
            tool_call_id="call_123",
            content="",
            is_error=True,
            error_message="City not found",
        )

        assert result.is_error is True
        assert result.error_message == "City not found"

    def test_to_openai_message(self) -> None:
        """测试转换为 OpenAI 消息。"""
        result = ToolResult(
            tool_call_id="call_123",
            content={"temp": 25},
        )

        msg = result.to_openai_message()

        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert "temp" in msg["content"]

    def test_to_anthropic_content(self) -> None:
        """测试转换为 Anthropic 内容。"""
        result = ToolResult(
            tool_call_id="call_123",
            content="Sunny",
        )

        content = result.to_anthropic_content()

        assert content["type"] == "tool_result"
        assert content["tool_use_id"] == "call_123"
        assert content["content"] == "Sunny"


class TestToolExecution:
    """ToolExecution 测试。"""

    def test_create(self) -> None:
        """测试创建。"""
        call = ToolCall(id="1", name="test", arguments={})
        execution = ToolExecution(tool_call=call)

        assert execution.tool_call == call
        assert execution.status == ToolExecutionStatus.PENDING
        assert execution.result is None

    def test_with_result(self) -> None:
        """测试带结果。"""
        call = ToolCall(id="1", name="test", arguments={})
        result = ToolResult(tool_call_id="1", content="done")

        execution = ToolExecution(
            tool_call=call,
            status=ToolExecutionStatus.SUCCESS,
            result=result,
            duration_ms=100.5,
        )

        assert execution.status == ToolExecutionStatus.SUCCESS
        assert execution.result.content == "done"
        assert execution.duration_ms == 100.5
