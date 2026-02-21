"""工具类型测试。"""

from __future__ import annotations

from dawn_shuttle.dawn_shuttle_intelligence.src.tools.types import (
    ToolCall,
    ToolDefinition,
    ToolExecution,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
    normalize_content,
    content_to_string,
)


class TestHelperFunctions:
    """辅助函数测试。"""

    def test_normalize_content_str(self) -> None:
        """测试规范化字符串内容。"""
        assert normalize_content("hello") == "hello"

    def test_normalize_content_dict(self) -> None:
        """测试规范化字典内容。"""
        data = {"key": "value"}
        assert normalize_content(data) == data

    def test_normalize_content_bytes(self) -> None:
        """测试规范化字节内容。"""
        data = b"hello"
        assert normalize_content(data) == data

    def test_normalize_content_other(self) -> None:
        """测试规范化其他类型内容。"""
        assert normalize_content(123) == "123"
        assert normalize_content([1, 2, 3]) == "[1, 2, 3]"

    def test_content_to_string_str(self) -> None:
        """测试字符串内容转换。"""
        assert content_to_string("hello") == "hello"

    def test_content_to_string_dict(self) -> None:
        """测试字典内容转换。"""
        assert content_to_string({"key": "value"}) == '{"key": "value"}'

    def test_content_to_string_bytes(self) -> None:
        """测试字节内容转换。"""
        assert content_to_string(b"hello") == "hello"

    def test_content_to_string_other(self) -> None:
        """测试其他类型内容转换。"""
        assert content_to_string(123) == "123"


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

    def test_to_json_schema_with_items(self) -> None:
        """测试数组类型的 JSON Schema。"""
        param = ToolParameter(
            name="tags",
            type="array",
            items={"type": "string"},
        )

        schema = param.to_json_schema()

        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

    def test_to_json_schema_with_properties(self) -> None:
        """测试对象类型的 JSON Schema。"""
        param = ToolParameter(
            name="config",
            type="object",
            properties={
                "name": {"type": "string"},
                "value": {"type": "number"},
            },
        )

        schema = param.to_json_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "name" in schema["properties"]


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

    def test_from_openai_with_dict_args(self) -> None:
        """测试从 OpenAI 格式创建（参数已是字典）。"""
        call = ToolCall.from_openai({
            "id": "call_789",
            "function": {
                "name": "test",
                "arguments": {"key": "value"},  # 已经是字典
            },
        })

        assert call.arguments == {"key": "value"}

    def test_from_openai_no_id(self) -> None:
        """测试从 OpenAI 格式创建（无 ID）。"""
        call = ToolCall.from_openai({
            "function": {
                "name": "test",
                "arguments": "{}",
            },
        })

        assert call.id == ""

    def test_from_anthropic(self) -> None:
        """测试从 Anthropic 格式创建。"""
        # Mock block
        block = type("Block", (), {})()
        block.id = "toolu_123"
        block.name = "get_weather"
        block.input = {"city": "Beijing"}

        call = ToolCall.from_anthropic(block)

        assert call.id == "toolu_123"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Beijing"}

    def test_from_anthropic_non_dict_input(self) -> None:
        """测试从 Anthropic 格式创建（非字典输入）。"""
        block = type("Block", (), {})()
        block.id = "toolu_456"
        block.name = "test"
        block.input = "not a dict"

        call = ToolCall.from_anthropic(block)

        assert call.arguments == {}

    def test_from_google(self) -> None:
        """测试从 Google 格式创建。"""
        # Mock part
        part = type("Part", (), {})()
        fc = type("FunctionCall", (), {})()
        fc.name = "search"
        fc.args = {"query": "hello"}
        part.function_call = fc

        call = ToolCall.from_google(part)

        assert call.id == "call_search"
        assert call.name == "search"
        assert call.arguments == {"query": "hello"}

    def test_from_google_no_args(self) -> None:
        """测试从 Google 格式创建（无参数）。"""
        part = type("Part", (), {})()
        fc = type("FunctionCall", (), {})()
        fc.name = "test"
        fc.args = None
        part.function_call = fc

        call = ToolCall.from_google(part)

        assert call.arguments == {}


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

    def test_to_anthropic_content_bytes(self) -> None:
        """测试转换为 Anthropic 内容（字节）。"""
        result = ToolResult(
            tool_call_id="call_123",
            content=b"binary data",
        )

        content = result.to_anthropic_content()

        assert content["content"] == "binary data"

    def test_to_anthropic_content_other(self) -> None:
        """测试转换为 Anthropic 内容（其他类型）。"""
        result = ToolResult(
            tool_call_id="call_123",
            content=12345,  # 非 str/dict/bytes
        )

        content = result.to_anthropic_content()

        assert content["content"] == "12345"

    def test_to_google_content_str(self) -> None:
        """测试转换为 Google 内容（字符串）。"""
        result = ToolResult(
            tool_call_id="call_123",
            content="success",
        )

        content = result.to_google_content("test_tool")

        assert "function_response" in content
        assert content["function_response"]["name"] == "test_tool"
        assert content["function_response"]["response"] == {"result": "success"}

    def test_to_google_content_json_str(self) -> None:
        """测试转换为 Google 内容（JSON 字符串）。"""
        result = ToolResult(
            tool_call_id="call_123",
            content='{"temp": 25}',
        )

        content = result.to_google_content("test_tool")

        assert content["function_response"]["response"] == {"temp": 25}

    def test_to_google_content_bytes(self) -> None:
        """测试转换为 Google 内容（字节）。"""
        result = ToolResult(
            tool_call_id="call_123",
            content=b"binary",
        )

        content = result.to_google_content("test_tool")

        assert content["function_response"]["response"] == {"result": "binary"}

    def test_to_google_content_other(self) -> None:
        """测试转换为 Google 内容（其他类型）。"""
        result = ToolResult(
            tool_call_id="call_123",
            content=12345,
        )

        content = result.to_google_content("test_tool")

        assert content["function_response"]["response"] == {"result": "12345"}

    def test_to_google_content_error(self) -> None:
        """测试转换为 Google 内容（错误）。"""
        result = ToolResult(
            tool_call_id="call_123",
            content="",
            is_error=True,
            error_message="Tool failed",
        )

        content = result.to_google_content("test_tool")

        assert content["function_response"]["response"] == {"error": "Tool failed"}

    def test_to_google_content_error_no_message(self) -> None:
        """测试转换为 Google 内容（错误无消息）。"""
        result = ToolResult(
            tool_call_id="call_123",
            content="",
            is_error=True,
        )

        content = result.to_google_content("test_tool")

        assert content["function_response"]["response"] == {"error": "Tool execution failed"}


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
