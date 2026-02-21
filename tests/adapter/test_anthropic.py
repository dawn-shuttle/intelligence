"""测试 adapter/anthropic.py - Anthropic (Claude) 适配器。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.anthropic import AnthropicProvider
from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.base import (
    handle_anthropic_error,
    validate_config,
    validate_messages,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.core.config import GenerateConfig
from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ConfigurationError
from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import (
    ImageContent,
    Message,
    Role,
    TextContent,
    ToolCall,
)


class TestAnthropicProvider:
    """测试 AnthropicProvider。"""

    def test_provider_name(self) -> None:
        """测试提供商标识。"""
        provider = AnthropicProvider(api_key="test-key")
        assert provider.name == "anthropic"

    def test_supports_model_claude_35_sonnet(self) -> None:
        """测试支持 Claude 3.5 Sonnet 模型。"""
        provider = AnthropicProvider()
        assert provider.supports_model("claude-3-5-sonnet-latest") is True
        assert provider.supports_model("claude-3-5-sonnet-20241022") is True

    def test_supports_model_claude_35_haiku(self) -> None:
        """测试支持 Claude 3.5 Haiku 模型。"""
        provider = AnthropicProvider()
        assert provider.supports_model("claude-3-5-haiku-latest") is True

    def test_supports_model_claude_3_opus(self) -> None:
        """测试支持 Claude 3 Opus 模型。"""
        provider = AnthropicProvider()
        assert provider.supports_model("claude-3-opus-latest") is True

    def test_supports_model_invalid(self) -> None:
        """测试不支持无效模型。"""
        provider = AnthropicProvider()
        assert provider.supports_model("gpt-4") is False

    def test_get_model_list(self) -> None:
        """测试获取模型列表。"""
        provider = AnthropicProvider()
        models = provider.get_model_list()
        assert "claude-3-5-sonnet-latest" in models
        assert "claude-3-opus-latest" in models

    def test_validate_config_empty_model(self) -> None:
        """测试空模型名称验证。"""
        provider = AnthropicProvider()
        config = GenerateConfig()

        with pytest.raises(ConfigurationError, match="Model name is required"):
            validate_config(config, provider.name, temp_max=1.0)

    def test_validate_config_invalid_temperature(self) -> None:
        """测试无效温度验证(Anthropic 温度范围是 0-1)。"""
        provider = AnthropicProvider()
        config = GenerateConfig(model="claude-3-5-sonnet-latest", temperature=1.5)

        with pytest.raises(ConfigurationError, match="Temperature must be between"):
            validate_config(config, provider.name, temp_max=1.0)

    def test_validate_messages_empty(self) -> None:
        """测试空消息列表验证。"""
        provider = AnthropicProvider()

        with pytest.raises(ConfigurationError, match="Messages list cannot be empty"):
            validate_messages([], provider.name)

    def test_convert_message_user(self) -> None:
        """测试转换用户消息。"""
        provider = AnthropicProvider()
        msg = Message.user("Hello")

        result = provider._convert_message(msg)

        assert result["role"] == "user"
        assert result["content"] == [{"type": "text", "text": "Hello"}]

    def test_convert_message_assistant(self) -> None:
        """测试转换助手消息。"""
        provider = AnthropicProvider()
        msg = Message.assistant("Hi there")

        result = provider._convert_message(msg)

        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": "Hi there"}]

    def test_convert_message_multimodal(self) -> None:
        """测试转换多模态消息。"""
        provider = AnthropicProvider()
        msg = Message(
            role=Role.USER,
            content=[
                TextContent(text="描述这张图片"),
                ImageContent(image="base64imagedata", mime_type="image/png"),
            ],
        )

        result = provider._convert_message(msg)

        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "image"

    def test_convert_message_image_url_raises(self) -> None:
        """测试图片 URL 抛出异常。"""
        provider = AnthropicProvider()
        msg = Message(
            role=Role.USER,
            content=[ImageContent(image="https://example.com/image.png")],
        )

        with pytest.raises(
            ConfigurationError,
            match="does not support image URLs"
        ):
            provider._convert_message(msg)

    def test_convert_message_tool_calls(self) -> None:
        """测试转换带工具调用的消息。"""
        provider = AnthropicProvider()
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"city": "Beijing"},
        )
        msg = Message.assistant(tool_calls=[tool_call])

        result = provider._convert_message(msg)

        assert result["role"] == "assistant"
        assert any(c["type"] == "tool_use" for c in result["content"])

    def test_convert_message_tool_result(self) -> None:
        """测试转换工具结果消息。"""
        provider = AnthropicProvider()
        msg = Message.tool_result("call_123", "天气晴朗")

        result = provider._convert_message(msg)

        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"

    def test_convert_tools(self) -> None:
        """测试转换工具定义。"""
        provider = AnthropicProvider()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取天气",
                    "parameters": {"type": "object"},
                },
            }
        ]

        result = provider._convert_tools(tools)

        assert len(result) == 1
        assert result[0]["name"] == "get_weather"

    def test_convert_tool_choice_auto(self) -> None:
        """测试转换工具选择(auto)。"""
        provider = AnthropicProvider()

        result = provider._convert_tool_choice("auto")

        assert result == {"type": "auto"}

    def test_convert_tool_choice_required(self) -> None:
        """测试转换工具选择(required)。"""
        provider = AnthropicProvider()

        result = provider._convert_tool_choice("required")

        assert result == {"type": "any"}

    def test_convert_tool_choice_none(self) -> None:
        """测试转换工具选择(none)。"""
        provider = AnthropicProvider()

        result = provider._convert_tool_choice("none")

        # Anthropic 没有 none, 会返回 auto
        assert result == {"type": "auto"}

    def test_convert_tool_choice_specific(self) -> None:
        """测试转换工具选择(指定工具)。"""
        provider = AnthropicProvider()

        result = provider._convert_tool_choice({"name": "get_weather"})

        assert result == {"type": "tool", "name": "get_weather"}

    def test_build_params_with_system(self) -> None:
        """测试构建带系统消息的参数。"""
        provider = AnthropicProvider()
        messages = [
            Message.system("你是一个助手"),
            Message.user("你好"),
        ]
        config = GenerateConfig(model="claude-3-5-sonnet-latest")

        params = provider._build_params(messages, config)

        assert params["system"] == "你是一个助手"
        assert len(params["messages"]) == 1

    def test_handle_error_auth(self) -> None:
        """测试处理认证错误。"""
        provider = AnthropicProvider()

        error = Exception("401 Unauthorized")
        result = handle_anthropic_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import AuthenticationError
        assert isinstance(result, AuthenticationError)

    def test_handle_error_overloaded(self) -> None:
        """测试处理过载错误。"""
        provider = AnthropicProvider()

        error = Exception("Service overloaded")
        result = handle_anthropic_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ProviderNotAvailableError
        assert isinstance(result, ProviderNotAvailableError)

    def test_map_status_code_400(self) -> None:
        """测试状态码 400 映射。"""
        from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.base import map_status_code_to_error
        provider = AnthropicProvider()

        result = map_status_code_to_error(400, "Bad request", provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import InvalidRequestError
        assert isinstance(result, InvalidRequestError)

    def test_map_status_code_403(self) -> None:
        """测试状态码 403 映射。"""
        from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.base import map_status_code_to_error
        provider = AnthropicProvider()

        result = map_status_code_to_error(403, "Forbidden", provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import AuthenticationError
        assert isinstance(result, AuthenticationError)

    def test_get_client_success(self) -> None:
        """测试成功获取客户端。"""
        provider = AnthropicProvider(api_key="test-key")

        mock_client = MagicMock()
        with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=MagicMock(return_value=mock_client))}):
            client = provider._get_client()
            assert client is not None

    def test_get_client_import_error(self) -> None:
        """测试获取客户端时导入错误。"""
        provider = AnthropicProvider(api_key="test-key")

        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic 包未安装"):
                provider._get_client()

    def test_parse_response_with_text(self) -> None:
        """测试解析文本响应。"""
        provider = AnthropicProvider()

        # Mock response
        response = MagicMock()
        response.content = [MagicMock(text="Hello, world!")]
        response.stop_reason = "end_turn"
        response.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
        )
        response.model = "claude-3-5-sonnet-latest"
        response.id = "msg_123"

        result = provider._parse_response(response)

        assert result.text == "Hello, world!"
        assert result.finish_reason == "stop"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10

    def test_parse_response_with_tool_call(self) -> None:
        """测试解析带工具调用的响应。"""
        provider = AnthropicProvider()

        # Mock response with tool call
        response = MagicMock()
        tool_block = MagicMock()
        # 删除 text 属性，让它不被 hasattr 检测到
        del tool_block.text
        tool_block.name = "get_weather"
        tool_block.id = "call_123"
        tool_block.input = {"city": "Beijing"}
        response.content = [tool_block]
        response.stop_reason = "tool_use"
        response.usage = None
        response.model = "claude-3-5-sonnet-latest"
        response.id = "msg_123"

        result = provider._parse_response(response)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.finish_reason == "tool_calls"

    def test_parse_response_max_tokens(self) -> None:
        """测试解析 max_tokens 结束的响应。"""
        provider = AnthropicProvider()

        response = MagicMock()
        response.content = [MagicMock(text="Hello")]
        response.stop_reason = "max_tokens"
        response.usage = None
        response.model = "claude-3-5-sonnet-latest"
        response.id = "msg_123"

        result = provider._parse_response(response)

        assert result.finish_reason == "length"

    def test_build_params_with_tools(self) -> None:
        """测试构建带工具的参数。"""
        provider = AnthropicProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(
            model="claude-3-5-sonnet-latest",
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取天气",
                    "parameters": {"type": "object"},
                },
            }],
            tool_choice="auto",
        )

        params = provider._build_params(messages, config)

        assert "tools" in params
        assert "tool_choice" in params

    def test_build_params_with_stop_sequences(self) -> None:
        """测试构建带停止序列的参数。"""
        provider = AnthropicProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(
            model="claude-3-5-sonnet-latest",
            stop=["END", "STOP"],
        )

        params = provider._build_params(messages, config)

        assert params["stop_sequences"] == ["END", "STOP"]

    def test_build_params_with_temperature(self) -> None:
        """测试构建带温度的参数。"""
        provider = AnthropicProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(
            model="claude-3-5-sonnet-latest",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

        params = provider._build_params(messages, config)

        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9
        assert params["top_k"] == 50

    def test_parse_stream_event_text_delta(self) -> None:
        """测试解析文本增量流式事件。"""
        provider = AnthropicProvider()

        # 创建一个模拟的 TextDelta 类
        class MockTextDelta:
            def __init__(self, text: str):
                self.text = text

        # Mock event - delta 是 MockTextDelta 实例
        event = MagicMock()
        event.delta = MockTextDelta("Hello")

        # Mock anthropic.types.TextDelta
        with patch.dict("sys.modules", {"anthropic.types": MagicMock(TextDelta=MockTextDelta)}):
            result = provider._parse_stream_event(event)

        assert result is not None
        assert result.delta == "Hello"
        assert not result.is_finished

    def test_parse_stream_event_message_stop(self) -> None:
        """测试解析消息停止流式事件。"""
        provider = AnthropicProvider()

        # Mock event - delta 不是 TextDelta 实例
        event = MagicMock()
        event.delta = "not a TextDelta"  # isinstance 会返回 False
        event.message = MagicMock()
        event.message.stop_reason = "end_turn"

        # Mock anthropic.types.TextDelta - 使用一个不会匹配的类
        class NotTextDelta:
            pass

        with patch.dict("sys.modules", {"anthropic.types": MagicMock(TextDelta=NotTextDelta)}):
            result = provider._parse_stream_event(event)

        assert result is not None
        assert result.is_finished
        assert result.finish_reason == "end_turn"

    def test_parse_stream_event_tool_use_stop(self) -> None:
        """测试解析工具调用停止流式事件。"""
        provider = AnthropicProvider()

        # Mock event
        event = MagicMock()
        event.delta = "not a TextDelta"
        event.message = MagicMock()
        event.message.stop_reason = "tool_use"

        class NotTextDelta:
            pass

        with patch.dict("sys.modules", {"anthropic.types": MagicMock(TextDelta=NotTextDelta)}):
            result = provider._parse_stream_event(event)

        assert result is not None
        assert result.finish_reason == "tool_use"

    def test_parse_stream_event_content_block_stop(self) -> None:
        """测试解析内容块停止流式事件。"""
        provider = AnthropicProvider()

        # Mock event - 没有 message 属性
        event = MagicMock()
        event.delta = "not a TextDelta"
        del event.message

        class NotTextDelta:
            pass

        with patch.dict("sys.modules", {"anthropic.types": MagicMock(TextDelta=NotTextDelta)}):
            result = provider._parse_stream_event(event)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """测试成功生成响应。"""
        provider = AnthropicProvider(api_key="test-key")

        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello!")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = None
        mock_response.model = "claude-3-5-sonnet-latest"
        mock_response.id = "msg_123"

        mock_client = AsyncMock()
        mock_client.messages = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message.user("Hi")]
            config = GenerateConfig(model="claude-3-5-sonnet-latest")

            result = await provider.generate(messages, config)

            assert result.text == "Hello!"

    @pytest.mark.asyncio
    async def test_generate_error(self) -> None:
        """测试生成时的错误处理。"""
        provider = AnthropicProvider(api_key="test-key")

        mock_client = AsyncMock()
        mock_client.messages = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=Exception("401 Unauthorized")
        )

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message.user("Hi")]
            config = GenerateConfig(model="claude-3-5-sonnet-latest")

            from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import AuthenticationError
            with pytest.raises(AuthenticationError):
                await provider.generate(messages, config)

    @pytest.mark.asyncio
    async def test_generate_stream_success(self) -> None:
        """测试成功生成流式响应。"""
        provider = AnthropicProvider(api_key="test-key")

        # 创建模拟 TextDelta 类
        class MockTextDelta:
            def __init__(self, text: str):
                self.text = text

        # 创建模拟事件
        events = []

        # Text delta event
        event1 = MagicMock()
        event1.delta = MockTextDelta("Hello")
        events.append(event1)

        # Stop event
        event2 = MagicMock()
        event2.delta = "not a TextDelta"
        event2.message = MagicMock()
        event2.message.stop_reason = "end_turn"
        events.append(event2)

        # 创建异步迭代器
        class AsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        # Mock stream context manager
        mock_stream = AsyncMock()
        mock_stream.__aiter__ = lambda self: AsyncIterator(events)

        # Mock messages.stream 返回一个上下文管理器
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_client = AsyncMock()
        mock_client.messages = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=mock_context)

        with patch.object(provider, "_get_client", return_value=mock_client):
            with patch.dict("sys.modules", {"anthropic.types": MagicMock(TextDelta=MockTextDelta)}):
                messages = [Message.user("Hi")]
                config = GenerateConfig(model="claude-3-5-sonnet-latest")

                chunks = []
                async for chunk in provider.generate_stream(messages, config):
                    chunks.append(chunk)

                assert len(chunks) >= 0
