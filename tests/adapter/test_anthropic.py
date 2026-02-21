"""测试 adapter/anthropic.py - Anthropic (Claude) 适配器。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.anthropic import AnthropicProvider
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
            provider._validate_config(config)

    def test_validate_config_invalid_temperature(self) -> None:
        """测试无效温度验证(Anthropic 温度范围是 0-1)。"""
        provider = AnthropicProvider()
        config = GenerateConfig(model="claude-3-5-sonnet-latest", temperature=1.5)

        with pytest.raises(ConfigurationError, match="Temperature must be between"):
            provider._validate_config(config)

    def test_validate_messages_empty(self) -> None:
        """测试空消息列表验证。"""
        provider = AnthropicProvider()

        with pytest.raises(ConfigurationError, match="Messages list cannot be empty"):
            provider._validate_messages([])

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
        result = provider._handle_error(error)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import AuthenticationError
        assert isinstance(result, AuthenticationError)

    def test_handle_error_overloaded(self) -> None:
        """测试处理过载错误。"""
        provider = AnthropicProvider()

        error = Exception("Service overloaded")
        result = provider._handle_error(error)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ProviderNotAvailableError
        assert isinstance(result, ProviderNotAvailableError)

    def test_map_status_code_400(self) -> None:
        """测试状态码 400 映射。"""
        provider = AnthropicProvider()

        result = provider._map_status_code(400, "Bad request")

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import InvalidRequestError
        assert isinstance(result, InvalidRequestError)

    def test_map_status_code_403(self) -> None:
        """测试状态码 403 映射。"""
        provider = AnthropicProvider()

        result = provider._map_status_code(403, "Forbidden")

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import AuthenticationError
        assert isinstance(result, AuthenticationError)
