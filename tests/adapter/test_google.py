"""测试 adapter/google.py - Google (Gemini) 适配器。"""

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.google import GoogleProvider
from dawn_shuttle.dawn_shuttle_intelligence.src.core.config import GenerateConfig
from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ConfigurationError
from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import (
    ImageContent,
    Message,
    Role,
    TextContent,
    ToolCall,
)


class TestGoogleProvider:
    """测试 GoogleProvider。"""

    def test_provider_name(self) -> None:
        """测试提供商标识。"""
        provider = GoogleProvider(api_key="test-key")
        assert provider.name == "google"

    def test_supports_model_gemini_20_flash(self) -> None:
        """测试支持 Gemini 2.0 Flash 模型。"""
        provider = GoogleProvider()
        assert provider.supports_model("gemini-2.0-flash") is True

    def test_supports_model_gemini_15_pro(self) -> None:
        """测试支持 Gemini 1.5 Pro 模型。"""
        provider = GoogleProvider()
        assert provider.supports_model("gemini-1.5-pro") is True

    def test_supports_model_gemini_15_flash(self) -> None:
        """测试支持 Gemini 1.5 Flash 模型。"""
        provider = GoogleProvider()
        assert provider.supports_model("gemini-1.5-flash") is True

    def test_supports_model_invalid(self) -> None:
        """测试不支持无效模型。"""
        provider = GoogleProvider()
        assert provider.supports_model("gpt-4") is False

    def test_get_model_list(self) -> None:
        """测试获取模型列表。"""
        provider = GoogleProvider()
        models = provider.get_model_list()
        assert "gemini-2.0-flash" in models
        assert "gemini-1.5-pro" in models

    def test_validate_config_empty_model(self) -> None:
        """测试空模型名称验证。"""
        provider = GoogleProvider()
        config = GenerateConfig()

        with pytest.raises(ConfigurationError, match="Model name is required"):
            provider._validate_config(config)

    def test_validate_config_invalid_temperature(self) -> None:
        """测试无效温度验证。"""
        provider = GoogleProvider()
        config = GenerateConfig(model="gemini-2.0-flash", temperature=3.0)

        with pytest.raises(ConfigurationError, match="Temperature must be between"):
            provider._validate_config(config)

    def test_validate_messages_empty(self) -> None:
        """测试空消息列表验证。"""
        provider = GoogleProvider()

        with pytest.raises(ConfigurationError, match="Messages list cannot be empty"):
            provider._validate_messages([])

    def test_build_params_basic(self) -> None:
        """测试构建基本参数。"""
        provider = GoogleProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="gemini-2.0-flash")

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert system_instruction is None

    def test_build_params_with_system(self) -> None:
        """测试构建带系统消息的参数。"""
        provider = GoogleProvider()
        messages = [
            Message.system("你是一个助手"),
            Message.user("你好"),
        ]
        config = GenerateConfig(model="gemini-2.0-flash")

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert system_instruction == "你是一个助手"
        assert len(contents) == 1

    def test_build_params_with_temperature(self) -> None:
        """测试构建带温度参数。"""
        provider = GoogleProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="gemini-2.0-flash", temperature=0.7)

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert generation_config["temperature"] == 0.7

    def test_build_params_with_max_tokens(self) -> None:
        """测试构建带 max_tokens 参数。"""
        provider = GoogleProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="gemini-2.0-flash", max_tokens=1000)

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert generation_config["max_output_tokens"] == 1000

    def test_convert_tools(self) -> None:
        """测试转换工具定义。"""
        provider = GoogleProvider()
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
        assert "function_declarations" in result[0]

    def test_handle_error_invalid_api_key(self) -> None:
        """测试处理无效 API Key 错误。"""
        provider = GoogleProvider()

        error = Exception("InvalidAPIKey: Invalid API key")
        result = provider._handle_error(error)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import AuthenticationError
        assert isinstance(result, AuthenticationError)

    def test_handle_error_resource_exhausted(self) -> None:
        """测试处理资源耗尽错误。"""
        provider = GoogleProvider()

        error = Exception("ResourceExhausted: Rate limit exceeded")
        result = provider._handle_error(error)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import RateLimitError
        assert isinstance(result, RateLimitError)

    def test_handle_error_safety(self) -> None:
        """测试处理安全过滤错误。"""
        provider = GoogleProvider()

        error = Exception("Safety: Content blocked")
        result = provider._handle_error(error)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ContentFilterError
        assert isinstance(result, ContentFilterError)

    def test_handle_error_deadline_exceeded(self) -> None:
        """测试处理超时错误。"""
        provider = GoogleProvider()

        error = Exception("DeadlineExceeded: Timeout")
        result = provider._handle_error(error)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import TimeoutError
        assert isinstance(result, TimeoutError)
