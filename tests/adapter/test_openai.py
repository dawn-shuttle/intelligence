"""测试 adapter/openai.py - OpenAI 适配器。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.openai import OpenAIProvider
from dawn_shuttle.dawn_shuttle_intelligence.src.core.config import GenerateConfig
from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import Message, Role


class TestOpenAIProvider:
    """测试 OpenAIProvider。"""

    def test_provider_name(self) -> None:
        """测试提供商标识。"""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.name == "openai"

    def test_supports_model_gpt4(self) -> None:
        """测试支持 GPT-4 模型。"""
        provider = OpenAIProvider()
        assert provider.supports_model("gpt-4") is True
        assert provider.supports_model("gpt-4-turbo") is True
        assert provider.supports_model("gpt-4o") is True

    def test_supports_model_gpt35(self) -> None:
        """测试支持 GPT-3.5 模型。"""
        provider = OpenAIProvider()
        assert provider.supports_model("gpt-3.5-turbo") is True

    def test_supports_model_invalid(self) -> None:
        """测试不支持无效模型。"""
        provider = OpenAIProvider()
        assert provider.supports_model("invalid-model") is False

    def test_get_model_list(self) -> None:
        """测试获取模型列表。"""
        provider = OpenAIProvider()
        models = provider.get_model_list()
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models
        assert len(models) > 0

    def test_init_with_params(self) -> None:
        """测试带参数初始化。"""
        provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://custom.api.com",
            organization="org-123",
        )
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://custom.api.com"
        assert provider.organization == "org-123"

    def test_build_params_basic(self) -> None:
        """测试构建基本参数。"""
        provider = OpenAIProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="gpt-4")

        params = provider._build_params(messages, config)

        assert params["model"] == "gpt-4"
        assert len(params["messages"]) == 1
        assert params["messages"][0]["role"] == "user"

    def test_build_params_with_temperature(self) -> None:
        """测试构建带温度参数。"""
        provider = OpenAIProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="gpt-4", temperature=0.7)

        params = provider._build_params(messages, config)

        assert params["temperature"] == 0.7

    def test_build_params_with_tools(self) -> None:
        """测试构建带工具参数。"""
        provider = OpenAIProvider()
        messages = [Message.user("hello")]
        tools = [{"type": "function", "function": {"name": "test"}}]
        config = GenerateConfig(model="gpt-4", tools=tools)

        params = provider._build_params(messages, config)

        assert params["tools"] == tools

    def test_build_params_stream(self) -> None:
        """测试构建流式参数。"""
        provider = OpenAIProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="gpt-4")

        params = provider._build_params(messages, config, stream=True)

        assert params["stream"] is True

    def test_parse_response_simple(self) -> None:
        """测试解析简单响应。"""
        provider = OpenAIProvider()

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello back!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4"
        mock_response.id = "resp_123"

        result = provider._parse_response(mock_response)

        assert result.text == "Hello back!"
        assert result.finish_reason == "stop"
        assert result.model == "gpt-4"
        assert result.request_id == "resp_123"
        assert result.usage is not None
        assert result.usage.total_tokens == 15

    def test_parse_stream_chunk(self) -> None:
        """测试解析流式块。"""
        provider = OpenAIProvider()

        # Mock chunk
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello"
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].finish_reason = None

        result = provider._parse_stream_chunk(mock_chunk)

        assert result is not None
        assert result.delta == "Hello"
        assert result.is_finished is False

    def test_parse_stream_chunk_empty(self) -> None:
        """测试解析空流式块。"""
        provider = OpenAIProvider()

        # Mock empty chunk
        mock_chunk = MagicMock()
        mock_chunk.choices = []

        result = provider._parse_stream_chunk(mock_chunk)

        assert result is None

    def test_handle_error_auth(self) -> None:
        """测试处理认证错误。"""
        provider = OpenAIProvider()

        error = Exception("401 Unauthorized")
        result = provider._handle_error(error)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import AuthenticationError
        assert isinstance(result, AuthenticationError)

    def test_handle_error_rate_limit(self) -> None:
        """测试处理速率限制错误。"""
        provider = OpenAIProvider()

        error = Exception("429 Rate limit exceeded")
        result = provider._handle_error(error)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import RateLimitError
        assert isinstance(result, RateLimitError)

    def test_handle_error_not_found(self) -> None:
        """测试处理未找到错误。"""
        provider = OpenAIProvider()

        error = Exception("404 Model not found")
        result = provider._handle_error(error)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ModelNotFoundError
        assert isinstance(result, ModelNotFoundError)
