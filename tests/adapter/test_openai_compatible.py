"""测试 adapter/openai_compatible.py - OpenAI 兼容格式基类。"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.base import (
    validate_config,
    validate_messages,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.openai_compatible import (
    OpenAICompatibleProvider,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.core.config import GenerateConfig
from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ConfigurationError
from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import Message, Role


class MockCompatibleProvider(OpenAICompatibleProvider):
    """模拟的兼容提供商用于测试。"""

    name = "mock_provider"
    DEFAULT_BASE_URL = "https://api.mock.com/v1"
    SUPPORTED_MODELS = ["mock-1", "mock-2"]


class TestOpenAICompatibleProvider:
    """测试 OpenAICompatibleProvider 基类。"""

    def test_provider_name(self) -> None:
        """测试提供商标识。"""
        provider = MockCompatibleProvider(api_key="test-key")
        assert provider.name == "mock_provider"

    def test_default_base_url(self) -> None:
        """测试默认 API 端点。"""
        provider = MockCompatibleProvider(api_key="test-key")
        assert provider.base_url == "https://api.mock.com/v1"

    def test_custom_base_url(self) -> None:
        """测试自定义 API 端点。"""
        provider = MockCompatibleProvider(
            api_key="test-key",
            base_url="https://custom.api.com/v1",
        )
        assert provider.base_url == "https://custom.api.com/v1"

    def test_supports_model(self) -> None:
        """测试模型支持检查。"""
        provider = MockCompatibleProvider()
        assert provider.supports_model("mock-1") is True
        assert provider.supports_model("mock-2-pro") is True  # 前缀匹配
        assert provider.supports_model("other-model") is False

    def test_get_model_list(self) -> None:
        """测试获取模型列表。"""
        provider = MockCompatibleProvider()
        models = provider.get_model_list()
        assert "mock-1" in models
        assert "mock-2" in models
        # 返回副本，修改不影响原始
        models.append("new-model")
        assert "new-model" not in provider.get_model_list()

    def test_validate_config_no_model(self) -> None:
        """测试配置验证：无模型。"""
        provider = MockCompatibleProvider()
        config = GenerateConfig(model="")

        with pytest.raises(ConfigurationError):
            validate_config(config, provider.name)

    def test_validate_config_invalid_top_p_high(self) -> None:
        """测试配置验证：top_p 过高。"""
        provider = MockCompatibleProvider()
        config = GenerateConfig(model="mock-1", top_p=1.5)

        with pytest.raises(ConfigurationError, match="top_p"):
            validate_config(config, provider.name)

    def test_validate_config_invalid_top_p_low(self) -> None:
        """测试配置验证：top_p 过低。"""
        provider = MockCompatibleProvider()
        config = GenerateConfig(model="mock-1", top_p=-0.1)

        with pytest.raises(ConfigurationError, match="top_p"):
            validate_config(config, provider.name)

    def test_validate_config_invalid_max_tokens(self) -> None:
        """测试配置验证：无效 max_tokens。"""
        provider = MockCompatibleProvider()
        config = GenerateConfig(model="mock-1", max_tokens=-100)

        with pytest.raises(ConfigurationError, match="max_tokens"):
            validate_config(config, provider.name)

    def test_validate_config_valid(self) -> None:
        """测试配置验证：有效配置。"""
        provider = MockCompatibleProvider()
        config = GenerateConfig(
            model="mock-1",
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
        )
        # 不应抛出异常
        validate_config(config, provider.name)

    def test_validate_messages_empty(self) -> None:
        """测试消息验证：空消息列表。"""
        provider = MockCompatibleProvider()

        with pytest.raises(ConfigurationError, match="empty"):
            validate_messages([], provider.name)

    def test_validate_messages_valid(self) -> None:
        """测试消息验证：有效消息。"""
        provider = MockCompatibleProvider()
        messages = [Message.user("Hello")]

        # 不应抛出异常
        validate_messages(messages, provider.name)

    def test_build_params_basic(self) -> None:
        """测试构建基本参数。"""
        provider = MockCompatibleProvider()
        messages = [Message.user("Hello")]
        config = GenerateConfig(model="mock-1")

        params = provider._build_params(messages, config)

        assert params["model"] == "mock-1"
        assert len(params["messages"]) == 1
        assert params["messages"][0]["role"] == "user"

    def test_build_params_with_temperature(self) -> None:
        """测试构建带温度参数。"""
        provider = MockCompatibleProvider()
        messages = [Message.user("Hello")]
        config = GenerateConfig(model="mock-1", temperature=0.5)

        params = provider._build_params(messages, config)

        assert params["temperature"] == 0.5

    def test_build_params_with_max_tokens(self) -> None:
        """测试构建带 max_tokens 参数。"""
        provider = MockCompatibleProvider()
        messages = [Message.user("Hello")]
        config = GenerateConfig(model="mock-1", max_tokens=500)

        params = provider._build_params(messages, config)

        assert params["max_tokens"] == 500

    def test_build_params_with_system_message(self) -> None:
        """测试构建带系统消息。"""
        provider = MockCompatibleProvider()
        messages = [
            Message.system("You are helpful"),
            Message.user("Hello"),
        ]
        config = GenerateConfig(model="mock-1")

        params = provider._build_params(messages, config)

        assert len(params["messages"]) == 2
        assert params["messages"][0]["role"] == "system"

    def test_build_params_with_stop(self) -> None:
        """测试构建带 stop 参数。"""
        provider = MockCompatibleProvider()
        messages = [Message.user("Hello")]
        config = GenerateConfig(model="mock-1", stop=["END"])

        params = provider._build_params(messages, config)

        assert params["stop"] == ["END"]

    def test_build_params_with_tools(self) -> None:
        """测试构建带工具参数。"""
        provider = MockCompatibleProvider()
        messages = [Message.user("Hello")]
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        config = GenerateConfig(model="mock-1", tools=tools)

        params = provider._build_params(messages, config)

        assert "tools" in params
        assert params["tools"][0]["function"]["name"] == "test"

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """测试成功生成响应。"""
        provider = MockCompatibleProvider(api_key="test-key")

        # Mock 客户端
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.id = "test-id"

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider._client = mock_client

        messages = [Message.user("Hello")]
        config = GenerateConfig(model="mock-1")

        response = await provider.generate(messages, config)

        assert response.text == "Hello, world!"
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_generate_with_empty_content(self) -> None:
        """测试空内容响应。"""
        provider = MockCompatibleProvider(api_key="test-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = None
        mock_response.id = "test-id"

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        messages = [Message.user("Hello")]
        config = GenerateConfig(model="mock-1")

        response = await provider.generate(messages, config)

        assert response.text == ""
