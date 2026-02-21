"""测试 adapter/deepseek.py - DeepSeek 适配器。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.deepseek import DeepSeekProvider
from dawn_shuttle.dawn_shuttle_intelligence.src.core.config import GenerateConfig
from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import Message


class TestDeepSeekProvider:
    """测试 DeepSeekProvider。"""

    def test_provider_name(self) -> None:
        """测试提供商标识。"""
        provider = DeepSeekProvider(api_key="test-key")
        assert provider.name == "deepseek"

    def test_default_base_url(self) -> None:
        """测试默认 API 端点。"""
        provider = DeepSeekProvider()
        assert provider.base_url == "https://api.deepseek.com/v1"

    def test_custom_base_url(self) -> None:
        """测试自定义 API 端点。"""
        provider = DeepSeekProvider(base_url="https://custom.api.com/v1")
        assert provider.base_url == "https://custom.api.com/v1"

    def test_supports_model_deepseek_chat(self) -> None:
        """测试支持 deepseek-chat 模型。"""
        provider = DeepSeekProvider()
        assert provider.supports_model("deepseek-chat") is True

    def test_supports_model_deepseek_coder(self) -> None:
        """测试支持 deepseek-coder 模型。"""
        provider = DeepSeekProvider()
        assert provider.supports_model("deepseek-coder") is True

    def test_supports_model_invalid(self) -> None:
        """测试不支持无效模型。"""
        provider = DeepSeekProvider()
        assert provider.supports_model("gpt-4") is False

    def test_get_model_list(self) -> None:
        """测试获取模型列表。"""
        provider = DeepSeekProvider()
        models = provider.get_model_list()
        assert "deepseek-chat" in models
        assert "deepseek-coder" in models

    def test_validate_config_empty_model(self) -> None:
        """测试空模型名称验证。"""
        provider = DeepSeekProvider()
        config = GenerateConfig()

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ConfigurationError
        with pytest.raises(ConfigurationError, match="Model name is required"):
            provider._validate_config(config)

    def test_validate_messages_empty(self) -> None:
        """测试空消息列表验证。"""
        provider = DeepSeekProvider()

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ConfigurationError
        with pytest.raises(ConfigurationError, match="Messages list cannot be empty"):
            provider._validate_messages([])

    def test_build_params_basic(self) -> None:
        """测试构建基本参数。"""
        provider = DeepSeekProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="deepseek-chat")

        params = provider._build_params(messages, config)

        assert params["model"] == "deepseek-chat"
        assert len(params["messages"]) == 1
