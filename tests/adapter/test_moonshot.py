"""测试 adapter/moonshot.py - Moonshot (Kimi) 适配器。"""

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.base import (
    validate_config,
    validate_messages,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.moonshot import MoonshotProvider
from dawn_shuttle.dawn_shuttle_intelligence.src.core.config import GenerateConfig
from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import Message


class TestMoonshotProvider:
    """测试 MoonshotProvider。"""

    def test_provider_name(self) -> None:
        """测试提供商标识。"""
        provider = MoonshotProvider(api_key="test-key")
        assert provider.name == "moonshot"

    def test_default_base_url(self) -> None:
        """测试默认 API 端点。"""
        provider = MoonshotProvider()
        assert provider.base_url == "https://api.moonshot.cn/v1"

    def test_custom_base_url(self) -> None:
        """测试自定义 API 端点。"""
        provider = MoonshotProvider(base_url="https://custom.api.com/v1")
        assert provider.base_url == "https://custom.api.com/v1"

    def test_supports_model_moonshot_v1_8k(self) -> None:
        """测试支持 moonshot-v1-8k 模型。"""
        provider = MoonshotProvider()
        assert provider.supports_model("moonshot-v1-8k") is True

    def test_supports_model_moonshot_v1_32k(self) -> None:
        """测试支持 moonshot-v1-32k 模型。"""
        provider = MoonshotProvider()
        assert provider.supports_model("moonshot-v1-32k") is True

    def test_supports_model_moonshot_v1_128k(self) -> None:
        """测试支持 moonshot-v1-128k 模型。"""
        provider = MoonshotProvider()
        assert provider.supports_model("moonshot-v1-128k") is True

    def test_supports_model_invalid(self) -> None:
        """测试不支持无效模型。"""
        provider = MoonshotProvider()
        assert provider.supports_model("gpt-4") is False

    def test_get_model_list(self) -> None:
        """测试获取模型列表。"""
        provider = MoonshotProvider()
        models = provider.get_model_list()
        assert "moonshot-v1-8k" in models
        assert "moonshot-v1-32k" in models
        assert "moonshot-v1-128k" in models

    def test_validate_config_empty_model(self) -> None:
        """测试空模型名称验证。"""
        provider = MoonshotProvider()
        config = GenerateConfig()

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ConfigurationError
        with pytest.raises(ConfigurationError, match="Model name is required"):
            validate_config(config, provider.name)

    def test_validate_messages_empty(self) -> None:
        """测试空消息列表验证。"""
        provider = MoonshotProvider()

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ConfigurationError
        with pytest.raises(ConfigurationError, match="Messages list cannot be empty"):
            validate_messages([], provider.name)

    def test_build_params_basic(self) -> None:
        """测试构建基本参数。"""
        provider = MoonshotProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="moonshot-v1-8k")

        params = provider._build_params(messages, config)

        assert params["model"] == "moonshot-v1-8k"
        assert len(params["messages"]) == 1
