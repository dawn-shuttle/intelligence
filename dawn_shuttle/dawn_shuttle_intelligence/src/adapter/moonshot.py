"""Moonshot (Kimi) 适配器 - 对接 Moonshot API。

Moonshot API 兼容 OpenAI 格式。
"""

from typing import ClassVar

from .openai_compatible import OpenAICompatibleProvider


class MoonshotProvider(OpenAICompatibleProvider):
    """Moonshot (Kimi) API 适配器。

    支持的模型包括 moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k。
    """

    name: str = "moonshot"

    DEFAULT_BASE_URL: ClassVar[str] = "https://api.moonshot.cn/v1"

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
    ]


# 便捷别名
moonshot: type[MoonshotProvider] = MoonshotProvider
