"""DeepSeek 适配器 - 对接 DeepSeek API。

DeepSeek API 兼容 OpenAI 格式。
"""

from typing import ClassVar

from .openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API 适配器。

    支持的模型包括 deepseek-chat, deepseek-coder 等。
    """

    name: str = "deepseek"

    DEFAULT_BASE_URL: ClassVar[str] = "https://api.deepseek.com/v1"

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "deepseek-chat",
        "deepseek-coder",
        "deepseek-reasoner",
    ]


# 便捷别名
deepseek: type[DeepSeekProvider] = DeepSeekProvider
