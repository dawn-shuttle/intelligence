"""Provider 协议/基类 - 定义 AI 提供商的统一接口。"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from .config import GenerateConfig
from .response import GenerateResponse, StreamChunk
from .types import Message


class BaseProvider(ABC):
    """AI 提供商基类，所有适配器必须实现此接口。"""

    # 提供商标识
    name: str = "base"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs):
        """初始化提供商。

        Args:
            api_key: API 密钥
            base_url: 自定义 API 端点
            **kwargs: 其他提供商特定参数
        """
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        config: GenerateConfig,
    ) -> GenerateResponse:
        """生成文本响应（非流式）。

        Args:
            messages: 消息列表
            config: 生成配置

        Returns:
            GenerateResponse: 统一格式的响应
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerateConfig,
    ) -> AsyncIterator[StreamChunk]:
        """生成流式响应。

        Args:
            messages: 消息列表
            config: 生成配置

        Yields:
            StreamChunk: 流式响应块
        """
        pass

    def supports_model(self, model: str) -> bool:
        """检查是否支持指定模型。

        Args:
            model: 模型标识

        Returns:
            bool: 是否支持
        """
        return True  # 默认支持所有模型，子类可重写

    def get_model_list(self) -> list[str]:
        """获取支持的模型列表。

        Returns:
            list[str]: 模型标识列表
        """
        return []  # 子类应重写此方法
