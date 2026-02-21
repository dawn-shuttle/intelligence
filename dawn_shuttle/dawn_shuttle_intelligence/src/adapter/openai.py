"""OpenAI 适配器 - 对接 OpenAI API。"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, ClassVar

from ..core.config import GenerateConfig
from ..core.error import (
    AIError,
    AuthenticationError,
    ContentFilterError,
    InvalidRequestError,
    ModelNotFoundError,
    RateLimitError,
    TimeoutError,
)
from ..core.provider import BaseProvider
from ..core.response import GenerateResponse, StreamChunk, Usage
from ..core.types import Message
from .base import message_to_openai_format, openai_tool_to_dict

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk


class OpenAIProvider(BaseProvider):
    """OpenAI API 适配器。

    支持的模型包括 GPT-4, GPT-3.5, O1 系列等。
    """

    name: str = "openai"

    # 支持的模型列表
    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-4-32k",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "o1",
        "o1-mini",
        "o1-preview",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 OpenAI 适配器。

        Args:
            api_key: OpenAI API 密钥(也可通过环境变量 OPENAI_API_KEY 设置)。
            base_url: 自定义 API 端点(用于代理或兼容服务)。
            organization: OpenAI 组织 ID。
            **kwargs: 其他参数传递给基类。
        """
        super().__init__(api_key, base_url, **kwargs)
        self.organization: str | None = organization
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """获取 OpenAI 客户端(延迟初始化)。

        Returns:
            AsyncOpenAI: OpenAI 异步客户端实例。

        Raises:
            ImportError: openai 包未安装。
        """
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "openai 包未安装, 请运行: pip install openai"
                ) from e

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
            )
        return self._client

    def supports_model(self, model: str) -> bool:
        """检查是否支持指定模型。

        Args:
            model: 模型标识。

        Returns:
            bool: 是否支持该模型。
        """
        return any(model.startswith(m) for m in self.SUPPORTED_MODELS)

    def get_model_list(self) -> list[str]:
        """获取支持的模型列表。

        Returns:
            list[str]: 模型标识列表的副本。
        """
        return self.SUPPORTED_MODELS.copy()

    async def generate(
        self,
        messages: list[Message],
        config: GenerateConfig,
    ) -> GenerateResponse:
        """生成文本响应(非流式)。

        Args:
            messages: 消息列表。
            config: 生成配置。

        Returns:
            GenerateResponse: 统一格式的响应。

        Raises:
            AIError: API 调用错误。
        """
        client = self._get_client()
        params = self._build_params(messages, config)

        try:
            response = await client.chat.completions.create(**params)
        except Exception as e:
            raise self._handle_error(e) from e

        return self._parse_response(response)

    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerateConfig,
    ) -> AsyncGenerator[StreamChunk, None]:
        """生成流式响应。

        Args:
            messages: 消息列表。
            config: 生成配置。

        Yields:
            StreamChunk: 流式响应块。

        Raises:
            AIError: API 调用错误。
        """
        client = self._get_client()
        params = self._build_params(messages, config, stream=True)

        try:
            stream = await client.chat.completions.create(**params)
        except Exception as e:
            raise self._handle_error(e) from e

        async for chunk in stream:
            parsed = self._parse_stream_chunk(chunk)
            if parsed:
                yield parsed

    def _build_params(
        self,
        messages: list[Message],
        config: GenerateConfig,
        stream: bool = False,
    ) -> dict[str, Any]:
        """构建 OpenAI API 请求参数。

        Args:
            messages: 消息列表。
            config: 生成配置。
            stream: 是否启用流式输出。

        Returns:
            dict[str, Any]: OpenAI API 请求参数字典。
        """
        # 转换消息格式
        openai_messages: list[dict[str, Any]] = [
            message_to_openai_format(m) for m in messages
        ]

        params: dict[str, Any] = {
            "model": config.model,
            "messages": openai_messages,
        }

        # 采样参数
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p

        # 输出控制
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        if config.stop is not None:
            params["stop"] = config.stop

        # 惩罚参数
        if config.frequency_penalty is not None:
            params["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            params["presence_penalty"] = config.presence_penalty

        # 种子
        if config.seed is not None:
            params["seed"] = config.seed

        # 流式
        if stream:
            params["stream"] = True

        # 工具
        if config.tools:
            params["tools"] = config.tools
        if config.tool_choice:
            params["tool_choice"] = config.tool_choice

        # 响应格式
        if config.response_format:
            params["response_format"] = config.response_format

        # 额外参数
        params.update(config.extra)

        return params

    def _parse_response(self, response: ChatCompletion) -> GenerateResponse:
        """解析 OpenAI 响应为统一格式。

        Args:
            response: OpenAI ChatCompletion 响应对象。

        Returns:
            GenerateResponse: 统一格式的响应。
        """
        choice = response.choices[0]

        # 提取文本内容
        text: str = choice.message.content or ""

        # 提取工具调用
        tool_calls: list[dict[str, Any]] = []
        if choice.message.tool_calls:
            tool_calls = [
                openai_tool_to_dict(tc.model_dump())
                for tc in choice.message.tool_calls
            ]

        # 构建使用统计
        usage: Usage | None = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return GenerateResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            raw=response,
            usage=usage,
            model=response.model,
            request_id=response.id,
        )

    def _parse_stream_chunk(
        self,
        chunk: ChatCompletionChunk,
    ) -> StreamChunk | None:
        """解析流式响应块。

        Args:
            chunk: OpenAI ChatCompletionChunk 对象。

        Returns:
            StreamChunk | None: 解析后的流式块, 无效时返回 None。
        """
        if not chunk.choices:
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        # 提取增量文本
        text: str = delta.content or ""

        # 提取工具调用增量
        tool_calls: list[dict[str, Any]] = []
        if delta.tool_calls:
            for tc in delta.tool_calls:
                tool_call: dict[str, Any] = {"id": tc.id or ""}
                if tc.function:
                    tool_call["name"] = tc.function.name or ""
                    tool_call["arguments"] = tc.function.arguments or ""
                tool_calls.append(tool_call)

        return StreamChunk(
            delta=text,
            tool_calls=tool_calls,
            is_finished=choice.finish_reason is not None,
            finish_reason=choice.finish_reason,
        )

    def _handle_error(self, error: Exception) -> AIError:
        """将 OpenAI 错误转换为统一错误类型。

        Args:
            error: 原始异常对象。

        Returns:
            AIError: 统一格式的错误对象。
        """
        error_type: str = type(error).__name__
        error_message: str = str(error)

        # 根据 OpenAI 错误类型映射
        if "AuthenticationError" in error_type or "401" in error_message:
            return AuthenticationError(error_message, provider=self.name)
        elif "RateLimitError" in error_type or "429" in error_message:
            return RateLimitError(error_message, provider=self.name)
        elif "BadRequestError" in error_type or "400" in error_message:
            return InvalidRequestError(error_message, provider=self.name)
        elif "NotFoundError" in error_type or "404" in error_message:
            return ModelNotFoundError(error_message, provider=self.name)
        elif "ContentFilter" in error_message:
            return ContentFilterError(error_message, provider=self.name)
        elif "Timeout" in error_type or "timeout" in error_message.lower():
            return TimeoutError(error_message, provider=self.name)
        else:
            return AIError(error_message, provider=self.name)


# 便捷别名
openai: type[OpenAIProvider] = OpenAIProvider
