"""OpenAI 适配器 - 对接 OpenAI API。"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, ClassVar

from ..core.config import GenerateConfig
from ..core.error import (
    AIError,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ContentFilterError,
    InternalServerError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderNotAvailableError,
    QuotaExceededError,
    RateLimitError,
    ResponseParseError,
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

    def _validate_config(self, config: GenerateConfig) -> None:
        """验证配置参数。

        Args:
            config: 生成配置。

        Raises:
            ConfigurationError: 配置无效。
        """
        if not config.model:
            raise ConfigurationError(
                "Model name is required",
                provider=self.name,
            )

        if config.temperature is not None and not 0.0 <= config.temperature <= 2.0:
            raise ConfigurationError(
                f"Temperature must be between 0.0 and 2.0, got {config.temperature}",
                provider=self.name,
            )

        if config.top_p is not None and not 0.0 <= config.top_p <= 1.0:
            raise ConfigurationError(
                f"top_p must be between 0.0 and 1.0, got {config.top_p}",
                provider=self.name,
            )

        if config.max_tokens is not None and config.max_tokens <= 0:
            raise ConfigurationError(
                f"max_tokens must be positive, got {config.max_tokens}",
                provider=self.name,
            )

    def _validate_messages(self, messages: list[Message]) -> None:
        """验证消息列表。

        Args:
            messages: 消息列表。

        Raises:
            ConfigurationError: 消息无效。
        """
        if not messages:
            raise ConfigurationError(
                "Messages list cannot be empty",
                provider=self.name,
            )

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
            ConfigurationError: 配置错误。
            ConnectionError: 连接错误。
            RateLimitError: 速率限制。
            AuthenticationError: 认证错误。
            InvalidRequestError: 请求无效。
            ModelNotFoundError: 模型不存在。
            ContentFilterError: 内容过滤。
            TimeoutError: 超时。
            QuotaExceededError: 配额用尽。
            ProviderNotAvailableError: 服务不可用。
            InternalServerError: 服务器错误。
            ResponseParseError: 响应解析错误。
        """
        self._validate_config(config)
        self._validate_messages(messages)

        client = self._get_client()
        params = self._build_params(messages, config)

        try:
            response = await client.chat.completions.create(**params)
        except Exception as e:
            raise self._handle_error(e) from e

        try:
            return self._parse_response(response)
        except (KeyError, IndexError, AttributeError) as e:
            raise ResponseParseError(
                f"Failed to parse response: {e}",
                provider=self.name,
            ) from e

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
            ConfigurationError: 配置错误。
            ConnectionError: 连接错误。
            RateLimitError: 速率限制。
            AuthenticationError: 认证错误。
            InvalidRequestError: 请求无效。
            ModelNotFoundError: 模型不存在。
            ContentFilterError: 内容过滤。
            TimeoutError: 超时。
            QuotaExceededError: 配额用尽。
            ProviderNotAvailableError: 服务不可用。
            InternalServerError: 服务器错误。
            ResponseParseError: 响应解析错误。
        """
        self._validate_config(config)
        self._validate_messages(messages)

        client = self._get_client()
        params = self._build_params(messages, config, stream=True)

        try:
            stream = await client.chat.completions.create(**params)
        except Exception as e:
            raise self._handle_error(e) from e

        async for chunk in stream:
            try:
                parsed = self._parse_stream_chunk(chunk)
            except (KeyError, IndexError, AttributeError) as e:
                raise ResponseParseError(
                    f"Failed to parse stream chunk: {e}",
                    provider=self.name,
                ) from e
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
            具体的错误类型实例。
        """
        error_type: str = type(error).__name__
        error_message: str = str(error)

        # 提取额外信息
        status_code: int | None = getattr(error, "status_code", None)
        request_id: str | None = getattr(error, "request_id", None)

        # OpenAI SDK 具体异常类型映射
        if "AuthenticationError" in error_type:
            return AuthenticationError(
                error_message,
                provider=self.name,
                status_code=status_code,
                request_id=request_id,
                cause=error,
            )

        if "RateLimitError" in error_type:
            # 尝试提取 retry-after
            retry_after: int | None = None
            if hasattr(error, "response"):
                resp = getattr(error, "response", None)
                if resp and hasattr(resp, "headers"):
                    ra = resp.headers.get("retry-after")
                    if ra:
                        with contextlib.suppress(ValueError):
                            retry_after = int(ra)
            return RateLimitError(
                error_message,
                provider=self.name,
                status_code=status_code or 429,
                request_id=request_id,
                retry_after=retry_after,
                cause=error,
            )

        if "BadRequestError" in error_type:
            return InvalidRequestError(
                error_message,
                provider=self.name,
                status_code=status_code or 400,
                request_id=request_id,
                cause=error,
            )

        if "NotFoundError" in error_type:
            return ModelNotFoundError(
                error_message,
                provider=self.name,
                status_code=status_code or 404,
                request_id=request_id,
                cause=error,
            )

        if "APIStatusError" in error_type and status_code:
            return self._map_status_code(status_code, error_message, error)

        # HTTP 状态码映射(从错误消息中提取)
        if "401" in error_message:
            return AuthenticationError(
                error_message,
                provider=self.name,
                status_code=401,
                cause=error,
            )

        if "403" in error_message:
            return AuthenticationError(
                f"Access forbidden: {error_message}",
                provider=self.name,
                status_code=403,
                cause=error,
            )

        if "429" in error_message:
            return RateLimitError(
                error_message,
                provider=self.name,
                status_code=429,
                cause=error,
            )

        if "400" in error_message:
            return InvalidRequestError(
                error_message,
                provider=self.name,
                status_code=400,
                cause=error,
            )

        if "404" in error_message:
            return ModelNotFoundError(
                error_message,
                provider=self.name,
                status_code=404,
                cause=error,
            )

        if "500" in error_message or "Internal" in error_type:
            return InternalServerError(
                error_message,
                provider=self.name,
                status_code=500,
                request_id=request_id,
                cause=error,
            )

        if "503" in error_message or "Service Unavailable" in error_message:
            return ProviderNotAvailableError(
                error_message,
                provider=self.name,
                status_code=503,
                cause=error,
            )

        if "502" in error_message or "Bad Gateway" in error_message:
            return ProviderNotAvailableError(
                error_message,
                provider=self.name,
                status_code=502,
                cause=error,
            )

        # 内容过滤
        if "ContentFilter" in error_message:
            return ContentFilterError(
                error_message,
                provider=self.name,
                cause=error,
            )

        # 超时
        if "Timeout" in error_type or "timeout" in error_message.lower():
            return TimeoutError(
                error_message,
                provider=self.name,
                cause=error,
            )

        # 连接错误
        if "Connection" in error_type or "connect" in error_message.lower():
            return ConnectionError(
                error_message,
                provider=self.name,
                cause=error,
            )

        # 配额错误
        if "quota" in error_message.lower() or "insufficient_quota" in error_message:
            return QuotaExceededError(
                error_message,
                provider=self.name,
                cause=error,
            )

        # 默认返回服务器错误
        return InternalServerError(
            f"Unexpected error: {error_type}: {error_message}",
            provider=self.name,
            status_code=status_code,
            request_id=request_id,
            cause=error,
        ).with_context(original_type=error_type)

    def _map_status_code(
        self, status_code: int, message: str, cause: Exception | None = None
    ) -> AIError:
        """根据 HTTP 状态码映射到具体错误类型。

        Args:
            status_code: HTTP 状态码。
            message: 错误消息。
            cause: 原始异常。

        Returns:
            具体的错误类型实例。
        """
        error_map: dict[int, type[AIError]] = {
            400: InvalidRequestError,
            401: AuthenticationError,
            403: AuthenticationError,
            404: ModelNotFoundError,
            429: RateLimitError,
            500: InternalServerError,
            502: ProviderNotAvailableError,
            503: ProviderNotAvailableError,
            504: TimeoutError,
        }

        error_class = error_map.get(status_code, InternalServerError)

        return error_class(
            message,
            provider=self.name,
            status_code=status_code,
            cause=cause,
        )


# 便捷别名
openai: type[OpenAIProvider] = OpenAIProvider
