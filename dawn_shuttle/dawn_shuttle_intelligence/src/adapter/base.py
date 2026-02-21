"""适配器基础工具 - 提供消息转换等通用功能。"""

from __future__ import annotations

import contextlib
from typing import Any

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
    TimeoutError,
)
from ..core.types import (
    ImageContent,
    Message,
    Role,
    TextContent,
)


def validate_config(
    config: GenerateConfig,
    provider_name: str,
    *,
    temp_max: float = 2.0,
) -> None:
    """验证配置参数。

    Args:
        config: 生成配置。
        provider_name: 供应商标识。
        temp_max: temperature 最大值(Anthropic 为 1.0，其他为 2.0)。

    Raises:
        ConfigurationError: 配置无效。
    """
    if not config.model:
        raise ConfigurationError(
            "Model name is required",
            provider=provider_name,
        )

    if config.temperature is not None and not 0.0 <= config.temperature <= temp_max:
        raise ConfigurationError(
            f"Temperature must be between 0.0 and {temp_max}, got {config.temperature}",
            provider=provider_name,
        )

    if config.top_p is not None and not 0.0 <= config.top_p <= 1.0:
        raise ConfigurationError(
            f"top_p must be between 0.0 and 1.0, got {config.top_p}",
            provider=provider_name,
        )

    if config.max_tokens is not None and config.max_tokens <= 0:
        raise ConfigurationError(
            f"max_tokens must be positive, got {config.max_tokens}",
            provider=provider_name,
        )


def validate_messages(messages: list[Message], provider_name: str) -> None:
    """验证消息列表。

    Args:
        messages: 消息列表。
        provider_name: 供应商标识。

    Raises:
        ConfigurationError: 消息无效。
    """
    if not messages:
        raise ConfigurationError(
            "Messages list cannot be empty",
            provider=provider_name,
        )


def extract_error_info(error: Exception) -> dict[str, Any]:
    """从异常中提取错误信息。

    Args:
        error: 异常对象。

    Returns:
        dict[str, Any]: 错误信息字典。
    """
    info: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
        "status_code": None,
        "request_id": None,
        "retry_after": None,
    }

    # 提取状态码
    if hasattr(error, "status_code"):
        info["status_code"] = error.status_code

    # 提取请求 ID
    if hasattr(error, "request_id"):
        info["request_id"] = error.request_id

    # 提取 retry-after
    if hasattr(error, "response"):
        resp = getattr(error, "response", None)
        if resp and hasattr(resp, "headers"):
            ra = resp.headers.get("retry-after")
            if ra:
                with contextlib.suppress(ValueError):
                    info["retry_after"] = int(ra)

    return info


def map_status_code_to_error(
    status_code: int,
    message: str,
    provider_name: str,
    cause: Exception | None = None,
) -> AIError:
    """根据 HTTP 状态码映射到具体错误类型。

    Args:
        status_code: HTTP 状态码。
        message: 错误消息。
        provider_name: 供应商标识。
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
        provider=provider_name,
        status_code=status_code,
        cause=cause,
    )


def handle_openai_error(
    error: Exception,
    provider_name: str,
) -> AIError:
    """处理 OpenAI 格式的错误。

    Args:
        error: 原始异常对象。
        provider_name: 供应商标识。

    Returns:
        具体的错误类型实例。
    """
    error_type: str = type(error).__name__
    error_message: str = str(error)
    info = extract_error_info(error)
    status_code = info["status_code"]
    request_id = info["request_id"]
    retry_after = info["retry_after"]

    # OpenAI SDK 具体异常类型映射
    if "AuthenticationError" in error_type:
        return AuthenticationError(
            error_message,
            provider=provider_name,
            status_code=status_code,
            request_id=request_id,
            cause=error,
        )

    if "RateLimitError" in error_type:
        return RateLimitError(
            error_message,
            provider=provider_name,
            status_code=status_code or 429,
            request_id=request_id,
            retry_after=retry_after,
            cause=error,
        )

    if "BadRequestError" in error_type:
        return InvalidRequestError(
            error_message,
            provider=provider_name,
            status_code=status_code or 400,
            request_id=request_id,
            cause=error,
        )

    if "NotFoundError" in error_type:
        return ModelNotFoundError(
            error_message,
            provider=provider_name,
            status_code=status_code or 404,
            request_id=request_id,
            cause=error,
        )

    if "APIStatusError" in error_type and status_code:
        return map_status_code_to_error(
            status_code, error_message, provider_name, error
        )

    # HTTP 状态码映射(从错误消息中提取)
    if "401" in error_message:
        return AuthenticationError(
            error_message,
            provider=provider_name,
            status_code=401,
            cause=error,
        )

    if "403" in error_message:
        return AuthenticationError(
            f"Access forbidden: {error_message}",
            provider=provider_name,
            status_code=403,
            cause=error,
        )

    if "429" in error_message:
        return RateLimitError(
            error_message,
            provider=provider_name,
            status_code=429,
            cause=error,
        )

    if "400" in error_message:
        return InvalidRequestError(
            error_message,
            provider=provider_name,
            status_code=400,
            cause=error,
        )

    if "404" in error_message:
        return ModelNotFoundError(
            error_message,
            provider=provider_name,
            status_code=404,
            cause=error,
        )

    if "500" in error_message or "Internal" in error_type:
        return InternalServerError(
            error_message,
            provider=provider_name,
            status_code=500,
            cause=error,
        )

    if "503" in error_message or "Service Unavailable" in error_message:
        return ProviderNotAvailableError(
            error_message,
            provider=provider_name,
            status_code=503,
            cause=error,
        )

    if "502" in error_message or "Bad Gateway" in error_message:
        return ProviderNotAvailableError(
            error_message,
            provider=provider_name,
            status_code=502,
            cause=error,
        )

    if "ContentFilter" in error_message:
        return InvalidRequestError(
            error_message,
            provider=provider_name,
            cause=error,
        )

    if "Timeout" in error_type or "timeout" in error_message.lower():
        return TimeoutError(
            error_message,
            provider=provider_name,
            cause=error,
        )

    # 默认返回服务器错误
    return InternalServerError(
        f"Unexpected error: {error_type}: {error_message}",
        provider=provider_name,
        status_code=status_code,
        request_id=request_id,
        cause=error,
    ).with_context(original_type=error_type)


def message_to_openai_format(message: Message) -> dict[str, Any]:
    """将统一消息格式转换为 OpenAI API 格式。

    Args:
        message: 统一消息对象。

    Returns:
        dict[str, Any]: OpenAI API 格式的消息字典。
    """
    result: dict[str, Any] = {"role": message.role.value}

    # 处理内容
    if message.content is not None:
        if isinstance(message.content, str):
            result["content"] = message.content
        else:
            # 多模态内容
            parts: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextContent):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContent):
                    if part.image.startswith("http"):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": part.image},
                            }
                        )
                    else:
                        # base64
                        mime = part.mime_type or "image/png"
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{part.image}"
                                },
                            }
                        )
            result["content"] = parts

    # 处理 name
    if message.name:
        result["name"] = message.name

    # 处理工具调用
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": (
                        tc.arguments
                        if isinstance(tc.arguments, str)
                        else __import__("json").dumps(tc.arguments)
                    ),
                },
            }
            for tc in message.tool_calls
        ]

    # 处理 tool 角色的消息
    if message.role == Role.TOOL and message.tool_call_id:
        result["tool_call_id"] = message.tool_call_id

    return result


def openai_tool_to_dict(tool: dict[str, Any]) -> dict[str, Any]:
    """将 OpenAI 工具调用格式转换为统一格式。

    Args:
        tool: OpenAI 格式的工具调用字典。

    Returns:
        dict[str, Any]: 统一格式的工具调用字典。
    """
    import json

    arguments = tool["function"]["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    return {
        "id": tool["id"],
        "name": tool["function"]["name"],
        "arguments": arguments,
    }


def handle_anthropic_error(error: Exception, provider_name: str) -> AIError:
    """处理 Anthropic 格式的错误。

    Args:
        error: 原始异常对象。
        provider_name: 供应商标识。

    Returns:
        具体的错误类型实例。
    """
    error_type: str = type(error).__name__
    error_message: str = str(error)
    info = extract_error_info(error)
    status_code = info["status_code"]
    request_id = info["request_id"]

    # Anthropic SDK 特有异常类型
    if "AuthenticationError" in error_type or "401" in error_message:
        return AuthenticationError(
            error_message,
            provider=provider_name,
            status_code=401,
            cause=error,
        )
    if "RateLimitError" in error_type or "429" in error_message:
        return RateLimitError(
            error_message,
            provider=provider_name,
            status_code=429,
            cause=error,
        )
    if "BadRequestError" in error_type or "400" in error_message:
        return InvalidRequestError(
            error_message,
            provider=provider_name,
            status_code=400,
            cause=error,
        )
    if "NotFoundError" in error_type or "404" in error_message:
        return ModelNotFoundError(
            error_message,
            provider=provider_name,
            status_code=404,
            cause=error,
        )
    if "APIStatusError" in error_type and status_code:
        return map_status_code_to_error(status_code, error_message, provider_name, error)

    # Anthropic 特有错误模式
    if "overloaded" in error_message.lower():
        return ProviderNotAvailableError(
            error_message,
            provider=provider_name,
            cause=error,
        )
    if "ContentFilter" in error_message or "content_filter" in error_message:
        return ContentFilterError(
            error_message,
            provider=provider_name,
            cause=error,
        )
    if "credit" in error_message.lower() or "quota" in error_message.lower():
        return QuotaExceededError(
            error_message,
            provider=provider_name,
            cause=error,
        )

    # 通用错误处理
    if "403" in error_message:
        return AuthenticationError(
            f"Access forbidden: {error_message}",
            provider=provider_name,
            status_code=403,
            cause=error,
        )
    if "500" in error_message:
        return InternalServerError(
            error_message,
            provider=provider_name,
            status_code=500,
            cause=error,
        )
    if "503" in error_message or "Service Unavailable" in error_message:
        return ProviderNotAvailableError(
            error_message,
            provider=provider_name,
            status_code=503,
            cause=error,
        )
    if "502" in error_message:
        return ProviderNotAvailableError(
            error_message,
            provider=provider_name,
            status_code=502,
            cause=error,
        )
    if "Timeout" in error_type or "timeout" in error_message.lower():
        return TimeoutError(
            error_message,
            provider=provider_name,
            cause=error,
        )
    if "Connection" in error_type or "connect" in error_message.lower():
        return ConnectionError(
            error_message,
            provider=provider_name,
            cause=error,
        )

    return InternalServerError(
        f"Unexpected error: {error_type}: {error_message}",
        provider=provider_name,
        status_code=status_code,
        request_id=request_id,
        cause=error,
    ).with_context(original_type=error_type)


def handle_google_error(error: Exception, provider_name: str) -> AIError:
    """处理 Google (Gemini) 格式的错误。

    Args:
        error: 原始异常对象。
        provider_name: 供应商标识。

    Returns:
        具体的错误类型实例。
    """
    error_type: str = type(error).__name__
    error_message: str = str(error)
    error_message_lower = error_message.lower()

    # Google SDK 特有异常类型
    if "InvalidAPIKey" in error_type or (
        "invalid" in error_message_lower and "api" in error_message_lower
    ):
        return AuthenticationError(error_message, provider=provider_name)

    if "ResourceExhausted" in error_type or "resourceexhausted" in error_message_lower.replace(" ", "").replace("-", ""):
        if "quota" in error_message_lower:
            return QuotaExceededError(error_message, provider=provider_name)
        return RateLimitError(error_message, provider=provider_name)

    if "InvalidArgument" in error_type or "400" in error_message or "invalidargument" in error_message_lower.replace(" ", ""):
        return InvalidRequestError(error_message, provider=provider_name)

    if "NotFound" in error_type or "404" in error_message or "notfound" in error_message_lower.replace(" ", ""):
        return ModelNotFoundError(error_message, provider=provider_name)

    if "PermissionDenied" in error_type or "403" in error_message or "permissiondenied" in error_message_lower.replace(" ", ""):
        return AuthenticationError(
            f"Access forbidden: {error_message}",
            provider=provider_name,
        )

    if "Unavailable" in error_type or "503" in error_message or "unavailable" in error_message_lower.replace(" ", ""):
        return ProviderNotAvailableError(
            error_message,
            provider=provider_name,
            status_code=503,
            cause=error,
        )

    if "Internal" in error_type or "500" in error_message or "internalerror" in error_message_lower.replace(" ", ""):
        return InternalServerError(
            error_message,
            provider=provider_name,
            status_code=500,
            cause=error,
        )

    if "DeadlineExceeded" in error_type or "timeout" in error_message_lower:
        return TimeoutError(
            error_message,
            provider=provider_name,
            cause=error,
        )

    # Google 特有错误模式
    if (
        "Safety" in error_type
        or "Blocked" in error_message
        or "safety" in error_message_lower
    ):
        return ContentFilterError(
            error_message,
            provider=provider_name,
            cause=error,
        )

    if "quota" in error_message_lower or "exhausted" in error_message_lower:
        return QuotaExceededError(
            error_message,
            provider=provider_name,
            cause=error,
        )

    if "429" in error_message:
        return RateLimitError(error_message, provider=provider_name)

    return InternalServerError(
        f"Unexpected error: {error_type}: {error_message}",
        provider=provider_name,
        cause=error,
    ).with_context(original_type=error_type)
