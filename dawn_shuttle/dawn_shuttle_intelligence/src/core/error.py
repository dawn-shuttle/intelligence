"""错误类型定义 - 统一的异常体系。

提供丰富的错误信息和用户友好的反馈。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """错误代码枚举。"""

    # 认证相关 (1xx)
    AUTH_INVALID_KEY = "AUTH_INVALID_KEY"
    AUTH_EXPIRED = "AUTH_EXPIRED"
    AUTH_MISSING = "AUTH_MISSING"

    # 速率限制 (2xx)
    RATE_LIMIT = "RATE_LIMIT"

    # 模型相关 (3xx)
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_OVERLOADED = "MODEL_OVERLOADED"

    # 请求相关 (4xx)
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_PARAM = "INVALID_PARAM"
    CONTEXT_TOO_LONG = "CONTEXT_TOO_LONG"

    # 内容相关 (5xx)
    CONTENT_FILTER = "CONTENT_FILTER"
    CONTENT_FLAGGED = "CONTENT_FLAGGED"

    # 配额相关 (6xx)
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    BILLING_REQUIRED = "BILLING_REQUIRED"

    # 网络/服务相关 (7xx)
    TIMEOUT = "TIMEOUT"
    CONNECTION = "CONNECTION"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    # 解析相关 (8xx)
    PARSE_ERROR = "PARSE_ERROR"
    STREAM_ERROR = "STREAM_ERROR"

    # 配置相关 (9xx)
    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_INVALID = "CONFIG_INVALID"
    PROVIDER_NOT_FOUND = "PROVIDER_NOT_FOUND"


@dataclass
class ErrorDetail:
    """错误详情。"""

    field: str | None = None
    """相关字段名。"""

    value: Any = None
    """相关值。"""

    reason: str | None = None
    """具体原因。"""

    suggestion: str | None = None
    """建议解决方案。"""


class AIError(Exception):
    """AI 调用基础异常。

    所有具体错误类型的基类，提供丰富的错误信息。

    Attributes:
        code: 错误代码。
        message: 错误消息。
        provider: 提供商名称。
        model: 模型名称。
        request_id: 请求 ID。
        status_code: HTTP 状态码。
        retry_after: 重试等待秒数。
        details: 错误详情列表。
        raw_response: 原始响应数据。
    """

    # 子类应重写这些属性
    default_code: ErrorCode = ErrorCode.INTERNAL_ERROR
    default_message: str = "An error occurred"
    user_guide: str = "Please check your configuration and try again."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: ErrorCode | None = None,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        status_code: int | None = None,
        retry_after: int | None = None,
        details: list[ErrorDetail] | None = None,
        raw_response: dict[str, Any] | None = None,
        cause: Exception | None = None,
        **context: Any,
    ) -> None:
        self.code = code or self.default_code
        self.message = message or self.default_message
        self.provider = provider
        self.model = model
        self.request_id = request_id
        self.status_code = status_code
        self.retry_after = retry_after
        self.details = details or []
        self.raw_response = raw_response
        self.context = context
        self.cause = cause

        super().__init__(self.message)

    def __str__(self) -> str:
        """格式化错误信息。"""
        parts = [f"[{self.code.value}]"]

        if self.provider:
            parts.append(f"provider={self.provider}")

        if self.model:
            parts.append(f"model={self.model}")

        parts.append(self.message)

        if self.request_id:
            parts.append(f"(request_id: {self.request_id})")

        return " ".join(parts)

    def __repr__(self) -> str:
        """详细错误信息。"""
        return (
            f"{self.__class__.__name__}("
            f"code={self.code.value!r}, "
            f"message={self.message!r}, "
            f"provider={self.provider!r}, "
            f"status_code={self.status_code})"
        )

    def format(
        self, *, include_guide: bool = False, include_details: bool = False
    ) -> str:
        """格式化错误信息。

        Args:
            include_guide: 是否包含用户指南。
            include_details: 是否包含详细信息。

        Returns:
            格式化的错误信息。
        """
        lines = [str(self)]

        if include_details:
            if self.retry_after:
                lines.append(f"  重试等待: {self.retry_after} 秒")

            for detail in self.details:
                if detail.field:
                    line = f"  - 字段 '{detail.field}'"
                    if detail.reason:
                        line += f": {detail.reason}"
                    lines.append(line)

            if self.context:
                for key, value in self.context.items():
                    lines.append(f"  {key}: {value}")

        if include_guide:
            lines.append(f"  建议: {self.user_guide}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，便于序列化。

        Returns:
            错误信息字典。
        """
        result: dict[str, Any] = {
            "error": {
                "code": self.code.value,
                "type": self.__class__.__name__,
                "message": self.message,
            }
        }

        if self.provider:
            result["error"]["provider"] = self.provider

        if self.model:
            result["error"]["model"] = self.model

        if self.request_id:
            result["error"]["request_id"] = self.request_id

        if self.status_code:
            result["error"]["status_code"] = self.status_code

        if self.retry_after:
            result["error"]["retry_after"] = self.retry_after

        if self.details:
            result["error"]["details"] = [
                {
                    "field": d.field,
                    "value": d.value,
                    "reason": d.reason,
                    "suggestion": d.suggestion,
                }
                for d in self.details
            ]

        if self.raw_response:
            result["error"]["raw_response"] = self.raw_response

        return result

    def with_detail(
        self,
        field: str | None = None,
        value: Any = None,
        reason: str | None = None,
        suggestion: str | None = None,
    ) -> AIError:
        """添加错误详情并返回 self。

        Args:
            field: 相关字段名。
            value: 相关值。
            reason: 具体原因。
            suggestion: 建议解决方案。

        Returns:
            self，便于链式调用。
        """
        self.details.append(ErrorDetail(
            field=field,
            value=value,
            reason=reason,
            suggestion=suggestion,
        ))
        return self

    def with_context(self, **kwargs: Any) -> AIError:
        """添加上下文信息并返回 self。

        Returns:
            self，便于链式调用。
        """
        self.context.update(kwargs)
        return self


class AuthenticationError(AIError):
    """认证失败(API Key 无效或过期)。"""

    default_code = ErrorCode.AUTH_INVALID_KEY
    default_message = "Authentication failed"
    user_guide = "请检查 API Key 是否正确，或重新生成 API Key。"


class RateLimitError(AIError):
    """速率限制。"""

    default_code = ErrorCode.RATE_LIMIT
    default_message = "Rate limit exceeded"
    user_guide = "请等待后重试，或升级账户以获得更高的速率限制。"


class ModelNotFoundError(AIError):
    """模型不存在。"""

    default_code = ErrorCode.MODEL_NOT_FOUND
    default_message = "Model not found"
    user_guide = "请检查模型名称是否正确，或确认该模型是否可用。"


class InvalidRequestError(AIError):
    """请求参数无效。"""

    default_code = ErrorCode.INVALID_REQUEST
    default_message = "Invalid request"
    user_guide = "请检查请求参数是否符合 API 要求。"


class ContentFilterError(AIError):
    """内容过滤触发。"""

    default_code = ErrorCode.CONTENT_FILTER
    default_message = "Content filtered"
    user_guide = "您的内容触发了安全过滤，请修改内容后重试。"


class QuotaExceededError(AIError):
    """配额用尽。"""

    default_code = ErrorCode.QUOTA_EXCEEDED
    default_message = "Quota exceeded"
    user_guide = "您的 API 配额已用尽，请充值或升级账户。"


class TimeoutError(AIError):
    """请求超时。"""

    default_code = ErrorCode.TIMEOUT
    default_message = "Request timed out"
    user_guide = "请求超时，请检查网络连接或稍后重试。"


class ConnectionError(AIError):
    """连接失败。"""

    default_code = ErrorCode.CONNECTION
    default_message = "Connection failed"
    user_guide = "无法连接到服务器，请检查网络连接。"


class ProviderNotAvailableError(AIError):
    """提供商服务不可用(如 503 错误)。"""

    default_code = ErrorCode.SERVICE_UNAVAILABLE
    default_message = "Provider service unavailable"
    user_guide = "服务暂时不可用，请稍后重试。"


class InternalServerError(AIError):
    """服务器内部错误(如 500 错误)。"""

    default_code = ErrorCode.INTERNAL_ERROR
    default_message = "Internal server error"
    user_guide = "服务器内部错误，请稍后重试或联系支持。"


class ResponseParseError(AIError):
    """响应解析失败。"""

    default_code = ErrorCode.PARSE_ERROR
    default_message = "Failed to parse response"
    user_guide = "响应格式异常，请稍后重试。"


class ConfigurationError(AIError):
    """配置错误(如缺少必要参数)。"""

    default_code = ErrorCode.CONFIG_MISSING
    default_message = "Configuration error"
    user_guide = "请检查配置是否完整，确保所有必要参数已设置。"


__all__ = [
    "AIError",
    "AuthenticationError",
    "ConfigurationError",
    "ConnectionError",
    "ContentFilterError",
    "ErrorCode",
    "ErrorDetail",
    "InternalServerError",
    "InvalidRequestError",
    "ModelNotFoundError",
    "ProviderNotAvailableError",
    "QuotaExceededError",
    "RateLimitError",
    "ResponseParseError",
    "TimeoutError",
]
