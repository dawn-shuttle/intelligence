"""错误类型定义 - 统一的异常体系。"""

from typing import Any


class AIError(Exception):
    """AI 调用基础异常。"""

    def __init__(self, message: str, provider: str | None = None, **context: Any):
        super().__init__(message)
        self.provider = provider
        self.context = context

    def __str__(self) -> str:
        result = super().__str__()
        if self.provider:
            result = f"[{self.provider}] {result}"
        return result


class AuthenticationError(AIError):
    """认证失败（API Key 无效或过期）。"""

    pass


class RateLimitError(AIError):
    """速率限制。"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        provider: str | None = None,
        retry_after: int | None = None,
        **context: Any,
    ):
        super().__init__(message, provider, **context)
        self.retry_after = retry_after


class ModelNotFoundError(AIError):
    """模型不存在。"""

    pass


class InvalidRequestError(AIError):
    """请求参数无效。"""

    pass


class ContentFilterError(AIError):
    """内容过滤触发。"""

    pass


class QuotaExceededError(AIError):
    """配额用尽。"""

    pass


class TimeoutError(AIError):
    """请求超时。"""

    pass


class ConnectionError(AIError):
    """连接失败。"""

    pass


class ProviderNotAvailableError(AIError):
    """提供商服务不可用。"""

    pass
