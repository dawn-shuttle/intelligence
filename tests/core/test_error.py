"""测试 core/error.py - 错误类型定义。"""

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import (
    AIError,
    AuthenticationError,
    ConnectionError,
    ContentFilterError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderNotAvailableError,
    QuotaExceededError,
    RateLimitError,
    TimeoutError,
)


class TestAIError:
    """测试 AIError。"""

    def test_basic_error(self) -> None:
        """测试基本错误。"""
        error = AIError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_error_with_provider(self) -> None:
        """测试带 provider 的错误。"""
        error = AIError("Error message", provider="openai")
        assert str(error) == "[openai] Error message"

    def test_error_with_context(self) -> None:
        """测试带上下文的错误。"""
        error = AIError("Error", provider="test", code=500, detail="detail")
        assert error.provider == "test"
        assert error.context == {"code": 500, "detail": "detail"}

    def test_error_without_provider(self) -> None:
        """测试不带 provider 的错误。"""
        error = AIError("Plain error")
        assert str(error) == "Plain error"
        assert error.provider is None


class TestAuthenticationError:
    """测试 AuthenticationError。"""

    def test_auth_error(self) -> None:
        """测试认证错误。"""
        error = AuthenticationError("Invalid API key", provider="openai")
        assert isinstance(error, AIError)
        assert "Invalid API key" in str(error)


class TestRateLimitError:
    """测试 RateLimitError。"""

    def test_rate_limit_default_message(self) -> None:
        """测试默认消息。"""
        error = RateLimitError()
        assert "Rate limit exceeded" in str(error)

    def test_rate_limit_with_retry_after(self) -> None:
        """测试带 retry_after。"""
        error = RateLimitError(
            message="Too many requests",
            provider="openai",
            retry_after=60,
        )
        assert error.retry_after == 60
        assert error.provider == "openai"


class TestModelNotFoundError:
    """测试 ModelNotFoundError。"""

    def test_model_not_found(self) -> None:
        """测试模型不存在错误。"""
        error = ModelNotFoundError("Model gpt-5 not found", provider="openai")
        assert isinstance(error, AIError)


class TestInvalidRequestError:
    """测试 InvalidRequestError。"""

    def test_invalid_request(self) -> None:
        """测试无效请求错误。"""
        error = InvalidRequestError("Missing parameter", provider="openai")
        assert isinstance(error, AIError)


class TestContentFilterError:
    """测试 ContentFilterError。"""

    def test_content_filter(self) -> None:
        """测试内容过滤错误。"""
        error = ContentFilterError("Content blocked", provider="openai")
        assert isinstance(error, AIError)


class TestQuotaExceededError:
    """测试 QuotaExceededError。"""

    def test_quota_exceeded(self) -> None:
        """测试配额错误。"""
        error = QuotaExceededError("Quota exceeded", provider="openai")
        assert isinstance(error, AIError)


class TestTimeoutError:
    """测试 TimeoutError。"""

    def test_timeout(self) -> None:
        """测试超时错误。"""
        error = TimeoutError("Request timed out", provider="openai")
        assert isinstance(error, AIError)


class TestConnectionError:
    """测试 ConnectionError。"""

    def test_connection_error(self) -> None:
        """测试连接错误。"""
        error = ConnectionError("Failed to connect", provider="openai")
        assert isinstance(error, AIError)


class TestProviderNotAvailableError:
    """测试 ProviderNotAvailableError。"""

    def test_provider_not_available(self) -> None:
        """测试提供商不可用错误。"""
        error = ProviderNotAvailableError("Service unavailable", provider="openai")
        assert isinstance(error, AIError)
