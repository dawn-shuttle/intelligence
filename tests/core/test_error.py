"""测试 core/error.py - 错误类型定义。"""

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import (
    AIError,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ContentFilterError,
    ErrorCode,
    ErrorDetail,
    InternalServerError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderNotAvailableError,
    QuotaExceededError,
    RateLimitError,
    ResponseParseError,
    TimeoutError,
)


class TestErrorCode:
    """测试 ErrorCode 枚举。"""

    def test_error_codes_exist(self) -> None:
        """测试错误代码存在。"""
        assert ErrorCode.AUTH_INVALID_KEY.value == "AUTH_INVALID_KEY"
        assert ErrorCode.RATE_LIMIT.value == "RATE_LIMIT"
        assert ErrorCode.MODEL_NOT_FOUND.value == "MODEL_NOT_FOUND"

    def test_error_code_is_string(self) -> None:
        """测试错误代码是字符串。"""
        assert isinstance(ErrorCode.AUTH_INVALID_KEY.value, str)


class TestErrorDetail:
    """测试 ErrorDetail。"""

    def test_error_detail_basic(self) -> None:
        """测试基本错误详情。"""
        detail = ErrorDetail(field="api_key", reason="Invalid format")
        assert detail.field == "api_key"
        assert detail.reason == "Invalid format"
        assert detail.value is None
        assert detail.suggestion is None

    def test_error_detail_full(self) -> None:
        """测试完整错误详情。"""
        detail = ErrorDetail(
            field="temperature",
            value=3.0,
            reason="Must be between 0 and 2",
            suggestion="Use a value between 0 and 2",
        )
        assert detail.field == "temperature"
        assert detail.value == 3.0
        assert detail.reason == "Must be between 0 and 2"
        assert detail.suggestion == "Use a value between 0 and 2"


class TestAIError:
    """测试 AIError。"""

    def test_basic_error(self) -> None:
        """测试基本错误。"""
        error = AIError("Something went wrong")
        assert "[INTERNAL_ERROR]" in str(error)
        assert "Something went wrong" in str(error)

    def test_error_with_provider(self) -> None:
        """测试带 provider 的错误。"""
        error = AIError("Error message", provider="openai")
        assert "provider=openai" in str(error)

    def test_error_with_model(self) -> None:
        """测试带 model 的错误。"""
        error = AIError("Error", provider="openai", model="gpt-4")
        assert "model=gpt-4" in str(error)

    def test_error_with_request_id(self) -> None:
        """测试带 request_id 的错误。"""
        error = AIError("Error", request_id="req-123")
        assert "request_id: req-123" in str(error)

    def test_error_with_status_code(self) -> None:
        """测试带 status_code 的错误。"""
        error = AIError("Error", status_code=500)
        assert error.status_code == 500

    def test_error_with_cause(self) -> None:
        """测试带原始异常的错误。"""
        original = ValueError("Original error")
        error = AIError("Wrapped", cause=original)
        assert error.cause is original

    def test_error_context(self) -> None:
        """测试上下文信息。"""
        error = AIError("Error", extra_info="some info")
        assert error.context == {"extra_info": "some info"}

    def test_repr(self) -> None:
        """测试 repr 输出。"""
        error = AIError("Test error", provider="openai", status_code=500)
        repr_str = repr(error)
        assert "AIError" in repr_str
        assert "openai" in repr_str
        assert "500" in repr_str

    def test_format_basic(self) -> None:
        """测试基本格式化。"""
        error = AIError("Test error", provider="openai")
        formatted = error.format()
        assert "[INTERNAL_ERROR]" in formatted
        assert "openai" in formatted

    def test_format_with_guide(self) -> None:
        """测试带指南的格式化。"""
        error = AuthenticationError("Invalid key", provider="openai")
        formatted = error.format(include_guide=True)
        assert "建议:" in formatted
        assert "API Key" in formatted

    def test_format_with_details(self) -> None:
        """测试带详情的格式化。"""
        error = RateLimitError(
            "Too many requests",
            provider="openai",
            retry_after=60,
        )
        formatted = error.format(include_details=True)
        assert "重试等待: 60 秒" in formatted

    def test_to_dict(self) -> None:
        """测试转换为字典。"""
        error = AIError(
            "Test error",
            code=ErrorCode.INVALID_REQUEST,
            provider="openai",
            model="gpt-4",
            request_id="req-123",
            status_code=400,
        )
        d = error.to_dict()
        assert d["error"]["code"] == "INVALID_REQUEST"
        assert d["error"]["type"] == "AIError"
        assert d["error"]["message"] == "Test error"
        assert d["error"]["provider"] == "openai"
        assert d["error"]["model"] == "gpt-4"
        assert d["error"]["request_id"] == "req-123"
        assert d["error"]["status_code"] == 400

    def test_to_dict_with_details(self) -> None:
        """测试带详情的字典转换。"""
        error = AIError(
            "Invalid request",
            details=[
                ErrorDetail(field="temperature", reason="Out of range"),
            ],
        )
        d = error.to_dict()
        assert "details" in d["error"]
        assert len(d["error"]["details"]) == 1

    def test_to_dict_with_raw_response(self) -> None:
        """测试带原始响应的字典转换。"""
        error = AIError("Error", raw_response={"error": "something"})
        d = error.to_dict()
        assert d["error"]["raw_response"] == {"error": "something"}

    def test_with_detail(self) -> None:
        """测试 with_detail 方法。"""
        error = AIError("Error").with_detail(
            field="api_key",
            reason="Invalid format",
        )
        assert len(error.details) == 1
        assert error.details[0].field == "api_key"

    def test_with_context(self) -> None:
        """测试 with_context 方法。"""
        error = AIError("Error").with_context(key="value")
        assert error.context == {"key": "value"}

    def test_chaining(self) -> None:
        """测试链式调用。"""
        error = (
            AIError("Error")
            .with_detail(field="field1", reason="reason1")
            .with_detail(field="field2", reason="reason2")
            .with_context(extra="info")
        )
        assert len(error.details) == 2
        assert error.context == {"extra": "info"}


class TestAuthenticationError:
    """测试 AuthenticationError。"""

    def test_auth_error(self) -> None:
        """测试认证错误。"""
        error = AuthenticationError("Invalid API key", provider="openai")
        assert isinstance(error, AIError)
        assert error.code == ErrorCode.AUTH_INVALID_KEY

    def test_auth_error_default_message(self) -> None:
        """测试默认消息。"""
        error = AuthenticationError()
        assert error.message == "Authentication failed"

    def test_auth_error_user_guide(self) -> None:
        """测试用户指南。"""
        error = AuthenticationError()
        formatted = error.format(include_guide=True)
        assert "API Key" in formatted


class TestRateLimitError:
    """测试 RateLimitError。"""

    def test_rate_limit_default_message(self) -> None:
        """测试默认消息。"""
        error = RateLimitError()
        assert error.code == ErrorCode.RATE_LIMIT
        assert "Rate limit exceeded" in error.message

    def test_rate_limit_with_retry_after(self) -> None:
        """测试带 retry_after。"""
        error = RateLimitError(
            message="Too many requests",
            provider="openai",
            retry_after=60,
        )
        assert error.retry_after == 60
        assert error.provider == "openai"

    def test_rate_limit_format_with_details(self) -> None:
        """测试带详情的格式化。"""
        error = RateLimitError(retry_after=30)
        formatted = error.format(include_details=True)
        assert "30 秒" in formatted


class TestModelNotFoundError:
    """测试 ModelNotFoundError。"""

    def test_model_not_found(self) -> None:
        """测试模型不存在错误。"""
        error = ModelNotFoundError("Model gpt-5 not found", provider="openai")
        assert isinstance(error, AIError)
        assert error.code == ErrorCode.MODEL_NOT_FOUND


class TestInvalidRequestError:
    """测试 InvalidRequestError。"""

    def test_invalid_request(self) -> None:
        """测试无效请求错误。"""
        error = InvalidRequestError("Missing parameter", provider="openai")
        assert isinstance(error, AIError)
        assert error.code == ErrorCode.INVALID_REQUEST

    def test_invalid_request_with_details(self) -> None:
        """测试带详情的无效请求。"""
        error = (
            InvalidRequestError("Invalid parameters")
            .with_detail(field="temperature", value=3.0, reason="Must be <= 2")
        )
        assert len(error.details) == 1


class TestContentFilterError:
    """测试 ContentFilterError。"""

    def test_content_filter(self) -> None:
        """测试内容过滤错误。"""
        error = ContentFilterError("Content blocked", provider="openai")
        assert isinstance(error, AIError)
        assert error.code == ErrorCode.CONTENT_FILTER


class TestQuotaExceededError:
    """测试 QuotaExceededError。"""

    def test_quota_exceeded(self) -> None:
        """测试配额错误。"""
        error = QuotaExceededError("Quota exceeded", provider="openai")
        assert isinstance(error, AIError)
        assert error.code == ErrorCode.QUOTA_EXCEEDED


class TestTimeoutError:
    """测试 TimeoutError。"""

    def test_timeout(self) -> None:
        """测试超时错误。"""
        error = TimeoutError("Request timed out", provider="openai")
        assert isinstance(error, AIError)
        assert error.code == ErrorCode.TIMEOUT


class TestConnectionError:
    """测试 ConnectionError。"""

    def test_connection_error(self) -> None:
        """测试连接错误。"""
        error = ConnectionError("Failed to connect", provider="openai")
        assert isinstance(error, AIError)
        assert error.code == ErrorCode.CONNECTION


class TestProviderNotAvailableError:
    """测试 ProviderNotAvailableError。"""

    def test_provider_not_available(self) -> None:
        """测试提供商不可用错误。"""
        error = ProviderNotAvailableError("Service unavailable", provider="openai")
        assert isinstance(error, AIError)
        assert error.code == ErrorCode.SERVICE_UNAVAILABLE


class TestInternalServerError:
    """测试 InternalServerError。"""

    def test_internal_server_error(self) -> None:
        """测试服务器内部错误。"""
        error = InternalServerError("500 Internal Error", provider="openai")
        assert isinstance(error, AIError)
        assert error.code == ErrorCode.INTERNAL_ERROR


class TestConfigurationError:
    """测试 ConfigurationError。"""

    def test_configuration_error(self) -> None:
        """测试配置错误。"""
        error = ConfigurationError("Missing model name", provider="openai")
        assert isinstance(error, AIError)
        assert error.code == ErrorCode.CONFIG_MISSING


class TestResponseParseError:
    """测试 ResponseParseError。"""

    def test_response_parse_error(self) -> None:
        """测试响应解析错误。"""
        error = ResponseParseError("Invalid response format", provider="openai")
        assert isinstance(error, AIError)
        assert error.code == ErrorCode.PARSE_ERROR