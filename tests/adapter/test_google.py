"""测试 adapter/google.py - Google (Gemini) 适配器。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.google import GoogleProvider
from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.base import (
    handle_google_error,
    validate_config,
    validate_messages,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.core.config import GenerateConfig
from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import (
    ConfigurationError,
    ResponseParseError,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import (
    ImageContent,
    Message,
    Role,
    TextContent,
    ToolCall,
)


class TestGoogleProvider:
    """测试 GoogleProvider。"""

    def test_provider_name(self) -> None:
        """测试提供商标识。"""
        provider = GoogleProvider(api_key="test-key")
        assert provider.name == "google"

    def test_supports_model_gemini_20_flash(self) -> None:
        """测试支持 Gemini 2.0 Flash 模型。"""
        provider = GoogleProvider()
        assert provider.supports_model("gemini-2.0-flash") is True

    def test_supports_model_gemini_15_pro(self) -> None:
        """测试支持 Gemini 1.5 Pro 模型。"""
        provider = GoogleProvider()
        assert provider.supports_model("gemini-1.5-pro") is True

    def test_supports_model_gemini_15_flash(self) -> None:
        """测试支持 Gemini 1.5 Flash 模型。"""
        provider = GoogleProvider()
        assert provider.supports_model("gemini-1.5-flash") is True

    def test_supports_model_invalid(self) -> None:
        """测试不支持无效模型。"""
        provider = GoogleProvider()
        assert provider.supports_model("gpt-4") is False

    def test_get_model_list(self) -> None:
        """测试获取模型列表。"""
        provider = GoogleProvider()
        models = provider.get_model_list()
        assert "gemini-2.0-flash" in models
        assert "gemini-1.5-pro" in models

    def test_validate_config_empty_model(self) -> None:
        """测试空模型名称验证。"""
        provider = GoogleProvider()
        config = GenerateConfig()

        with pytest.raises(ConfigurationError, match="Model name is required"):
            validate_config(config, provider.name)

    def test_validate_config_invalid_temperature(self) -> None:
        """测试无效温度验证。"""
        provider = GoogleProvider()
        config = GenerateConfig(model="gemini-2.0-flash", temperature=3.0)

        with pytest.raises(ConfigurationError, match="Temperature must be between"):
            validate_config(config, provider.name)

    def test_validate_messages_empty(self) -> None:
        """测试空消息列表验证。"""
        provider = GoogleProvider()

        with pytest.raises(ConfigurationError, match="Messages list cannot be empty"):
            validate_messages([], provider.name)

    def test_build_params_basic(self) -> None:
        """测试构建基本参数。"""
        provider = GoogleProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="gemini-2.0-flash")

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert system_instruction is None

    def test_build_params_with_system(self) -> None:
        """测试构建带系统消息的参数。"""
        provider = GoogleProvider()
        messages = [
            Message.system("你是一个助手"),
            Message.user("你好"),
        ]
        config = GenerateConfig(model="gemini-2.0-flash")

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert system_instruction == "你是一个助手"
        assert len(contents) == 1

    def test_build_params_with_temperature(self) -> None:
        """测试构建带温度参数。"""
        provider = GoogleProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="gemini-2.0-flash", temperature=0.7)

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert generation_config["temperature"] == 0.7

    def test_build_params_with_max_tokens(self) -> None:
        """测试构建带 max_tokens 参数。"""
        provider = GoogleProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="gemini-2.0-flash", max_tokens=1000)

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert generation_config["max_output_tokens"] == 1000

    def test_convert_tools(self) -> None:
        """测试转换工具定义。"""
        provider = GoogleProvider()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取天气",
                    "parameters": {"type": "object"},
                },
            }
        ]

        result = provider._convert_tools(tools)

        assert len(result) == 1
        assert "function_declarations" in result[0]

    def test_handle_error_invalid_api_key(self) -> None:
        """测试处理无效 API Key 错误。"""
        provider = GoogleProvider()

        error = Exception("InvalidAPIKey: Invalid API key")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import AuthenticationError
        assert isinstance(result, AuthenticationError)

    def test_handle_error_resource_exhausted(self) -> None:
        """测试处理资源耗尽错误。"""
        provider = GoogleProvider()

        error = Exception("ResourceExhausted: Rate limit exceeded")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import RateLimitError
        assert isinstance(result, RateLimitError)

    def test_handle_error_safety(self) -> None:
        """测试处理安全过滤错误。"""
        provider = GoogleProvider()

        error = Exception("Safety: Content blocked")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ContentFilterError
        assert isinstance(result, ContentFilterError)

    def test_handle_error_deadline_exceeded(self) -> None:
        """测试处理超时错误。"""
        provider = GoogleProvider()

        error = Exception("DeadlineExceeded: Timeout")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import TimeoutError
        assert isinstance(result, TimeoutError)

    def test_handle_error_not_found(self) -> None:
        """测试处理资源未找到错误。"""
        provider = GoogleProvider()

        error = Exception("NotFound: Resource not found")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ModelNotFoundError
        assert isinstance(result, ModelNotFoundError)

    def test_handle_error_permission_denied(self) -> None:
        """测试处理权限拒绝错误。"""
        provider = GoogleProvider()

        error = Exception("PermissionDenied: Access denied")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import AuthenticationError
        assert isinstance(result, AuthenticationError)

    def test_handle_error_unavailable(self) -> None:
        """测试处理服务不可用错误。"""
        provider = GoogleProvider()

        error = Exception("Unavailable: Service unavailable")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import ProviderNotAvailableError
        assert isinstance(result, ProviderNotAvailableError)

    def test_handle_error_internal(self) -> None:
        """测试处理内部错误。"""
        provider = GoogleProvider()

        error = Exception("Internal: Internal error")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import InternalServerError
        assert isinstance(result, InternalServerError)

    def test_handle_error_quota_exhausted(self) -> None:
        """测试处理配额耗尽错误。"""
        provider = GoogleProvider()

        error = Exception("quota exhausted")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import QuotaExceededError
        assert isinstance(result, QuotaExceededError)

    def test_handle_error_rate_limit_429(self) -> None:
        """测试处理 429 限流错误。"""
        provider = GoogleProvider()

        error = Exception("429 Too Many Requests")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import RateLimitError
        assert isinstance(result, RateLimitError)

    def test_handle_error_invalid_argument(self) -> None:
        """测试处理无效参数错误。"""
        provider = GoogleProvider()

        error = Exception("InvalidArgument: Bad argument")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import InvalidRequestError
        assert isinstance(result, InvalidRequestError)

    def test_handle_error_unknown(self) -> None:
        """测试处理未知错误。"""
        provider = GoogleProvider()

        error = Exception("Some unknown error")
        result = handle_google_error(error, provider.name)

        from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import InternalServerError
        assert isinstance(result, InternalServerError)

    def test_build_params_with_stop_sequences(self) -> None:
        """测试构建带停止序列的参数。"""
        provider = GoogleProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="gemini-2.0-flash", stop=["END", "STOP"])

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert generation_config["stop_sequences"] == ["END", "STOP"]

    def test_build_params_with_top_k(self) -> None:
        """测试构建带 top_k 参数。"""
        provider = GoogleProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(model="gemini-2.0-flash", top_k=40)

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert generation_config["top_k"] == 40

    def test_build_params_with_tools(self) -> None:
        """测试构建带工具的参数。"""
        provider = GoogleProvider()
        messages = [Message.user("hello")]
        config = GenerateConfig(
            model="gemini-2.0-flash",
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取天气",
                    "parameters": {"type": "object"},
                },
            }],
        )

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert "tools" in generation_config

    def test_build_params_with_multimodal_content(self) -> None:
        """测试构建带多模态内容的参数。"""
        provider = GoogleProvider()
        messages = [Message(
            role=Role.USER,
            content=[
                TextContent(text="描述这张图片"),
                ImageContent(image="base64imagedata", mime_type="image/png"),
            ],
        )]
        config = GenerateConfig(model="gemini-2.0-flash")

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert len(contents[0]["parts"]) == 2

    def test_build_params_with_image_url(self) -> None:
        """测试构建带图片 URL 的参数。"""
        provider = GoogleProvider()
        messages = [Message(
            role=Role.USER,
            content=[ImageContent(image="https://example.com/image.png")],
        )]
        config = GenerateConfig(model="gemini-2.0-flash")

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert "file_data" in contents[0]["parts"][0]

    def test_build_params_with_tool_result(self) -> None:
        """测试构建带工具结果消息的参数。"""
        provider = GoogleProvider()
        messages = [Message(
            role=Role.TOOL,
            content="result",
            tool_call_id="call_123",
            name="get_weather",
        )]
        config = GenerateConfig(model="gemini-2.0-flash")

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert "function_response" in contents[0]["parts"][0]

    def test_build_params_with_tool_calls(self) -> None:
        """测试构建带工具调用消息的参数。"""
        provider = GoogleProvider()
        tool_call = ToolCall(id="call_123", name="get_weather", arguments={"city": "Beijing"})
        messages = [Message.assistant(tool_calls=[tool_call])]
        config = GenerateConfig(model="gemini-2.0-flash")

        contents, system_instruction, generation_config = provider._build_params(
            messages, config
        )

        assert "function_call" in contents[0]["parts"][0]

    def test_parse_response_basic(self) -> None:
        """测试解析基本响应。"""
        provider = GoogleProvider()

        # Mock response
        response = MagicMock()
        response.candidates = [MagicMock()]
        response.candidates[0].content.parts = [MagicMock(text="Hello, world!")]
        response.candidates[0].finish_reason = "STOP"
        response.usage_metadata = MagicMock(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        )

        result = provider._parse_response(response, "gemini-2.0-flash")

        assert result.text == "Hello, world!"
        assert result.finish_reason == "stop"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10

    def test_parse_response_with_tool_call(self) -> None:
        """测试解析带工具调用的响应。"""
        provider = GoogleProvider()

        # Mock response with tool call
        response = MagicMock()
        response.candidates = [MagicMock()]
        part = MagicMock()
        part.text = None
        part.function_call = MagicMock()
        part.function_call.name = "get_weather"
        part.function_call.args = {"city": "Beijing"}
        response.candidates[0].content.parts = [part]
        response.candidates[0].finish_reason = "STOP"
        response.usage_metadata = None

        result = provider._parse_response(response, "gemini-2.0-flash")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_parse_response_max_tokens(self) -> None:
        """测试解析 max_tokens 结束的响应。"""
        provider = GoogleProvider()

        response = MagicMock()
        response.candidates = [MagicMock()]
        response.candidates[0].content.parts = [MagicMock(text="Hello")]
        response.candidates[0].finish_reason = "MAX_TOKENS"
        response.usage_metadata = None

        result = provider._parse_response(response, "gemini-2.0-flash")

        assert result.finish_reason == "length"

    def test_parse_response_safety(self) -> None:
        """测试解析安全过滤的响应。"""
        provider = GoogleProvider()

        response = MagicMock()
        response.candidates = [MagicMock()]
        response.candidates[0].content.parts = [MagicMock(text="")]
        response.candidates[0].finish_reason = "SAFETY"
        response.usage_metadata = None

        result = provider._parse_response(response, "gemini-2.0-flash")

        assert result.finish_reason == "content_filter"

    def test_parse_stream_chunk_basic(self) -> None:
        """测试解析基本流式块。"""
        provider = GoogleProvider()

        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content.parts = [MagicMock(text="Hello")]
        chunk.candidates[0].finish_reason = None

        result = provider._parse_stream_chunk(chunk)

        assert result is not None
        assert result.delta == "Hello"
        assert not result.is_finished

    def test_parse_stream_chunk_finished(self) -> None:
        """测试解析结束的流式块。"""
        provider = GoogleProvider()

        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content.parts = [MagicMock(text="")]
        chunk.candidates[0].finish_reason = "STOP"

        result = provider._parse_stream_chunk(chunk)

        assert result is not None
        assert result.is_finished
        assert result.finish_reason == "stop"

    def test_parse_stream_chunk_no_candidates(self) -> None:
        """测试解析无候选的流式块。"""
        provider = GoogleProvider()

        chunk = MagicMock()
        chunk.candidates = []

        result = provider._parse_stream_chunk(chunk)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """测试成功生成响应。"""
        provider = GoogleProvider(api_key="test-key")

        # Mock response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Hello!")]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.usage_metadata = None

        with patch.object(provider, "_get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_get_model.return_value = mock_model

            messages = [Message.user("Hi")]
            config = GenerateConfig(model="gemini-2.0-flash")

            result = await provider.generate(messages, config)

            assert result.text == "Hello!"

    @pytest.mark.asyncio
    async def test_generate_with_system_instruction(self) -> None:
        """测试带系统指令的生成。"""
        provider = GoogleProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Response")]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.usage_metadata = None

        mock_model = AsyncMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        # Mock genai.GenerativeModel
        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            messages = [
                Message.system("Be helpful"),
                Message.user("Hi"),
            ]
            config = GenerateConfig(model="gemini-2.0-flash")

            result = await provider.generate(messages, config)

            assert result.text == "Response"

    @pytest.mark.asyncio
    async def test_generate_error(self) -> None:
        """测试生成时的错误处理。"""
        provider = GoogleProvider(api_key="test-key")

        with patch.object(provider, "_get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate_content_async = AsyncMock(
                side_effect=Exception("InvalidAPIKey: Invalid key")
            )
            mock_get_model.return_value = mock_model

            messages = [Message.user("Hi")]
            config = GenerateConfig(model="gemini-2.0-flash")

            from dawn_shuttle.dawn_shuttle_intelligence.src.core.error import AuthenticationError
            with pytest.raises(AuthenticationError):
                await provider.generate(messages, config)
