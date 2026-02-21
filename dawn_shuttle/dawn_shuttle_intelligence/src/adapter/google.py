"""Google (Gemini) 适配器 - 对接 Google Generative AI API。"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, ClassVar

from ..core.config import GenerateConfig
from ..core.error import (
    AIError,
    AuthenticationError,
    ConfigurationError,
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
from ..core.types import ImageContent, Message, Role, TextContent

if TYPE_CHECKING:
    from google.generativeai import GenerativeModel
    from google.generativeai.types import GenerateContentResponse


class GoogleProvider(BaseProvider):
    """Google (Gemini) API 适配器。

    支持的模型包括 Gemini 1.5, Gemini 2.0 系列。
    """

    name: str = "google"

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
        "gemini-pro",
        "gemini-pro-vision",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Google 适配器。

        Args:
            api_key: Google API 密钥(也可通过环境变量 GOOGLE_API_KEY 设置)。
            **kwargs: 其他参数。
        """
        super().__init__(api_key, **kwargs)
        self._configured: bool = False

    def _configure(self) -> None:
        """配置 Google SDK。"""
        if not self._configured:
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise ImportError(
                    "google-generativeai 包未安装, "
                    "请运行: pip install google-generativeai"
                ) from e

            genai.configure(api_key=self.api_key)
            self._configured = True

    def _get_model(self, model_name: str) -> GenerativeModel:
        """获取生成模型。

        Args:
            model_name: 模型名称。

        Returns:
            GenerativeModel: 生成模型实例。
        """
        self._configure()
        import google.generativeai as genai

        return genai.GenerativeModel(model_name)

    def supports_model(self, model: str) -> bool:
        """检查是否支持指定模型。"""
        return any(model.startswith(m) for m in self.SUPPORTED_MODELS)

    def get_model_list(self) -> list[str]:
        """获取支持的模型列表。"""
        return self.SUPPORTED_MODELS.copy()

    def _validate_config(self, config: GenerateConfig) -> None:
        """验证配置参数。"""
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
        """验证消息列表。"""
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
        """生成文本响应(非流式)。"""
        self._validate_config(config)
        self._validate_messages(messages)

        model = self._get_model(config.model)
        params = self._build_params(messages, config)
        contents, system_instruction, generation_config = params

        try:
            if system_instruction:
                model = self._get_model(config.model)
                import google.generativeai as genai
                model = genai.GenerativeModel(
                    config.model,
                    system_instruction=system_instruction,
                )

            response = await model.generate_content_async(
                contents,
                generation_config=generation_config,
            )
        except Exception as e:
            raise self._handle_error(e) from e

        try:
            return self._parse_response(response, config.model)
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
        """生成流式响应。"""
        self._validate_config(config)
        self._validate_messages(messages)

        params = self._build_params(messages, config)
        contents, system_instruction, generation_config = params

        if system_instruction:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                config.model,
                system_instruction=system_instruction,
            )
        else:
            model = self._get_model(config.model)

        try:
            response = await model.generate_content_async(
                contents,
                generation_config=generation_config,
                stream=True,
            )
        except Exception as e:
            raise self._handle_error(e) from e

        async for chunk in response:
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
    ) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
        """构建 Google API 请求参数。

        Returns:
            tuple: (contents, system_instruction, generation_config)
        """

        contents: list[dict[str, Any]] = []
        system_instruction: str | None = None

        for msg in messages:
            if msg.role == Role.SYSTEM:
                if isinstance(msg.content, str):
                    system_instruction = msg.content
                continue

            role = "user" if msg.role == Role.USER else "model"

            parts: list[dict[str, Any]] = []
            if msg.content is not None:
                if isinstance(msg.content, str):
                    parts.append({"text": msg.content})
                else:
                    for part in msg.content:
                        if isinstance(part, TextContent):
                            parts.append({"text": part.text})
                        elif isinstance(part, ImageContent):
                            if part.image.startswith("http"):
                                # Gemini 支持 URL
                                parts.append({
                                    "file_data": {
                                        "file_uri": part.image,
                                        "mime_type": part.mime_type or "image/png",
                                    }
                                })
                            else:
                                # base64
                                import base64
                                try:
                                    decoded = base64.b64decode(part.image)
                                except Exception:
                                    decoded = part.image.encode()
                                parts.append({
                                    "inline_data": {
                                        "mime_type": part.mime_type or "image/png",
                                        "data": decoded,
                                    }
                                })

            # 工具调用处理
            if msg.role == Role.TOOL and msg.tool_call_id:
                # 工具结果
                response_content = (
                    msg.content
                    if isinstance(msg.content, dict)
                    else {"result": str(msg.content)}
                )
                parts.append({
                    "function_response": {
                        "name": msg.name or "",
                        "response": response_content,
                    }
                })
            elif msg.tool_calls:
                for tc in msg.tool_calls:
                    args = (
                        tc.arguments
                        if isinstance(tc.arguments, dict)
                        else json.loads(tc.arguments)
                    )
                    parts.append({
                        "function_call": {
                            "name": tc.name,
                            "args": args,
                        }
                    })

            contents.append({"role": role, "parts": parts})

        generation_config: dict[str, Any] = {}
        if config.temperature is not None:
            generation_config["temperature"] = config.temperature
        if config.top_p is not None:
            generation_config["top_p"] = config.top_p
        if config.top_k is not None:
            generation_config["top_k"] = config.top_k
        if config.max_tokens is not None:
            generation_config["max_output_tokens"] = config.max_tokens
        if config.stop is not None:
            stop_sequences = (
                config.stop if isinstance(config.stop, list) else [config.stop]
            )
            generation_config["stop_sequences"] = stop_sequences

        # 工具配置
        if config.tools:
            generation_config["tools"] = self._convert_tools(config.tools)

        return contents, system_instruction, generation_config

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将统一工具格式转换为 Google 格式。"""
        google_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                google_tools.append({
                    "function_declarations": [{
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {"type": "object"}),
                    }]
                })
        return google_tools

    def _parse_response(
        self,
        response: GenerateContentResponse,
        model: str,
    ) -> GenerateResponse:
        """解析 Google 响应为统一格式。"""
        text = ""
        tool_calls: list[dict[str, Any]] = []

        if response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        "id": f"call_{fc.name}",
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    })

        finish_reason = "stop"
        if response.candidates:
            finish_reason_val = response.candidates[0].finish_reason
            if finish_reason_val:
                reason_name = str(finish_reason_val)
                if "MAX_TOKENS" in reason_name or "LENGTH" in reason_name:
                    finish_reason = "length"
                elif "SAFETY" in reason_name or "RECITATION" in reason_name:
                    finish_reason = "content_filter"

        usage: Usage | None = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = Usage(
                prompt_tokens=response.usage_metadata.prompt_token_count,
                completion_tokens=response.usage_metadata.candidates_token_count,
                total_tokens=response.usage_metadata.total_token_count,
            )

        return GenerateResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw=response,
            usage=usage,
            model=model,
        )

    def _parse_stream_chunk(
        self,
        chunk: GenerateContentResponse,
    ) -> StreamChunk | None:
        """解析流式响应块。"""
        if not chunk.candidates:
            return None

        candidate = chunk.candidates[0]
        text = ""

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text

        finish_reason = None
        if candidate.finish_reason:
            reason_name = str(candidate.finish_reason)
            if "STOP" in reason_name:
                finish_reason = "stop"
            elif "MAX_TOKENS" in reason_name:
                finish_reason = "length"
            elif "SAFETY" in reason_name:
                finish_reason = "content_filter"

        return StreamChunk(
            delta=text,
            is_finished=finish_reason is not None,
            finish_reason=finish_reason,
        )

    def _handle_error(self, error: Exception) -> AIError:
        """将 Google 错误转换为统一错误类型。"""
        error_type: str = type(error).__name__
        error_message: str = str(error)
        error_message_lower = error_message.lower()

        # 检查类型名或错误消息中的关键词
        if "InvalidAPIKey" in error_type or (
            "invalid" in error_message_lower and "api" in error_message_lower
        ):
            return AuthenticationError(error_message, provider=self.name)
        if "ResourceExhausted" in error_type or "resourc" in error_message_lower:
            # 先检查 quota, 再返回 RateLimitError
            if "quota" in error_message_lower:
                return QuotaExceededError(error_message, provider=self.name)
            return RateLimitError(error_message, provider=self.name)
        if "429" in error_message:
            return RateLimitError(error_message, provider=self.name)
        if "InvalidArgument" in error_type or "400" in error_message:
            return InvalidRequestError(error_message, provider=self.name)
        if "NotFound" in error_type or "404" in error_message:
            return ModelNotFoundError(error_message, provider=self.name)
        if "PermissionDenied" in error_type or "403" in error_message:
            return AuthenticationError(
                f"Access forbidden: {error_message}",
                provider=self.name,
            )
        if "Unavailable" in error_type or "503" in error_message:
            return ProviderNotAvailableError(error_message, provider=self.name)
        if "Internal" in error_type or "500" in error_message:
            return InternalServerError(error_message, provider=self.name)
        if "DeadlineExceeded" in error_type or "timeout" in error_message_lower:
            return TimeoutError(error_message, provider=self.name)
        if (
            "Safety" in error_type
            or "Blocked" in error_message
            or "safety" in error_message_lower
        ):
            return ContentFilterError(error_message, provider=self.name)
        if "quota" in error_message_lower or "exhausted" in error_message_lower:
            return QuotaExceededError(error_message, provider=self.name)

        return InternalServerError(
            f"Unexpected error: {error_type}: {error_message}",
            provider=self.name,
            original_error=error_type,
        )


# 便捷别名
google: type[GoogleProvider] = GoogleProvider
