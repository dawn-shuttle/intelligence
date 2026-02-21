"""Anthropic (Claude) 适配器 - 对接 Anthropic API。"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, ClassVar

from .base import (
    handle_anthropic_error,
    validate_config,
    validate_messages,
)
from ..core.config import GenerateConfig
from ..core.error import (
    AIError,
    ConfigurationError,
    ResponseParseError,
)
from ..core.provider import BaseProvider
from ..core.response import GenerateResponse, StreamChunk, Usage
from ..core.types import ImageContent, Message, Role, TextContent

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import RawMessageStreamEvent


class AnthropicProvider(BaseProvider):
    """Anthropic (Claude) API 适配器。

    支持的模型包括 Claude 3.5, Claude 3 系列。
    """

    name: str = "anthropic"

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-latest",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-latest",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Anthropic 适配器。

        Args:
            api_key: Anthropic API 密钥(也可通过环境变量 ANTHROPIC_API_KEY 设置)。
            base_url: 自定义 API 端点。
            **kwargs: 其他参数。
        """
        super().__init__(api_key, base_url, **kwargs)
        self._client: AsyncAnthropic | None = None

    def _get_client(self) -> AsyncAnthropic:
        """获取 Anthropic 客户端(延迟初始化)。

        Returns:
            AsyncAnthropic: Anthropic 异步客户端实例。

        Raises:
            ImportError: anthropic 包未安装。
        """
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic 包未安装, 请运行: pip install anthropic"
                ) from e

            self._client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def supports_model(self, model: str) -> bool:
        """检查是否支持指定模型。"""
        return any(model.startswith(m) for m in self.SUPPORTED_MODELS)

    def get_model_list(self) -> list[str]:
        """获取支持的模型列表。"""
        return self.SUPPORTED_MODELS.copy()

    async def generate(
        self,
        messages: list[Message],
        config: GenerateConfig,
    ) -> GenerateResponse:
        """生成文本响应(非流式)。"""
        validate_config(config, self.name, temp_max=1.0)
        validate_messages(messages, self.name)

        client = self._get_client()
        params = self._build_params(messages, config)

        try:
            response = await client.messages.create(**params)
        except Exception as e:
            raise handle_anthropic_error(e, self.name) from e

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
        """生成流式响应。"""
        validate_config(config, self.name, temp_max=1.0)
        validate_messages(messages, self.name)

        client = self._get_client()
        params = self._build_params(messages, config)

        try:
            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    parsed = self._parse_stream_event(event)
                    if parsed:
                        yield parsed
        except Exception as e:
            raise handle_anthropic_error(e, self.name) from e

    def _build_params(
        self,
        messages: list[Message],
        config: GenerateConfig,
    ) -> dict[str, Any]:
        """构建 Anthropic API 请求参数。"""
        # 分离系统消息和对话消息
        system_content: str | None = None
        conversation_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                if isinstance(msg.content, str):
                    system_content = msg.content
            else:
                conversation_messages.append(self._convert_message(msg))

        params: dict[str, Any] = {
            "model": config.model,
            "messages": conversation_messages,
            "max_tokens": config.max_tokens or 4096,
        }

        if system_content:
            params["system"] = system_content

        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.top_k is not None:
            params["top_k"] = config.top_k
        if config.stop is not None:
            stop_sequences = (
                config.stop if isinstance(config.stop, list) else [config.stop]
            )
            params["stop_sequences"] = stop_sequences

        if config.tools:
            params["tools"] = self._convert_tools(config.tools)
        if config.tool_choice:
            params["tool_choice"] = self._convert_tool_choice(config.tool_choice)

        params.update(config.extra)

        return params

    def _convert_message(self, message: Message) -> dict[str, Any]:
        """将统一消息格式转换为 Anthropic 格式。"""
        role = "user" if message.role == Role.USER else "assistant"

        content: list[dict[str, Any]] = []

        if message.content is not None:
            if isinstance(message.content, str):
                content.append({"type": "text", "text": message.content})
            else:
                for part in message.content:
                    if isinstance(part, TextContent):
                        content.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        # Anthropic 使用 base64 图片
                        if part.image.startswith("http"):
                            # 需要下载图片转 base64, 暂不支持
                            raise ConfigurationError(
                                "Anthropic does not support image URLs, "
                                "use base64 instead",
                                provider=self.name,
                            )
                        mime = part.mime_type or "image/png"
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": part.image,
                            },
                        })

        # 工具结果消息
        if message.role == Role.TOOL and message.tool_call_id:
            tool_content = (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            )
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": tool_content,
                }],
            }

        # 助手消息带工具调用
        if message.role == Role.ASSISTANT and message.tool_calls:
            for tc in message.tool_calls:
                args = (
                    tc.arguments
                    if isinstance(tc.arguments, str)
                    else json.dumps(tc.arguments)
                )
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": (
                        json.loads(args) if isinstance(args, str) else args
                    ),
                })

        return {"role": role, "content": content}

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将统一工具格式转换为 Anthropic 格式。"""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object"}),
                })
        return anthropic_tools

    def _convert_tool_choice(self, tool_choice: str | dict[str, Any]) -> dict[str, Any]:
        """转换工具选择策略。"""
        if isinstance(tool_choice, str):
            choice_map = {
                "auto": {"type": "auto"},
                "required": {"type": "any"},
                "none": {"type": "auto"},  # Anthropic 没有 none
            }
            return choice_map.get(tool_choice, {"type": "auto"})

        if isinstance(tool_choice, dict) and "name" in tool_choice:
            return {"type": "tool", "name": tool_choice["name"]}

        return {"type": "auto"}

    def _parse_response(self, response: AnthropicMessage) -> GenerateResponse:
        """解析 Anthropic 响应为统一格式。"""
        text = ""
        tool_calls: list[dict[str, Any]] = []

        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
            elif hasattr(block, "name") and hasattr(block, "id"):
                # tool_use block
                args = (
                    block.input
                    if isinstance(block.input, dict)
                    else json.loads(block.input)
                )
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": args,
                })

        finish_reason = "stop"
        if response.stop_reason == "max_tokens":
            finish_reason = "length"
        elif response.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif response.stop_reason == "end_turn":
            finish_reason = "stop"

        usage: Usage | None = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=(
                    response.usage.input_tokens + response.usage.output_tokens
                ),
            )

        return GenerateResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw=response,
            usage=usage,
            model=response.model,
            request_id=response.id,
        )

    def _parse_stream_event(
        self,
        event: RawMessageStreamEvent,
    ) -> StreamChunk | None:
        """解析流式事件。"""
        from anthropic.types import TextDelta

        if hasattr(event, "delta") and isinstance(event.delta, TextDelta):
            return StreamChunk(
                delta=event.delta.text,
                is_finished=False,
            )

        if (
            hasattr(event, "message")
            and hasattr(event.message, "stop_reason")
        ):
            # 最终消息
            return StreamChunk(
                delta="",
                is_finished=True,
                finish_reason=event.message.stop_reason,
            )

        return None


# 便捷别名
anthropic: type[AnthropicProvider] = AnthropicProvider
