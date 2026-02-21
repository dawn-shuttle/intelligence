"""统一入口函数 - 提供给用户的简洁 API。"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from .config import GenerateConfig, ResponseFormat, StopSequences, ToolChoice
from .provider import BaseProvider
from .response import GenerateResponse, StreamChunk
from .types import Message


async def generate_text(
    messages: list[Message],
    provider: BaseProvider,
    *,
    model: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop: StopSequences | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: ToolChoice | None = None,
    response_format: ResponseFormat | None = None,
    seed: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    **kwargs: Any,
) -> GenerateResponse:
    """生成文本响应(非流式)。

    Args:
        messages: 消息列表。
        provider: AI 提供商实例。
        model: 模型标识。
        temperature: 采样温度(0.0-2.0)。
        max_tokens: 最大输出 token 数。
        top_p: Top-p 采样参数。
        stop: 停止词。
        tools: 工具定义列表。
        tool_choice: 工具选择策略。
        response_format: 响应格式配置。
        seed: 随机种子。
        frequency_penalty: 频率惩罚。
        presence_penalty: 存在惩罚。
        **kwargs: 其他参数, 传递给提供商。

    Returns:
        GenerateResponse: 统一格式的响应。

    Raises:
        AIError: AI 调用相关错误。
    """
    config = GenerateConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        extra=kwargs,
    )

    return await provider.generate(messages, config)


async def stream_text(
    messages: list[Message],
    provider: BaseProvider,
    *,
    model: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop: StopSequences | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: ToolChoice | None = None,
    response_format: ResponseFormat | None = None,
    seed: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    **kwargs: Any,
) -> AsyncGenerator[StreamChunk, None]:
    """生成流式文本响应。

    Args:
        messages: 消息列表。
        provider: AI 提供商实例。
        model: 模型标识。
        temperature: 采样温度(0.0-2.0)。
        max_tokens: 最大输出 token 数。
        top_p: Top-p 采样参数。
        stop: 停止词。
        tools: 工具定义列表。
        tool_choice: 工具选择策略。
        response_format: 响应格式配置。
        seed: 随机种子。
        frequency_penalty: 频率惩罚。
        presence_penalty: 存在惩罚。
        **kwargs: 其他参数, 传递给提供商。

    Yields:
        StreamChunk: 流式响应块。

    Raises:
        AIError: AI 调用相关错误。
    """
    config = GenerateConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stream=True,
        extra=kwargs,
    )

    async for chunk in provider.generate_stream(messages, config):
        yield chunk
