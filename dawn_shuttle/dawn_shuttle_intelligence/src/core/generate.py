"""统一入口函数 - 提供给用户的简洁 API。"""

from typing import AsyncIterator

from .config import GenerateConfig
from .error import AIError
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
    stop: list[str] | str | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    **kwargs,
) -> GenerateResponse:
    """生成文本响应（非流式）。

    Args:
        messages: 消息列表
        provider: AI 提供商实例
        model: 模型标识
        temperature: 采样温度
        max_tokens: 最大输出 token 数
        top_p: Top-p 采样
        stop: 停止词
        tools: 工具定义
        tool_choice: 工具选择策略
        **kwargs: 其他参数

    Returns:
        GenerateResponse: 统一格式的响应

    Raises:
        AIError: AI 调用相关错误
    """
    config = GenerateConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
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
    stop: list[str] | str | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    **kwargs,
) -> AsyncIterator[StreamChunk]:
    """生成流式文本响应。

    Args:
        messages: 消息列表
        provider: AI 提供商实例
        model: 模型标识
        temperature: 采样温度
        max_tokens: 最大输出 token 数
        top_p: Top-p 采样
        stop: 停止词
        tools: 工具定义
        tool_choice: 工具选择策略
        **kwargs: 其他参数

    Yields:
        StreamChunk: 流式响应块

    Raises:
        AIError: AI 调用相关错误
    """
    config = GenerateConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        stream=True,
        extra=kwargs,
    )

    async for chunk in provider.generate_stream(messages, config):
        yield chunk
