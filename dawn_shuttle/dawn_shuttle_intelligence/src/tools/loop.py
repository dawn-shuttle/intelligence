"""工具调用循环 - 自动处理 AI 返回的工具调用并执行。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.config import GenerateConfig
from ..core.provider import BaseProvider
from ..core.response import GenerateResponse
from ..core.types import Message, Role
from .converter import ProviderType, ToolConverter
from .executor import ToolExecutionError, ToolExecutor
from .registry import ToolRegistry
from .tool import Tool
from .types import ToolCall, ToolResult


class LoopStatus(str, Enum):
    """循环状态。"""

    COMPLETED = "completed"  # 正常完成
    MAX_ITERATIONS = "max_iterations"  # 达到最大迭代次数
    ERROR = "error"  # 发生错误


@dataclass
class LoopResult:
    """循环执行结果。

    Attributes:
        response: 最终的 AI 响应。
        status: 循环状态。
        iterations: 迭代次数。
        tool_calls: 所有工具调用记录。
        errors: 错误列表。
    """

    response: GenerateResponse | None = None
    status: LoopStatus = LoopStatus.COMPLETED
    iterations: int = 0
    tool_calls: list[tuple[ToolCall, ToolResult]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """是否成功完成。"""
        return self.status == LoopStatus.COMPLETED


@dataclass
class LoopConfig:
    """循环配置。

    Attributes:
        max_iterations: 最大迭代次数。
        execute_tools: 是否自动执行工具。
        on_tool_call: 工具调用回调。
        on_tool_result: 工具结果回调。
        on_iteration: 迭代回调。
    """

    max_iterations: int = 10
    execute_tools: bool = True
    on_tool_call: Callable[[ToolCall, int], None] | None = None
    on_tool_result: Callable[[ToolCall, ToolResult, int], None] | None = None
    on_iteration: Callable[[int, GenerateResponse], None] | None = None


async def run_with_tools(
    messages: list[Message],
    provider: BaseProvider,
    tools: list[Tool] | ToolRegistry,
    *,
    config: GenerateConfig | None = None,
    loop_config: LoopConfig | None = None,
    executor: ToolExecutor | None = None,
) -> LoopResult:
    """运行对话并自动处理工具调用循环。

    Args:
        messages: 消息列表。
        provider: AI 提供商。
        tools: 工具列表或注册表。
        config: 生成配置。
        loop_config: 循环配置。
        executor: 工具执行器。

    Returns:
        LoopResult: 循环执行结果。
    """
    # 准备工具注册表
    registry = _prepare_registry(tools)

    # 准备配置
    cfg = loop_config or LoopConfig()
    exec_ = executor or ToolExecutor(registry=registry)

    # 准备生成配置
    gen_config = config or GenerateConfig(model="")

    # 转换工具定义为提供商格式
    tool_definitions = [t.get_definition() for t in registry]
    provider_type = _get_provider_type(provider)
    tools_param = ToolConverter.definitions_to_provider(tool_definitions, provider_type)

    # 结果
    result = LoopResult()

    try:
        for iteration in range(cfg.max_iterations):
            result.iterations = iteration + 1

            # 调用 AI
            response = await _generate_with_tools(
                messages=messages,
                provider=provider,
                config=gen_config,
                tools_param=tools_param,
            )

            # 迭代回调
            if cfg.on_iteration:
                cfg.on_iteration(iteration, response)

            # 检查是否有工具调用
            if not response.tool_calls:
                result.response = response
                result.status = LoopStatus.COMPLETED
                return result

            # 如果不自动执行工具, 直接返回
            if not cfg.execute_tools:
                result.response = response
                result.status = LoopStatus.COMPLETED
                return result

            # 执行所有工具调用
            for tc_data in response.tool_calls:
                # 解析工具调用
                tool_call = _parse_tool_call(tc_data, provider_type)

                # 回调
                if cfg.on_tool_call:
                    cfg.on_tool_call(tool_call, iteration)

                # 执行工具
                execution = await exec_.execute(tool_call)
                tool_result = execution.result

                # 记录
                result.tool_calls.append((tool_call, tool_result))

                # 回调
                if cfg.on_tool_result:
                    cfg.on_tool_result(tool_call, tool_result, iteration)

                # 添加助手消息(带工具调用)
                messages.append(Message(
                    role=Role.ASSISTANT,
                    content=response.text or None,
                    tool_calls=[tool_call],
                ))

                # 添加工具结果消息
                tool_name = tool_call.name
                ToolConverter.result_to_provider(
                    tool_result, provider_type, tool_name
                )

                if provider_type == "openai":
                    messages.append(Message(
                        role=Role.TOOL,
                        content=tool_result.content,
                        tool_call_id=tool_result.tool_call_id,
                    ))
                elif provider_type == "anthropic":
                    messages.append(Message(
                        role=Role.USER,
                        content=[{
                            "type": "tool_result",
                            "tool_use_id": tool_result.tool_call_id,
                            "content": tool_result.content,
                            "is_error": tool_result.is_error,
                        }],
                    ))
                elif provider_type == "google":
                    # Google 需要在下一请求的 contents 中添加
                    pass

        # 达到最大迭代次数
        result.status = LoopStatus.MAX_ITERATIONS
        return result

    except Exception as e:
        result.status = LoopStatus.ERROR
        result.errors.append(str(e))
        return result


async def _generate_with_tools(
    messages: list[Message],
    provider: BaseProvider,
    config: GenerateConfig,
    tools_param: list[dict[str, Any]],
) -> GenerateResponse:
    """调用 AI 生成。"""
    # 构建带工具的配置
    config_dict = config.to_dict()
    config_dict["tools"] = tools_param

    return await provider.generate(messages, config)


def _prepare_registry(tools: list[Tool] | ToolRegistry) -> ToolRegistry:
    """准备工具注册表。"""
    if isinstance(tools, ToolRegistry):
        return tools

    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)

    return registry


def _get_provider_type(provider: BaseProvider) -> ProviderType:
    """获取提供商类型。"""
    name = provider.name.lower()

    if "openai" in name or "deepseek" in name or "moonshot" in name:
        return "openai"
    if "anthropic" in name or "claude" in name:
        return "anthropic"
    if "google" in name or "gemini" in name:
        return "google"

    # 默认使用 OpenAI 格式
    return "openai"


def _parse_tool_call(data: Any, provider: ProviderType) -> ToolCall:
    """解析工具调用数据。"""
    if isinstance(data, ToolCall):
        return data

    if isinstance(data, dict):
        # 已经是字典格式
        import json

        args = data.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)

        return ToolCall(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=args,
        )

    # 使用转换器解析
    return ToolConverter.call_from_provider(data, provider)


async def execute_and_continue(
    messages: list[Message],
    provider: BaseProvider,
    tools: list[Tool] | ToolRegistry,
    config: GenerateConfig,
    *,
    max_iterations: int = 10,
    on_tool_call: Callable[[ToolCall, ToolResult], None] | None = None,
) -> GenerateResponse:
    """执行工具并继续对话的简化函数。

    Args:
        messages: 消息列表。
        provider: AI 提供商。
        tools: 工具列表或注册表。
        config: 生成配置。
        max_iterations: 最大迭代次数。
        on_tool_call: 工具调用回调。

    Returns:
        GenerateResponse: 最终响应。

    Raises:
        ToolExecutorError: 执行失败。
    """
    loop_config = LoopConfig(
        max_iterations=max_iterations,
        on_tool_result=lambda tc, tr, _: on_tool_call(tc, tr) if on_tool_call else None,
    )

    result = await run_with_tools(
        messages=messages,
        provider=provider,
        tools=tools,
        config=config,
        loop_config=loop_config,
    )

    if result.status == LoopStatus.ERROR:
        raise ToolExecutionError("; ".join(result.errors))

    if result.status == LoopStatus.MAX_ITERATIONS:
        raise ToolExecutionError(f"Max iterations ({max_iterations}) exceeded")

    if result.response is None:
        raise ToolExecutionError("No response generated")

    return result.response
