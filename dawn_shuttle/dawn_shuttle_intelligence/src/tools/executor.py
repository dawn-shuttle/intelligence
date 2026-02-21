"""工具执行器 - 负责执行工具调用并处理超时、错误、重试。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .registry import ToolRegistry, get_default_registry
from .tool import Tool
from .types import (
    ToolCall,
    ToolExecution,
    ToolExecutionStatus,
    ToolResult,
)


class ToolExecutionError(Exception):
    """工具执行错误。"""
    pass


class ToolTimeoutError(ToolExecutionError):
    """工具执行超时。"""
    pass


class ToolNotFoundError(ToolExecutionError):
    """工具未找到。"""
    pass


@dataclass
class ExecutorConfig:
    """执行器配置。

    Attributes:
        timeout: 默认超时时间(秒)。
        max_retries: 最大重试次数。
        retry_delay: 重试延迟(秒)。
        validate_args: 是否验证参数。
        on_before_execute: 执行前回调。
        on_after_execute: 执行后回调。
        on_error: 错误回调。
    """

    timeout: float = 30.0
    max_retries: int = 0
    retry_delay: float = 1.0
    validate_args: bool = True
    on_before_execute: Callable[[ToolCall], None] | None = None
    on_after_execute: Callable[[ToolCall, ToolResult], None] | None = None
    on_error: Callable[[ToolCall, Exception], None] | None = None


@dataclass
class ToolExecutor:
    """工具执行器。

    负责执行工具调用, 处理超时、错误、重试等。

    Attributes:
        registry: 工具注册表。
        config: 执行器配置。
    """

    registry: ToolRegistry = field(default_factory=get_default_registry)
    config: ExecutorConfig = field(default_factory=ExecutorConfig)

    async def execute(
        self,
        tool_call: ToolCall,
        *,
        timeout: float | None = None,
        validate: bool | None = None,
    ) -> ToolExecution:
        """执行工具调用。

        Args:
            tool_call: 工具调用请求。
            timeout: 超时时间(覆盖配置)。
            validate: 是否验证参数(覆盖配置)。

        Returns:
            ToolExecution: 执行记录。
        """
        execution = ToolExecution(tool_call=tool_call)
        start_time = time.monotonic()

        # 获取工具
        tool = await self._get_tool(tool_call.name)
        if tool is None:
            execution.status = ToolExecutionStatus.FAILED
            execution.result = ToolResult(
                tool_call_id=tool_call.id,
                content="",
                is_error=True,
                error_message=f"Tool '{tool_call.name}' not found",
            )
            return execution

        # 验证参数
        should_validate = (
            validate if validate is not None else self.config.validate_args
        )
        if should_validate:
            valid, errors = tool.validate_arguments(tool_call.arguments)
            if not valid:
                execution.status = ToolExecutionStatus.FAILED
                execution.result = ToolResult(
                    tool_call_id=tool_call.id,
                    content="",
                    is_error=True,
                    error_message=f"Invalid arguments: {'; '.join(errors)}",
                )
                return execution

        # 执行前回调
        if self.config.on_before_execute:
            self.config.on_before_execute(tool_call)

        # 执行
        result = await self._execute_with_retry(
            tool=tool,
            tool_call=tool_call,
            timeout=timeout or self.config.timeout,
        )

        # 记录结果
        execution.result = result
        execution.duration_ms = (time.monotonic() - start_time) * 1000
        execution.status = (
            ToolExecutionStatus.SUCCESS
            if not result.is_error
            else ToolExecutionStatus.FAILED
        )

        # 执行后回调
        if self.config.on_after_execute:
            self.config.on_after_execute(tool_call, result)

        return execution

    async def execute_batch(
        self,
        tool_calls: list[ToolCall],
        *,
        parallel: bool = True,
    ) -> list[ToolExecution]:
        """批量执行工具调用。

        Args:
            tool_calls: 工具调用列表。
            parallel: 是否并行执行。

        Returns:
            list[ToolExecution]: 执行记录列表。
        """
        if parallel:
            tasks = [self.execute(tc) for tc in tool_calls]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for tc in tool_calls:
                results.append(await self.execute(tc))
            return results

    async def _get_tool(self, name: str) -> Tool | None:
        """获取工具。"""
        try:
            return self.registry.get(name)
        except Exception:
            return None

    async def _execute_with_retry(
        self,
        tool: Tool,
        tool_call: ToolCall,
        timeout: float,
    ) -> ToolResult:
        """带重试的执行。"""
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # 带超时执行
                result = await asyncio.wait_for(
                    tool.execute(**tool_call.arguments),
                    timeout=timeout,
                )

                # 设置 tool_call_id
                result.tool_call_id = tool_call.id

                # 成功则返回
                if not result.is_error:
                    return result

                # 如果是错误结果, 但可以重试
                msg = result.error_message or "Tool returned error"
                last_error = ToolExecutionError(msg)

            except asyncio.TimeoutError:
                last_error = ToolTimeoutError(
                    f"Tool '{tool_call.name}' timed out after {timeout}s"
                )

            except Exception as e:
                last_error = e

            # 错误回调
            if self.config.on_error:
                self.config.on_error(tool_call, last_error)

            # 重试延迟
            if attempt < self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay)

        # 所有重试失败
        return ToolResult(
            tool_call_id=tool_call.id,
            content="",
            is_error=True,
            error_message=str(last_error),
        )


async def execute_tool_call(
    tool_call: ToolCall,
    registry: ToolRegistry | None = None,
    **kwargs: Any,
) -> ToolResult:
    """执行单个工具调用的便捷函数。

    Args:
        tool_call: 工具调用请求。
        registry: 工具注册表。
        **kwargs: 传递给执行器的其他参数。

    Returns:
        ToolResult: 执行结果。
    """
    target_registry = registry or get_default_registry()
    executor = ToolExecutor(registry=target_registry)
    execution = await executor.execute(tool_call, **kwargs)
    return execution.result


async def execute_tool_calls(
    tool_calls: list[ToolCall],
    registry: ToolRegistry | None = None,
    parallel: bool = True,
    **kwargs: Any,
) -> list[ToolResult]:
    """批量执行工具调用的便捷函数。

    Args:
        tool_calls: 工具调用列表。
        registry: 工具注册表。
        parallel: 是否并行执行。
        **kwargs: 传递给执行器的其他参数。

    Returns:
        list[ToolResult]: 执行结果列表。
    """
    target_registry = registry or get_default_registry()
    executor = ToolExecutor(registry=target_registry)
    executions = await executor.execute_batch(tool_calls, parallel=parallel)
    return [e.result for e in executions]
