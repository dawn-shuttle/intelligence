"""工具执行器测试。"""

from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.tools import (
    ExecutorConfig,
    Tool,
    ToolCall,
    ToolExecution,
    ToolExecutionStatus,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    tool,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.tools.executor import (
    execute_tool_call,
    execute_tool_calls,
)


class SlowTool(Tool):
    """慢速工具。"""

    name: ClassVar[str] = "slow_tool"
    description: ClassVar[str] = "A slow tool for testing timeout"

    async def execute(self, **kwargs):
        await asyncio.sleep(2)
        return ToolResult(tool_call_id="", content="done")


class FailingTool(Tool):
    """失败工具。"""

    name: ClassVar[str] = "failing_tool"
    description: ClassVar[str] = "A tool that always fails"

    async def execute(self, **kwargs):
        raise ValueError("Tool failed intentionally")


class FlakyTool(Tool):
    """不稳定工具，前几次失败后成功。"""

    name: ClassVar[str] = "flaky_tool"
    description: ClassVar[str] = "A tool that fails first few times"
    call_count: int = 0

    async def execute(self, **kwargs):
        FlakyTool.call_count += 1
        if FlakyTool.call_count < 3:
            raise RuntimeError(f"Attempt {FlakyTool.call_count} failed")
        return ToolResult(tool_call_id="", content="finally worked")


class ErrorReturningTool(Tool):
    """返回错误结果的工具。"""

    name: ClassVar[str] = "error_tool"
    description: ClassVar[str] = "Returns error result"
    fail_count: int = 0

    async def execute(self, **kwargs):
        ErrorReturningTool.fail_count += 1
        if ErrorReturningTool.fail_count < 3:
            return ToolResult(
                tool_call_id="",
                content="",
                is_error=True,
                error_message="Temporary error",
            )
        return ToolResult(tool_call_id="", content="success")


class TestToolExecutor:
    """ToolExecutor 测试。"""

    @pytest.mark.asyncio
    async def test_execute_simple_tool(self) -> None:
        """测试执行简单工具。"""
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        registry = ToolRegistry()
        registry.register(add)

        executor = ToolExecutor(registry=registry)
        call = ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})

        execution = await executor.execute(call)

        assert execution.status == ToolExecutionStatus.SUCCESS
        assert execution.result is not None
        assert execution.result.content == "3"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self) -> None:
        """测试工具不存在。"""
        registry = ToolRegistry()
        executor = ToolExecutor(registry=registry)
        call = ToolCall(id="1", name="nonexistent", arguments={})

        execution = await executor.execute(call)

        assert execution.status == ToolExecutionStatus.FAILED
        assert execution.result is not None
        assert "not found" in execution.result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_with_invalid_arguments(self) -> None:
        """测试无效参数。"""
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}"

        registry = ToolRegistry()
        registry.register(greet)

        executor = ToolExecutor(registry=registry)
        call = ToolCall(id="1", name="greet", arguments={})  # 缺少必需参数

        execution = await executor.execute(call)

        assert execution.status == ToolExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_batch_parallel(self) -> None:
        """测试并行批量执行。"""
        @tool
        def double(x: int) -> int:
            """Double a number."""
            return x * 2

        registry = ToolRegistry()
        registry.register(double)

        executor = ToolExecutor(registry=registry)
        calls = [
            ToolCall(id=str(i), name="double", arguments={"x": i})
            for i in range(3)
        ]

        executions = await executor.execute_batch(calls, parallel=True)

        assert len(executions) == 3
        assert all(e.status == ToolExecutionStatus.SUCCESS for e in executions)

    @pytest.mark.asyncio
    async def test_execute_batch_sequential(self) -> None:
        """测试串行批量执行。"""
        @tool
        def identity(x: int) -> int:
            """Return input."""
            return x

        registry = ToolRegistry()
        registry.register(identity)

        executor = ToolExecutor(registry=registry)
        calls = [
            ToolCall(id=str(i), name="identity", arguments={"x": i})
            for i in range(3)
        ]

        executions = await executor.execute_batch(calls, parallel=False)

        assert len(executions) == 3

    @pytest.mark.asyncio
    async def test_execute_async_tool(self) -> None:
        """测试执行异步工具。"""
        @tool
        async def async_operation(value: str) -> str:
            """Async operation."""
            await asyncio.sleep(0.01)
            return f"processed: {value}"

        registry = ToolRegistry()
        registry.register(async_operation)

        executor = ToolExecutor(registry=registry)
        call = ToolCall(id="1", name="async_operation", arguments={"value": "test"})

        execution = await executor.execute(call)

        assert execution.status == ToolExecutionStatus.SUCCESS
        assert "processed: test" in execution.result.content

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self) -> None:
        """测试超时。"""
        registry = ToolRegistry()
        registry.register(SlowTool())

        config = ExecutorConfig(timeout=0.1)
        executor = ToolExecutor(registry=registry, config=config)
        call = ToolCall(id="1", name="slow_tool", arguments={})

        execution = await executor.execute(call)

        assert execution.status == ToolExecutionStatus.FAILED
        assert execution.result is not None
        assert "timed out" in execution.result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self) -> None:
        """测试重试成功。"""
        FlakyTool.call_count = 0
        registry = ToolRegistry()
        registry.register(FlakyTool())

        config = ExecutorConfig(max_retries=5, retry_delay=0.01)
        executor = ToolExecutor(registry=registry, config=config)
        call = ToolCall(id="1", name="flaky_tool", arguments={})

        execution = await executor.execute(call)

        assert execution.status == ToolExecutionStatus.SUCCESS
        assert execution.result.content == "finally worked"

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self) -> None:
        """测试重试次数用尽。"""
        registry = ToolRegistry()
        registry.register(FailingTool())

        config = ExecutorConfig(max_retries=2, retry_delay=0.01)
        executor = ToolExecutor(registry=registry, config=config)
        call = ToolCall(id="1", name="failing_tool", arguments={})

        execution = await executor.execute(call)

        assert execution.status == ToolExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_with_callback(self) -> None:
        """测试回调函数。"""
        before_calls: list[ToolCall] = []
        after_results: list[ToolResult] = []
        error_calls: list[tuple[ToolCall, Exception]] = []

        @tool
        def simple_tool(x: int) -> int:
            """Simple tool."""
            return x * 2

        registry = ToolRegistry()
        registry.register(simple_tool)

        config = ExecutorConfig(
            on_before_execute=lambda tc: before_calls.append(tc),
            on_after_execute=lambda tc, r: after_results.append(r),
        )
        executor = ToolExecutor(registry=registry, config=config)
        call = ToolCall(id="1", name="simple_tool", arguments={"x": 5})

        await executor.execute(call)

        assert len(before_calls) == 1
        assert len(after_results) == 1

    @pytest.mark.asyncio
    async def test_execute_with_error_callback(self) -> None:
        """测试错误回调。"""
        error_calls: list[tuple[ToolCall, Exception]] = []

        registry = ToolRegistry()
        registry.register(FailingTool())

        config = ExecutorConfig(
            max_retries=0,
            on_error=lambda tc, e: error_calls.append((tc, e)),
        )
        executor = ToolExecutor(registry=registry, config=config)
        call = ToolCall(id="1", name="failing_tool", arguments={})

        await executor.execute(call)

        assert len(error_calls) == 1
        assert isinstance(error_calls[0][1], ValueError)

    @pytest.mark.asyncio
    async def test_execute_with_validation(self) -> None:
        """测试参数验证。"""
        @tool
        def divide(a: int, b: int) -> float:
            """Divide two numbers."""
            return a / b

        registry = ToolRegistry()
        registry.register(divide)

        config = ExecutorConfig(validate_args=True)
        executor = ToolExecutor(registry=registry, config=config)

        # 有效参数
        call = ToolCall(id="1", name="divide", arguments={"a": 10, "b": 2})
        execution = await executor.execute(call)
        assert execution.status == ToolExecutionStatus.SUCCESS

        # 无效参数（字符串而非整数）
        call2 = ToolCall(id="2", name="divide", arguments={"a": "not_a_number", "b": 2})
        execution2 = await executor.execute(call2)
        assert execution2.status == ToolExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_skip_validation(self) -> None:
        """测试跳过参数验证。"""
        @tool
        def echo(value: str) -> str:
            """Echo value."""
            return str(value)

        registry = ToolRegistry()
        registry.register(echo)

        config = ExecutorConfig(validate_args=False)
        executor = ToolExecutor(registry=registry, config=config)
        call = ToolCall(id="1", name="echo", arguments={"value": "test"})

        execution = await executor.execute(call)
        assert execution.status == ToolExecutionStatus.SUCCESS


class TestExecutorConfig:
    """ExecutorConfig 测试。"""

    def test_default_config(self) -> None:
        """测试默认配置。"""
        config = ExecutorConfig()

        assert config.timeout == 30.0
        assert config.max_retries == 0
        assert config.retry_delay == 1.0
        assert config.validate_args is True

    def test_custom_config(self) -> None:
        """测试自定义配置。"""
        config = ExecutorConfig(
            timeout=60.0,
            max_retries=3,
            retry_delay=2.0,
            validate_args=False,
        )

        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.retry_delay == 2.0
        assert config.validate_args is False


class TestToolExecution:
    """ToolExecution 测试。"""

    def test_execution_creation(self) -> None:
        """测试执行记录创建。"""
        call = ToolCall(id="test", name="tool", arguments={})
        execution = ToolExecution(
            tool_call=call,
            status=ToolExecutionStatus.SUCCESS,
            result=ToolResult(tool_call_id="test", content="done"),
            duration_ms=100.5,
        )

        assert execution.tool_call.id == "test"
        assert execution.status == ToolExecutionStatus.SUCCESS
        assert execution.duration_ms == 100.5


class TestConvenienceFunctions:
    """便捷函数测试。"""

    @pytest.mark.asyncio
    async def test_execute_tool_call(self) -> None:
        """测试 execute_tool_call 便捷函数。"""
        @tool
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        registry = ToolRegistry()
        registry.register(add)

        call = ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})
        result = await execute_tool_call(call, registry=registry)

        assert result.content == "3"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_execute_tool_calls_parallel(self) -> None:
        """测试 execute_tool_calls 并行执行。"""
        @tool
        def multiply(x: int, y: int) -> int:
            """Multiply numbers."""
            return x * y

        registry = ToolRegistry()
        registry.register(multiply)

        calls = [
            ToolCall(id=str(i), name="multiply", arguments={"x": i, "y": 2})
            for i in range(3)
        ]
        results = await execute_tool_calls(calls, registry=registry, parallel=True)

        assert len(results) == 3
        assert all(not r.is_error for r in results)

    @pytest.mark.asyncio
    async def test_execute_tool_calls_sequential(self) -> None:
        """测试 execute_tool_calls 串行执行。"""
        @tool
        def square(x: int) -> int:
            """Square a number."""
            return x * x

        registry = ToolRegistry()
        registry.register(square)

        calls = [
            ToolCall(id=str(i), name="square", arguments={"x": i})
            for i in range(3)
        ]
        results = await execute_tool_calls(calls, registry=registry, parallel=False)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_execute_tool_call_not_found(self) -> None:
        """测试工具不存在时的便捷函数。"""
        call = ToolCall(id="1", name="nonexistent", arguments={})
        result = await execute_tool_call(call)

        assert result.is_error is True
        assert "not found" in result.error_message.lower()


class TestToolCall:
    """ToolCall 测试。"""

    def test_create(self) -> None:
        """测试创建。"""
        call = ToolCall(id="call_123", name="test", arguments={"x": 1})

        assert call.id == "call_123"
        assert call.name == "test"
        assert call.arguments == {"x": 1}

    def test_from_dict(self) -> None:
        """测试从字典创建。"""
        call = ToolCall(id="1", name="test", arguments={"a": 1})
        assert call.arguments == {"a": 1}


class TestToolResult:
    """ToolResult 测试。"""

    def test_success_result(self) -> None:
        """测试成功结果。"""
        result = ToolResult(tool_call_id="1", content="success")

        assert result.tool_call_id == "1"
        assert result.content == "success"
        assert result.is_error is False

    def test_error_result(self) -> None:
        """测试错误结果。"""
        result = ToolResult(
            tool_call_id="1",
            content="",
            is_error=True,
            error_message="Something went wrong",
        )

        assert result.is_error is True
        assert result.error_message == "Something went wrong"

    def test_bytes_content(self) -> None:
        """测试字节内容。"""
        result = ToolResult(tool_call_id="1", content=b"binary data")

        assert result.content == b"binary data"

    def test_dict_content(self) -> None:
        """测试字典内容。"""
        result = ToolResult(
            tool_call_id="1",
            content={"key": "value"},
        )

        assert result.content == {"key": "value"}