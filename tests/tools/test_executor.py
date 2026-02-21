"""工具执行器测试。"""

from __future__ import annotations

import asyncio

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.tools import (
    Tool,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    tool,
)


class SlowTool(Tool):
    """慢速工具。"""

    name = "slow_tool"
    description = "A slow tool for testing timeout"

    async def execute(self, **kwargs):
        await asyncio.sleep(2)
        return ToolResult(tool_call_id="", content="done")


class FailingTool(Tool):
    """失败工具。"""

    name = "failing_tool"
    description = "A tool that always fails"

    async def execute(self, **kwargs):
        raise ValueError("Tool failed intentionally")


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

        assert execution.status.value == "success"
        assert execution.result is not None
        assert execution.result.content == "3"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self) -> None:
        """测试工具不存在。"""
        registry = ToolRegistry()
        executor = ToolExecutor(registry=registry)
        call = ToolCall(id="1", name="nonexistent", arguments={})

        execution = await executor.execute(call)

        assert execution.status.value == "failed"
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

        assert execution.status.value == "failed"

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
        assert all(e.status.value == "success" for e in executions)

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

        assert execution.status.value == "success"
        assert "processed: test" in execution.result.content


class TestToolCall:
    """ToolCall 测试。"""

    def test_create(self) -> None:
        """测试创建。"""
        call = ToolCall(id="call_123", name="test", arguments={"x": 1})

        assert call.id == "call_123"
        assert call.name == "test"
        assert call.arguments == {"x": 1}


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
