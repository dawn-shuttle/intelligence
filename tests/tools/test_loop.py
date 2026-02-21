"""工具调用循环测试 - 使用 mock 模拟 AI 调用。"""

from __future__ import annotations

from dataclasses import field
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.core.config import GenerateConfig
from dawn_shuttle.dawn_shuttle_intelligence.src.core.provider import BaseProvider
from dawn_shuttle.dawn_shuttle_intelligence.src.core.response import (
    GenerateResponse,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import Message, Role
from dawn_shuttle.dawn_shuttle_intelligence.src.tools import (
    LoopConfig,
    LoopResult,
    LoopStatus,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    execute_and_continue,
    run_with_tools,
)


class MockProvider(BaseProvider):
    """模拟 Provider 用于测试。"""

    name = "mock"
    _call_count = 0
    _responses: list[GenerateResponse] = field(default_factory=list)

    def __init__(self, responses: list[GenerateResponse] | None = None):
        super().__init__(api_key="test")
        self._call_count = 0
        self._responses = responses or []

    async def generate(
        self,
        messages: list[Message],
        config: GenerateConfig,
    ) -> GenerateResponse:
        self._call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return GenerateResponse(
            text="Default response",
            tool_calls=[],
            finish_reason="stop",
            raw={},
        )

    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerateConfig,
    ):
        yield None

    def supports_model(self, model: str) -> bool:
        return True

    def get_model_list(self) -> list[str]:
        return ["mock-model"]


class AddTool(Tool):
    """加法工具。"""

    name = "add"
    description = "Add two numbers"

    async def execute(self, a: int, b: int) -> ToolResult:
        return ToolResult(tool_call_id="", content=str(a + b))


class MultiplyTool(Tool):
    """乘法工具。"""

    name = "multiply"
    description = "Multiply two numbers"

    async def execute(self, a: int, b: int) -> ToolResult:
        return ToolResult(tool_call_id="", content=str(a * b))


class FailingTool(Tool):
    """失败工具。"""

    name = "fail"
    description = "Always fails"

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            tool_call_id="",
            content="",
            is_error=True,
            error_message="Tool intentionally failed",
        )


def make_tool_call_response(
    call_id: str,
    name: str,
    args: dict[str, Any],
    text: str = "",
) -> GenerateResponse:
    """创建带工具调用的响应。"""
    return GenerateResponse(
        text=text,
        tool_calls=[
            ToolCall(id=call_id, name=name, arguments=args),
        ],
        finish_reason="tool_calls",
        raw={},
    )


def make_text_response(text: str) -> GenerateResponse:
    """创建文本响应。"""
    return GenerateResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        raw={},
    )


class TestRunWithTools:
    """run_with_tools 测试。"""

    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        """测试简单文本响应(无工具调用)。"""
        provider = MockProvider(responses=[make_text_response("Hello, world!")])
        registry = ToolRegistry()
        registry.register(AddTool())

        messages = [Message(role=Role.USER, content="Say hello")]

        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
        )

        assert result.status == LoopStatus.COMPLETED
        assert result.response is not None
        assert result.response.text == "Hello, world!"
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_single_tool_call(self) -> None:
        """测试单个工具调用。"""
        provider = MockProvider(responses=[
            make_tool_call_response("call_1", "add", {"a": 2, "b": 3}),
            make_text_response("The result is 5"),
        ])
        registry = ToolRegistry()
        registry.register(AddTool())

        messages = [Message(role=Role.USER, content="What is 2 + 3?")]

        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
        )

        assert result.status == LoopStatus.COMPLETED
        assert result.response is not None
        assert result.response.text == "The result is 5"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0][0].name == "add"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self) -> None:
        """测试多个工具调用。"""
        provider = MockProvider(responses=[
            make_tool_call_response("call_1", "add", {"a": 2, "b": 3}),
            make_tool_call_response("call_2", "multiply", {"a": 5, "b": 4}),
            make_text_response("Results: 5 and 20"),
        ])
        registry = ToolRegistry()
        registry.register(AddTool())
        registry.register(MultiplyTool())

        messages = [Message(role=Role.USER, content="Calculate")]

        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
        )

        assert result.status == LoopStatus.COMPLETED
        assert len(result.tool_calls) == 2
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_tool_not_found(self) -> None:
        """测试工具不存在。"""
        provider = MockProvider(responses=[
            make_tool_call_response("call_1", "nonexistent", {}),
            make_text_response("Tool not found, but continuing"),
        ])
        registry = ToolRegistry()

        messages = [Message(role=Role.USER, content="Test")]

        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
        )

        # 工具不存在应该记录错误但继续
        assert result.status == LoopStatus.COMPLETED
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0][1].is_error is True

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self) -> None:
        """测试超过最大迭代次数。"""
        # 永远返回工具调用
        provider = MockProvider(responses=[
            make_tool_call_response("call_1", "add", {"a": 1, "b": 1}),
            make_tool_call_response("call_2", "add", {"a": 2, "b": 2}),
            make_tool_call_response("call_3", "add", {"a": 3, "b": 3}),
        ])
        registry = ToolRegistry()
        registry.register(AddTool())

        messages = [Message(role=Role.USER, content="Keep calculating")]

        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
            loop_config=LoopConfig(max_iterations=2),
        )

        assert result.status == LoopStatus.MAX_ITERATIONS
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """测试错误处理。"""
        provider = MockProvider()
        provider.generate = AsyncMock(side_effect=ValueError("Provider error"))

        registry = ToolRegistry()
        registry.register(AddTool())

        messages = [Message(role=Role.USER, content="Test")]

        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
        )

        assert result.status == LoopStatus.ERROR
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_callbacks(self) -> None:
        """测试回调函数。"""
        provider = MockProvider(responses=[
            make_tool_call_response("call_1", "add", {"a": 1, "b": 2}),
            make_text_response("Done"),
        ])
        registry = ToolRegistry()
        registry.register(AddTool())

        tool_calls_log = []
        tool_results_log = []
        iterations_log = []

        def on_tool_call(call: ToolCall, iteration: int) -> None:
            tool_calls_log.append((call.name, iteration))

        def on_tool_result(call: ToolCall, result: ToolResult, iteration: int) -> None:
            tool_results_log.append((call.name, result.content, iteration))

        def on_iteration(iteration: int, response: GenerateResponse) -> None:
            iterations_log.append(iteration)

        messages = [Message(role=Role.USER, content="Test")]

        await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
            loop_config=LoopConfig(
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                on_iteration=on_iteration,
            ),
        )

        assert len(tool_calls_log) == 1
        assert len(tool_results_log) == 1
        assert len(iterations_log) == 2

    @pytest.mark.asyncio
    async def test_execute_tools_false(self) -> None:
        """测试不自动执行工具。"""
        provider = MockProvider(responses=[
            make_tool_call_response("call_1", "add", {"a": 1, "b": 2}),
        ])
        registry = ToolRegistry()
        registry.register(AddTool())

        messages = [Message(role=Role.USER, content="Test")]

        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
            loop_config=LoopConfig(execute_tools=False),
        )

        assert result.status == LoopStatus.COMPLETED
        assert len(result.tool_calls) == 0  # 没有执行工具


class TestExecuteAndContinue:
    """execute_and_continue 测试。"""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        """测试成功执行。"""
        provider = MockProvider(responses=[make_text_response("Done")])
        registry = ToolRegistry()

        messages = [Message(role=Role.USER, content="Test")]

        response = await execute_and_continue(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
        )

        assert response.text == "Done"

    @pytest.mark.asyncio
    async def test_max_iterations_error(self) -> None:
        """测试超过最大迭代次数抛出错误。"""
        provider = MockProvider(responses=[
            make_tool_call_response("call_1", "add", {"a": 1, "b": 1}),
            make_tool_call_response("call_2", "add", {"a": 2, "b": 2}),
        ])
        registry = ToolRegistry()
        registry.register(AddTool())

        messages = [Message(role=Role.USER, content="Test")]

        with pytest.raises(Exception) as exc_info:
            await execute_and_continue(
                messages=messages,
                provider=provider,
                tools=registry,
                config=GenerateConfig(model="mock-model"),
                max_iterations=1,
            )

        assert "Max iterations" in str(exc_info.value)


class TestLoopResult:
    """LoopResult 测试。"""

    def test_success_property(self) -> None:
        """测试 success 属性。"""
        result = LoopResult(
            response=GenerateResponse(
                text="Done", tool_calls=[], finish_reason="stop", raw={}
            ),
            status=LoopStatus.COMPLETED,
        )

        assert result.success is True

        result.status = LoopStatus.ERROR
        assert result.success is False

        result.status = LoopStatus.MAX_ITERATIONS
        assert result.success is False


class TestLoopConfig:
    """LoopConfig 测试。"""

    def test_default_values(self) -> None:
        """测试默认值。"""
        config = LoopConfig()

        assert config.max_iterations == 10
        assert config.execute_tools is True
        assert config.on_tool_call is None
        assert config.on_tool_result is None
        assert config.on_iteration is None


class TestProviderTypeDetection:
    """提供商类型检测测试。"""

    @pytest.mark.asyncio
    async def test_anthropic_provider_type(self) -> None:
        """测试 Anthropic 提供商类型。"""
        # 创建模拟的 anthropic provider
        class AnthropicMockProvider(BaseProvider):
            name = "anthropic"

            async def generate(self, messages, config):
                return make_text_response("done")

            async def generate_stream(self, messages, config):
                yield None

            def supports_model(self, model):
                return True

            def get_model_list(self):
                return []

        provider = AnthropicMockProvider(api_key="test")
        registry = ToolRegistry()

        # 使用 bytes 内容的工具结果
        tool_call = ToolCall(id="call_1", name="test", arguments={})
        tool_result = ToolResult(
            tool_call_id="call_1",
            content=b"binary result",
        )

        # 注册一个测试工具
        class TestTool(Tool):
            name = "test"
            description = "test"

            async def execute(self, **kwargs):
                return ToolResult(tool_call_id="", content="ok")

        registry.register(TestTool())

        messages = [Message(role=Role.USER, content="test")]

        # 这个测试主要覆盖 _get_provider_type 的 anthropic 分支
        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="claude-3"),
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_google_provider_type(self) -> None:
        """测试 Google 提供商类型。"""
        class GoogleMockProvider(BaseProvider):
            name = "google"

            async def generate(self, messages, config):
                return make_text_response("done")

            async def generate_stream(self, messages, config):
                yield None

            def supports_model(self, model):
                return True

            def get_model_list(self):
                return []

        provider = GoogleMockProvider(api_key="test")
        registry = ToolRegistry()

        class TestTool(Tool):
            name = "test"
            description = "test"

            async def execute(self, **kwargs):
                return ToolResult(tool_call_id="", content={"key": "value"})

        registry.register(TestTool())

        messages = [Message(role=Role.USER, content="test")]

        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="gemini"),
        )

        assert result is not None


class TestToolResultContentTypes:
    """测试不同类型的工具结果内容。"""

    @pytest.mark.asyncio
    async def test_bytes_content(self) -> None:
        """测试字节内容。"""
        class BytesTool(Tool):
            name = "bytes_tool"
            description = "Returns bytes"

            async def execute(self, **kwargs):
                return ToolResult(tool_call_id="", content=b"binary data")

        provider = MockProvider(responses=[
            make_tool_call_response("call_1", "bytes_tool", {}),
            make_text_response("Done"),
        ])
        registry = ToolRegistry()
        registry.register(BytesTool())

        messages = [Message(role=Role.USER, content="Test")]

        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
        )

        assert result.status == LoopStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_dict_content(self) -> None:
        """测试字典内容。"""
        class DictTool(Tool):
            name = "dict_tool"
            description = "Returns dict"

            async def execute(self, **kwargs):
                return ToolResult(tool_call_id="", content={"result": "success"})

        provider = MockProvider(responses=[
            make_tool_call_response("call_1", "dict_tool", {}),
            make_text_response("Done"),
        ])
        registry = ToolRegistry()
        registry.register(DictTool())

        messages = [Message(role=Role.USER, content="Test")]

        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
        )

        assert result.status == LoopStatus.COMPLETED


class TestFailingToolExecution:
    """测试工具执行失败情况。"""

    @pytest.mark.asyncio
    async def test_tool_execution_exception(self) -> None:
        """测试工具执行抛出异常。"""
        class ExceptionTool(Tool):
            name = "exception_tool"
            description = "Throws exception"

            async def execute(self, **kwargs):
                raise RuntimeError("Tool crashed")

        provider = MockProvider(responses=[
            make_tool_call_response("call_1", "exception_tool", {}),
            make_text_response("Recovered"),
        ])
        registry = ToolRegistry()
        registry.register(ExceptionTool())

        messages = [Message(role=Role.USER, content="Test")]

        result = await run_with_tools(
            messages=messages,
            provider=provider,
            tools=registry,
            config=GenerateConfig(model="mock-model"),
        )

        # 工具执行失败应该记录错误
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0][1].is_error is True
