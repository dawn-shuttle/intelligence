"""Skill 系统完整测试 - 使用 mock 模拟 Provider。"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.core.config import GenerateConfig
from dawn_shuttle.dawn_shuttle_intelligence.src.core.provider import BaseProvider
from dawn_shuttle.dawn_shuttle_intelligence.src.core.response import GenerateResponse
from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import Message, Role
from dawn_shuttle.dawn_shuttle_intelligence.src.tools import (
    Skill,
    SkillContext,
    Tool,
    ToolRegistry,
    ToolResult,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.tools.skill import skill
from dawn_shuttle.dawn_shuttle_intelligence.src.tools.types import (
    ToolParameter,
)


class MockProvider(BaseProvider):
    """模拟 Provider 用于测试。"""

    name = "mock"

    def __init__(self):
        super().__init__(api_key="test")
        self.generate_mock = AsyncMock(
            return_value=GenerateResponse(
                text="Generated text",
                tool_calls=[],
                finish_reason="stop",
                raw={},
            )
        )

    async def generate(
        self,
        messages: list[Message],
        config: GenerateConfig,
    ) -> GenerateResponse:
        return await self.generate_mock(messages, config)

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


class EchoTool(Tool):
    """回显工具。"""

    name = "echo"
    description = "Echo input"

    async def execute(self, message: str) -> ToolResult:
        return ToolResult(tool_call_id="", content=f"Echo: {message}")


class CounterSkill(Skill):
    """计数技能。"""

    name: ClassVar[str] = "counter"
    description: ClassVar[str] = "Count and track numbers"
    tools: ClassVar[list[Tool]] = [EchoTool()]
    parameters: ClassVar[list[ToolParameter]] = [
        ToolParameter(name="start", type="integer", description="Start number"),
        ToolParameter(name="end", type="integer", description="End number"),
    ]

    async def run(self, context: SkillContext, start: int = 0, end: int = 10) -> dict:
        results = []
        for i in range(start, end):
            result = await context.call_tool("echo", message=str(i))
            results.append(result.content)

        return {"count": end - start, "results": results}


class FailingSkill(Skill):
    """失败技能。"""

    name = "failing_skill"
    description = "A skill that fails"

    async def run(self, context: SkillContext, **kwargs) -> Any:
        raise ValueError("Skill intentionally failed")


class TestSkillContext:
    """SkillContext 完整测试。"""

    @pytest.mark.asyncio
    async def test_call_tool(self) -> None:
        """测试调用工具。"""
        registry = ToolRegistry()
        registry.register(EchoTool())

        context = SkillContext(provider=None, registry=registry)  # type: ignore

        result = await context.call_tool("echo", message="Hello")

        assert result.is_error is False
        assert result.content == "Echo: Hello"

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self) -> None:
        """测试调用不存在的工具。"""
        registry = ToolRegistry()
        context = SkillContext(provider=None, registry=registry)  # type: ignore

        result = await context.call_tool("nonexistent", arg="value")

        assert result.is_error is True
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_generate(self) -> None:
        """测试生成文本。"""
        provider = MockProvider()
        context = SkillContext(provider=provider, registry=ToolRegistry())

        text = await context.generate("Hello, AI!")

        assert text == "Generated text"
        provider.generate_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_system(self) -> None:
        """测试带系统提示生成。"""
        provider = MockProvider()
        context = SkillContext(provider=provider, registry=ToolRegistry())

        text = await context.generate("Hello", system="You are helpful")

        assert text == "Generated text"

    @pytest.mark.asyncio
    async def test_generate_with_tools(self) -> None:
        """测试带工具生成。"""
        provider = MockProvider()
        provider.generate_mock.return_value = GenerateResponse(
            text="Used tool",
            tool_calls=[],
            finish_reason="stop",
            raw={},
        )

        registry = ToolRegistry()
        registry.register(EchoTool())

        context = SkillContext(provider=provider, registry=registry)

        response = await context.generate_with_tools(
            "Test",
            model="mock-model",
            tools=[EchoTool()],
        )

        assert response.text == "Used tool"

    def test_state_operations(self) -> None:
        """测试状态操作。"""
        context = SkillContext(provider=None, registry=ToolRegistry())  # type: ignore

        context.set_state("counter", 0)
        context.set_state("data", {"key": "value"})

        assert context.get_state("counter") == 0
        assert context.get_state("data") == {"key": "value"}
        assert context.get_state("missing") is None
        assert context.get_state("missing", "default") == "default"

        # 修改状态
        context.set_state("counter", 1)
        assert context.get_state("counter") == 1

    def test_add_message(self) -> None:
        """测试添加消息。"""
        context = SkillContext(provider=None, registry=ToolRegistry())  # type: ignore

        msg1 = Message(role=Role.USER, content="Hello")
        msg2 = Message(role=Role.ASSISTANT, content="Hi")

        context.add_message(msg1)
        context.add_message(msg2)

        assert len(context.messages) == 2
        assert context.messages[0].content == "Hello"
        assert context.messages[1].content == "Hi"


class TestSkill:
    """Skill 完整测试。"""

    @pytest.mark.asyncio
    async def test_run_skill(self) -> None:
        """测试运行技能。"""
        registry = ToolRegistry()
        registry.register(EchoTool())

        context = SkillContext(provider=None, registry=registry)  # type: ignore
        skill_instance = CounterSkill()

        result = await skill_instance.run(context, start=0, end=3)

        assert result["count"] == 3
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_skill_error(self) -> None:
        """测试技能错误。"""
        context = SkillContext(provider=None, registry=ToolRegistry())  # type: ignore
        skill_instance = FailingSkill()

        with pytest.raises(ValueError) as exc_info:
            await skill_instance.run(context)

        assert "intentionally failed" in str(exc_info.value)

    def test_to_tool(self) -> None:
        """测试转换为工具。"""
        skill_instance = CounterSkill()
        tool = skill_instance.to_tool()

        assert tool.name == "counter"
        assert tool.description == "Count and track numbers"

    def test_get_parameters(self) -> None:
        """测试获取参数。"""
        skill_instance = CounterSkill()
        params = skill_instance.get_parameters()

        assert len(params) == 2
        assert params[0].name == "start"
        assert params[1].name == "end"


class TestSkillToolWrapper:
    """SkillToolWrapper 测试。"""

    @pytest.mark.asyncio
    async def test_execute_skill(self) -> None:
        """测试执行技能包装器。"""
        registry = ToolRegistry()
        registry.register(EchoTool())

        context = SkillContext(provider=None, registry=registry)  # type: ignore
        skill_instance = CounterSkill()
        tool = skill_instance.to_tool()

        # 直接调用 execute
        result = await tool.execute(_context=context, start=0, end=2)

        assert result.is_error is False
        assert "count" in result.content

    @pytest.mark.asyncio
    async def test_execute_no_context(self) -> None:
        """测试无上下文执行。"""
        skill_instance = CounterSkill()
        tool = skill_instance.to_tool()

        result = await tool.execute(start=0, end=2)

        assert result.is_error is True
        assert "Context required" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_no_skill(self) -> None:
        """测试无技能执行。"""
        from dawn_shuttle.dawn_shuttle_intelligence.src.tools.skill.base import (
            SkillToolWrapper,
        )

        wrapper = SkillToolWrapper(skill=None)
        result = await wrapper.execute()

        assert result.is_error is True
        assert "not attached" in result.error_message.lower()


class TestSkillDecorator:
    """@skill 装饰器完整测试。"""

    def test_basic(self) -> None:
        """测试基本装饰器。"""
        @skill(name="test_skill", description="Test skill")
        async def my_skill(context: SkillContext, value: str) -> str:
            return f"Processed: {value}"

        assert my_skill.name == "test_skill"
        assert my_skill.description == "Test skill"

    def test_auto_name_from_function(self) -> None:
        """测试自动从函数获取名称。"""
        @skill()
        async def auto_named_skill(context: SkillContext) -> str:
            """Auto named skill."""
            return "done"

        assert auto_named_skill.name == "auto_named_skill"

    def test_auto_description_from_docstring(self) -> None:
        """测试自动从 docstring 获取描述。"""
        @skill()
        async def docstring_skill(context: SkillContext) -> str:
            """This is the description."""
            return "done"

        assert docstring_skill.description == "This is the description."

    @pytest.mark.asyncio
    async def test_run_decorated_skill(self) -> None:
        """测试运行装饰的技能。"""
        @skill(name="echo_skill", description="Echo skill")
        async def echo_skill(context: SkillContext, message: str) -> str:
            return f"Echoed: {message}"

        context = SkillContext(provider=None, registry=ToolRegistry())  # type: ignore

        result = await echo_skill.run(context, message="Hello")

        assert result == "Echoed: Hello"

    @pytest.mark.asyncio
    async def test_skill_with_tools(self) -> None:
        """测试带工具的技能。"""
        @skill(
            name="tool_user",
            description="Uses tools",
            tools=[EchoTool()],
        )
        async def tool_user_skill(context: SkillContext, msg: str) -> str:
            result = await context.call_tool("echo", message=msg)
            return result.content

        registry = ToolRegistry()
        registry.register(EchoTool())
        context = SkillContext(provider=None, registry=registry)

        result = await tool_user_skill.run(context, msg="Test")

        assert result == "Echo: Test"
