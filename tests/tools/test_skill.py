"""Skill 系统测试。"""

from __future__ import annotations

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.tools import (
    Skill,
    SkillContext,
    SkillError,
    Tool,
    ToolRegistry,
    ToolResult,
    tool,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.tools.skill import skill as skill_decorator


class HelloTool(Tool):
    """问候工具。"""

    name = "hello"
    description = "Say hello"

    async def execute(self, **kwargs):
        return ToolResult(
            tool_call_id="",
            content=f"Hello, {kwargs.get('name', 'World')}!",
        )


class TestSkill:
    """Skill 测试。"""

    def test_skill_definition(self) -> None:
        """测试技能定义。"""
        class MySkill(Skill):
            name = "my_skill"
            description = "A test skill"

            async def run(self, context: SkillContext, **kwargs):
                return "done"

        assert MySkill.name == "my_skill"
        assert MySkill.description == "A test skill"

    def test_skill_to_tool(self) -> None:
        """测试技能转换为工具。"""
        class SimpleSkill(Skill):
            name = "simple"
            description = "Simple skill"

            async def run(self, context: SkillContext, **kwargs):
                return "result"

        skill_instance = SimpleSkill()
        tool = skill_instance.to_tool()

        assert tool.name == "simple"
        assert tool.description == "Simple skill"


class TestSkillContext:
    """SkillContext 测试。"""

    def test_create_context(self) -> None:
        """测试创建上下文。"""
        registry = ToolRegistry()
        context = SkillContext(provider=None, registry=registry)  # type: ignore

        assert context.registry is registry
        assert context.messages == []
        assert context.state == {}

    def test_state_operations(self) -> None:
        """测试状态操作。"""
        context = SkillContext(provider=None, registry=ToolRegistry())  # type: ignore

        context.set_state("key1", "value1")
        context.set_state("key2", {"nested": "data"})

        assert context.get_state("key1") == "value1"
        assert context.get_state("key2") == {"nested": "data"}
        assert context.get_state("nonexistent") is None
        assert context.get_state("nonexistent", "default") == "default"

    def test_add_message(self) -> None:
        """测试添加消息。"""
        from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import Message, Role

        context = SkillContext(provider=None, registry=ToolRegistry())  # type: ignore
        msg = Message(role=Role.USER, content="Hello")

        context.add_message(msg)

        assert len(context.messages) == 1
        assert context.messages[0].content == "Hello"


class TestSkillDecorator:
    """@skill 装饰器测试。"""

    def test_basic_decorator(self) -> None:
        """测试基本装饰器。"""
        @skill_decorator(name="test_skill", description="Test skill")
        async def my_skill(context: SkillContext, value: str) -> str:
            return f"Processed: {value}"

        assert my_skill.name == "test_skill"
        assert my_skill.description == "Test skill"

    def test_skill_execution(self) -> None:
        """测试技能执行。"""
        registry = ToolRegistry()
        registry.register(HelloTool())

        @skill_decorator(name="greet")
        async def greet_skill(context: SkillContext, name: str) -> str:
            result = await context.call_tool("hello", name=name)
            return result.content

        # 技能可以转换为工具
        tool = greet_skill.to_tool()
        assert tool.name == "greet"
