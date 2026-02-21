"""Skill 系统 - 工具的组合与编排。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ...core.provider import BaseProvider
from ...core.response import GenerateResponse
from ...core.types import Message, Role
from ..executor import ToolExecutor
from ..registry import ToolRegistry
from ..tool import Tool
from ..types import ToolCall, ToolParameter, ToolResult


class SkillError(Exception):
    """Skill 错误。"""
    pass


@dataclass
class SkillContext:
    """Skill 执行上下文。

    提供 Skill 执行时需要的所有依赖和能力。

    Attributes:
        provider: AI 提供商。
        registry: 工具注册表。
        executor: 工具执行器。
        messages: 消息历史。
        state: 状态存储。
    """

    provider: BaseProvider
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    executor: ToolExecutor | None = None
    messages: list[Message] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    async def call_tool(
        self,
        name: str,
        **kwargs: Any,
    ) -> ToolResult:
        """调用工具。

        Args:
            name: 工具名称。
            **kwargs: 工具参数。

        Returns:
            ToolResult: 执行结果。

        Raises:
            SkillError: 工具调用失败。
        """
        if self.executor is None:
            self.executor = ToolExecutor(registry=self.registry)

        tool_call = ToolCall(
            id=f"skill_{name}",
            name=name,
            arguments=kwargs,
        )

        execution = await self.executor.execute(tool_call)
        if execution.result is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                content="",
                is_error=True,
                error_message="Tool execution returned no result",
            )
        return execution.result

    async def generate(
        self,
        prompt: str,
        *,
        model: str = "",
        system: str | None = None,
    ) -> str:
        """生成文本。

        Args:
            prompt: 用户提示。
            model: 模型名称。
            system: 系统提示。

        Returns:
            str: 生成的文本。

        Raises:
            SkillError: 生成失败。
        """
        from ...core.config import GenerateConfig

        messages = []

        if system:
            messages.append(Message(role=Role.SYSTEM, content=system))

        messages.append(Message(role=Role.USER, content=prompt))

        config = GenerateConfig(model=model)

        try:
            response = await self.provider.generate(messages, config)
            return response.text
        except Exception as e:
            raise SkillError(f"Generation failed: {e}") from e

    async def generate_with_tools(
        self,
        prompt: str,
        *,
        model: str = "",
        system: str | None = None,
        tools: list[Tool] | None = None,
    ) -> GenerateResponse:
        """使用工具生成文本。

        Args:
            prompt: 用户提示。
            model: 模型名称。
            system: 系统提示。
            tools: 可用工具列表。

        Returns:
            GenerateResponse: 生成响应。

        Raises:
            SkillError: 生成失败。
        """
        from ...core.config import GenerateConfig
        from ..loop import run_with_tools

        messages = []

        if system:
            messages.append(Message(role=Role.SYSTEM, content=system))

        messages.append(Message(role=Role.USER, content=prompt))

        config = GenerateConfig(model=model)

        if tools:
            result = await run_with_tools(
                messages=messages,
                provider=self.provider,
                tools=tools,
                config=config,
            )

            if result.response is None:
                raise SkillError("No response generated")

            return result.response

        response = await self.provider.generate(messages, config)
        return response

    def set_state(self, key: str, value: Any) -> None:
        """设置状态。"""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态。"""
        return self.state.get(key, default)

    def add_message(self, message: Message) -> None:
        """添加消息到历史。"""
        self.messages.append(message)


class Skill(ABC):
    """技能抽象基类。

    技能是一组工具的组合与编排, 可以实现复杂的多步骤任务。

    子类需要实现:
    - name: 技能名称
    - description: 技能描述
    - run: 执行方法

    可选:
    - tools: 使用的工具列表
    - parameters: 技能参数
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    tools: ClassVar[list[Tool]] = []
    parameters: ClassVar[list[ToolParameter]] = []

    @abstractmethod
    async def run(self, context: SkillContext, **kwargs: Any) -> Any:
        """执行技能。

        Args:
            context: 执行上下文。
            **kwargs: 技能参数。

        Returns:
            Any: 执行结果。
        """
        pass

    def to_tool(self) -> Tool:
        """将技能转换为工具。

        Returns:
            Tool: 工具对象。
        """
        return SkillToolWrapper(skill=self)

    def get_parameters(self) -> list[ToolParameter]:
        """获取参数列表。"""
        return self.parameters.copy()


@dataclass
class SkillToolWrapper(Tool):
    """Skill 工具包装器 - 将 Skill 包装为 Tool 对象。"""

    skill: Skill | None = field(default=None)

    @property
    def name(self) -> str:
        return self.skill.name if self.skill else ""

    @name.setter
    def name(self, value: str) -> None:
        # Skill.name 是类变量，不能通过实例设置
        pass

    @property
    def description(self) -> str:
        return self.skill.description if self.skill else ""

    @description.setter
    def description(self, value: str) -> None:
        # Skill.description 是类变量，不能通过实例设置
        pass

    async def execute(
        self,
        _context: SkillContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """执行技能。"""
        if self.skill is None:
            return ToolResult(
                tool_call_id="",
                content="",
                is_error=True,
                error_message="Skill not attached",
            )

        if _context is None:
            return ToolResult(
                tool_call_id="",
                content="",
                is_error=True,
                error_message="SkillContext required",
            )

        try:
            result = await self.skill.run(_context, **kwargs)

            if isinstance(result, ToolResult):
                return result

            content = result if isinstance(result, (str, dict, bytes)) else str(result)

            return ToolResult(
                tool_call_id="",
                content=content,
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content="",
                is_error=True,
                error_message=f"Skill execution failed: {e}",
            )

    def get_parameters(self) -> list[ToolParameter]:
        """获取参数列表。"""
        if self.skill is None:
            return []
        return self.skill.get_parameters()


def skill(
    name: str | None = None,
    description: str | None = None,
    tools: list[Tool] | None = None,
    parameters: list[ToolParameter] | None = None,
) -> Callable[[Callable[..., Any]], Skill]:
    """装饰器: 将函数转换为技能。

    Args:
        name: 技能名称。
        description: 技能描述。
        tools: 工具列表。
        parameters: 参数列表。

    Returns:
        Callable: 装饰器函数。

    Example:
        @skill(name="research", description="研究主题")
        async def research_skill(context: SkillContext, topic: str) -> str:
            results = await context.call_tool("search", query=topic)
            return results.content
    """
    def decorator(func: Callable[..., Any]) -> Skill:
        # 闭包捕获外部变量
        _skill_name = name or func.__name__
        _skill_description = description or func.__doc__ or ""
        _skill_tools = tools or []
        _skill_parameters = parameters or []

        @dataclass
        class FunctionSkill(Skill):
            _func: Callable[..., Any] = field(default=func, repr=False)

            @property
            def name(self) -> str:  # type: ignore
                return _skill_name

            @property
            def description(self) -> str:  # type: ignore
                return _skill_description

            @property
            def tools(self) -> list[Tool]:  # type: ignore
                return _skill_tools

            async def run(self, context: SkillContext, **kwargs: Any) -> Any:
                return await self._func(context, **kwargs)

        return FunctionSkill()

    return decorator
