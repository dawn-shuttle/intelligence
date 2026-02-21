"""工具注册表 - 管理工具的注册、发现和查找。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .tool import FunctionTool, Tool
from .types import ToolDefinition, ToolParameter


class ToolRegistryError(Exception):
    """工具注册表错误。"""
    pass


class DuplicateToolError(ToolRegistryError):
    """工具名称重复。"""
    pass


class ToolNotFoundError(ToolRegistryError):
    """工具未找到。"""
    pass


@dataclass
class ToolRegistry:
    """工具注册表。

    管理工具的注册、发现和查找。

    Attributes:
        tools: 工具字典(name -> Tool)。
        case_sensitive: 名称是否区分大小写。
    """

    tools: dict[str, Tool] = field(default_factory=dict)
    case_sensitive: bool = False

    def _normalize_name(self, name: str) -> str:
        """规范化工具名称。"""
        return name if self.case_sensitive else name.lower()

    def register(
        self,
        tool: Tool | Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: list[ToolParameter] | None = None,
        override: bool = False,
    ) -> Tool:
        """注册工具。

        Args:
            tool: 工具对象或函数。
            name: 工具名称(函数时使用)。
            description: 工具描述(函数时使用)。
            parameters: 参数列表(函数时使用)。
            override: 是否覆盖已存在的工具。

        Returns:
            Tool: 注册的工具对象。

        Raises:
            DuplicateToolError: 工具名称已存在。
        """
        # 如果是函数, 转换为工具
        if callable(tool) and not isinstance(tool, Tool):
            tool = FunctionTool(
                func=tool,
                _name=name,
                _description=description,
                _parameters=parameters,
            )

        tool_name = self._normalize_name(tool.name)

        if tool_name in self.tools and not override:
            raise DuplicateToolError(f"Tool '{tool.name}' already registered")

        self.tools[tool_name] = tool
        return tool

    def unregister(self, name: str) -> Tool:
        """注销工具。

        Args:
            name: 工具名称。

        Returns:
            Tool: 被注销的工具。

        Raises:
            ToolNotFoundError: 工具未找到。
        """
        normalized = self._normalize_name(name)

        if normalized not in self.tools:
            raise ToolNotFoundError(f"Tool '{name}' not found")

        return self.tools.pop(normalized)

    def get(self, name: str) -> Tool:
        """获取工具。

        Args:
            name: 工具名称。

        Returns:
            Tool: 工具对象。

        Raises:
            ToolNotFoundError: 工具未找到。
        """
        normalized = self._normalize_name(name)

        if normalized not in self.tools:
            raise ToolNotFoundError(f"Tool '{name}' not found")

        return self.tools[normalized]

    def has(self, name: str) -> bool:
        """检查工具是否存在。

        Args:
            name: 工具名称。

        Returns:
            bool: 是否存在。
        """
        return self._normalize_name(name) in self.tools

    def list_tools(self) -> list[str]:
        """列出所有工具名称。

        Returns:
            list[str]: 工具名称列表。
        """
        return [tool.name for tool in self.tools.values()]

    def list_definitions(self) -> list[ToolDefinition]:
        """列出所有工具定义。

        Returns:
            list[ToolDefinition]: 工具定义列表。
        """
        return [tool.get_definition() for tool in self.tools.values()]

    def get_definitions_dict(self) -> dict[str, ToolDefinition]:
        """获取工具定义字典。

        Returns:
            dict[str, ToolDefinition]: 工具定义字典。
        """
        return {tool.name: tool.get_definition() for tool in self.tools.values()}

    def clear(self) -> None:
        """清空所有工具。"""
        self.tools.clear()

    def merge(self, other: ToolRegistry, *, override: bool = False) -> None:
        """合并另一个注册表。

        Args:
            other: 另一个注册表。
            override: 是否覆盖已存在的工具。
        """
        for tool in other.tools.values():
            self.register(tool, override=override)

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __getitem__(self, name: str) -> Tool:
        return self.get(name)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self.tools.values())

    def __repr__(self) -> str:
        tool_names = ", ".join(self.list_tools()[:5])
        if len(self.tools) > 5:
            tool_names += f", ... ({len(self.tools)} total)"
        return f"ToolRegistry([{tool_names}])"


# 全局默认注册表
_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """获取默认注册表。

    Returns:
        ToolRegistry: 默认注册表实例。
    """
    global _default_registry

    if _default_registry is None:
        _default_registry = ToolRegistry()

    return _default_registry


def register(
    tool: Tool | Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: list[ToolParameter] | None = None,
    registry: ToolRegistry | None = None,
) -> Tool:
    """注册工具到注册表。

    Args:
        tool: 工具对象或函数。
        name: 工具名称。
        description: 工具描述。
        parameters: 参数列表。
        registry: 目标注册表(默认使用全局注册表)。

    Returns:
        Tool: 注册的工具对象。
    """
    target = registry or get_default_registry()
    return target.register(
        tool,
        name=name,
        description=description,
        parameters=parameters,
    )


def get_tool(name: str, registry: ToolRegistry | None = None) -> Tool:
    """从注册表获取工具。

    Args:
        name: 工具名称。
        registry: 目标注册表(默认使用全局注册表)。

    Returns:
        Tool: 工具对象。
    """
    target = registry or get_default_registry()
    return target.get(name)


def list_tools(registry: ToolRegistry | None = None) -> list[str]:
    """列出注册表中的工具。

    Args:
        registry: 目标注册表(默认使用全局注册表)。

    Returns:
        list[str]: 工具名称列表。
    """
    target = registry or get_default_registry()
    return target.list_tools()
