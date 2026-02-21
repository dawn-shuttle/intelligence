"""Tool 和 Registry 测试。"""

from __future__ import annotations

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.tools import (
    DuplicateToolError,
    Tool,
    ToolNotFoundError,
    ToolRegistry,
    tool,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.tools.types import ToolParameter


# 定义一个简单的工具函数
@tool
def get_weather(city: str) -> str:
    """获取城市天气。

    Args:
        city: 城市名称
    """
    return f"Weather in {city}: Sunny"


@tool(name="custom_search", description="自定义搜索工具")
def search(query: str, limit: int = 10) -> str:
    """搜索内容。"""
    return f"Results for: {query}"


class SimpleTool(Tool):
    """简单工具用于测试。"""

    name = "simple"
    description = "A simple tool"

    async def execute(self, **kwargs):
        from dawn_shuttle.dawn_shuttle_intelligence.src.tools.types import ToolResult

        return ToolResult(
            tool_call_id="",
            content=f"Executed with: {kwargs}",
        )


class TestToolDecorator:
    """@tool 装饰器测试。"""

    def test_basic_decorator(self) -> None:
        """测试基本装饰器。"""
        assert get_weather.name == "get_weather"
        assert "天气" in get_weather.description or "weather" in get_weather.description.lower()

    def test_decorator_with_options(self) -> None:
        """测试带选项的装饰器。"""
        assert search.name == "custom_search"
        assert search.description == "自定义搜索工具"

    def test_get_definition(self) -> None:
        """测试获取工具定义。"""
        definition = get_weather.get_definition()

        assert definition.name == "get_weather"
        assert len(definition.parameters) >= 1


class TestToolRegistry:
    """ToolRegistry 测试。"""

    def test_register_tool(self) -> None:
        """测试注册工具。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())

        assert registry.has("simple")
        assert len(registry) == 1

    def test_register_function(self) -> None:
        """测试注册函数。"""
        registry = ToolRegistry()

        def my_func(x: int) -> str:
            """测试函数。"""
            return str(x)

        tool = registry.register(my_func, name="my_tool")

        assert tool.name == "my_tool"
        assert registry.has("my_tool")

    def test_duplicate_tool(self) -> None:
        """测试重复注册。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())

        with pytest.raises(DuplicateToolError):
            registry.register(SimpleTool())

    def test_unregister(self) -> None:
        """测试注销工具。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())

        removed = registry.unregister("simple")

        assert removed.name == "simple"
        assert not registry.has("simple")

    def test_unregister_not_found(self) -> None:
        """测试注销不存在的工具。"""
        registry = ToolRegistry()

        with pytest.raises(ToolNotFoundError):
            registry.unregister("nonexistent")

    def test_get_tool(self) -> None:
        """测试获取工具。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())

        tool = registry.get("simple")

        assert tool.name == "simple"

    def test_list_tools(self) -> None:
        """测试列出工具。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())

        tools = registry.list_tools()

        assert "simple" in tools

    def test_list_definitions(self) -> None:
        """测试列出工具定义。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())

        definitions = registry.list_definitions()

        assert len(definitions) == 1
        assert definitions[0].name == "simple"

    def test_case_insensitive(self) -> None:
        """测试不区分大小写。"""
        registry = ToolRegistry(case_sensitive=False)
        registry.register(SimpleTool())

        assert registry.has("Simple")
        assert registry.has("SIMPLE")
        assert registry.has("simple")

    def test_merge(self) -> None:
        """测试合并注册表。"""
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()

        registry1.register(SimpleTool())

        def another_tool() -> str:
            """Another tool."""
            return "done"

        registry2.register(another_tool, name="another")

        registry1.merge(registry2)

        assert registry1.has("simple")
        assert registry1.has("another")

    def test_contains(self) -> None:
        """测试包含检查。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())

        assert "simple" in registry
        assert "nonexistent" not in registry

    def test_iteration(self) -> None:
        """测试迭代。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())

        tools = list(registry)

        assert len(tools) == 1
        assert tools[0].name == "simple"
