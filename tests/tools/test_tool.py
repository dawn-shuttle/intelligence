"""Tool 和 Registry 测试。"""

from __future__ import annotations

import asyncio

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.tools import (
    DictTool,
    DuplicateToolError,
    FunctionTool,
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


class TestFunctionTool:
    """FunctionTool 测试。"""

    @pytest.mark.asyncio
    async def test_execute_sync_function(self) -> None:
        """测试执行同步函数。"""
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool_instance = FunctionTool(func=add)
        result = await tool_instance.execute(a=1, b=2)

        assert result.is_error is False
        assert result.content == "3"

    @pytest.mark.asyncio
    async def test_execute_async_function(self) -> None:
        """测试执行异步函数。"""
        async def async_add(a: int, b: int) -> int:
            """Async add."""
            await asyncio.sleep(0.01)
            return a + b

        tool_instance = FunctionTool(func=async_add)
        result = await tool_instance.execute(a=1, b=2)

        assert result.is_error is False
        assert result.content == "3"

    @pytest.mark.asyncio
    async def test_execute_with_error(self) -> None:
        """测试执行出错。"""
        def failing_func() -> str:
            """Fail."""
            raise ValueError("Test error")

        tool_instance = FunctionTool(func=failing_func)
        result = await tool_instance.execute()

        assert result.is_error is True
        assert "Test error" in result.error_message

    def test_from_function(self) -> None:
        """测试从函数创建。"""
        def my_func(x: str) -> str:
            """My function."""
            return x

        tool_instance = Tool.from_function(my_func, name="my_tool")

        assert tool_instance.name == "my_tool"


class TestDictTool:
    """DictTool 测试。"""

    def test_create(self) -> None:
        """测试创建。"""
        tool_instance = DictTool(
            data={
                "name": "dict_tool",
                "description": "A dict tool",
                "parameters": [
                    {"name": "input", "type": "string", "description": "Input value"},
                ],
            }
        )

        assert tool_instance.name == "dict_tool"
        assert tool_instance.description == "A dict tool"

    @pytest.mark.asyncio
    async def test_execute(self) -> None:
        """测试执行。"""
        def executor(x: int) -> int:
            return x * 2

        tool_instance = DictTool(
            data={"name": "double", "description": "Double a number"},
            executor=executor,
        )

        result = await tool_instance.execute(x=5)

        assert result.is_error is False
        assert result.content == "10"  # 结果会被转为字符串

    @pytest.mark.asyncio
    async def test_execute_no_executor(self) -> None:
        """测试无执行器。"""
        tool_instance = DictTool(data={"name": "test", "description": ""})

        result = await tool_instance.execute()

        assert result.is_error is True
        assert "No executor" in result.error_message

    def test_from_dict(self) -> None:
        """测试从字典创建。"""
        tool_instance = Tool.from_dict({
            "name": "from_dict",
            "description": "Created from dict",
        })

        assert tool_instance.name == "from_dict"

    def test_get_parameters(self) -> None:
        """测试获取参数。"""
        tool_instance = DictTool(
            data={
                "name": "test",
                "description": "",
                "parameters": [
                    ToolParameter(name="arg1", type="string"),
                    {"name": "arg2", "type": "integer", "required": False},
                ],
            }
        )

        params = tool_instance.get_parameters()

        assert len(params) == 2
        assert params[0].name == "arg1"
        assert params[1].name == "arg2"


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

    def test_override_tool(self) -> None:
        """测试覆盖注册。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())

        # 使用 override=True 应该成功
        new_tool = SimpleTool()
        registry.register(new_tool, override=True)

        assert registry.has("simple")

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

    def test_get_definitions_dict(self) -> None:
        """测试获取定义字典。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())

        definitions = registry.get_definitions_dict()

        assert "simple" in definitions

    def test_clear(self) -> None:
        """测试清空。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())
        registry.clear()

        assert len(registry) == 0

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

    def test_repr(self) -> None:
        """测试字符串表示。"""
        registry = ToolRegistry()
        registry.register(SimpleTool())

        repr_str = repr(registry)

        assert "ToolRegistry" in repr_str
