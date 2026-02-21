"""Tool 基类和 @tool 装饰器。"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .schema import extract_function_schema, validate_arguments
from .types import (
    ToolDefinition,
    ToolParameter,
    ToolResult,
)


class Tool(ABC):
    """工具抽象基类。

    所有工具必须实现:
    - name: 工具名称
    - description: 工具描述
    - execute: 执行方法
    """

    name: str = ""
    description: str = ""
    _definition: ToolDefinition | None = None

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行工具。

        Args:
            **kwargs: 工具参数。

        Returns:
            ToolResult: 执行结果。
        """
        pass

    def get_definition(self) -> ToolDefinition:
        """获取工具定义。

        Returns:
            ToolDefinition: 工具定义对象。
        """
        if self._definition is not None:
            return self._definition

        self._definition = ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.get_parameters(),
        )
        return self._definition

    def get_parameters(self) -> list[ToolParameter]:
        """获取参数列表(子类可覆盖)。

        Returns:
            list[ToolParameter]: 参数列表。
        """
        return []

    def validate_arguments(self, arguments: dict[str, Any]) -> tuple[bool, list[str]]:
        """验证参数。

        Args:
            arguments: 参数字典。

        Returns:
            tuple[bool, list[str]]: (是否有效, 错误列表)。
        """
        schema = self.get_definition().to_json_schema()
        return validate_arguments(arguments, schema)

    @classmethod
    def from_function(
        cls,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: list[ToolParameter] | None = None,
    ) -> FunctionTool:
        """从函数创建工具。

        Args:
            func: 目标函数。
            name: 工具名称。
            description: 工具描述。
            parameters: 参数列表(可选, 自动推断)。

        Returns:
            FunctionTool: 函数工具实例。
        """
        return FunctionTool(
            func=func,
            name=name,
            description=description,
            parameters=parameters,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        executor: Callable | None = None,
    ) -> DictTool:
        """从字典创建工具。

        Args:
            data: 工具定义字典。
            executor: 执行函数。

        Returns:
            DictTool: 字典工具实例。
        """
        return DictTool(data=data, executor=executor)


@dataclass
class FunctionTool(Tool):
    """函数工具 - 从函数创建的工具。"""

    func: Callable | None = field(default=None)
    _name: str | None = field(default=None)
    _description: str | None = field(default=None)
    _parameters: list[ToolParameter] | None = field(default=None)
    _schema: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self._name or (self.func.__name__ if self.func else "")

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def description(self) -> str:
        if self._description:
            return self._description
        if self.func and self.func.__doc__:
            # 提取第一行
            lines = self.func.__doc__.strip().split("\n")
            return lines[0].strip()
        return ""

    @description.setter
    def description(self, value: str) -> None:
        self._description = value

    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行函数。"""
        if self.func is None:
            return ToolResult(
                tool_call_id="",
                content="",
                is_error=True,
                error_message="No function attached",
            )

        try:
            # 验证参数
            valid, errors = self.validate_arguments(kwargs)
            if not valid:
                return ToolResult(
                    tool_call_id="",
                    content="",
                    is_error=True,
                    error_message=f"Invalid arguments: {'; '.join(errors)}",
                )

            # 执行函数
            result = self.func(**kwargs)

            # 处理异步函数
            if asyncio.iscoroutine(result):
                result = await result

            # 构建结果
            return ToolResult(
                tool_call_id="",
                content=(
                    result
                    if isinstance(result, (str, dict, bytes))
                    else str(result)
                ),
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content="",
                is_error=True,
                error_message=f"Execution failed: {e}",
            )

    def get_parameters(self) -> list[ToolParameter]:
        """从函数签名推断参数。"""
        if self._parameters is not None:
            return self._parameters

        if self._schema is None:
            self._schema = extract_function_schema(self.func)

        params: list[ToolParameter] = []
        func_schema = self._schema.get("function", {})
        params_schema = func_schema.get("parameters", {})
        properties = params_schema.get("properties", {})
        required = params_schema.get("required", [])

        for name, prop in properties.items():
            params.append(ToolParameter(
                name=name,
                type=prop.get("type", "string"),
                description=prop.get("description"),
                required=name in required,
                enum=prop.get("enum"),
            ))

        return params


@dataclass
class DictTool(Tool):
    """字典工具 - 从字典定义创建的工具。"""

    data: dict[str, Any] = field(default_factory=dict)
    executor: Callable | None = field(default=None)

    @property
    def name(self) -> str:
        return self.data.get("name", "")

    @name.setter
    def name(self, value: str) -> None:
        self.data["name"] = value

    @property
    def description(self) -> str:
        return self.data.get("description", "")

    @description.setter
    def description(self, value: str) -> None:
        self.data["description"] = value

    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行工具。"""
        if self.executor is None:
            return ToolResult(
                tool_call_id="",
                content="",
                is_error=True,
                error_message="No executor attached",
            )

        try:
            result = self.executor(**kwargs)

            if asyncio.iscoroutine(result):
                result = await result

            return ToolResult(
                tool_call_id="",
                content=(
                    result
                    if isinstance(result, (str, dict, bytes))
                    else str(result)
                ),
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content="",
                is_error=True,
                error_message=f"Execution failed: {e}",
            )

    def get_parameters(self) -> list[ToolParameter]:
        """从字典提取参数。"""
        params: list[ToolParameter] = []
        param_defs = self.data.get("parameters", [])

        for p in param_defs:
            if isinstance(p, ToolParameter):
                params.append(p)
            elif isinstance(p, dict):
                params.append(ToolParameter(
                    name=p.get("name", ""),
                    type=p.get("type", "string"),
                    description=p.get("description"),
                    required=p.get("required", True),
                    default=p.get("default"),
                    enum=p.get("enum"),
                ))

        return params


def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: list[ToolParameter] | None = None,
) -> Callable:
    """装饰器: 将函数转换为工具。

    可以作为装饰器使用:
        @tool
        def my_func(x: int) -> str:
            ...

    或带参数:
        @tool(name="custom_name", description="Custom tool")
        def my_func(x: int) -> str:
            ...

    Args:
        func: 目标函数。
        name: 工具名称。
        description: 工具描述。
        parameters: 参数列表。

    Returns:
        Callable: 工具对象或装饰器函数。
    """
    def decorator(fn: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            func=fn,
            _name=name,
            _description=description,
            _parameters=parameters,
        )

    if func is not None:
        # 直接 @tool 用法
        return decorator(func)

    # @tool(...) 用法
    return decorator


# 类型别名
AnyTool = Tool | FunctionTool | DictTool
"""任意工具类型。"""
