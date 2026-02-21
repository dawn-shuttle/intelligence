"""JSON Schema 工具函数 - 从 Python 类型注解生成 Schema。"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


def python_type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """将 Python 类型转换为 JSON Schema。

    Args:
        python_type: Python 类型注解。

    Returns:
        dict[str, Any]: JSON Schema 字典。
    """
    # 处理 None
    if python_type is None or python_type is type(None):
        return {"type": "null"}

    # 处理基础类型
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    if python_type in type_mapping:
        return {"type": type_mapping[python_type]}

    # 处理 Literal
    origin = get_origin(python_type)
    if origin is Literal:
        args = get_args(python_type)
        return {
            "type": _infer_literal_type(args),
            "enum": list(args),
        }

    # 处理 Union (包括 Optional) 和 types.UnionType (Python 3.10+ 的 X | Y 语法)
    if origin is Union or origin is types.UnionType:
        args = get_args(python_type)
        # Optional[T] = Union[T, None]
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            # Optional 情况
            schema = python_type_to_json_schema(non_none_args[0])
            schema["nullable"] = True
            return schema
        else:
            # 多类型 Union
            return {
                "anyOf": [
                    python_type_to_json_schema(arg)
                    for arg in non_none_args
                ]
            }

    # 处理 list[T]
    if origin is list:
        args = get_args(python_type)
        if args:
            return {
                "type": "array",
                "items": python_type_to_json_schema(args[0]),
            }
        return {"type": "array"}

    # 处理 dict[K, V]
    if origin is dict:
        args = get_args(python_type)
        if len(args) == 2:
            return {
                "type": "object",
                "additionalProperties": python_type_to_json_schema(args[1]),
            }
        return {"type": "object"}

    # 处理 Annotated[type, description]
    if origin is Annotated:
        args = get_args(python_type)
        base_type = args[0]
        schema = python_type_to_json_schema(base_type)

        # 提取描述
        for annotation in args[1:]:
            if isinstance(annotation, str):
                schema["description"] = annotation
            elif isinstance(annotation, dict):
                schema.update(annotation)

        return schema

    # 默认返回 object
    return {"type": "object"}


def _infer_literal_type(args: tuple[Any, ...]) -> str:
    """推断 Literal 的类型。"""
    if not args:
        return "string"

    first_type = type(args[0])

    if first_type is str:
        return "string"
    elif first_type is int:
        return "integer"
    elif first_type is float:
        return "number"
    elif first_type is bool:
        return "boolean"

    return "string"


def extract_function_schema(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """从函数提取参数 Schema。

    Args:
        func: 目标函数。
        name: 工具名称(默认使用函数名)。
        description: 工具描述(默认使用函数 docstring)。

    Returns:
        dict[str, Any]: OpenAI tools 格式的工具定义。
    """
    # 获取函数名和描述
    tool_name = name or func.__name__
    tool_description = description or _extract_docstring(func)

    # 获取类型提示
    hints = get_type_hints(func, include_extras=True)
    sig = inspect.signature(func)

    # 构建参数 Schema
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # 跳过 self, cls, *args, **kwargs
        if param_name in ("self", "cls"):
            continue
        var_kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        if param.kind in var_kinds:
            continue

        # 获取参数类型
        param_type = hints.get(param_name, Any)
        if param_type is Any:
            param_type = str  # 默认为 string

        # 构建 Schema
        param_schema = python_type_to_json_schema(param_type)

        # 从 Annotated 提取描述
        origin = get_origin(param_type)
        if origin is Annotated:
            args = get_args(param_type)
            for annotation in args[1:]:
                if isinstance(annotation, str):
                    param_schema["description"] = annotation

        properties[param_name] = param_schema

        # 检查是否必需
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    # 构建最终 Schema
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }

    if required:
        parameters_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": parameters_schema,
        },
    }


def _extract_docstring(func: Callable[..., Any]) -> str:
    """提取函数 docstring 作为描述。"""
    doc = func.__doc__
    if not doc:
        return ""

    # 清理 docstring
    lines = doc.strip().split("\n")

    # 提取第一段作为描述(到第一个空行或 Args: 为止)
    description_lines = []
    for line in lines:
        stripped = line.strip()
        skip_prefixes = ("Args:", "Returns:", "Raises:", "Example:", "Note:")
        if stripped and not stripped.startswith(skip_prefixes):
            description_lines.append(stripped)
        elif description_lines:
            break

    return " ".join(description_lines)


def validate_arguments(
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[bool, list[str]]:
    """验证参数是否符合 Schema。

    Args:
        arguments: 待验证的参数字典。
        schema: JSON Schema。

    Returns:
        tuple[bool, list[str]]: (是否有效, 错误消息列表)。
    """
    errors: list[str] = []

    # 检查必需参数
    required = schema.get("required", [])
    for req in required:
        if req not in arguments:
            errors.append(f"Missing required parameter: {req}")

    # 检查参数类型
    properties = schema.get("properties", {})
    for key, value in arguments.items():
        if key not in properties:
            continue

        prop_schema = properties[key]
        expected_type = prop_schema.get("type")

        if not _validate_type(value, expected_type, prop_schema):
            errors.append(
                f"Parameter '{key}' has invalid type, expected {expected_type}"
            )

    return len(errors) == 0, errors


def _validate_type(
    value: Any, expected_type: str | None, schema: dict[str, Any]
) -> bool:
    """验证单个值的类型。"""
    if expected_type is None:
        return True

    type_validators = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "null": lambda v: v is None,
    }

    validator = type_validators.get(expected_type)
    if validator is None:
        return True

    # 检查 nullable
    if schema.get("nullable") and value is None:
        return True

    # 检查 enum
    if "enum" in schema and value not in schema["enum"]:
        return False

    return validator(value)
