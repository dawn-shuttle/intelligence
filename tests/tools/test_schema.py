"""Schema 工具测试。"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.tools.schema import (
    extract_function_schema,
    python_type_to_json_schema,
    validate_arguments,
)


class TestPythonTypeToJsonSchema:
    """python_type_to_json_schema 测试。"""

    def test_basic_types(self) -> None:
        """测试基础类型。"""
        assert python_type_to_json_schema(str) == {"type": "string"}
        assert python_type_to_json_schema(int) == {"type": "integer"}
        assert python_type_to_json_schema(float) == {"type": "number"}
        assert python_type_to_json_schema(bool) == {"type": "boolean"}
        assert python_type_to_json_schema(list) == {"type": "array"}
        assert python_type_to_json_schema(dict) == {"type": "object"}

    def test_none_type(self) -> None:
        """测试 None 类型。"""
        assert python_type_to_json_schema(None) == {"type": "null"}
        assert python_type_to_json_schema(type(None)) == {"type": "null"}

    def test_literal_type(self) -> None:
        """测试 Literal 类型。"""
        schema = python_type_to_json_schema(Literal["a", "b", "c"])
        assert schema["type"] == "string"
        assert schema["enum"] == ["a", "b", "c"]

        schema = python_type_to_json_schema(Literal[1, 2, 3])
        assert schema["type"] == "integer"
        assert schema["enum"] == [1, 2, 3]

    def test_optional_type(self) -> None:
        """测试 Optional 类型。"""
        schema = python_type_to_json_schema(Optional[str])
        assert schema["type"] == "string"
        assert schema["nullable"] is True

    def test_list_type(self) -> None:
        """测试 List 类型。"""
        schema = python_type_to_json_schema(list[str])
        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

        schema = python_type_to_json_schema(list[int])
        assert schema["type"] == "array"
        assert schema["items"] == {"type": "integer"}

    def test_dict_type(self) -> None:
        """测试 Dict 类型。"""
        schema = python_type_to_json_schema(dict[str, int])
        assert schema["type"] == "object"
        assert schema["additionalProperties"] == {"type": "integer"}

    def test_annotated_type(self) -> None:
        """测试 Annotated 类型。"""
        schema = python_type_to_json_schema(
            Annotated[str, "这是一个描述"]
        )
        assert schema["type"] == "string"
        assert schema["description"] == "这是一个描述"

    def test_unknown_type(self) -> None:
        """测试未知类型。"""
        schema = python_type_to_json_schema(object)
        assert schema["type"] == "object"


class TestExtractFunctionSchema:
    """extract_function_schema 测试。"""

    def test_simple_function(self) -> None:
        """测试简单函数。"""
        def greet(name: str) -> str:
            """问候函数。"""
            return f"Hello, {name}"

        schema = extract_function_schema(greet)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "greet"
        assert "问候" in schema["function"]["description"]

    def test_function_with_default(self) -> None:
        """测试带默认值的函数。"""
        def search(query: str, limit: int = 10) -> list:
            """搜索函数。"""
            return []

        schema = extract_function_schema(search)

        params = schema["function"]["parameters"]
        assert "query" in params["properties"]
        assert "limit" in params["properties"]
        assert params["required"] == ["query"]

    def test_function_with_docstring(self) -> None:
        """测试带 docstring 的函数。"""
        def calculate(a: int, b: int) -> int:
            """计算两个数的和。

            Args:
                a: 第一个数
                b: 第二个数

            Returns:
                两数之和
            """
            return a + b

        schema = extract_function_schema(calculate)

        assert schema["function"]["description"] == "计算两个数的和。"

    def test_function_with_custom_name(self) -> None:
        """测试自定义名称。"""
        def func() -> None:
            pass

        schema = extract_function_schema(func, name="custom_name")

        assert schema["function"]["name"] == "custom_name"

    def test_function_with_custom_description(self) -> None:
        """测试自定义描述。"""
        def func() -> None:
            pass

        schema = extract_function_schema(func, description="自定义描述")

        assert schema["function"]["description"] == "自定义描述"


class TestValidateArguments:
    """validate_arguments 测试。"""

    def test_valid_string(self) -> None:
        """测试有效的字符串。"""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        valid, errors = validate_arguments({"name": "test"}, schema)

        assert valid is True
        assert errors == []

    def test_valid_integer(self) -> None:
        """测试有效的整数。"""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }

        valid, errors = validate_arguments({"count": 42}, schema)

        assert valid is True

    def test_missing_required(self) -> None:
        """测试缺少必需参数。"""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        valid, errors = validate_arguments({}, schema)

        assert valid is False
        assert any("Missing required" in e for e in errors)

    def test_invalid_type(self) -> None:
        """测试无效类型。"""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }

        valid, errors = validate_arguments({"count": "not a number"}, schema)

        assert valid is False

    def test_valid_enum(self) -> None:
        """测试有效的枚举值。"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]},
            },
        }

        valid, errors = validate_arguments({"status": "active"}, schema)

        assert valid is True

    def test_invalid_enum(self) -> None:
        """测试无效的枚举值。"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]},
            },
        }

        valid, errors = validate_arguments({"status": "unknown"}, schema)

        assert valid is False

    def test_nullable(self) -> None:
        """测试可空值。"""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "string", "nullable": True},
            },
        }

        valid, errors = validate_arguments({"value": None}, schema)

        assert valid is True

    def test_array_type(self) -> None:
        """测试数组类型。"""
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array"}},
        }

        valid, _ = validate_arguments({"items": [1, 2, 3]}, schema)
        assert valid is True

        valid, _ = validate_arguments({"items": "not array"}, schema)
        assert valid is False

    def test_object_type(self) -> None:
        """测试对象类型。"""
        schema = {
            "type": "object",
            "properties": {"data": {"type": "object"}},
        }

        valid, _ = validate_arguments({"data": {"key": "value"}}, schema)
        assert valid is True
