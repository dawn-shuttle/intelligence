"""格式转换器 - 工具定义和结果在各提供商格式间转换。"""

from __future__ import annotations

from typing import Any, Literal

from .types import (
    ToolCall,
    ToolDefinition,
    ToolResult,
    content_to_string,
)

ProviderType = Literal["openai", "anthropic", "google"]
"""提供商类型。"""


class ToolConverter:
    """工具格式转换器。

    负责将统一的工具定义/调用/结果转换为各提供商的格式。
    """

    @staticmethod
    def definition_to_provider(
        definition: ToolDefinition,
        provider: ProviderType,
    ) -> dict[str, Any]:
        """将工具定义转换为提供商格式。

        Args:
            definition: 统一工具定义。
            provider: 目标提供商。

        Returns:
            dict[str, Any]: 提供商格式的工具定义。
        """
        converters = {
            "openai": ToolConverter._definition_to_openai,
            "anthropic": ToolConverter._definition_to_anthropic,
            "google": ToolConverter._definition_to_google,
        }

        converter = converters.get(provider)
        if converter is None:
            raise ValueError(f"Unknown provider: {provider}")

        return converter(definition)

    @staticmethod
    def definitions_to_provider(
        definitions: list[ToolDefinition],
        provider: ProviderType,
    ) -> list[dict[str, Any]]:
        """批量转换工具定义。

        Args:
            definitions: 工具定义列表。
            provider: 目标提供商。

        Returns:
            list[dict[str, Any]]: 提供商格式的工具定义列表。
        """
        return [
            ToolConverter.definition_to_provider(d, provider)
            for d in definitions
        ]

    @staticmethod
    def call_from_provider(
        data: Any,
        provider: ProviderType,
    ) -> ToolCall:
        """从提供商格式解析工具调用。

        Args:
            data: 提供商返回的工具调用数据。
            provider: 来源提供商。

        Returns:
            ToolCall: 统一格式的工具调用。
        """
        import json

        if provider == "openai":
            # OpenAI: {id, type, function: {name, arguments}}
            func_data = data.get("function", {})
            args = func_data.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)

            return ToolCall(
                id=data.get("id", ""),
                name=func_data.get("name", ""),
                arguments=args,
            )

        if provider == "anthropic":
            # Anthropic: content block with type="tool_use"
            args = getattr(data, "input", {})
            if not isinstance(args, dict):
                args = json.loads(args) if isinstance(args, str) else {}

            return ToolCall(
                id=getattr(data, "id", ""),
                name=getattr(data, "name", ""),
                arguments=args,
            )

        if provider == "google":
            # Google: part with function_call
            fc = getattr(data, "function_call", None)
            if fc is None:
                raise ValueError("No function_call in Google response")

            args = getattr(fc, "args", {})
            if not isinstance(args, dict):
                args = dict(args) if args else {}

            return ToolCall(
                id=f"call_{fc.name}",
                name=getattr(fc, "name", ""),
                arguments=args,
            )

        raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def calls_from_provider(
        data: list[Any],
        provider: ProviderType,
    ) -> list[ToolCall]:
        """批量解析工具调用。

        Args:
            data: 提供商返回的工具调用列表。
            provider: 来源提供商。

        Returns:
            list[ToolCall]: 统一格式的工具调用列表。
        """
        return [
            ToolConverter.call_from_provider(d, provider)
            for d in data
        ]

    @staticmethod
    def result_to_provider(
        result: ToolResult,
        provider: ProviderType,
        tool_name: str = "",
    ) -> dict[str, Any]:
        """将工具结果转换为提供商消息格式。

        Args:
            result: 工具执行结果。
            provider: 目标提供商。
            tool_name: 工具名称(Google 需要)。

        Returns:
            dict[str, Any]: 提供商格式的消息。
        """
        converters = {
            "openai": ToolConverter._result_to_openai,
            "anthropic": ToolConverter._result_to_anthropic,
            "google": ToolConverter._result_to_google,
        }

        converter = converters.get(provider)
        if converter is None:
            raise ValueError(f"Unknown provider: {provider}")

        return converter(result, tool_name)

    # ============ OpenAI 格式 ============

    @staticmethod
    def _definition_to_openai(definition: ToolDefinition) -> dict[str, Any]:
        """转换为 OpenAI tools 格式。"""
        return {
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.to_json_schema(),
            },
        }

    @staticmethod
    def _result_to_openai(
        result: ToolResult,
        tool_name: str,
    ) -> dict[str, Any]:
        """转换为 OpenAI tool message 格式。"""
        return {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": content_to_string(result.content),
        }

    # ============ Anthropic 格式 ============

    @staticmethod
    def _definition_to_anthropic(definition: ToolDefinition) -> dict[str, Any]:
        """转换为 Anthropic tools 格式。"""
        return {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.to_json_schema(),
        }

    @staticmethod
    def _result_to_anthropic(
        result: ToolResult,
        tool_name: str,
    ) -> dict[str, Any]:
        """转换为 Anthropic tool_result 格式。"""
        content = result.content

        if isinstance(content, dict):
            pass  # Anthropic 接受 dict
        elif isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        else:
            content = str(content)

        return {
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": content,
            "is_error": result.is_error,
        }

    # ============ Google 格式 ============

    @staticmethod
    def _definition_to_google(definition: ToolDefinition) -> dict[str, Any]:
        """转换为 Google tools 格式。"""
        return {
            "function_declarations": [{
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.to_json_schema(),
            }]
        }

    @staticmethod
    def _result_to_google(
        result: ToolResult,
        tool_name: str,
    ) -> dict[str, Any]:
        """转换为 Google function_response 格式。"""
        import json

        content = result.content

        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = {"result": content}
        elif isinstance(content, bytes):
            content = {"result": content.decode("utf-8", errors="replace")}
        elif not isinstance(content, dict):
            content = {"result": str(content)}

        if result.is_error:
            content = {"error": result.error_message or "Tool execution failed"}

        return {
            "function_response": {
                "name": tool_name,
                "response": content,
            }
        }


def convert_tools(
    definitions: list[ToolDefinition],
    provider: ProviderType,
) -> list[dict[str, Any]]:
    """批量转换工具定义的便捷函数。

    Args:
        definitions: 工具定义列表。
        provider: 目标提供商。

    Returns:
        list[dict[str, Any]]: 提供商格式的工具定义列表。
    """
    return ToolConverter.definitions_to_provider(definitions, provider)


def convert_tool_call(
    data: Any,
    provider: ProviderType,
) -> ToolCall:
    """解析工具调用的便捷函数。

    Args:
        data: 提供商返回的工具调用数据。
        provider: 来源提供商。

    Returns:
        ToolCall: 统一格式的工具调用。
    """
    return ToolConverter.call_from_provider(data, provider)


def convert_tool_result(
    result: ToolResult,
    provider: ProviderType,
    tool_name: str = "",
) -> dict[str, Any]:
    """转换工具结果的便捷函数。

    Args:
        result: 工具执行结果。
        provider: 目标提供商。
        tool_name: 工具名称(Google 需要)。

    Returns:
        dict[str, Any]: 提供商格式的消息。
    """
    return ToolConverter.result_to_provider(result, provider, tool_name)
