"""适配器基础工具 - 提供消息转换等通用功能。"""

from __future__ import annotations

import contextlib
from typing import Any

from ..core.types import (
    ImageContent,
    Message,
    Role,
    TextContent,
)


def extract_error_info(error: Exception) -> dict[str, Any]:
    """从异常中提取错误信息。

    Args:
        error: 异常对象。

    Returns:
        dict[str, Any]: 错误信息字典。
    """
    info: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
        "status_code": None,
        "request_id": None,
        "retry_after": None,
    }

    # 提取状态码
    if hasattr(error, "status_code"):
        info["status_code"] = error.status_code

    # 提取请求 ID
    if hasattr(error, "request_id"):
        info["request_id"] = error.request_id

    # 提取 retry-after
    if hasattr(error, "response"):
        resp = getattr(error, "response", None)
        if resp and hasattr(resp, "headers"):
            ra = resp.headers.get("retry-after")
            if ra:
                with contextlib.suppress(ValueError):
                    info["retry_after"] = int(ra)

    return info


def message_to_openai_format(message: Message) -> dict[str, Any]:
    """将统一消息格式转换为 OpenAI API 格式。

    Args:
        message: 统一消息对象。

    Returns:
        dict[str, Any]: OpenAI API 格式的消息字典。
    """
    result: dict[str, Any] = {"role": message.role.value}

    # 处理内容
    if message.content is not None:
        if isinstance(message.content, str):
            result["content"] = message.content
        else:
            # 多模态内容
            parts: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextContent):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContent):
                    if part.image.startswith("http"):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": part.image},
                            }
                        )
                    else:
                        # base64
                        mime = part.mime_type or "image/png"
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{part.image}"
                                },
                            }
                        )
            result["content"] = parts

    # 处理 name
    if message.name:
        result["name"] = message.name

    # 处理工具调用
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": (
                        tc.arguments
                        if isinstance(tc.arguments, str)
                        else __import__("json").dumps(tc.arguments)
                    ),
                },
            }
            for tc in message.tool_calls
        ]

    # 处理 tool 角色的消息
    if message.role == Role.TOOL and message.tool_call_id:
        result["tool_call_id"] = message.tool_call_id

    return result


def openai_tool_to_dict(tool: dict[str, Any]) -> dict[str, Any]:
    """将 OpenAI 工具调用格式转换为统一格式。

    Args:
        tool: OpenAI 格式的工具调用字典。

    Returns:
        dict[str, Any]: 统一格式的工具调用字典。
    """
    import json

    arguments = tool["function"]["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    return {
        "id": tool["id"],
        "name": tool["function"]["name"],
        "arguments": arguments,
    }
