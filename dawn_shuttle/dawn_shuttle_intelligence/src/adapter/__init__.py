"""Adapter 模块 - AI 提供商适配器。"""

from .base import message_to_openai_format, openai_tool_to_dict
from .openai import OpenAIProvider, openai

__all__ = [
    "OpenAIProvider",
    "message_to_openai_format",
    "openai",
    "openai_tool_to_dict",
]
