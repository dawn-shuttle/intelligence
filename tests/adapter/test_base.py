"""测试 adapter/base.py - 适配器基础工具。"""

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.adapter.base import (
    message_to_openai_format,
    openai_tool_to_dict,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import (
    ImageContent,
    Message,
    Role,
    TextContent,
    ToolCall,
)


class TestMessageToOpenAIFormat:
    """测试 message_to_openai_format。"""

    def test_simple_user_message(self) -> None:
        """测试简单用户消息。"""
        msg = Message.user("你好")
        result = message_to_openai_format(msg)
        assert result["role"] == "user"
        assert result["content"] == "你好"

    def test_system_message(self) -> None:
        """测试系统消息。"""
        msg = Message.system("你是一个助手")
        result = message_to_openai_format(msg)
        assert result["role"] == "system"
        assert result["content"] == "你是一个助手"

    def test_assistant_message(self) -> None:
        """测试助手消息。"""
        msg = Message.assistant("回复内容")
        result = message_to_openai_format(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "回复内容"

    def test_message_with_name(self) -> None:
        """测试带 name 的消息。"""
        msg = Message(role=Role.USER, content="hello", name="Alice")
        result = message_to_openai_format(msg)
        assert result["name"] == "Alice"

    def test_multimodal_message_with_text(self) -> None:
        """测试多模态消息(文本)。"""
        msg = Message(
            role=Role.USER,
            content=[TextContent(text="描述图片")],
        )
        result = message_to_openai_format(msg)
        assert result["content"] == [{"type": "text", "text": "描述图片"}]

    def test_multimodal_message_with_image_url(self) -> None:
        """测试多模态消息(图片URL)。"""
        msg = Message(
            role=Role.USER,
            content=[ImageContent(image="https://example.com/img.png")],
        )
        result = message_to_openai_format(msg)
        assert result["content"] == [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/img.png"},
            }
        ]

    def test_multimodal_message_with_base64_image(self) -> None:
        """测试多模态消息(base64图片)。"""
        msg = Message(
            role=Role.USER,
            content=[ImageContent(image="base64data", mime_type="image/jpeg")],
        )
        result = message_to_openai_format(msg)
        assert result["content"] == [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,base64data"},
            }
        ]

    def test_message_with_tool_calls(self) -> None:
        """测试带工具调用的消息。"""
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"city": "Beijing"},
        )
        msg = Message.assistant(tool_calls=[tool_call])
        result = message_to_openai_format(msg)
        assert "tool_calls" in result
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["type"] == "function"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_result_message(self) -> None:
        """测试工具结果消息。"""
        msg = Message.tool_result("call_123", "天气晴朗")
        result = message_to_openai_format(msg)
        assert result["role"] == "tool"
        assert result["content"] == "天气晴朗"
        assert result["tool_call_id"] == "call_123"


class TestOpenaiToolToDict:
    """测试 openai_tool_to_dict。"""

    def test_tool_with_dict_arguments(self) -> None:
        """测试字典参数。"""
        openai_tool = {
            "id": "call_123",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Beijing"}',
            },
        }
        result = openai_tool_to_dict(openai_tool)
        assert result["id"] == "call_123"
        assert result["name"] == "get_weather"
        assert result["arguments"] == {"city": "Beijing"}

    def test_tool_with_complex_arguments(self) -> None:
        """测试复杂参数。"""
        openai_tool = {
            "id": "call_456",
            "function": {
                "name": "search",
                "arguments": '{"query": "test", "limit": 10}',
            },
        }
        result = openai_tool_to_dict(openai_tool)
        assert result["arguments"]["query"] == "test"
        assert result["arguments"]["limit"] == 10
