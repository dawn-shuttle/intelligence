"""测试 core/types.py - 核心类型定义。"""

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import (
    ContentPart,
    ImageContent,
    Message,
    MessageDict,
    Messages,
    Role,
    TextContent,
    ToolCall,
    ToolResult,
)


class TestRole:
    """测试 Role 枚举。"""

    def test_role_values(self) -> None:
        """测试角色值。"""
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.SYSTEM.value == "system"
        assert Role.TOOL.value == "tool"

    def test_role_is_string(self) -> None:
        """测试角色是字符串枚举。"""
        assert isinstance(Role.USER, str)
        assert Role.USER == "user"


class TestTextContent:
    """测试 TextContent。"""

    def test_text_content_creation(self) -> None:
        """测试创建文本内容。"""
        content = TextContent(text="Hello")
        assert content.type == "text"
        assert content.text == "Hello"

    def test_text_content_default(self) -> None:
        """测试默认值。"""
        content = TextContent()
        assert content.type == "text"
        assert content.text == ""


class TestImageContent:
    """测试 ImageContent。"""

    def test_image_content_url(self) -> None:
        """测试图片 URL。"""
        content = ImageContent(image="https://example.com/image.png")
        assert content.type == "image"
        assert content.image == "https://example.com/image.png"
        assert content.mime_type is None

    def test_image_content_base64(self) -> None:
        """测试 base64 图片。"""
        content = ImageContent(image="base64data", mime_type="image/png")
        assert content.type == "image"
        assert content.mime_type == "image/png"


class TestToolCall:
    """测试 ToolCall。"""

    def test_tool_call_with_dict_args(self) -> None:
        """测试字典参数。"""
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"city": "Beijing"},
        )
        assert tool_call.id == "call_123"
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == {"city": "Beijing"}

    def test_tool_call_with_str_args(self) -> None:
        """测试字符串参数。"""
        tool_call = ToolCall(
            id="call_456",
            name="search",
            arguments='{"query": "test"}',
        )
        assert tool_call.arguments == '{"query": "test"}'


class TestToolResult:
    """测试 ToolResult。"""

    def test_tool_result(self) -> None:
        """测试工具结果。"""
        result = ToolResult(tool_call_id="call_123", content="result data")
        assert result.tool_call_id == "call_123"
        assert result.content == "result data"


class TestMessage:
    """测试 Message。"""

    def test_user_message(self) -> None:
        """测试用户消息。"""
        msg = Message.user("你好")
        assert msg.role == Role.USER
        assert msg.content == "你好"

    def test_assistant_message(self) -> None:
        """测试助手消息。"""
        msg = Message.assistant("回复内容")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "回复内容"

    def test_system_message(self) -> None:
        """测试系统消息。"""
        msg = Message.system("你是一个助手")
        assert msg.role == Role.SYSTEM
        assert msg.content == "你是一个助手"

    def test_tool_result_message(self) -> None:
        """测试工具结果消息。"""
        msg = Message.tool_result("call_123", "结果")
        assert msg.role == Role.TOOL
        assert msg.tool_call_id == "call_123"
        assert msg.content == "结果"

    def test_message_with_tool_calls(self) -> None:
        """测试带工具调用的消息。"""
        tool_call = ToolCall(id="call_1", name="test", arguments={})
        msg = Message.assistant(tool_calls=[tool_call])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_to_dict_simple(self) -> None:
        """测试简单消息转字典。"""
        msg = Message.user("test")
        result = msg.to_dict()
        assert result["role"] == "user"
        assert result["content"] == "test"

    def test_to_dict_with_multimodal(self) -> None:
        """测试多模态消息转字典。"""
        msg = Message(
            role=Role.USER,
            content=[
                TextContent(text="描述这张图片"),
                ImageContent(image="https://example.com/img.png"),
            ],
        )
        result = msg.to_dict()
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2

    def test_to_dict_with_tool_calls(self) -> None:
        """测试带工具调用的消息转字典。"""
        tool_call = ToolCall(id="call_1", name="test", arguments={"a": 1})
        msg = Message.assistant(tool_calls=[tool_call])
        result = msg.to_dict()
        assert "tool_calls" in result
        assert result["tool_calls"][0]["id"] == "call_1"


class TestContentPart:
    """测试 ContentPart 类型别名。"""

    def test_content_part_text(self) -> None:
        """测试文本内容块。"""
        part: ContentPart = TextContent(text="hello")
        assert part.type == "text"

    def test_content_part_image(self) -> None:
        """测试图片内容块。"""
        part: ContentPart = ImageContent(image="url")
        assert part.type == "image"
