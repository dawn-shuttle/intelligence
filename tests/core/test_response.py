"""测试 core/response.py - 统一响应格式。"""

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.core.response import (
    GenerateResponse,
    StreamChunk,
    Usage,
)


class TestUsage:
    """测试 Usage。"""

    def test_default_usage(self) -> None:
        """测试默认值。"""
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_usage_with_values(self) -> None:
        """测试带值的 Usage。"""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestGenerateResponse:
    """测试 GenerateResponse。"""

    def test_default_response(self) -> None:
        """测试默认响应。"""
        response = GenerateResponse()
        assert response.text == ""
        assert response.tool_calls == []
        assert response.finish_reason is None
        assert response.raw is None
        assert response.usage is None
        assert response.model is None
        assert response.request_id is None

    def test_response_with_text(self) -> None:
        """测试带文本的响应。"""
        response = GenerateResponse(text="Hello, world!")
        assert response.text == "Hello, world!"

    def test_response_with_tool_calls(self) -> None:
        """测试带工具调用的响应。"""
        tool_calls = [
            {"id": "call_1", "name": "test", "arguments": {}}
        ]
        response = GenerateResponse(tool_calls=tool_calls)
        assert response.tool_calls == tool_calls
        assert response.is_tool_call is True

    def test_is_tool_call_false(self) -> None:
        """测试无工具调用时 is_tool_call。"""
        response = GenerateResponse(text="Hello")
        assert response.is_tool_call is False

    def test_is_tool_call_true(self) -> None:
        """测试有工具调用时 is_tool_call。"""
        response = GenerateResponse(
            tool_calls=[{"id": "call_1", "name": "test", "arguments": {}}]
        )
        assert response.is_tool_call is True

    def test_to_dict_minimal(self) -> None:
        """测试最小字典转换。"""
        response = GenerateResponse(text="Hello")
        result = response.to_dict()
        assert result == {"text": "Hello"}

    def test_to_dict_full(self) -> None:
        """测试完整字典转换。"""
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        response = GenerateResponse(
            text="Hello",
            tool_calls=[{"id": "call_1", "name": "test", "arguments": {}}],
            finish_reason="stop",
            usage=usage,
            model="gpt-4",
            request_id="req_123",
        )
        result = response.to_dict()
        assert result["text"] == "Hello"
        assert result["tool_calls"] == [{"id": "call_1", "name": "test", "arguments": {}}]
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["model"] == "gpt-4"
        assert result["request_id"] == "req_123"


class TestStreamChunk:
    """测试 StreamChunk。"""

    def test_default_chunk(self) -> None:
        """测试默认值。"""
        chunk = StreamChunk()
        assert chunk.delta == ""
        assert chunk.tool_calls == []
        assert chunk.is_finished is False
        assert chunk.finish_reason is None
        assert chunk.usage is None

    def test_chunk_with_delta(self) -> None:
        """测试带增量的块。"""
        chunk = StreamChunk(delta="Hello")
        assert chunk.delta == "Hello"

    def test_chunk_finished(self) -> None:
        """测试结束块。"""
        chunk = StreamChunk(
            delta="",
            is_finished=True,
            finish_reason="stop",
        )
        assert chunk.is_finished is True
        assert chunk.finish_reason == "stop"

    def test_chunk_with_usage(self) -> None:
        """测试带使用统计的块。"""
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk = StreamChunk(
            is_finished=True,
            finish_reason="stop",
            usage=usage,
        )
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 15

    def test_chunk_with_tool_call_delta(self) -> None:
        """测试带工具调用增量的块。"""
        chunk = StreamChunk(
            tool_calls=[{"id": "call_1", "name": "test", "arguments": ""}]
        )
        assert len(chunk.tool_calls) == 1
