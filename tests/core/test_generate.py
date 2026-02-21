"""测试 core/generate.py - 统一入口函数。"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from dawn_shuttle.dawn_shuttle_intelligence.src.core.generate import (
    generate_text,
    stream_text,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.core.response import (
    GenerateResponse,
    StreamChunk,
    Usage,
)
from dawn_shuttle.dawn_shuttle_intelligence.src.core.types import Message


class TestGenerateText:
    """测试 generate_text。"""

    @pytest.mark.asyncio
    async def test_generate_text_basic(self) -> None:
        """测试基本生成。"""
        # Mock provider
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            return_value=GenerateResponse(
                text="Hello!",
                finish_reason="stop",
                model="gpt-4",
            )
        )

        messages = [Message.user("Hi")]
        result = await generate_text(messages=messages, provider=mock_provider, model="gpt-4")

        assert result.text == "Hello!"
        assert result.finish_reason == "stop"
        mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_text_with_temperature(self) -> None:
        """测试带温度参数。"""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value=GenerateResponse(text="test"))

        messages = [Message.user("Hi")]
        await generate_text(
            messages=messages,
            provider=mock_provider,
            model="gpt-4",
            temperature=0.7,
        )

        # 验证 config 参数被传递
        call_args = mock_provider.generate.call_args
        config = call_args[0][1]
        assert config.temperature == 0.7

    @pytest.mark.asyncio
    async def test_generate_text_with_max_tokens(self) -> None:
        """测试带 max_tokens 参数。"""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value=GenerateResponse(text="test"))

        messages = [Message.user("Hi")]
        await generate_text(
            messages=messages,
            provider=mock_provider,
            model="gpt-4",
            max_tokens=100,
        )

        call_args = mock_provider.generate.call_args
        config = call_args[0][1]
        assert config.max_tokens == 100

    @pytest.mark.asyncio
    async def test_generate_text_with_tools(self) -> None:
        """测试带工具参数。"""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value=GenerateResponse(text="test"))

        tools = [{"type": "function", "function": {"name": "test"}}]
        messages = [Message.user("Hi")]
        await generate_text(
            messages=messages,
            provider=mock_provider,
            model="gpt-4",
            tools=tools,
            tool_choice="auto",
        )

        call_args = mock_provider.generate.call_args
        config = call_args[0][1]
        assert config.tools == tools
        assert config.tool_choice == "auto"

    @pytest.mark.asyncio
    async def test_generate_text_with_all_params(self) -> None:
        """测试所有参数。"""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value=GenerateResponse(text="test"))

        messages = [Message.user("Hi")]
        await generate_text(
            messages=messages,
            provider=mock_provider,
            model="gpt-4",
            temperature=0.5,
            max_tokens=200,
            top_p=0.9,
            stop=["END"],
            seed=42,
            frequency_penalty=0.3,
            presence_penalty=0.2,
        )

        call_args = mock_provider.generate.call_args
        config = call_args[0][1]
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 200
        assert config.top_p == 0.9
        assert config.stop == ["END"]
        assert config.seed == 42
        assert config.frequency_penalty == 0.3
        assert config.presence_penalty == 0.2

    @pytest.mark.asyncio
    async def test_generate_text_with_extra_kwargs(self) -> None:
        """测试额外参数。"""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value=GenerateResponse(text="test"))

        messages = [Message.user("Hi")]
        await generate_text(
            messages=messages,
            provider=mock_provider,
            model="gpt-4",
            custom_param="value",
        )

        call_args = mock_provider.generate.call_args
        config = call_args[0][1]
        assert config.extra["custom_param"] == "value"


class TestStreamText:
    """测试 stream_text。"""

    @pytest.mark.asyncio
    async def test_stream_text_basic(self) -> None:
        """测试基本流式生成。"""
        # Mock generator
        async def mock_generator(*args, **kwargs):
            yield StreamChunk(delta="Hello")
            yield StreamChunk(delta=" world")
            yield StreamChunk(delta="!", is_finished=True, finish_reason="stop")

        mock_provider = MagicMock()
        mock_provider.generate_stream = mock_generator

        messages = [Message.user("Hi")]
        chunks = []
        async for chunk in stream_text(
            messages=messages,
            provider=mock_provider,
            model="gpt-4",
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " world"
        assert chunks[2].is_finished is True

    @pytest.mark.asyncio
    async def test_stream_text_with_params(self) -> None:
        """测试带参数的流式生成。"""
        chunks_yielded = []

        async def mock_generator(messages, config):
            chunks_yielded.append(config)
            yield StreamChunk(delta="test")

        mock_provider = MagicMock()
        mock_provider.generate_stream = mock_generator

        messages = [Message.user("Hi")]
        async for _ in stream_text(
            messages=messages,
            provider=mock_provider,
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
        ):
            pass

        # 验证 stream=True
        assert chunks_yielded[0].stream is True
        assert chunks_yielded[0].temperature == 0.7
        assert chunks_yielded[0].max_tokens == 100
