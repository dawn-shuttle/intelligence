"""测试 core/config.py - 生成配置类。"""

import pytest

from dawn_shuttle.dawn_shuttle_intelligence.src.core.config import (
    GenerateConfig,
    ResponseFormat,
    StopSequences,
    ToolChoice,
)


class TestGenerateConfig:
    """测试 GenerateConfig。"""

    def test_default_config(self) -> None:
        """测试默认配置。"""
        config = GenerateConfig()
        assert config.model == ""
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.stream is False

    def test_config_with_model(self) -> None:
        """测试带模型的配置。"""
        config = GenerateConfig(model="gpt-4")
        assert config.model == "gpt-4"

    def test_config_with_all_params(self) -> None:
        """测试完整参数配置。"""
        config = GenerateConfig(
            model="gpt-4",
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            stop=["END"],
            frequency_penalty=0.5,
            presence_penalty=0.3,
            seed=42,
            stream=True,
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_tokens == 1000
        assert config.stop == ["END"]
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.3
        assert config.seed == 42
        assert config.stream is True

    def test_config_with_tools(self) -> None:
        """测试带工具的配置。"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {},
                },
            }
        ]
        config = GenerateConfig(model="gpt-4", tools=tools)
        assert config.tools == tools

    def test_to_dict_filters_none(self) -> None:
        """测试 to_dict 过滤 None 值。"""
        config = GenerateConfig(model="gpt-4", temperature=0.5)
        result = config.to_dict()
        assert "model" in result
        assert "temperature" in result
        assert "max_tokens" not in result
        assert "stop" not in result

    def test_to_dict_includes_extra(self) -> None:
        """测试 to_dict 包含额外参数。"""
        config = GenerateConfig(
            model="gpt-4",
            extra={"custom_param": "value"},
        )
        result = config.to_dict()
        assert result["custom_param"] == "value"

    def test_to_dict_stop_string(self) -> None:
        """测试 stop 为字符串。"""
        config = GenerateConfig(model="gpt-4", stop="END")
        result = config.to_dict()
        assert result["stop"] == "END"

    def test_to_dict_stop_list(self) -> None:
        """测试 stop 为列表。"""
        config = GenerateConfig(model="gpt-4", stop=["END", "STOP"])
        result = config.to_dict()
        assert result["stop"] == ["END", "STOP"]


class TestTypeAliases:
    """测试类型别名。"""

    def test_stop_sequences_str(self) -> None:
        """测试 StopSequences 字符串。"""
        stop: StopSequences = "END"
        assert stop == "END"

    def test_stop_sequences_list(self) -> None:
        """测试 StopSequences 列表。"""
        stop: StopSequences = ["END", "STOP"]
        assert stop == ["END", "STOP"]

    def test_tool_choice_str(self) -> None:
        """测试 ToolChoice 字符串。"""
        choice: ToolChoice = "auto"
        assert choice == "auto"

    def test_tool_choice_dict(self) -> None:
        """测试 ToolChoice 字典。"""
        choice: ToolChoice = {"type": "function", "name": "test"}
        assert choice["type"] == "function"

    def test_response_format(self) -> None:
        """测试 ResponseFormat。"""
        fmt: ResponseFormat = {"type": "json_object"}
        assert fmt["type"] == "json_object"
