"""生成配置类 - 定义调用 AI 模型时的各种参数。"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GenerateConfig:
    """生成配置，包含所有 AI 调用的通用参数。"""

    # 模型标识（如 "gpt-4", "claude-3-opus"）
    model: str = ""

    # 采样参数
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    # 输出控制
    max_tokens: int | None = None
    stop: list[str] | str | None = None

    # 频率惩罚
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    # 种子（可复现输出）
    seed: int | None = None

    # 流式输出
    stream: bool = False

    # 工具定义
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    # 响应格式
    response_format: dict[str, Any] | None = None

    # 额外参数（提供商特定）
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，过滤掉 None 值。"""
        result: dict[str, Any] = {}

        if self.model:
            result["model"] = self.model

        # 采样参数
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.top_k is not None:
            result["top_k"] = self.top_k

        # 输出控制
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.stop is not None:
            result["stop"] = self.stop

        # 惩罚参数
        if self.frequency_penalty is not None:
            result["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            result["presence_penalty"] = self.presence_penalty

        # 其他
        if self.seed is not None:
            result["seed"] = self.seed
        if self.tools is not None:
            result["tools"] = self.tools
        if self.tool_choice is not None:
            result["tool_choice"] = self.tool_choice
        if self.response_format is not None:
            result["response_format"] = self.response_format

        # 合并额外参数
        result.update(self.extra)

        return result
