"""生成配置类 - 定义调用 AI 模型时的各种参数。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

StopSequences = str | Sequence[str]
"""停止序列类型, 可以是单个字符串或字符串列表。"""

ToolChoice = str | dict[str, Any]
"""工具选择策略类型。"""

ResponseFormat = dict[str, Any]
"""响应格式类型。"""


@dataclass
class GenerateConfig:
    """生成配置, 包含所有 AI 调用的通用参数。

    Attributes:
        model: 模型标识(如 "gpt-4", "claude-3-opus")。
        temperature: 采样温度, 控制输出的随机性。范围 0.0-2.0。
        top_p: Top-p 采样参数, 控制多样性。
        top_k: Top-k 采样参数, 限制候选 token 数量。
        max_tokens: 最大输出 token 数。
        stop: 停止词, 遇到时停止生成。
        frequency_penalty: 频率惩罚, 降低重复内容。
        presence_penalty: 存在惩罚, 鼓励新话题。
        seed: 随机种子, 用于可复现输出。
        stream: 是否启用流式输出。
        tools: 工具定义列表。
        tool_choice: 工具选择策略。
        response_format: 响应格式配置。
        extra: 额外参数, 用于提供商特定选项。
    """

    # 模型标识
    model: str = ""

    # 采样参数
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    # 输出控制
    max_tokens: int | None = None
    stop: StopSequences | None = None

    # 频率惩罚
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    # 种子(可复现输出)
    seed: int | None = None

    # 流式输出
    stream: bool = False

    # 工具定义
    tools: list[dict[str, Any]] | None = None
    tool_choice: ToolChoice | None = None

    # 响应格式
    response_format: ResponseFormat | None = None

    # 额外参数(提供商特定)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典, 过滤掉 None 值。

        Returns:
            dict[str, Any]: 不包含 None 值的配置字典。
        """
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
            if isinstance(self.stop, Sequence):
                result["stop"] = list(self.stop)
            else:
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
