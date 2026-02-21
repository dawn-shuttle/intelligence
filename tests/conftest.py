"""测试配置和共享 fixtures。"""

import pytest


@pytest.fixture
def sample_message_dict() -> dict:
    """返回示例消息字典。"""
    return {
        "role": "user",
        "content": "你好",
    }
