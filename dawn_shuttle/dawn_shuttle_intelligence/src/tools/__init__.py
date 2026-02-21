"""工具模块 - 工具定义、注册、执行和 MCP 支持。"""

from __future__ import annotations

from .converter import (
    ProviderType,
    ToolConverter,
    convert_tool_call,
    convert_tool_result,
    convert_tools,
)
from .executor import (
    ExecutorConfig,
    ToolExecutionError,
    ToolExecutor,
    ToolTimeoutError,
    execute_tool_call,
    execute_tool_calls,
)
from .loop import (
    LoopConfig,
    LoopResult,
    LoopStatus,
    execute_and_continue,
    run_with_tools,
)
from .registry import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistryError,
    get_default_registry,
    get_tool,
    list_tools,
    register,
)
from .schema import (
    extract_function_schema,
    python_type_to_json_schema,
    validate_arguments,
)
from .skill import Skill, SkillContext, SkillError, SkillToolWrapper, skill
from .tool import AnyTool, DictTool, FunctionTool, Tool, tool
from .types import (
    JSONSchemaType,
    ToolCall,
    ToolDefinition,
    ToolExecuteAsyncFunc,
    ToolExecuteFunc,
    ToolExecution,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)

__all__ = [
    "AnyTool",
    "DictTool",
    "DuplicateToolError",
    "ExecutorConfig",
    "FunctionTool",
    "JSONSchemaType",
    "LoopConfig",
    "LoopResult",
    "LoopStatus",
    "ProviderType",
    "Skill",
    "SkillContext",
    "SkillError",
    "SkillToolWrapper",
    "Tool",
    "ToolCall",
    "ToolConverter",
    "ToolDefinition",
    "ToolExecuteAsyncFunc",
    "ToolExecuteFunc",
    "ToolExecution",
    "ToolExecutionError",
    "ToolExecutionStatus",
    "ToolExecutor",
    "ToolNotFoundError",
    "ToolParameter",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolResult",
    "ToolTimeoutError",
    "convert_tool_call",
    "convert_tool_result",
    "convert_tools",
    "execute_and_continue",
    "execute_tool_call",
    "execute_tool_calls",
    "extract_function_schema",
    "get_default_registry",
    "get_tool",
    "list_tools",
    "python_type_to_json_schema",
    "register",
    "run_with_tools",
    "skill",
    "tool",
    "validate_arguments",
]
