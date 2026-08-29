"""Coding-agent tool definitions and the Agent-layer adapter."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

from lsm_harness.agent.tools import (
    AgentTool,
    AfterHook,
    BeforeHook,
    Effect,
    ToolExecutionMode,
)


ToolRenderer = Callable[[dict[str, Any]], str]
ToolResultRenderer = Callable[[str, dict[str, Any] | None], str]


@dataclass(frozen=True, init=False)
class ToolDefinition:
    """Product tool with prompt/UI metadata and bound dependencies."""

    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[..., Any]
    prompt_snippet: str
    render_call: ToolRenderer | None
    render_result: ToolResultRenderer | None
    product_context: Any
    prepare_arguments: Callable[[dict[str, Any]], dict[str, Any]] | None
    execution_mode: ToolExecutionMode
    effect: Effect
    timeout: float
    before_hook: BeforeHook | None
    after_hook: AfterHook | None
    terminate_on_success: bool

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        execute: Callable[..., Any] | None = None,
        effect: Effect = "read",
        *,
        label: str = "",
        prompt_snippet: str = "",
        render_call: ToolRenderer | None = None,
        render_result: ToolResultRenderer | None = None,
        product_context: Any = None,
        prepare_arguments: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        execution_mode: ToolExecutionMode | None = None,
        timeout: float = 0.0,
        before_hook: BeforeHook | None = None,
        after_hook: AfterHook | None = None,
        terminate_on_success: bool = False,
        input_schema: dict[str, Any] | None = None,
        fn: Callable[..., Any] | None = None,
        prepare_args: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        parallel_safe: bool | None = None,
    ) -> None:
        """Accept canonical fields plus old spellings during native migration."""
        resolved_parameters = parameters if parameters is not None else input_schema
        resolved_execute = execute if execute is not None else fn
        if resolved_parameters is None or resolved_execute is None:
            raise TypeError("ToolDefinition requires parameters and execute")
        if execution_mode is None:
            if parallel_safe is not None:
                execution_mode = "parallel" if parallel_safe else "sequential"
            else:
                execution_mode = "parallel" if effect == "read" else "sequential"
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "label", label or name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "parameters", resolved_parameters)
        object.__setattr__(self, "execute", resolved_execute)
        object.__setattr__(self, "prompt_snippet", prompt_snippet)
        object.__setattr__(self, "render_call", render_call)
        object.__setattr__(self, "render_result", render_result)
        object.__setattr__(self, "product_context", product_context)
        object.__setattr__(
            self,
            "prepare_arguments",
            prepare_arguments if prepare_arguments is not None else prepare_args,
        )
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "before_hook", before_hook)
        object.__setattr__(self, "after_hook", after_hook)
        object.__setattr__(self, "terminate_on_success", terminate_on_success)


def wrap_tool_definition(definition: ToolDefinition) -> AgentTool:
    """Erase product-only metadata and produce an executable AgentTool."""
    execute = definition.execute
    if definition.product_context is not None:
        parameters = inspect.signature(execute).parameters
        if "_product_context" in parameters:
            original = execute

            @wraps(original)
            def execute_with_context(**kwargs: Any) -> Any:
                return original(
                    **kwargs,
                    _product_context=definition.product_context,
                )

            execute = execute_with_context

    return AgentTool(
        name=definition.name,
        label=definition.label,
        description=definition.description,
        parameters=definition.parameters,
        execute=execute,
        prepare_arguments=definition.prepare_arguments,
        execution_mode=definition.execution_mode,
        effect=definition.effect,
        timeout=definition.timeout,
        before_hook=definition.before_hook,
        after_hook=definition.after_hook,
        terminate_on_success=definition.terminate_on_success,
    )


__all__ = ["ToolDefinition", "wrap_tool_definition"]
