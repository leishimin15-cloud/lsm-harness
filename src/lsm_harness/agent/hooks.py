"""Agent lifecycle and Turn-control extension points.

Usage::

    hooks = LoopHooks(
        on_trace_start=lambda msg, emit: print(f"Trace start: {msg[:50]}"),
        on_turn_end=lambda turn, emit: print(f"Turn {turn.turn_index}: {turn.status}"),
        on_tool_error=lambda name, err, emit: print(f"Tool {name} failed: {err}"),
        on_trace_end=lambda result, emit: print(f"Trace done: {result.reply[:50]}"),
    )
    run_loop(..., hooks=hooks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from lsm_harness.ai.types import Model

if TYPE_CHECKING:
    from lsm_harness.agent.messages import (
        AgentMessage,
        AssistantMessage,
        ToolResultMessage,
    )
    # Agent-layer concept (refactor plan §9.1); imported lazily to avoid
    # the agent.types → agent.hooks import cycle.
    from lsm_harness.agent.types import TraceResult


Emit = Callable[[str, dict], None]


@dataclass
class TurnContext:
    """Snapshot of one model call and the tool batch it caused."""

    turn_index: int
    model: str
    stop_reason: str
    usage: dict[str, int]
    tool_count: int
    tool_error_count: int
    accumulated_text: str = ""
    status: str = "running"

    @property
    def iteration(self) -> int:
        """Legacy alias for integrations that still display iterations."""
        return self.turn_index


IterationContext = TurnContext


@dataclass
class TurnControlContext:
    """State exposed to control callbacks after one Turn fully completes."""

    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    system: str
    messages: list[AgentMessage]
    result: TraceResult
    turn: TurnContext


@dataclass
class NextTurnUpdate:
    """Optional runtime replacements applied before the next model call."""

    system: str | None = None
    messages: list[AgentMessage] | None = None
    model: Model | str | None = None
    thinking: str | None = None


PrepareNextTurnContext = TurnControlContext
ShouldStopAfterTurnContext = TurnControlContext
PrepareNextTurn = Callable[[PrepareNextTurnContext], NextTurnUpdate | None]
ShouldStopAfterTurn = Callable[[ShouldStopAfterTurnContext], bool]


@dataclass
class LoopHooks:
    """Collection of callbacks invoked at specific points in the agent loop.

    Every hook receives an ``emit`` callable as its last argument so it
    can write events to the trace.  Hooks should never raise — exceptions
    are caught and logged.
    """

    # ── trace-level hooks ─────────────────────────────────────

    on_trace_start: Callable[[str, Emit], None] | None = None
    """Called once when the reason→act→observe trace starts.
    Args: (user_message, emit)"""

    on_trace_end: Callable[[Any, Emit], None] | None = None
    """Called once when the complete trace ends.
    Args: (trace_result, emit)"""

    # ── turn-level hooks ──────────────────────────────────────

    on_turn_start: Callable[[int, Emit], None] | None = None
    """Called before each model call.
    Args: (turn_index, emit)"""

    on_turn_end: Callable[[TurnContext, Emit], None] | None = None
    """Called after the model call and its caused tool batch finish.
    Args: (turn_context, emit)"""

    # ── iteration-level hooks ─────────────────────────────────

    on_iteration_start: Callable[[int, Emit], None] | None = None
    """Legacy alias called alongside ``on_turn_start``.
    Args: (iteration_number, emit)"""

    on_iteration_end: Callable[[TurnContext, Emit], None] | None = None
    """Legacy alias called alongside ``on_turn_end``.
    Args: (iteration_context, emit)"""

    # ── model hooks ───────────────────────────────────────────

    on_model_call: Callable[[str, int, Emit], None] | None = None
    """Called just before calling the model.
    Args: (model_name, iteration, emit)"""

    on_model_response: Callable[[str, str, dict, Emit], None] | None = None
    """Called after receiving a model response.
    Args: (stop_reason, text_preview, usage, emit)"""

    # ── tool hooks ────────────────────────────────────────────

    on_tool_request: Callable[[str, dict, Emit], None] | None = None
    """Called when a tool is about to be called.
    Args: (tool_name, arguments, emit)"""

    on_tool_result: Callable[[str, str, bool, Emit], None] | None = None
    """Called after a tool completes.
    Args: (tool_name, output_preview, is_error, emit)"""

    on_tool_error: Callable[[str, str, Emit], None] | None = None
    """Called when a tool returns an error.
    Args: (tool_name, error_message, emit)"""

    # ── error hooks ───────────────────────────────────────────

    on_loop_error: Callable[[str, Emit], None] | None = None
    """Called when the loop encounters an unexpected error.
    Args: (error_message, emit)"""


# ── helpers to safely invoke hooks ──────────────────────────────


def _safe_call(fn: Callable | None, *args: Any) -> None:
    """Call a hook if set, swallowing any exceptions."""
    if fn is None:
        return
    try:
        fn(*args)
    except Exception:
        pass  # hooks must never break the loop


def invoke_trace_start(hooks: LoopHooks | None, message: str, emit: Emit) -> None:
    _safe_call(hooks.on_trace_start if hooks else None, message, emit)


def invoke_trace_end(hooks: LoopHooks | None, result: Any, emit: Emit) -> None:
    _safe_call(hooks.on_trace_end if hooks else None, result, emit)


def invoke_turn_start(hooks: LoopHooks | None, turn_index: int, emit: Emit) -> None:
    _safe_call(hooks.on_turn_start if hooks else None, turn_index, emit)
    invoke_iteration_start(hooks, turn_index, emit)


def invoke_turn_end(
    hooks: LoopHooks | None,
    ctx: TurnContext,
    emit: Emit,
) -> None:
    _safe_call(hooks.on_turn_end if hooks else None, ctx, emit)
    invoke_iteration_end(hooks, ctx, emit)


def invoke_iteration_start(hooks: LoopHooks | None, iteration: int, emit: Emit) -> None:
    _safe_call(hooks.on_iteration_start if hooks else None, iteration, emit)


def invoke_iteration_end(
    hooks: LoopHooks | None,
    ctx: TurnContext,
    emit: Emit,
) -> None:
    _safe_call(hooks.on_iteration_end if hooks else None, ctx, emit)


def invoke_model_call(
    hooks: LoopHooks | None,
    model: str,
    iteration: int,
    emit: Emit,
) -> None:
    _safe_call(hooks.on_model_call if hooks else None, model, iteration, emit)


def invoke_model_response(
    hooks: LoopHooks | None,
    stop_reason: str,
    text_preview: str,
    usage: dict,
    emit: Emit,
) -> None:
    _safe_call(
        hooks.on_model_response if hooks else None,
        stop_reason,
        text_preview,
        usage,
        emit,
    )


def invoke_tool_request(
    hooks: LoopHooks | None,
    tool_name: str,
    args: dict,
    emit: Emit,
) -> None:
    _safe_call(hooks.on_tool_request if hooks else None, tool_name, args, emit)


def invoke_tool_result(
    hooks: LoopHooks | None,
    tool_name: str,
    output_preview: str,
    is_error: bool,
    emit: Emit,
) -> None:
    _safe_call(
        hooks.on_tool_result if hooks else None,
        tool_name,
        output_preview,
        is_error,
        emit,
    )


def invoke_tool_error(
    hooks: LoopHooks | None,
    tool_name: str,
    error_message: str,
    emit: Emit,
) -> None:
    _safe_call(hooks.on_tool_error if hooks else None, tool_name, error_message, emit)
    # Also invoke the general tool_result hook
    invoke_tool_result(hooks, tool_name, error_message, True, emit)


def invoke_loop_error(
    hooks: LoopHooks | None,
    error_message: str,
    emit: Emit,
) -> None:
    _safe_call(hooks.on_loop_error if hooks else None, error_message, emit)
