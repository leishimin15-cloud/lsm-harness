"""Structured contracts for the reusable Agent Loop package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from lsm_harness.agent.hooks import (
    LoopHooks,
    PrepareNextTurn,
    ShouldStopAfterTurn,
)
from lsm_harness.agent.events import AgentEventListener
from lsm_harness.agent.messages import (
    AgentMessage,
    AssistantMessage,
    ConvertToLlm,
    TransformContext,
    default_convert_to_llm,
)
from lsm_harness.agent.pending import PendingMessage
from lsm_harness.agent.tools import ToolExecutionMode, ToolRegistry, ToolResult
from lsm_harness.ai.types import CacheRetention, Model, StopReason

__all__ = [
    "AgentContext",
    "AgentLoopConfig",
    "AgentMessage",
    "AfterToolCall",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "BeforeToolCall",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "ConvertToLlm",
    "PendingMessageGetter",
    "TraceResult",
    "TraceStatus",
    "TransformContext",
    "TurnResult",
    "default_convert_to_llm",
]


PendingMessageGetter = Callable[[], list[PendingMessage]]


@dataclass
class AgentContext:
    """Provider-ready conversation state owned by one loop invocation."""

    system_prompt: str
    messages: list[AgentMessage]
    tools: ToolRegistry


@dataclass
class BeforeToolCallResult:
    """Decision returned by ``before_tool_call``."""

    block: bool = False
    reason: str | None = None


@dataclass
class BeforeToolCallContext:
    """Validated tool request exposed before execution."""

    assistant_message: AssistantMessage
    tool_call: dict[str, Any]
    args: dict[str, Any]
    context: AgentContext


@dataclass
class AfterToolCallResult:
    """Field-by-field overrides returned after tool execution."""

    output: str | None = None
    details: dict[str, Any] | None = None
    is_error: bool | None = None
    terminate: bool | None = None


@dataclass
class AfterToolCallContext:
    """Executed tool result exposed before it enters model context."""

    assistant_message: AssistantMessage
    tool_call: dict[str, Any]
    args: dict[str, Any]
    result: ToolResult
    is_error: bool
    context: AgentContext


BeforeToolCall = Callable[
    [BeforeToolCallContext],
    BeforeToolCallResult | None,
]
AfterToolCall = Callable[
    [AfterToolCallContext],
    AfterToolCallResult | None,
]


@dataclass
class AgentLoopConfig:
    """Runtime policy passed to the low-level Agent Loop as one boundary."""

    model: Model | str
    max_iterations: int
    max_tokens: int
    convert_to_llm: ConvertToLlm = default_convert_to_llm
    transform_context: TransformContext | None = None
    before_tool_call: BeforeToolCall | None = None
    after_tool_call: AfterToolCall | None = None
    tool_execution: ToolExecutionMode = "parallel"
    get_steering_messages: PendingMessageGetter | None = None
    get_follow_up_messages: PendingMessageGetter | None = None
    prepare_next_turn: PrepareNextTurn | None = None
    should_stop_after_turn: ShouldStopAfterTurn | None = None
    on_truncation: Callable[[], tuple[str, list[dict[str, Any]]]] | None = None
    governor: Any = None
    hooks: LoopHooks | None = None
    thinking: str = "disabled"
    cache_retention: CacheRetention = "short"
    max_model_retries: int = 2
    max_empty_retries: int = 2
    max_length_recoveries: int = 3
    approval_broker: Any = None
    trace_id: str = ""
    session_id: str = ""
    sandboxed: bool = False
    listeners: list[AgentEventListener] | None = None


# ── trace outcome ─────────────────────────────────────────────────
# Refactor plan §9.1: the trace result is an Agent-layer concept (it
# describes one reason→act→observe loop run), not a model concept, so
# it lives here rather than in the AI layer.

TraceStatus = Literal["completed", "failed", "aborted"]


@dataclass
class TraceResult:
    """Outcome of one complete reason→act→observe trace."""

    reply: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Retained until callers migrate from iteration terminology.
    iterations: int = 0
    aborted: bool = False
    status: TraceStatus = "completed"
    stop_reason: StopReason = "stop"
    error: str = ""

    @property
    def turn_count(self) -> int:
        return self.iterations


# Compatibility alias for persisted/session integrations that still import it.
TurnResult = TraceResult
