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
from lsm_harness.ai.types import (
    CacheRetention,
    Model,
    PayloadHook,
    ResponseHook,
    StopReason,
    ThinkingBudgets,
    Transport,
)
from lsm_harness.ai.types import ErrorCategory

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
ModelRetryListener = Callable[[int, ErrorCategory, str], None]


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
    """Runtime policy passed to the low-level Agent Loop as one boundary.

    批 4(config/state 切分):``model`` / ``thinking`` 是 per-run 覆盖,
    可为 None——经 ``Agent.run`` 进入时由 ``Agent._full_config`` 按
    "显式 config > agent.state > 默认"解析;直调 ``run_agent_loop``
    时 model 仍是硬要求(loop 对 None 抛 ValueError),thinking 的
    None 等价 "disabled"。预算字段的默认值与 Settings 一致。
    """

    model: Model | str | None = None
    max_iterations: int = 10
    max_tokens: int = 8192
    convert_to_llm: ConvertToLlm = default_convert_to_llm
    transform_context: TransformContext | None = None
    before_tool_call: BeforeToolCall | None = None
    after_tool_call: AfterToolCall | None = None
    tool_execution: ToolExecutionMode = "parallel"
    get_steering_messages: PendingMessageGetter | None = None
    get_follow_up_messages: PendingMessageGetter | None = None
    # Pi continue(): a batch already drained by the caller, delivered at
    # the run's first turn BEFORE the model call, with its source intact
    # (``loop.steered`` / ``loop.followed_up`` events, sink appends).
    # A "steering"-sourced batch also skips the loop's initial steering
    # poll (Pi skipInitialSteeringPoll — that queue was just drained).
    # 批 3: source="user" 的 initial batch 是 prompt 自己的 user 消息
    # (kernel 摄入通道),字符串层对它沉默。
    initial_pending_messages: list[PendingMessage] | None = None
    initial_pending_source: str = "follow_up"
    prepare_next_turn: PrepareNextTurn | None = None
    should_stop_after_turn: ShouldStopAfterTurn | None = None
    on_truncation: Callable[[], tuple[str, list[AgentMessage]]] | None = None
    governor: Any = None
    hooks: LoopHooks | None = None
    thinking: str | None = None
    cache_retention: CacheRetention = "short"
    max_model_retries: int = 2
    on_model_retry: ModelRetryListener | None = None
    on_payload: PayloadHook | None = None
    on_response: ResponseHook | None = None
    thinking_budgets: ThinkingBudgets | None = None
    transport: Transport = "auto"
    max_retry_delay_ms: int | None = None
    max_empty_retries: int = 2
    max_length_recoveries: int = 3
    approval_broker: Any = None
    trace_id: str = ""
    session_id: str = ""
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
