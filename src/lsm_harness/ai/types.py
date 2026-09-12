"""Provider-neutral model, message, and streaming contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Protocol

from lsm_harness.ai.messages import Message


StopReason = Literal["tool_calls", "stop", "length", "error", "aborted"]
ThinkingLevel = Literal[
    "off", "minimal", "low", "medium", "high", "xhigh", "max"
]
CacheRetention = Literal["none", "short", "long"]
Transport = Literal["auto", "sse", "websocket", "websocket-cached"]
AuthMode = Literal["api_key", "bearer"]
ThinkingBudgets = dict[ThinkingLevel, int]
PayloadHook = Callable[[dict[str, Any], "Model"], dict[str, Any] | None]
ResponseHook = Callable[[dict[str, Any], "Model"], None]
CacheControlFormat = Literal["none", "anthropic", "openai"]
ThinkingFormat = Literal["none", "reasoning_effort", "deepseek", "anthropic"]
ErrorCategory = Literal["arrearage", "rate_limit", "transient", "permanent", "aborted"]


def normalize_stop_reason(reason: str) -> StopReason:
    """Map provider-specific finish reasons to the Agent Loop contract."""
    normalized = (reason or "stop").strip().lower()
    if normalized in {"tool_calls", "tool_use", "tooluse", "function_call"}:
        return "tool_calls"
    if normalized in {"stop", "end_turn", "stop_sequence", "refusal"}:
        return "stop"
    if normalized in {
        "length",
        "max_tokens",
        "max_output_tokens",
        "model_context_window_exceeded",
    }:
        return "length"
    if normalized in {"aborted", "cancelled", "canceled"}:
        return "aborted"
    return "error"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Tool:
    """Pure model-visible tool descriptor owned by the AI layer."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_cache_read: float = 0.0
    cost_cache_write: float = 0.0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def cost_total(self) -> float:
        return (
            self.cost_input
            + self.cost_output
            + self.cost_cache_read
            + self.cost_cache_write
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "cost_input": self.cost_input,
            "cost_output": self.cost_output,
            "cost_cache_read": self.cost_cache_read,
            "cost_cache_write": self.cost_cache_write,
            "cost_total": self.cost_total,
        }

    def with_model_cost(self, model: "Model") -> "Usage":
        scale = 1_000_000
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            cost_input=self.input_tokens * model.input_cost_per_million / scale,
            cost_output=self.output_tokens * model.output_cost_per_million / scale,
            cost_cache_read=(
                self.cache_read_tokens
                * model.cache_read_cost_per_million
                / scale
            ),
            cost_cache_write=(
                self.cache_write_tokens
                * model.cache_write_cost_per_million
                / scale
            ),
        )


@dataclass(frozen=True)
class Model:
    """One model plus the API dialect and capabilities used to call it."""

    id: str
    api: str
    provider: str
    name: str = ""
    base_url: str | None = None
    reasoning: bool = False
    input_modalities: tuple[Literal["text", "image"], ...] = ("text",)
    context_window: int = 0
    max_tokens: int = 0
    thinking_level_map: dict[ThinkingLevel, str | None] = field(default_factory=dict)
    thinking_format: ThinkingFormat = "none"
    cache_control_format: CacheControlFormat = "none"
    supports_long_cache_retention: bool = False
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    cache_read_cost_per_million: float = 0.0
    cache_write_cost_per_million: float = 0.0
    auth_mode: AuthMode = "api_key"
    headers: dict[str, str] = field(default_factory=dict)
    allow_empty_thinking_signature: bool = False
    force_adaptive_thinking: bool = False


@dataclass(frozen=True)
class AIContext:
    """LLM-visible context after Agent messages have been converted."""

    system_prompt: str
    messages: list[Message]
    tools: list[Tool]


@dataclass(frozen=True)
class StreamOptions:
    """Provider-neutral options for one model request."""

    max_tokens: int
    api_key: str = ""
    reasoning: ThinkingLevel = "off"
    cache_retention: CacheRetention = "none"
    session_id: str = ""
    timeout: float = 120.0
    interrupt: Any = None
    max_retries: int = 2
    on_retry: Callable[[int, ErrorCategory, str], None] | None = None
    on_payload: PayloadHook | None = None
    on_response: ResponseHook | None = None
    thinking_budgets: ThinkingBudgets | None = None
    transport: Transport = "auto"
    max_retry_delay_ms: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    thinking: str = ""
    thinking_signature: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = "stop"
    usage: Usage = field(default_factory=Usage)
    error_message: str = ""


AssistantMessageEventKind = Literal[
    "start",
    "text_start",
    "text_delta",
    "text_end",
    "thinking_start",
    "thinking_delta",
    "thinking_signature_delta",
    "thinking_end",
    "toolcall_start",
    "toolcall_delta",
    "toolcall_end",
    "done",
    "error",
]


@dataclass(frozen=True)
class AssistantMessageEvent:
    """One event in the canonical model-response stream."""

    kind: AssistantMessageEventKind
    partial: ModelResponse
    text_delta: str = ""
    thinking_delta: str = ""
    tool_index: int = 0
    tool_id: str = ""
    tool_name: str = ""
    arguments_delta: str = ""
    error_category: ErrorCategory | None = None


# ── streaming types ──────────────────────────────────────────────

StreamDeltaKind = Literal["text_delta", "tool_call_start", "tool_call_delta", "done"]


@dataclass(frozen=True)
class StreamDelta:
    """Single event from a streaming model response.

    Lifecycle per response:
        text_delta* → (tool_call_start → tool_call_delta*) → done

    ``text_delta`` may interleave with ``tool_call_delta`` chunks
    when the model emits thinking text alongside tool calls.
    """

    kind: StreamDeltaKind
    # text_delta
    text: str = ""
    # tool_call_start
    tool_index: int = 0
    tool_id: str = ""
    tool_name: str = ""
    # tool_call_delta
    arguments_delta: str = ""
    # done
    stop_reason: StopReason | Literal[""] = ""
    usage: Usage | None = None


class ModelClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ModelResponse: ...

    def stream_complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> Iterator[StreamDelta]: ...


class StreamFunction(Protocol):
    """Canonical provider translator contract used by the Agent Loop."""

    def __call__(
        self,
        model: Model,
        context: AIContext,
        options: StreamOptions,
    ) -> Iterator[AssistantMessageEvent]: ...
