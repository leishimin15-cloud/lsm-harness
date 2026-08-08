"""Provider-neutral contracts shared by the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)


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
    stop_reason: str = ""
    usage: Usage | None = None


@dataclass
class TurnResult:
    reply: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    aborted: bool = False  # True when the loop was cancelled mid-turn


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

