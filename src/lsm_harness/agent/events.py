"""Typed kernel events and the synchronous event barrier (Chapter 7).

Ten kernel events model a 4-layer nested lifecycle::

    agent_start ─────────────────────────────── agent_end
      turn_start ─────────────────────── turn_end
        message_start → message_update ×N → message_end
        tool_execution_start → tool_execution_update ×N → tool_execution_end

Pi's ``await emit(...)`` translates to plain synchronous dispatch: calling
every listener in subscription order IS the barrier — there is no microtask
interleaving in sync code.  ``tool_execution_update`` needs no
collect-then-batch either (that pattern exists to batch microtask awaits);
its design intent is preserved by state exemption below plus the existing
``_ProgressGate`` in ``agent/tools.py`` (Pi's ``acceptingUpdates``).

Listener errors propagate (fail-fast fuse); pass ``wrap=True`` to
:meth:`AgentEventSink.subscribe` for untrusted third-party listeners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Union

from lsm_harness.agent.messages import AgentMessage, message_preview
from lsm_harness.ai.types import AssistantMessageEvent, StopReason

if TYPE_CHECKING:
    # Agent-layer concept (refactor plan §9.1); imported lazily to avoid
    # the agent.types → agent.events import cycle.
    from lsm_harness.agent.types import TraceStatus

# The pre-Chapter-7 string-typed emit channel, still used by product-tier
# events (trace.*/loop.*/llm.started/...). Triplicated ``Emit`` aliases in
# agent_loop.py / hooks.py / memory/facade.py name this same shape.
LegacyEmit = Callable[[str, dict], None]


# ---------------------------------------------------------------------------
# The ten kernel events.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentStartEvent:
    model: str
    kind: Literal["agent_start"] = "agent_start"


@dataclass(frozen=True)
class AgentEndEvent:
    status: TraceStatus
    stop_reason: StopReason
    error: str = ""
    kind: Literal["agent_end"] = "agent_end"


@dataclass(frozen=True)
class TurnStartEvent:
    turn_index: int
    model: str
    kind: Literal["turn_start"] = "turn_start"


@dataclass(frozen=True)
class TurnEndEvent:
    turn_index: int
    model: str
    stop_reason: str
    status: str
    usage: dict[str, int]
    tool_count: int
    tool_error_count: int
    kind: Literal["turn_end"] = "turn_end"


@dataclass(frozen=True)
class MessageStartEvent:
    message: AgentMessage
    source: str = "assistant"  # assistant | tool | steering | follow_up
    turn_index: int = 0
    kind: Literal["message_start"] = "message_start"


@dataclass(frozen=True)
class MessageUpdateEvent:
    """Streaming update: full snapshot + the ai-layer event passed through."""

    message: AgentMessage
    assistant_message_event: AssistantMessageEvent
    turn_index: int = 0
    kind: Literal["message_update"] = "message_update"


@dataclass(frozen=True)
class MessageEndEvent:
    message: AgentMessage
    source: str = "assistant"
    turn_index: int = 0
    kind: Literal["message_end"] = "message_end"


@dataclass(frozen=True)
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    label: str
    effect: str = ""
    args: dict[str, Any] | None = None
    kind: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass(frozen=True)
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    label: str
    partial: str
    kind: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass(frozen=True)
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    label: str
    is_error: bool
    kind: Literal["tool_execution_end"] = "tool_execution_end"


AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
]
AgentEventListener = Callable[[AgentEvent], None]


# ---------------------------------------------------------------------------
# AgentEventSink — Pi's Agent.processEvents translated to sync Python.
# ---------------------------------------------------------------------------


class AgentEventSink:
    """Owns run state derived from events and dispatches them in order.

    State first, listeners second: by the time a listener runs, the state
    it observes is already current.  Appending to ``messages`` happens here
    (on ``message_end``), not in the loop — Pi's ownership rule.
    """

    def __init__(self, messages: list[AgentMessage] | None = None) -> None:
        self._listeners: list[tuple[AgentEventListener, bool]] = []
        self.messages: list[AgentMessage] = messages if messages is not None else []
        self.streaming_message: AgentMessage | None = None
        self.current_turn: int = 0

    def subscribe(
        self,
        listener: AgentEventListener,
        *,
        wrap: bool = False,
    ) -> Callable[[], None]:
        """Register a listener; returns an unsubscribe function.

        ``wrap=True`` isolates untrusted listeners with try/except so a
        third-party extension cannot take down the run.
        """
        entry = (listener, wrap)
        self._listeners.append(entry)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(entry)
            except ValueError:
                pass

        return unsubscribe

    def replace_messages(self, messages: list[AgentMessage]) -> None:
        """Rebind the owned message list (truncation / next-turn rebuilds)."""
        self.messages = messages

    def process_event(self, event: AgentEvent) -> None:
        """Update state, then call every listener in subscription order."""
        self._update_state(event)
        for listener, wrapped in list(self._listeners):
            if wrapped:
                try:
                    listener(event)
                except Exception:
                    pass
            else:
                listener(event)

    def _update_state(self, event: AgentEvent) -> None:
        if isinstance(event, TurnStartEvent):
            self.current_turn = event.turn_index
        elif isinstance(event, MessageStartEvent):
            self.streaming_message = event.message
        elif isinstance(event, MessageUpdateEvent):
            self.streaming_message = event.message
        elif isinstance(event, MessageEndEvent):
            self.streaming_message = None
            self.messages.append(event.message)
        # ToolExecutionUpdateEvent is deliberately state-exempt: progress
        # updates are high-frequency, low-value and mergeable.


# ---------------------------------------------------------------------------
# Legacy adapter — reproduces the pre-Chapter-7 string events byte-for-byte
# so tracer / flow diagrams / CLI / TUI / Web and the existing test-suite
# observe an unchanged stream. Subscribed FIRST so legacy ordering holds.
# ---------------------------------------------------------------------------


def _preview(message: AgentMessage) -> str:
    return message_preview(message, limit=200)


def make_legacy_adapter(emit: LegacyEmit) -> AgentEventListener:
    """Adapt typed kernel events onto the legacy string-event channel."""

    def listener(event: AgentEvent) -> None:
        if isinstance(event, TurnStartEvent):
            emit("turn.started", {
                "turn_index": event.turn_index,
                "iteration": event.turn_index,
                "model": event.model,
            })
        elif isinstance(event, TurnEndEvent):
            emit("turn.completed", {
                "turn_index": event.turn_index,
                "iteration": event.turn_index,
                "model": event.model,
                "stop_reason": event.stop_reason,
                "status": event.status,
                "usage": event.usage,
                "tool_count": event.tool_count,
                "tool_error_count": event.tool_error_count,
            })
        elif isinstance(event, MessageStartEvent):
            if event.source in ("steering", "follow_up"):
                emit("message.started", {
                    "role": event.message.role,
                    "source": event.source,
                    "message": _preview(event.message),
                })
        elif isinstance(event, MessageUpdateEvent):
            _emit_stream_delta(emit, event)
        elif isinstance(event, MessageEndEvent):
            if event.source in ("steering", "follow_up"):
                emit("message.completed", {
                    "role": event.message.role,
                    "source": event.source,
                    "message": _preview(event.message),
                })
        elif isinstance(event, ToolExecutionStartEvent):
            emit("tool.started", {
                "tool": event.tool_name,
                "label": event.label,
                "tool_call_id": event.tool_call_id,
                "effect": event.effect,
            })
        elif isinstance(event, ToolExecutionUpdateEvent):
            emit("tool.progress", {
                "tool": event.tool_name,
                "label": event.label,
                "tool_call_id": event.tool_call_id,
                "delta": event.partial,
            })
        elif isinstance(event, ToolExecutionEndEvent):
            emit("tool.execution_end", {
                "tool": event.tool_name,
                "label": event.label,
                "tool_call_id": event.tool_call_id,
                "is_error": event.is_error,
            })
        # AgentStartEvent / AgentEndEvent are typed-only: trace.* product
        # events keep covering run boundaries on the legacy channel.

    return listener


def _emit_stream_delta(emit: LegacyEmit, event: MessageUpdateEvent) -> None:
    """Fold the ai-layer delta kinds back into the legacy llm.* names."""
    ai_event = event.assistant_message_event
    iteration = event.turn_index
    kind = ai_event.kind
    if kind == "text_start":
        emit("llm.text.start", {"iteration": iteration})
    elif kind == "text_delta":
        emit("llm.text.delta", {"text": ai_event.text_delta, "iteration": iteration})
    elif kind == "text_end":
        emit("llm.text.end", {"iteration": iteration, "text": ai_event.partial.text})
    elif kind == "thinking_start":
        emit("llm.thinking.start", {"iteration": iteration})
    elif kind == "thinking_delta":
        emit("llm.thinking.delta", {"text": ai_event.thinking_delta, "iteration": iteration})
    elif kind == "thinking_end":
        emit("llm.thinking.end", {"iteration": iteration, "text": ai_event.partial.thinking})
    elif kind == "toolcall_start":
        emit("llm.tool_call.start", {
            "iteration": iteration,
            "tool_index": ai_event.tool_index,
            "tool_id": ai_event.tool_id,
            "tool_name": ai_event.tool_name,
        })
    elif kind == "toolcall_delta":
        emit("llm.tool_call.delta", {
            "iteration": iteration,
            "tool_index": ai_event.tool_index,
            "arguments_delta": ai_event.arguments_delta,
        })
    elif kind == "toolcall_end":
        emit("llm.tool_call.end", {
            "iteration": iteration,
            "tool_index": ai_event.tool_index,
            "tool_id": ai_event.tool_id,
            "tool_name": ai_event.tool_name,
        })
    # start / done / error were never re-emitted as llm.* — keep it that way.
