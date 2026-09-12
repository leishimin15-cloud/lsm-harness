"""Typed events owned by the coding-session layer.

Agent lifecycle events pass through unchanged. This module adds product events
that do not belong in the reusable Agent package: queue snapshots, retries,
errors, state changes, and final settlement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Literal, Union

from lsm_harness.agent.events import AgentEvent


@dataclass(frozen=True)
class QueueUpdateEvent:
    steering: tuple[str, ...] = ()
    follow_up: tuple[str, ...] = ()
    kind: Literal["queue_update"] = "queue_update"


@dataclass(frozen=True)
class AutoRetryStartEvent:
    attempt: int
    category: str
    message: str
    max_attempts: int = 0
    delay_ms: int = 0
    kind: Literal["auto_retry_start"] = "auto_retry_start"


# Compatibility name retained for callers written against the first typed
# event draft. It is the same runtime type, not a second event vocabulary.
RetryEvent = AutoRetryStartEvent


@dataclass(frozen=True)
class AutoRetryEndEvent:
    success: bool
    attempt: int
    final_error: str | None = None
    kind: Literal["auto_retry_end"] = "auto_retry_end"


@dataclass(frozen=True)
class ErrorEvent:
    phase: str
    message: str
    kind: Literal["error"] = "error"


@dataclass(frozen=True)
class AgentSettledEvent:
    status: str
    kind: Literal["agent_settled"] = "agent_settled"


@dataclass(frozen=True)
class CompactionStartEvent:
    reason: Literal["manual", "threshold", "overflow"]
    kind: Literal["compaction_start"] = "compaction_start"


@dataclass(frozen=True)
class CompactionEndEvent:
    reason: Literal["manual", "threshold", "overflow"]
    result: Any = None
    aborted: bool = False
    will_retry: bool = False
    error_message: str | None = None
    kind: Literal["compaction_end"] = "compaction_end"


@dataclass(frozen=True)
class EntryAppendedEvent:
    entry: Any
    kind: Literal["entry_appended"] = "entry_appended"


@dataclass(frozen=True)
class SessionInfoChangedEvent:
    name: str | None = None
    kind: Literal["session_info_changed"] = "session_info_changed"


@dataclass(frozen=True)
class ToolHistoryRepairedEvent:
    synthesized_results: int = 0
    dropped_orphan_results: int = 0
    dropped_duplicate_results: int = 0
    reordered_results: int = 0
    kind: Literal["tool_history_repaired"] = "tool_history_repaired"


@dataclass(frozen=True)
class ModelChangedEvent:
    provider: str
    model: str
    kind: Literal["model_changed"] = "model_changed"


@dataclass(frozen=True)
class ThinkingLevelChangedEvent:
    level: str
    kind: Literal["thinking_level_changed"] = "thinking_level_changed"


@dataclass(frozen=True)
class SessionChangedEvent:
    session_id: str
    kind: Literal["session_changed"] = "session_changed"


CodingSessionOwnEvent = Union[
    QueueUpdateEvent,
    AutoRetryStartEvent,
    AutoRetryEndEvent,
    ErrorEvent,
    AgentSettledEvent,
    CompactionStartEvent,
    CompactionEndEvent,
    EntryAppendedEvent,
    SessionInfoChangedEvent,
    ToolHistoryRepairedEvent,
    ModelChangedEvent,
    ThinkingLevelChangedEvent,
    SessionChangedEvent,
]
CodingSessionEvent = Union[AgentEvent, CodingSessionOwnEvent]
CodingSessionEventListener = Callable[[CodingSessionEvent], None]


class CodingSessionEventSink:
    """Synchronous event barrier for product-level subscribers."""

    def __init__(self) -> None:
        self._listeners: list[tuple[CodingSessionEventListener, bool]] = []

    def subscribe(
        self,
        listener: CodingSessionEventListener,
        *,
        wrap: bool = False,
    ) -> Callable[[], None]:
        entry = (listener, wrap)
        self._listeners.append(entry)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(entry)
            except ValueError:
                pass

        return unsubscribe

    def publish(self, event: CodingSessionEvent) -> None:
        for listener, wrapped in list(self._listeners):
            if wrapped:
                try:
                    listener(event)
                except Exception:
                    pass
            else:
                listener(event)


def coding_session_event_to_dict(event: CodingSessionEvent) -> dict[str, Any]:
    """Serialize the typed SDK/RPC contract without legacy trace wrappers."""
    if not is_dataclass(event):
        raise TypeError(f"unsupported coding-session event: {type(event).__name__}")
    payload = asdict(event)
    kind = payload.pop("kind", type(event).__name__)
    return {"type": kind, **payload}


__all__ = [
    "AgentSettledEvent",
    "AutoRetryEndEvent",
    "AutoRetryStartEvent",
    "CodingSessionEvent",
    "CodingSessionEventListener",
    "CodingSessionEventSink",
    "coding_session_event_to_dict",
    "ErrorEvent",
    "CompactionEndEvent",
    "CompactionStartEvent",
    "EntryAppendedEvent",
    "ModelChangedEvent",
    "QueueUpdateEvent",
    "RetryEvent",
    "SessionChangedEvent",
    "SessionInfoChangedEvent",
    "ThinkingLevelChangedEvent",
    "ToolHistoryRepairedEvent",
]
