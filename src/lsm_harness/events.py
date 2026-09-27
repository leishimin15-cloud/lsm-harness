"""One JSON-serializable event shape for CLI observers and traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable


@dataclass(frozen=True)
class HarnessEvent:
    type: str
    # ``turn_id`` is retained as a transport compatibility alias.  The value
    # identifies the full Harness.respond() trace, not an individual model turn.
    turn_id: str
    timestamp: str
    data: dict[str, Any]
    event_id: str = ""
    sequence: int = 0
    session_id: str = ""
    duration_ms: int | None = None
    flow: dict[str, Any] = field(default_factory=dict)
    # Portable span metadata.  JSONL remains the source of truth; exporters
    # (OpenTelemetry/Langfuse/Phoenix) can project the same hierarchy without
    # coupling the Agent layer to an observability SDK.
    span_id: str = ""
    parent_span_id: str = ""
    span_kind: str = "CHAIN"

    @property
    def trace_id(self) -> str:
        return self.turn_id

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trace_id"] = self.trace_id
        return payload


Observer = Callable[[HarnessEvent], None]


def make_event(
    event_type: str,
    trace_id: str,
    data: dict[str, Any],
    *,
    session_id: str = "",
    sequence: int = 0,
    duration_ms: int | None = None,
) -> HarnessEvent:
    from lsm_harness.ops.flow import flow_for_event

    span_id, parent_span_id, span_kind = _span_metadata(
        event_type, trace_id, data
    )

    return HarnessEvent(
        type=event_type,
        turn_id=trace_id,
        timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
        data=data,
        event_id=f"{trace_id}:{sequence}",
        sequence=sequence,
        session_id=session_id,
        duration_ms=duration_ms,
        flow=flow_for_event(event_type, data),
        span_id=span_id,
        parent_span_id=parent_span_id,
        span_kind=span_kind,
    )


def _span_metadata(
    event_type: str,
    trace_id: str,
    data: dict[str, Any],
) -> tuple[str, str, str]:
    """Derive a stable, provider-neutral span hierarchy from an event.

    Event producers stay unchanged.  Start/end events for one model turn or
    tool call receive the same logical span id, while the full respond() call
    remains the root span.
    """
    root = trace_id
    if event_type.startswith("trace."):
        return root, "", "AGENT"
    if event_type.startswith("context."):
        return f"{root}:context", root, "CHAIN"
    iteration = data.get("iteration", data.get("turn_index"))
    turn_span = f"{root}:turn:{iteration}" if iteration is not None else root
    if event_type.startswith(("turn.", "llm.")):
        if event_type.startswith("llm.tool_call."):
            call_key = data.get("tool_id", data.get("tool_index", "unknown"))
            return f"{turn_span}:tool-call:{call_key}", turn_span, "TOOL"
        return turn_span, root, "LLM"
    if event_type.startswith("tool."):
        call_key = data.get("tool_call_id", data.get("tool", "unknown"))
        return f"{root}:tool:{call_key}", root, "TOOL"
    if event_type.startswith("persistence."):
        return f"{root}:persistence", root, "CHAIN"
    if "compaction" in event_type:
        return f"{root}:compaction", root, "CHAIN"
    return f"{root}:{event_type}", root, "CHAIN"
