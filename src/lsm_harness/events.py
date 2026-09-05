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
    )
