"""One JSON-serializable event shape for CLI observers and traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Callable


@dataclass(frozen=True)
class HarnessEvent:
    type: str
    turn_id: str
    timestamp: str
    data: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


Observer = Callable[[HarnessEvent], None]


def make_event(event_type: str, turn_id: str, data: dict[str, Any]) -> HarnessEvent:
    return HarnessEvent(
        type=event_type,
        turn_id=turn_id,
        timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
        data=data,
    )

