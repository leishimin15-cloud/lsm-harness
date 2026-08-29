"""Thread-safe coordination and read models for the local Web console."""

from __future__ import annotations

import json
import secrets
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from lsm_harness.app import Harness
from lsm_harness.ops.approval import ApprovalBroker
from lsm_harness.ops.flow import topology_snapshot
from lsm_harness.security import redact_data


class BusyError(RuntimeError):
    pass


class TurnCoordinator:
    """One shared Harness, one active trace, asynchronous controls.

    The HTTP transport still calls this a turn until its routes are migrated.
    """

    def __init__(self, app: Harness):
        self.app = app
        self.token = secrets.token_urlsafe(32)
        self.approvals = ApprovalBroker(
            timeout=app.settings.web_approval_timeout,
            redactor=lambda value: redact_data(value, (app.settings.api_key,)),
        )
        self._lock = threading.Lock()
        self.active_turn_id = ""
        self.active_source = ""

    def claim(self, turn_id: str, source: str = "web") -> None:
        with self._lock:
            if self.active_turn_id or self.app.is_running:
                raise BusyError("another turn is already running")
            self.active_turn_id = turn_id
            self.active_source = source

    def release(self, turn_id: str) -> None:
        with self._lock:
            if self.active_turn_id == turn_id:
                self.active_turn_id = ""
                self.active_source = ""

    def abort(self, turn_id: str) -> bool:
        with self._lock:
            matches = bool(self.active_turn_id and self.active_turn_id == turn_id)
        if matches:
            self.app.abort()
            self.approvals.reject_all("turn aborted")
        return matches

    def steer(self, turn_id: str, message: str) -> bool:
        with self._lock:
            matches = bool(self.active_turn_id and self.active_turn_id == turn_id)
        if matches and message.strip():
            self.app.steer(message.strip())
            return True
        return False

    def ensure_idle(self) -> None:
        with self._lock:
            if self.active_turn_id or self.app.is_running:
                raise BusyError("another turn is already running")

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "busy": bool(self.active_turn_id or self.app.is_running),
                "active_turn_id": self.active_turn_id,
                "source": self.active_source,
            }

    def bootstrap(self) -> dict[str, Any]:
        return {
            "runtime": self.state(),
            "session_id": self.app.session.session_id,
            "model": self.app.settings.model,
            "provider": self.app.settings.provider,
            "thinking": self.app.settings.thinking,
            "topology": topology_snapshot(self.app),
            "approvals": self.approvals.snapshot(),
        }


def read_trace_events(home: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    trace_dir = home / "traces"
    if not trace_dir.exists():
        return events
    for path in sorted(trace_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event.setdefault("event_id", f"legacy:{len(events) + 1}")
            event.setdefault("sequence", 0)
            event.setdefault("session_id", event.get("data", {}).get("session_id", ""))
            event.setdefault("trace_id", event.get("turn_id", ""))
            event.setdefault("duration_ms", None)
            event.setdefault("flow", {})
            event["cursor"] = len(events) + 1
            events.append(event)
    return events


def trace_slice(home: Path, cursor: int = 0, limit: int = 500) -> dict[str, Any]:
    events = read_trace_events(home)
    cursor = max(0, min(int(cursor), len(events)))
    selected = events[cursor:cursor + max(1, min(limit, 2000))]
    return {"events": selected, "cursor": cursor + len(selected), "total": len(events)}


def aggregate_turns(home: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in read_trace_events(home):
        trace_id = str(event.get("trace_id") or event.get("turn_id") or "")
        if trace_id:
            grouped[trace_id].append(event)
    turns = []
    for turn_id, events in grouped.items():
        first = events[0]
        terminal = next(
            (e for e in reversed(events) if e.get("type") in {
                "trace.done", "trace.completed", "trace.aborted", "trace.error", "trace.failed",
                "turn.done", "turn.completed", "turn.aborted", "turn.error", "turn.failed",
            }),
            events[-1],
        )
        status = "running"
        terminal_type = str(terminal.get("type", ""))
        if terminal_type in {"trace.error", "trace.failed", "turn.error", "turn.failed"}:
            status = "error"
        elif terminal_type in {"trace.aborted", "turn.aborted"} or terminal.get("data", {}).get("aborted"):
            status = "aborted"
        elif terminal_type in {"trace.done", "trace.completed", "turn.done", "turn.completed"}:
            status = "done"
        turns.append({
            "turn_id": turn_id,
            "trace_id": turn_id,
            "session_id": first.get("session_id", ""),
            "started_at": first.get("timestamp", ""),
            "duration_ms": terminal.get("duration_ms"),
            "source": first.get("data", {}).get("source", ""),
            "message": first.get("data", {}).get("message", ""),
            "reply": terminal.get("data", {}).get("reply", ""),
            "status": status,
            "event_count": len(events),
        })
    return sorted(turns, key=lambda item: item["started_at"], reverse=True)


def turn_events(home: Path, turn_id: str) -> list[dict[str, Any]]:
    return [event for event in read_trace_events(home) if event.get("turn_id") == turn_id]
