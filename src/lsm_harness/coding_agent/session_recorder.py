"""SessionRecorder — persist the live AgentMessage tree from kernel events.

Pi's model: the session file records what the Agent actually did, message
by message, as it happens — not a simplified reconstruction after the
Trace.  The recorder subscribes to typed kernel events (batch B)::

    MessageEndEvent(source=user/steering/follow_up)  → message entry
    MessageEndEvent(source=assistant)                → message entry
    MessageEndEvent(source=tool)                     → message entry
    CustomMessage payloads                           → custom_message entry

The EventSink still owns run-time message appends; the recorder only
persists — it never re-constructs messages.

Deferred first write (refactor plan §5.6): for a NEW session file the
entries buffer in memory until the first assistant message completes, then
the header plus buffered entries are written in one shot — a session file
never contains a lone user question with no answer.  If the run dies
before the first assistant message, nothing was ever written (and
``abandoned`` below records why).  Continuing sessions (file exists)
append line by line immediately.

Write failures are surfaced honestly: ``error`` is set and a
``session.jsonl_write_failed`` event is emitted — the recorder never
pretends a partial write was a full persistence.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from lsm_harness.agent.events import AgentEventListener, MessageEndEvent
from lsm_harness.agent.messages import (
    AgentMessage,
    AssistantMessage,
    CustomMessage,
)
from lsm_harness.ops.session_store import (
    CustomMessageEntry,
    MessageEntry,
    SessionEntry,
    SessionHeader,
    _entry_to_dict,
)

Emit = Callable[[str, dict], None]


class SessionRecorder:
    """Owns one session JSONL file's writes and parent chain."""

    def __init__(
        self,
        path: Path,
        session_id: str,
        *,
        cwd: str = "",
        emit: Emit | None = None,
        start_parent_id: str | None = None,
        initial_state: dict[str, str] | None = None,
    ) -> None:
        self.path = path
        self.session_id = session_id
        self._cwd = cwd
        self._emit = emit
        self._lock = threading.Lock()
        self.error: str | None = None
        # The session's INITIAL runtime state (provider/model/small_model/
        # thinking) — written into the header so resuming never depends on
        # whatever the global Settings happen to be later (issue 四).
        self._initial_state = dict(initial_state or {})
        # An existing non-empty file means this session continues: append
        # immediately.  A missing/empty file defers until the first
        # assistant message completes (no half-written sessions).
        self._flushed = path.exists() and path.stat().st_size > 0
        self._buffer: list[SessionEntry] = []
        # Parent chain starts where the previous run left off (or at the
        # header id for a fresh file).
        self.last_entry_id = start_parent_id or session_id

    @property
    def deferred(self) -> bool:
        """True while a fresh session's first exchange is still buffered."""
        return not self._flushed

    # ── event subscription ─────────────────────────────────────

    def set_emit(self, emit: Emit | None) -> None:
        """(Re)bind the event channel, e.g. to a run's tracer emit."""
        self._emit = emit

    def listener(self) -> AgentEventListener:
        """The kernel-event listener to plug into AgentLoopConfig.listeners."""
        return self._on_event

    def _on_event(self, event) -> None:
        # A listener must NEVER raise into the loop — a persistence hiccup
        # cannot take down the run.  Failures land in ``error`` + event.
        try:
            if isinstance(event, MessageEndEvent):
                self.record(event.message, source=event.source)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._notify("session.jsonl_write_failed", {
                "session_id": self.session_id,
                "error": self.error,
            })

    # ── recording ──────────────────────────────────────────────

    def make_id(self) -> str:
        return f"{self.session_id[:8]}-{uuid4().hex[:12]}"

    def record(
        self,
        message: AgentMessage,
        *,
        source: str,
        meta: dict | None = None,
    ) -> SessionEntry:
        """Turn one completed message into exactly one session entry."""
        if isinstance(message, CustomMessage):
            entry: SessionEntry = CustomMessageEntry(
                id=self.make_id(),
                parent_id=self.last_entry_id,
                timestamp=datetime.now(UTC).isoformat(),
                message=message,
                source=source,
            )
        else:
            entry = MessageEntry.create(
                self.make_id(),
                self.last_entry_id,
                message,
                source=source,
                meta=meta,
            )
        self.append_entry(entry)
        return entry

    def append_entry(self, entry: SessionEntry) -> None:
        """Append a non-message entry (compaction / branch / state change)
        or route a freshly recorded message entry through the buffer."""
        with self._lock:
            if self._flushed:
                self._write_lines([_entry_to_dict(entry)], mode="a")
            else:
                self._buffer.append(entry)
                if _is_first_assistant(entry):
                    self._flush_locked()
            self.last_entry_id = entry.id

    def abandon(self, reason: str) -> None:
        """Mark a never-flushed session as abandoned (failed trace).

        Nothing was written to disk; the reason is only surfaced via the
        event channel so a failed first exchange cannot be mistaken for a
        persisted conversation.
        """
        with self._lock:
            if self._flushed or not self._buffer:
                return
            dropped = len(self._buffer)
            self._buffer = []
        self._notify("session.jsonl_abandoned", {
            "session_id": self.session_id,
            "reason": reason,
            "dropped_entries": dropped,
        })

    # ── internal ───────────────────────────────────────────────

    def _flush_locked(self) -> None:
        """Atomically write header + buffered entries (first flush)."""
        header = SessionHeader.create(self.session_id, self._cwd, **{
            k: self._initial_state.get(k, "")
            for k in ("provider", "model", "small_model", "thinking")
        })
        lines = [_entry_to_dict(header)]
        lines.extend(_entry_to_dict(entry) for entry in self._buffer)
        self._write_lines(lines, mode="w")
        self._flushed = True
        self._buffer = []

    def _write_lines(self, rows: list[dict[str, Any]], *, mode: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open(mode, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            # Honest failure: flag + event, never silent partial success.
            self.error = f"{type(exc).__name__}: {exc}"
            self._notify("session.jsonl_write_failed", {
                "session_id": self.session_id,
                "error": self.error,
            })

    def _notify(self, kind: str, data: dict[str, Any]) -> None:
        if self._emit is not None:
            self._emit(kind, data)


def _is_first_assistant(entry: SessionEntry) -> bool:
    return isinstance(entry, MessageEntry) and isinstance(
        entry.message, AssistantMessage
    )


__all__ = ["SessionRecorder"]
