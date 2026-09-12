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
the header plus buffered entries are written in one shot.  If the run is
interrupted or fails before that, the app calls ``flush()`` instead: the
question WAS asked — the tree (the fact source) keeps it so a later
``continue()`` — possibly after a restart — can answer it.  A run that
died before recording anything flushes an empty buffer: no file, no half
a session.  Continuing sessions (file exists) append line by line
immediately.

Write failures are surfaced honestly: ``error`` is set and a
``session.jsonl_write_failed`` event is emitted — the recorder never
pretends a partial write was a full persistence.
"""

from __future__ import annotations

import json
import os
import tempfile
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
        persist_transform: Callable[[AgentMessage, str], AgentMessage] | None = None,
    ) -> None:
        self.path = path
        self.session_id = session_id
        self._cwd = cwd
        self._emit = emit
        self._entry_listener: Callable[[SessionEntry], None] | None = None
        # 批 3:kernel 事件携带 runtime 消息(图片含 base64);落树前过
        # 此变换(由 Session 注入,只动 source="user" 的 prompt 消息),
        # 会话文件格式与旧显式 record 路径逐字节一致。
        self._persist_transform = persist_transform
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

    def set_entry_listener(
        self, listener: Callable[[SessionEntry], None] | None
    ) -> None:
        """Observe logical appends without coupling storage to product events."""
        self._entry_listener = listener

    def listener(self) -> AgentEventListener:
        """The kernel-event listener to plug into AgentLoopConfig.listeners."""
        return self._on_event

    def _on_event(self, event) -> None:
        # A listener must NEVER raise into the loop — a persistence hiccup
        # cannot take down the run.  Failures land in ``error`` + event.
        try:
            if isinstance(event, MessageEndEvent):
                message = event.message
                # The kernel emits Pi's complete lifecycle for an aborted
                # partial assistant message, but this product's explicit
                # respond_continue() contract resumes from the unanswered
                # user/tool node. Keep the partial available to live UI and
                # Trace without advancing the durable branch tip past that
                # resumable node.
                if (
                    event.source == "assistant"
                    and isinstance(message, AssistantMessage)
                    and message.stop_reason == "aborted"
                ):
                    return
                if self._persist_transform is not None:
                    message = self._persist_transform(message, event.source)
                self.record(message, source=event.source)
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
        if self._entry_listener is not None:
            self._entry_listener(entry)

    def flush(self) -> None:
        """Force-write header + buffered entries (interrupted/failed first
        exchange).

        The question was asked even when its run never completed —
        persisting it keeps the tree truthful and lets a later
        ``continue()`` (possibly after a restart) answer it.  No-op once
        flushed or when nothing was ever recorded.
        """
        with self._lock:
            if self._flushed or not self._buffer:
                return
            self._flush_locked()

    # ── internal ───────────────────────────────────────────────

    def _flush_locked(self) -> None:
        """Atomically write header + buffered entries (first flush)."""
        header = SessionHeader.create(self.session_id, self._cwd, **{
            k: self._initial_state.get(k, "")
            for k in ("provider", "model", "small_model", "thinking")
        })
        lines = [_entry_to_dict(header)]
        lines.extend(_entry_to_dict(entry) for entry in self._buffer)
        if self._write_lines(lines, mode="w"):
            self._flushed = True
            self._buffer = []

    def _write_lines(self, rows: list[dict[str, Any]], *, mode: str) -> bool:
        temporary_name: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "w":
                handle = tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    delete=False,
                )
                temporary_name = handle.name
            else:
                handle = self.path.open("a", encoding="utf-8")
            with handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if temporary_name is not None:
                os.replace(temporary_name, self.path)
                temporary_name = None
            return True
        except OSError as exc:
            # Honest failure: flag + event, never silent partial success.
            self.error = f"{type(exc).__name__}: {exc}"
            self._notify("session.jsonl_write_failed", {
                "session_id": self.session_id,
                "error": self.error,
            })
            return False
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass

    def _notify(self, kind: str, data: dict[str, Any]) -> None:
        if self._emit is not None:
            self._emit(kind, data)


def _is_first_assistant(entry: SessionEntry) -> bool:
    return isinstance(entry, MessageEntry) and isinstance(
        entry.message, AssistantMessage
    )


__all__ = ["SessionRecorder"]
