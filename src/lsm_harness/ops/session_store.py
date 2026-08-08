"""JSONL-based session storage: one file per session, every entry typed.

Pi-style approach: the JSONL file IS the session.  It can be:
  - replayed to reconstruct the conversation without API calls
  - exported to share or archive
  - imported from external sources (including pi sessions)
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Literal

from lsm_harness.security import redact_data, redact_text


# ── entry types ──────────────────────────────────────────────────

EntryType = Literal[
    "session",
    "message",
    "compaction",
    "model_change",
    "thinking_level_change",
]


@dataclass
class SessionEntry:
    """Base: every entry has a type, id, parent_id, and timestamp."""

    id: str
    type: EntryType = "message"
    parent_id: str | None = None
    timestamp: str = ""


@dataclass
class SessionHeader(SessionEntry):
    """The first line of every session file."""

    type: str = "session"
    version: int = 1
    cwd: str = ""

    @classmethod
    def create(cls, session_id: str, cwd: str) -> "SessionHeader":
        return cls(
            id=session_id,
            timestamp=datetime.now(UTC).isoformat(),
            cwd=cwd,
        )


@dataclass
class MessageEntry(SessionEntry):
    """A user or assistant message (possibly with tool calls)."""

    type: str = "message"
    role: str = "user"
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] | None = None
    source: str = "cli"

    @classmethod
    def from_exchange(
        cls,
        entry_id: str,
        parent_id: str | None,
        role: str,
        content: str,
        source: str = "cli",
        meta: dict | None = None,
        tool_calls: list[dict] | None = None,
    ) -> "MessageEntry":
        return cls(
            id=entry_id,
            parent_id=parent_id,
            timestamp=datetime.now(UTC).isoformat(),
            role=role,
            content=content,
            source=source,
            meta=meta,
            tool_calls=tool_calls or [],
        )


@dataclass
class CompactionEntry(SessionEntry):
    """A compaction summary was generated for messages before through_chat_id."""

    type: str = "compaction"
    version: int = 1
    summary: str = ""
    through_chat_id: int = 0
    source_message_count: int = 0


@dataclass
class ModelChangeEntry(SessionEntry):
    """The model or provider changed at this point in the session."""

    type: str = "model_change"
    provider: str = ""
    model: str = ""


@dataclass
class ThinkingLevelChange(SessionEntry):
    """The thinking level changed at this point in the session."""

    type: str = "thinking_level_change"
    level: str = "off"


# ── serialisation ────────────────────────────────────────────────

_ENTRY_CLASSES: dict[str, type] = {
    "session": SessionHeader,
    "message": MessageEntry,
    "compaction": CompactionEntry,
    "model_change": ModelChangeEntry,
    "thinking_level_change": ThinkingLevelChange,
}


def _entry_to_dict(entry: SessionEntry) -> dict[str, Any]:
    """Convert a SessionEntry to a plain dict for JSON serialisation."""
    result = {}
    for field_name in entry.__dataclass_fields__:  # type: ignore[attr-defined]
        value = getattr(entry, field_name)
        if value is None:
            continue
        # Skip defaults that are empty
        if field_name == "tool_calls" and not value:
            continue
        if field_name == "meta" and value is None:
            continue
        result[field_name] = value
    return result


def _dict_to_entry(d: dict[str, Any]) -> SessionEntry:
    """Reconstruct a SessionEntry from a plain dict."""
    entry_type = d.get("type", "message")
    cls = _ENTRY_CLASSES.get(entry_type, MessageEntry)
    # Only pass fields the dataclass actually accepts
    valid_fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in d.items() if k in valid_fields}
    return cls(**kwargs)  # type: ignore[call-arg]


def write_session_header(path: Path, header: SessionHeader) -> None:
    _append(path, _entry_to_dict(header))


def append_session_entry(path: Path, entry: SessionEntry, lock: threading.Lock | None = None) -> None:
    _append(path, _entry_to_dict(entry), lock)


def read_session_entries(path: Path) -> list[SessionEntry]:
    """Read all entries from a session JSONL file."""
    if not path.exists():
        return []
    entries: list[SessionEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            entries.append(_dict_to_entry(d))
        except (json.JSONDecodeError, TypeError):
            continue
    return entries


def replay_session(
    entries: list[SessionEntry],
    emit,
    api_key: str = "",
) -> list[dict[str, str]]:
    """Replay a session's entries as conversation events.

    Returns the reconstructed message list (user/assistant pairs)
    suitable for feeding into prepare_context.
    """
    messages: list[dict[str, str]] = []
    current_model = "unknown"
    current_thinking = "off"

    for entry in entries:
        if entry.type == "session":
            emit("session.replay.header", {"session_id": entry.id})
        elif entry.type == "model_change":
            mce = entry  # type: ignore[assignment]
            current_model = f"{mce.provider}/{mce.model}"
            emit("session.replay.model_change", {"model": current_model})
        elif entry.type == "thinking_level_change":
            tlc = entry  # type: ignore[assignment]
            current_thinking = tlc.level
            emit("session.replay.thinking_change", {"level": current_thinking})
        elif entry.type == "compaction":
            ce = entry  # type: ignore[assignment]
            emit("session.replay.compaction", {
                "version": ce.version,
                "through_chat_id": ce.through_chat_id,
                "source_message_count": ce.source_message_count,
            })
        elif entry.type == "message":
            me = entry  # type: ignore[assignment]
            content = redact_text(me.content, (api_key,)) if api_key else me.content
            messages.append({"role": me.role, "content": content})
            emit("session.replay.message", {
                "role": me.role,
                "content": content[:120],
                "source": me.source,
            })

    return messages


# ── internal ─────────────────────────────────────────────────────


def _append(path: Path, data: dict[str, Any], lock: threading.Lock | None = None) -> None:
    line = json.dumps(data, ensure_ascii=False, default=str) + "\n"
    if lock:
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
