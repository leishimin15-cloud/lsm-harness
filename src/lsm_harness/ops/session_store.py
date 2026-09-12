"""JSONL-based session storage: one file per session, every entry typed.

Pi-style approach: the JSONL file IS the session.  It can be:
  - replayed to reconstruct the conversation without API calls
  - exported to share or archive
  - imported from external sources (including pi sessions)

Batch B: ``MessageEntry`` holds a complete typed ``AgentMessage`` — a real
tool turn persists as four entries (user → assistant(tool calls) →
tool_result → assistant(answer)), never a fused pair with a
``[tools used: ...]`` string.

The typed ↔ dict conversion lives HERE (``_message_from_dict`` /
``_message_to_dict``): the dict shape is this module's on-disk JSONL
format, not a general-purpose API.  v3.0 起不再迁移 v1 融合 tool_calls
记录——批次 B 之前写入的旧会话文件里那部分记录不可读。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping

from lsm_harness.agent.messages import (
    CUSTOM_ROLE,
    AgentMessage,
    AgentToolResultMessage,
    AssistantMessage,
    CustomMessage,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
    _tool_call_content,
    _user_content,
    message_preview,
)
from lsm_harness.ai.messages import ImageContent, TextContent
from lsm_harness.security import redact_text


class SessionFileError(ValueError):
    """A durable session cannot be parsed without hiding data loss."""

    def __init__(self, path: Path, line: int, detail: str) -> None:
        self.path = path
        self.line = line
        self.detail = detail
        super().__init__(f"invalid session {path} at line {line}: {detail}")


# ── entry types ──────────────────────────────────────────────────

EntryType = Literal[
    "session",
    "message",
    "custom_message",
    "compaction",
    "branch_summary",
    "model_change",
    "thinking_level_change",
    "label",
    "session_info",
    "custom",
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
    """The first line of every session file (file meta, not a tree node).

    Carries the session's INITIAL runtime state (code-review issue 四):
    a session that never switched models otherwise has no state nodes on
    the tree, and resume would fall back to whatever the global Settings
    happen to be now.  ``build_session_context`` starts from these values
    and applies model_change / thinking_level_change entries on top.
    """

    type: str = "session"
    version: int = 2
    cwd: str = ""
    provider: str = ""
    model: str = ""
    small_model: str = ""
    thinking: str = ""

    @classmethod
    def create(
        cls,
        session_id: str,
        cwd: str,
        *,
        provider: str = "",
        model: str = "",
        small_model: str = "",
        thinking: str = "",
    ) -> "SessionHeader":
        return cls(
            id=session_id,
            timestamp=datetime.now(UTC).isoformat(),
            cwd=cwd,
            provider=provider,
            model=model,
            small_model=small_model,
            thinking=thinking,
        )


def _new_id_prefix() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class MessageEntry(SessionEntry):
    """One typed agent message (user / assistant / tool result)."""

    type: str = "message"
    message: AgentMessage = field(
        default_factory=lambda: UserMessage(content="")
    )
    meta: dict[str, Any] | None = None
    source: str = "cli"

    @classmethod
    def create(
        cls,
        entry_id: str,
        parent_id: str | None,
        message: AgentMessage,
        source: str = "cli",
        meta: dict | None = None,
    ) -> "MessageEntry":
        return cls(
            id=entry_id,
            parent_id=parent_id,
            timestamp=_new_id_prefix(),
            message=message,
            source=source,
            meta=meta,
        )


@dataclass
class CustomMessageEntry(SessionEntry):
    """An application-defined CustomMessage, persisted with its fields."""

    type: str = "custom_message"
    message: CustomMessage = field(
        default_factory=lambda: CustomMessage(custom_type="")
    )
    source: str = "cli"


@dataclass
class CompactionEntry(SessionEntry):
    """A compaction summary was generated at this point in the tree.

    ``first_kept_entry_id`` marks coverage POSITIONALLY: every message
    entry before it on the current path is covered by ``summary``; entries
    from it onward survive.  ``through_chat_id`` remains as the legacy
    chat_log projection marker for pre-batch-B sessions.

    ``tokens_before`` is the context size that triggered this compaction
    (measured or estimated) — Pi records it for diagnostics.
    """

    type: str = "compaction"
    version: int = 1
    summary: str = ""
    first_kept_entry_id: str = ""
    through_chat_id: int = 0
    source_message_count: int = 0
    tokens_before: int = 0
    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)


@dataclass
class BranchSummaryEntry(SessionEntry):
    """Summary of an abandoned branch, injected at the fork point (ch10).

    ``from_id`` is the leaf of the branch that was abandoned; this entry's
    ``parent_id`` points at the fork node the user returned to.
    """

    type: str = "branch_summary"
    summary: str = ""
    from_id: str = ""
    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)


@dataclass
class ModelChangeEntry(SessionEntry):
    """The model or provider changed at this point in the session.

    ``small_model`` rides along because it feeds memory gate /
    consolidation / RAG rerank / compaction+branch summaries — restoring
    only the main model would leave those on the wrong model
    (code-review issue 三).  Old JSONL rows without the field
    deserialize with the empty default.
    """

    type: str = "model_change"
    provider: str = ""
    model: str = ""
    small_model: str = ""


@dataclass
class ThinkingLevelChange(SessionEntry):
    """The thinking level changed at this point in the session."""

    type: str = "thinking_level_change"
    level: str = "off"


@dataclass
class LabelEntry(SessionEntry):
    """A user-facing label pinned to an earlier entry (navigation aid)."""

    type: str = "label"
    target_id: str = ""
    label: str = ""


@dataclass
class SessionInfoEntry(SessionEntry):
    """Display metadata about the session (title etc.); never enters context."""

    type: str = "session_info"
    title: str = ""


@dataclass
class CustomEntry(SessionEntry):
    """Application extension slot for non-message state."""

    type: str = "custom"
    custom_type: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# ── serialisation ────────────────────────────────────────────────
#
# The dict shape produced/consumed below IS the on-disk JSONL format.
# It lives here (not in agent/messages.py) because nothing outside this
# module should ever see it.


def _message_from_dict(data: Mapping[str, Any]) -> AgentMessage:
    """Deserialise one stored dict message into the typed union."""
    role = data.get("role")
    if role == "user":
        return UserMessage(content=_user_content(data.get("content", "")))
    if role == "assistant":
        return AssistantMessage(
            text=data.get("content") or "",
            thinking=data.get("thinking") or "",
            thinking_signature=data.get("thinking_signature") or "",
            tool_calls=tuple(
                _tool_call_content(c) for c in data.get("tool_calls") or ()
            ),
            api=str(data.get("api", "")),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            usage={
                str(key): (
                    float(value) if str(key).startswith("cost_") else int(value)
                )
                for key, value in (data.get("usage") or {}).items()
            },
            stop_reason=str(data.get("stop_reason", "")),
            error_message=str(data.get("error_message", "")),
            timestamp=int(data.get("timestamp", 0) or 0),
        )
    if role == "tool":
        return AgentToolResultMessage(
            tool_call_id=str(data.get("tool_call_id", "")),
            tool_name=str(data.get("tool_name", "")),
            content=str(data.get("content", "")),
            is_error=bool(data.get("is_error", False)),
            details=data.get("details"),
            terminate=bool(data.get("terminate", False)),
        )
    if role == CUSTOM_ROLE:
        # Fields may arrive nested (the "fields" key) or spread at the top
        # level (older stored dicts) — accept both.
        extras = {
            key: value
            for key, value in data.items()
            if key
            not in ("role", "custom_type", "content", "exclude_from_context", "fields")
        }
        nested = data.get("fields")
        if isinstance(nested, Mapping):
            extras = {**nested, **extras}
        return CustomMessage(
            custom_type=str(data.get("custom_type", "")),
            content=data.get("content"),
            fields=extras,
            exclude_from_context=data.get("exclude_from_context"),
        )
    raise ValueError(f"cannot deserialise stored message with role: {role!r}")


def _message_to_dict(message: AgentMessage) -> dict[str, Any]:
    """Serialise a typed agent message into the stored dict shape."""
    if isinstance(message, CustomMessage):
        stored: dict[str, Any] = {
            "role": CUSTOM_ROLE,
            "custom_type": message.custom_type,
            "content": message.content,
            **message.fields,
        }
        if message.exclude_from_context is not None:
            stored["exclude_from_context"] = message.exclude_from_context
        return stored
    if isinstance(message, ToolResultMessage):
        stored = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": message.content,
            "is_error": message.is_error,
        }
        if isinstance(message, AgentToolResultMessage):
            stored["details"] = message.details
            stored["terminate"] = message.terminate
        return stored
    if isinstance(message, AssistantMessage):
        stored = {"role": "assistant", "content": message.text or None}
        if message.thinking:
            stored["thinking"] = message.thinking
        if message.thinking_signature:
            stored["thinking_signature"] = message.thinking_signature
        if message.tool_calls:
            stored["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments, ensure_ascii=False
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        if message.api:
            stored["api"] = message.api
        if message.provider:
            stored["provider"] = message.provider
        if message.model:
            stored["model"] = message.model
        if message.usage:
            stored["usage"] = message.usage
        if message.stop_reason:
            stored["stop_reason"] = message.stop_reason
        if message.error_message:
            stored["error_message"] = message.error_message
        if message.timestamp:
            stored["timestamp"] = message.timestamp
        return stored
    if isinstance(message, UserMessage):
        content = message.content
        if isinstance(content, str):
            return {"role": "user", "content": content}
        blocks: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, TextContent):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContent):
                blocks.append({"type": "image_url", "image_url": {"url": block.url}})
        return {"role": "user", "content": blocks}
    raise TypeError(f"unknown AgentMessage type: {type(message).__name__}")


_ENTRY_CLASSES: dict[str, type] = {
    "session": SessionHeader,
    "message": MessageEntry,
    "custom_message": CustomMessageEntry,
    "compaction": CompactionEntry,
    "branch_summary": BranchSummaryEntry,
    "model_change": ModelChangeEntry,
    "thinking_level_change": ThinkingLevelChange,
    "label": LabelEntry,
    "session_info": SessionInfoEntry,
    "custom": CustomEntry,
}


def _entry_to_dict(entry: SessionEntry) -> dict[str, Any]:
    """Convert a SessionEntry to a plain dict for JSON serialisation."""
    result = {}
    for field_name in entry.__dataclass_fields__:  # type: ignore[attr-defined]
        if field_name == "message":
            continue
        value = getattr(entry, field_name)
        if value is None:
            continue
        # Skip defaults that are empty
        if field_name in ("tool_calls", "read_files", "modified_files", "data") and not value:
            continue
        if field_name == "meta" and value is None:
            continue
        result[field_name] = value
    if isinstance(entry, (MessageEntry, CustomMessageEntry)):
        result["message"] = _message_to_dict(entry.message)
    return result


def _v1_to_message(d: dict[str, Any]) -> AgentMessage:
    """Reconstruct a typed message from a v1 flat entry (role/content/...)."""
    role = d.get("role", "user")
    content = d.get("content", "")
    if role == "assistant":
        return AssistantMessage(
            text=content or "",
            tool_calls=tuple(
                ToolCallContent(
                    id="",
                    name=str(call.get("tool", "")),
                    arguments=dict(call.get("args") or {}),
                )
                for call in d.get("tool_calls") or []
            ),
        )
    if role == "tool":
        return _message_from_dict(d)
    return UserMessage(content=content if isinstance(content, str) else str(content))


def _dict_to_entry(d: dict[str, Any]) -> SessionEntry:
    """Reconstruct a SessionEntry from a plain dict."""
    entry_type = d.get("type", "message")
    cls = _ENTRY_CLASSES.get(entry_type, MessageEntry)
    # Only pass fields the dataclass actually accepts
    valid_fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in d.items() if k in valid_fields and k != "message"}
    if issubclass(cls, MessageEntry):
        if "message" in d:
            kwargs["message"] = _message_from_dict(d["message"])
        else:
            kwargs["message"] = _v1_to_message(d)
    elif issubclass(cls, CustomMessageEntry):
        raw = d.get("message") or {
            "role": "custom",
            "custom_type": d.get("custom_type", ""),
            "content": d.get("content", ""),
        }
        typed = _message_from_dict(raw)
        kwargs["message"] = (
            typed
            if isinstance(typed, CustomMessage)
            else CustomMessage(
                custom_type=str(raw.get("custom_type", "")),
                content=raw.get("content", ""),
            )
        )
    return cls(**kwargs)  # type: ignore[call-arg]


def write_session_header(path: Path, header: SessionHeader) -> None:
    _append(path, _entry_to_dict(header))


def append_session_entry(path: Path, entry: SessionEntry, lock: threading.Lock | None = None) -> None:
    _append(path, _entry_to_dict(entry), lock)


def read_session_entries(path: Path) -> list[SessionEntry]:
    """Read a session strictly, repairing only a torn final record.

    A malformed final physical line can be the result of a process dying
    mid-write, so it is removed atomically. Any earlier malformed line is
    durable corruption and raises :class:`SessionFileError`; silently skipping
    it would splice unrelated parent-chain nodes together.
    """
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    if not content:
        return []
    physical_lines = content.splitlines()
    entries: list[SessionEntry] = []
    for index, raw_line in enumerate(physical_lines):
        line_number = index + 1
        line = raw_line.strip()
        if not line:
            raise SessionFileError(path, line_number, "empty record")
        try:
            d = json.loads(line)
            if not isinstance(d, dict):
                raise TypeError("record must be a JSON object")
            entries.append(_dict_to_entry(d))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            if index == len(physical_lines) - 1 and index > 0:
                valid_prefix = "\n".join(physical_lines[:index]) + "\n"
                _publish_text_atomically(path, valid_prefix)
                return entries
            raise SessionFileError(path, line_number, str(exc)) from exc
    if not content.endswith("\n"):
        _append_newline(path)
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
        elif entry.type == "branch_summary":
            bse = entry  # type: ignore[assignment]
            emit("session.replay.branch_summary", {
                "from_id": bse.from_id,
                "summary": bse.summary[:120],
            })
        elif entry.type == "custom_message":
            cme = entry  # type: ignore[assignment]
            emit("session.replay.custom_message", {
                "custom_type": cme.message.custom_type,
                "content": message_preview(cme.message, limit=120),
            })
        elif entry.type == "message":
            me = entry  # type: ignore[assignment]
            role = me.message.role
            text = message_preview(me.message, limit=1_000_000)
            content = redact_text(text, (api_key,)) if api_key else text
            messages.append({"role": role, "content": content})
            emit("session.replay.message", {
                "role": role,
                "content": content[:120],
                "source": me.source,
            })

    return messages


# ── tree operations (chapter 10) ─────────────────────────────────
#
# Entries form an append-only tree via parent_id ("认父不认子").  All
# navigation walks parent pointers through a by-id map; no entry is ever
# modified or deleted.


def build_by_id(entries: list[SessionEntry]) -> dict[str, SessionEntry]:
    return {entry.id: entry for entry in entries}


def path_to_leaf(
    entries: list[SessionEntry], leaf_id: str | None
) -> list[SessionEntry]:
    """Walk parent pointers from ``leaf_id`` back to the root.

    Returns the path in root → leaf order.  Only entries on this path are
    visible to the LLM — abandoned branches are simply never visited.
    """
    by_id = build_by_id(entries)
    path: list[SessionEntry] = []
    current = by_id.get(leaf_id) if leaf_id else None
    while current is not None:
        path.append(current)
        current = by_id.get(current.parent_id) if current.parent_id else None
    path.reverse()
    return path


def collect_abandoned_branch(
    entries: list[SessionEntry], old_leaf_id: str, new_leaf_id: str
) -> list[SessionEntry]:
    """Collect the entries unique to the old branch (LCA excluded).

    Used by branch summarisation: the LLM summarises exactly this segment —
    "what the user explored before coming back here".  Returns entries in
    chronological (root → leaf of the abandoned segment) order.
    """
    old_path = path_to_leaf(entries, old_leaf_id)
    new_path_ids = {entry.id for entry in path_to_leaf(entries, new_leaf_id)}
    abandoned = [entry for entry in old_path if entry.id not in new_path_ids]
    return abandoned


# ── internal ─────────────────────────────────────────────────────


def _append(path: Path, data: dict[str, Any], lock: threading.Lock | None = None) -> None:
    line = json.dumps(data, ensure_ascii=False, default=str) + "\n"
    if lock:
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def _append_newline(path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish_text_atomically(path: Path, content: str) -> None:
    """Publish a complete repaired sibling without exposing a partial file."""
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
