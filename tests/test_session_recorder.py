"""Batch B / plan §5.10: SessionRecorder acceptance tests.

- a real tool turn persists as a full per-message node chain
  (user → assistant(tool calls) → tool_result → assistant(answer));
- JSONL is the authoritative tree: context recovers from the file alone;
- state entries (model_change / thinking_level_change) are restored by path;
- custom messages persist as custom_message entries;
- write failures are surfaced honestly, never faked.
"""

from __future__ import annotations

from lsm_harness.agent.messages import (
    AssistantMessage,
    CustomMessage,
    UserMessage,
    custom_message,
    message_preview,
)
from lsm_harness.coding_agent.session_recorder import SessionRecorder
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.loop.agent import run_loop
from lsm_harness.memory.facade import Memory
from lsm_harness.ops.session_store import read_session_entries
from lsm_harness.runtime import Session
from lsm_harness.tools.registry import Tool, ToolRegistry
from lsm_harness.types import ModelResponse, ToolCall

from helpers import QueueClient


def build_session(tmp_path, client, session_id="session-rec", **overrides):
    settings = Settings(
        api_key=overrides.pop("api_key", "test-key"),
        home=tmp_path,
        consolidate_every=99,
        **overrides,
    )
    conn = connect(tmp_path)
    memory = Memory(conn, settings, client)
    return conn, memory, Session(settings, memory, session_id=session_id)


def echo_registry():
    tools = ToolRegistry()
    tools.register(
        Tool(
            "echo",
            "echo a value",
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            lambda value: f"ok:{value}",
        )
    )
    return tools


# ── full tool-turn node chain (§5.10) ────────────────────────────


def test_tool_turn_persists_full_node_chain(tmp_path):
    """A two-iteration tool request lands in the tree as four entries —
    user → assistant(tool call) → tool_result → assistant(answer) — each
    parented to the previous, never a fused record."""
    _, _, session = build_session(tmp_path, QueueClient())
    recorder = session.recorder
    loop_client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="最终回答"),
    )
    # app-level wiring in miniature: record the user message, then let the
    # recorder listener capture every kernel message_end event.
    recorder.record(UserMessage(content="帮我 echo"), source="test")
    result = run_loop(
        client=loop_client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "帮我 echo"}],
        tools=echo_registry(),
        max_iterations=4,
        max_tokens=100,
        emit=lambda *_: None,
        listeners=[recorder.listener()],
    )
    assert result.status == "completed"

    entries = [e for e in read_session_entries(session.jsonl_path) if e.type == "message"]
    assert len(entries) == 4
    roles = [e.message.role for e in entries]
    assert roles == ["user", "assistant", "tool", "assistant"]
    # the assistant entry carries structured tool calls
    call = entries[1].message.tool_calls[0]
    assert call.name == "echo" and call.arguments == {"value": "x"}
    # the tool result is its own entry, linked to the call
    assert entries[2].message.tool_call_id == "1"
    assert "ok:x" in message_preview(entries[2].message, limit=1_000_000)
    # parent chain: strictly linear, rooted at the user entry
    assert entries[1].parent_id == entries[0].id
    assert entries[2].parent_id == entries[1].id
    assert entries[3].parent_id == entries[2].id
    # no fused representation anywhere in the file
    assert "[tools used:" not in session.jsonl_path.read_text(encoding="utf-8")


# ── JSONL standalone recovery (§5.10) ────────────────────────────


def test_context_recovers_from_jsonl_alone(tmp_path):
    """With an empty chat_log (e.g. SQLite lost), the tree still rebuilds
    the full context — JSONL is authoritative, SQLite a projection."""
    _, _, session = build_session(tmp_path, QueueClient(), session_id="durable")
    recorder = session.recorder
    recorder.record(UserMessage(content="问题一"), source="test")
    recorder.record(AssistantMessage(text="回答一"), source="test")
    recorder.record(UserMessage(content="问题二"), source="test")
    recorder.record(AssistantMessage(text="回答二"), source="test")

    # a brand-new SQLite (chat_log lost) pointing at the same home
    import tempfile
    other_home = tmp_path / "elsewhere"
    settings = Settings(api_key="k", home=tmp_path, consolidate_every=99)
    conn = connect(other_home)  # fresh, empty database elsewhere
    memory = Memory(conn, settings, QueueClient())
    recovered = Session(settings, memory, session_id="durable")

    context = recovered.build_session_context()
    assert context is not None
    assert [message_preview(m, limit=1_000_000) for m in context.messages] == [
        "问题一", "回答一", "问题二", "回答二",
    ]


# ── state entries restore by path (§5.9 / §5.10) ─────────────────


def test_model_and_thinking_entries_restore_by_path(tmp_path):
    # Explicit initial state so the header baseline is deterministic —
    # branching back past the switch restores THESE values, not None
    # (code-review issue 四: the header carries the initial runtime state).
    _, _, session = build_session(
        tmp_path,
        QueueClient(),
        provider="initial-provider",
        model="initial-model",
        small_model="initial-small",
        thinking="off",
    )
    recorder = session.recorder
    recorder.record(UserMessage(content="阶段一"), source="test")
    recorder.record(AssistantMessage(text="回答一"), source="test")
    leaf_before_switch = recorder.last_entry_id

    session.record_model_change("anthropic", "claude-opus", small_model="claude-haiku")
    session.record_thinking_change("enabled")
    recorder.record(UserMessage(content="阶段二"), source="test")
    recorder.record(AssistantMessage(text="回答二"), source="test")

    # current path sees the new state
    context = session.build_session_context()
    assert context.model == "claude-opus"
    assert context.provider == "anthropic"
    assert context.small_model == "claude-haiku"
    assert context.thinking_level == "enabled"

    # branch back past the switch: the header's initial state returns
    session.branch(leaf_before_switch, lambda *_: None)
    restored = session.build_session_context()
    assert restored.model == "initial-model"
    assert restored.provider == "initial-provider"
    assert restored.small_model == "initial-small"
    assert restored.thinking_level == "off"
    assert [message_preview(m, limit=1_000_000) for m in restored.messages] == [
        "阶段一", "回答一",
    ]


def test_state_entries_never_enter_messages(tmp_path):
    _, _, session = build_session(tmp_path, QueueClient())
    session.record_model_change("openai", "gpt-x")
    session.record_thinking_change("auto")
    session.recorder.record(UserMessage(content="嗨"), source="test")
    session.recorder.record(AssistantMessage(text="嗨"), source="test")
    context = session.build_session_context()
    assert len(context.messages) == 2  # state entries are not messages


# ── custom message persistence (§5.10) ───────────────────────────


def test_custom_message_round_trips_as_custom_message_entry(tmp_path):
    _, _, session = build_session(tmp_path, QueueClient())
    recorder = session.recorder
    recorder.record(UserMessage(content="问题"), source="test")
    recorder.record(
        custom_message("progress_note", content="已完成 50%", fields={"pct": 50}),
        source="test",
    )
    recorder.record(AssistantMessage(text="回答"), source="test")

    entries = read_session_entries(session.jsonl_path)
    customs = [e for e in entries if e.type == "custom_message"]
    assert len(customs) == 1
    entry = customs[0]
    assert isinstance(entry.message, CustomMessage)
    assert entry.message.custom_type == "progress_note"
    assert entry.message.fields["pct"] == 50
    # it sits in the parent chain like any other entry
    messages = [e for e in entries if e.type == "message"]
    assert entry.parent_id == messages[0].id
    assert messages[1].parent_id == entry.id


# ── honest write failure (§5.10) ─────────────────────────────────


def test_write_failure_is_flagged_never_faked(tmp_path):
    events = []
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    # sessions dir exists as a FILE: mkdir/open must fail with OSError
    recorder = SessionRecorder(
        blocker / "sessions" / "s.jsonl",
        "s",
        cwd=str(tmp_path),
        emit=lambda k, d: events.append((k, d)),
        start_parent_id="s",
    )
    recorder.record(UserMessage(content="hi"), source="test")
    recorder.record(AssistantMessage(text="hello"), source="test")  # triggers flush

    assert recorder.error is not None
    assert any(k == "session.jsonl_write_failed" for k, _ in events)
    # the failure did not raise, and the parent chain kept moving
    assert recorder.last_entry_id != "s"


def test_recorder_listener_never_raises(tmp_path):
    """A listener exception cannot propagate into the agent loop."""
    recorder = SessionRecorder(
        tmp_path / "s.jsonl", "s", cwd=str(tmp_path), start_parent_id="s"
    )

    class Weird:
        pass

    listener = recorder.listener()
    listener(Weird())  # unknown event type: ignored
    # MessageEndEvent with an unserialisable message must not raise either
    from lsm_harness.agent.events import MessageEndEvent

    listener(MessageEndEvent(message=UserMessage(content="ok"), source="user"))
    assert recorder.error is None  # a valid record succeeded
