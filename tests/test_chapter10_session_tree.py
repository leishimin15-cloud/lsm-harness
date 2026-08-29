"""Chapter 10: session tree — append-only branching, path-based context
building, branch summaries, and the chat_log → JSONL backfill."""

from __future__ import annotations

from lsm_harness.agent.messages import (
    AssistantMessage,
    CustomMessage,
    UserMessage,
    default_convert_to_llm,
    message_preview,
)
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.memory.facade import Memory
from lsm_harness.ops.session_store import (
    MessageEntry,
    SessionHeader,
    collect_abandoned_branch,
    path_to_leaf,
    read_session_entries,
)
from lsm_harness.runtime import Session
from lsm_harness.types import ModelResponse

from helpers import QueueClient


def build_session(tmp_path, client, session_id="session-10", **overrides):
    settings = Settings(
        api_key=overrides.pop("api_key", "test-key"),
        home=tmp_path,
        consolidate_every=99,
        **overrides,
    )
    conn = connect(tmp_path)
    memory = Memory(conn, settings, client)
    return conn, memory, Session(settings, memory, session_id=session_id)


def gate_skip():
    return ModelResponse(text='{"retrieve":false,"query":"","reason":"测试"}')


def _message_entries(session):
    return [e for e in read_session_entries(session.jsonl_path) if e.type == "message"]


def run_exchange(session, user_text, reply_text, source="test"):
    """Simulate one app-level run: the recorder persists per-message
    entries (the loop's job via kernel events), then the ChatProjector
    projects the exchange into SQLite."""
    from lsm_harness.types import TurnResult

    recorder = session.recorder
    recorder.record(UserMessage(content=user_text), source=source)
    recorder.record(AssistantMessage(text=reply_text), source=source)
    return session.add_exchange(
        user_text, TurnResult(reply=reply_text, iterations=1), source
    )


def _tree_messages(session):
    context = session.build_session_context()
    return context.messages if context is not None else []


# ── tree primitives ──────────────────────────────────────────────


def _chain(*ids_roles):
    """Build a small entry chain/branched tree for primitive tests."""
    entries = [SessionHeader.create("s", "/tmp")]
    for entry_id, parent, role in ids_roles:
        message = (
            UserMessage(content=f"content-{entry_id}")
            if role == "user"
            else AssistantMessage(text=f"content-{entry_id}")
        )
        entries.append(MessageEntry.create(entry_id, parent, message))
    return entries


def test_path_to_leaf_excludes_abandoned_branches():
    entries = _chain(
        ("e1", None, "user"),
        ("e2", "e1", "assistant"),
        ("e3", "e2", "user"),      # old branch continues…
        ("e4", "e2", "user"),      # …while e4 forks off e2
        ("e5", "e4", "assistant"),
    )
    path = path_to_leaf(entries, "e5")
    assert [e.id for e in path] == ["e1", "e2", "e4", "e5"]
    path_old = path_to_leaf(entries, "e3")
    assert [e.id for e in path_old] == ["e1", "e2", "e3"]


def test_collect_abandoned_branch_stops_at_lca():
    entries = _chain(
        ("e1", None, "user"),
        ("e2", "e1", "assistant"),
        ("e3", "e2", "user"),
        ("e4", "e3", "assistant"),  # old leaf
        ("e5", "e2", "user"),       # new branch from e2
    )
    abandoned = collect_abandoned_branch(entries, old_leaf_id="e4", new_leaf_id="e2")
    assert [e.id for e in abandoned] == ["e3", "e4"]  # LCA (e2) excluded


# ── branch() on a live session ───────────────────────────────────


def test_branch_moves_leaf_and_rebuilds_history(tmp_path):
    _, _, session = build_session(tmp_path, QueueClient())
    run_exchange(session, "方案 A 试试", "A 的结果")
    run_exchange(session, "A 继续深入", "A 的更多细节")

    messages = _message_entries(session)
    fork = messages[0].id  # the first user message
    events = []
    target = session.branch(fork, lambda k, d: events.append((k, d)))

    assert target == fork
    # history rebuilt from the new path: only up to the fork
    assert session.history == [{"role": "user", "content": "方案 A 试试"}]
    assert ("session.branched", {"session_id": session.session_id,
                                 "from_entry": messages[-1].id,
                                 "to_entry": fork, "with_summary": False}) in events

    # a new exchange grows a sibling branch — nothing is deleted
    run_exchange(session, "换方案 B", "B 的结果")
    all_messages = _message_entries(session)
    assert len(all_messages) == 6  # 4 old + 2 new, abandoned branch intact
    new_user = all_messages[-2]
    assert new_user.parent_id == fork  # shares the fork parent → a branch

    # context built from the tree shows only the current path
    contents = [message_preview(m, limit=1_000_000) for m in _tree_messages(session)]
    assert "方案 A 试试" in contents and "换方案 B" in contents
    assert "A 继续深入" not in contents  # abandoned branch invisible


def test_branch_rejects_unknown_ref(tmp_path):
    _, _, session = build_session(tmp_path, QueueClient())
    run_exchange(session, "你好", "好")
    assert session.branch("nonexistent-entry", lambda *_: None) is None


# ── branch_with_summary ──────────────────────────────────────────


def test_branch_with_summary_leaves_branch_summary_entry(tmp_path):
    client = QueueClient(
        ModelResponse(text="## Goal\n- 试过方案 A\n\n## Key Decisions\n- A 太慢，放弃"),
    )
    _, _, session = build_session(tmp_path, client, small_model="small")
    run_exchange(session, "试方案 A", "A 分析")
    run_exchange(session, "A 的性能数据", "慢 3 倍")

    messages = _message_entries(session)
    fork = messages[0].id
    events = []
    result = session.branch_with_summary(fork, lambda k, d: events.append((k, d)))
    kinds = [k for k, _ in events]

    assert result is not None
    assert "session.branch_summary.started" in kinds
    assert "session.branch_summary.completed" in kinds

    entries = read_session_entries(session.jsonl_path)
    summaries = [e for e in entries if e.type == "branch_summary"]
    assert len(summaries) == 1
    summary_entry = summaries[0]
    assert summary_entry.parent_id == fork          # hung on the fork node
    assert summary_entry.from_id == messages[-1].id  # the abandoned leaf
    assert "方案 A" in summary_entry.summary

    # the summary shows up in the tree context of the new branch
    tree_messages = _tree_messages(session)
    branch_msgs = [
        m for m in tree_messages
        if isinstance(m, CustomMessage) and m.custom_type == "branch_summary"
    ]
    assert len(branch_msgs) == 1

    # LLM-facing translation carries the branch preamble
    from lsm_harness.coding_agent.messages import register_coding_agent_messages
    register_coding_agent_messages()
    converted = default_convert_to_llm(tree_messages)
    assert any(
        "explored a different conversation branch" in m.content
        for m in converted
    )


def test_branch_with_summary_failure_keeps_state(tmp_path):
    client = QueueClient(RuntimeError("llm down"))
    _, _, session = build_session(tmp_path, client)
    run_exchange(session, "试方案 A", "A")
    run_exchange(session, "A 继续", "A2")
    fork = _message_entries(session)[0].id
    old_leaf = session._last_entry_id

    events = []
    assert session.branch_with_summary(fork, lambda k, d: events.append(k)) is None
    assert session._last_entry_id == old_leaf  # unchanged
    assert "session.branch_summary.failed" in events


# ── tree context with compaction ─────────────────────────────────


def test_tree_context_applies_compaction_selective_skip(tmp_path):
    client = QueueClient(gate_skip(), ModelResponse(text="## Goal\n- 压缩测试"))
    _, _, session = build_session(
        tmp_path, client,
        context_budget_tokens=100000,
        context_compression_tokens=1,
        context_keep_recent_tokens=50,
    )
    run_exchange(session, "早期问题" * 100, "早期回答" * 200)
    run_exchange(session, "近期问题", "近期回答")

    session.prepare_context("继续", lambda *_: None)
    tree_messages = _tree_messages(session)
    assert tree_messages
    # first message is the compaction summary; covered messages are gone
    assert isinstance(tree_messages[0], CustomMessage)
    assert tree_messages[0].custom_type == "compaction_summary"
    contents = [message_preview(m, limit=1_000_000) for m in tree_messages]
    assert not any("早期问题" in c for c in contents)
    assert "近期问题" in contents


# ── backfill migration ───────────────────────────────────────────


def test_backfill_mirrors_chat_log_into_jsonl(tmp_path):
    # simulate a pre-JSONL session: rows in chat_log, empty JSONL file
    settings = Settings(api_key="k", home=tmp_path, consolidate_every=99)
    conn = connect(tmp_path)
    memory = Memory(conn, settings, QueueClient())
    memory.log_chat("老问题 1", "老回答 1", session_id="legacy", source="test")
    memory.log_chat("老问题 2", "老回答 2", session_id="legacy", source="test")

    session = Session(settings, memory, session_id="legacy")
    entries = read_session_entries(session.jsonl_path)
    messages = [e for e in entries if e.type == "message"]
    assert len(messages) == 4
    assert messages[0].meta["chat_id"] is not None
    assert messages[0].meta["backfilled"] is True
    # parent chain is intact
    assert messages[1].parent_id == messages[0].id

    tree_messages = _tree_messages(session)
    assert [message_preview(m, limit=1_000_000) for m in tree_messages] == [
        "老问题 1", "老回答 1", "老问题 2", "老回答 2",
    ]

    # idempotent: rebuilding the session must not duplicate the backfill
    session2 = Session(settings, memory, session_id="legacy")
    messages2 = [e for e in read_session_entries(session2.jsonl_path) if e.type == "message"]
    assert len(messages2) == 4


# ── batch F-4: session CLI commands (refactor plan §9.4) ─────────


def _noop_emit(_kind, _data):
    pass


def test_record_label_pins_navigation_entry(tmp_path):
    _conn, _memory, session = build_session(tmp_path, QueueClient(gate_skip()))
    run_exchange(session, "第一条", "回复一")
    run_exchange(session, "第二条", "回复二")
    entries = read_session_entries(session.jsonl_path)
    target = next(e for e in entries if e.type == "message")

    resolved = session.record_label(target.id, "里程碑")
    assert resolved == target.id

    labels = [e for e in read_session_entries(session.jsonl_path) if e.type == "label"]
    assert len(labels) == 1
    assert labels[0].target_id == target.id
    assert labels[0].label == "里程碑"
    # Labels never enter model context.
    assert all(
        "里程碑" not in getattr(m, "content", "") or not isinstance(getattr(m, "content", ""), str)
        for m in _tree_messages(session)
    )


def test_record_label_rejects_unknown_ref_and_empty_label(tmp_path):
    _conn, _memory, session = build_session(tmp_path, QueueClient(gate_skip()))
    run_exchange(session, "你好", "你好！")
    assert session.record_label("no-such-entry", "x") is None
    entries = read_session_entries(session.jsonl_path)
    target = next(e for e in entries if e.type == "message")
    assert session.record_label(target.id, "   ") is None
    assert not [e for e in read_session_entries(session.jsonl_path) if e.type == "label"]


def test_cli_branch_and_label_commands(tmp_path, capsys):
    from lsm_harness.coding_agent import cli as cli_module

    _conn, _memory, session = build_session(tmp_path, QueueClient(gate_skip()))
    run_exchange(session, "第一条", "回复一")
    run_exchange(session, "第二条", "回复二")
    entries = read_session_entries(session.jsonl_path)
    first_message = next(e for e in entries if e.type == "message")

    from types import SimpleNamespace

    app = SimpleNamespace(session=session)

    cli_module._cmd_branch(app, "", with_summary=False)
    cli_module._cmd_branch(app, first_message.id, with_summary=False)
    assert session._last_entry_id == first_message.id

    cli_module._cmd_label(app, "")
    cli_module._cmd_label(app, f"{first_message.id} 分叉点")
    labels = [e for e in read_session_entries(session.jsonl_path) if e.type == "label"]
    assert len(labels) == 1 and labels[0].label == "分叉点"

    out = capsys.readouterr().out
    assert "用法" in out
