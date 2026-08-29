from __future__ import annotations

from lsm_harness.agent.messages import (
    AssistantMessage,
    CustomMessage,
    UserMessage,
    message_preview,
)
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.memory.facade import Memory
from lsm_harness.ops.session_store import read_session_entries
from lsm_harness.runtime import Session, estimate_context_tokens, estimate_tokens
from lsm_harness.types import ModelResponse, TurnResult

from helpers import QueueClient


def build_session(tmp_path, client, session_id="session-a", **overrides):
    settings = Settings(
        api_key=overrides.pop("api_key", "test-key"),
        home=tmp_path,
        consolidate_every=99,
        **overrides,
    )
    conn = connect(tmp_path)
    memory = Memory(conn, settings, client)
    return conn, memory, Session(settings, memory, session_id=session_id)


def seed(session, pairs=3, width=80):
    """Seed `pairs` exchanges the way production does: the recorder writes
    per-message tree entries (with their chat_id for the legacy fallback),
    the projector writes chat_log."""
    for index in range(pairs):
        user = f"用户第 {index} 轮：" + "项目背景" * width
        reply = f"助手第 {index} 轮：" + "执行结果" * width
        user_chat_id, assistant_chat_id = session.memory.log_chat(
            user, reply, session_id=session.session_id, source="test"
        )
        session.recorder.record(
            UserMessage(content=user),
            source="test",
            meta={"chat_id": user_chat_id},
        )
        session.recorder.record(
            AssistantMessage(text=reply),
            source="test",
            meta={"chat_id": assistant_chat_id},
        )


def record_exchange(session, user_text, reply_text, source="test"):
    """One full run: recorder entries first, then the SQLite projection."""
    session.recorder.record(UserMessage(content=user_text), source=source)
    session.recorder.record(AssistantMessage(text=reply_text), source=source)
    return session.add_exchange(
        user_text, TurnResult(reply=reply_text, iterations=1), source
    )


def gate_skip():
    return ModelResponse(text='{"retrieve":false,"query":"","reason":"测试"}')


def test_mixed_chinese_token_estimate_is_deterministic():
    assert estimate_tokens("中文") == 2
    assert estimate_tokens("abcd") == 1
    assert estimate_context_tokens("系统", [{"role": "user", "content": "你好"}]) > 4


def test_context_compression_writes_version_and_keeps_raw_chat(tmp_path):
    secret = "sk-context-secret-12345678"
    client = QueueClient(
        gate_skip(),
        ModelResponse(text=f"- 当前目标：完成 LSM v1.1\n- 秘密：{secret}"),
    )
    conn, memory, session = build_session(
        tmp_path,
        client,
        api_key=secret,
        context_budget_tokens=2000,
        context_compression_tokens=1,
        context_keep_recent_tokens=400,
        summary_max_tokens=200,
    )
    seed(session, pairs=3)
    events = []

    system, messages = session.prepare_context("继续开发", lambda kind, data: events.append(kind))

    info = session.summary_info()
    assert info and info["version"] == 1
    # token-based cut: keep_recent=400 covers ~the last pair (each row ≈336
    # tokens), everything older is summarised
    assert info["source_message_count"] % 2 == 0
    assert secret not in info["summary"]
    assert "[REDACTED]" in info["summary"]
    # the summary is injected as the first *message* (chapter 9), not
    # concatenated into the system prompt
    assert isinstance(messages[0], CustomMessage)
    assert messages[0].custom_type == "compaction_summary"
    assert "当前会话历史摘要" not in system
    # summary + entries kept after first_kept_entry_id + current user
    remaining = 6 - info["source_message_count"]
    assert len(messages) == 1 + remaining + 1
    assert conn.execute("SELECT COUNT(*) FROM chat_log").fetchone()[0] == 6
    assert events.index("context.compression.started") < events.index(
        "context.compression.completed"
    )
    assert events[-1] == "context.built"


def test_compression_failure_keeps_raw_chat_for_retry(tmp_path):
    client = QueueClient(gate_skip(), RuntimeError("summary unavailable"))
    conn, memory, session = build_session(
        tmp_path,
        client,
        context_budget_tokens=1000,
        context_compression_tokens=1,
        context_keep_recent_tokens=400,
    )
    seed(session, pairs=3)
    events = []

    session.prepare_context("继续", lambda kind, data: events.append(kind))

    assert session.summary_info() is None
    assert conn.execute("SELECT COUNT(*) FROM chat_log").fetchone()[0] == 6
    assert "context.compression.failed" in events
    assert events[-1] == "context.built"


def test_rolling_summary_merges_previous_version_incrementally(tmp_path):
    client = QueueClient(
        gate_skip(),
        ModelResponse(text="- v1：已经确定项目目标"),
        gate_skip(),
        ModelResponse(text="- v2：项目目标不变，并完成数据库迁移"),
    )
    _, memory, session = build_session(
        tmp_path,
        client,
        context_budget_tokens=2000,
        context_compression_tokens=1,
        context_keep_recent_tokens=100,
    )
    seed(session, pairs=3, width=20)
    session.prepare_context("第一次继续", lambda *_: None)
    first = session.summary_info()

    seed(session, pairs=2, width=20)
    session.prepare_context("第二次继续", lambda *_: None)
    second = session.summary_info()

    assert first and second
    assert second["version"] == 2
    assert second["through_chat_id"] > first["through_chat_id"]
    assert "v1：已经确定项目目标" in client.calls[3]["messages"][0]["content"]
    assert len(session.conn.execute("SELECT id FROM session_summaries").fetchall()) == 2


def test_context_budget_drops_oldest_complete_exchanges(tmp_path):
    client = QueueClient(gate_skip())
    _, memory, session = build_session(
        tmp_path,
        client,
        context_budget_tokens=400,
        context_compression_tokens=400,
        context_keep_recent_tokens=999999,
    )
    seed(session, pairs=4, width=100)
    events = []

    _, messages = session.prepare_context(
        "最后的问题", lambda kind, data: events.append((kind, data))
    )

    built = next(data for kind, data in events if kind == "context.built")
    assert built["dropped_messages"] > 0
    assert built["dropped_messages"] % 2 == 0
    assert message_preview(messages[-1], limit=1_000_000) == "最后的问题"
    assert built["within_budget"] is True


def test_latest_session_and_history_are_restored_after_restart(tmp_path):
    first_client = QueueClient()
    conn, _, first = build_session(tmp_path, first_client, session_id="durable-session")
    first.add_exchange("继续开发上下文压缩", TurnResult(reply="已经记录", iterations=1), "test")
    conn.close()

    settings = Settings(api_key="x", home=tmp_path, consolidate_every=99)
    second_conn = connect(tmp_path)
    second_memory = Memory(second_conn, settings, QueueClient())
    restored = Session(settings, second_memory)

    assert restored.session_id == "durable-session"
    assert restored.history == [
        {"role": "user", "content": "继续开发上下文压缩"},
        {"role": "assistant", "content": "已经记录"},
    ]
    assert restored.list_sessions()[0]["message_count"] == 2


def test_new_and_resume_session_by_short_id(tmp_path):
    _, _, session = build_session(tmp_path, QueueClient(), session_id="aaaaaaaa-first")
    original = session.session_id
    created = session.start_new()

    assert created != original
    assert session.resume("aaaaaaaa") == original
    assert session.resume("missing") is None


# ── structured summary + turn-aware cutting ──────────────────────


def test_turn_aware_cutting_preserves_last_turn(tmp_path):
    """Token-based cutting still never splits a user+assistant pair."""
    client = QueueClient(
        gate_skip(),
        ModelResponse(text="## Goal\n- 测试项目\n\n## Progress\n- 完成初始化"),
    )
    _, _, session = build_session(
        tmp_path, client,
        context_budget_tokens=2000,
        context_compression_tokens=1,
        context_keep_recent_tokens=1000,
        summary_max_tokens=200,
    )
    seed(session, pairs=5)
    session.prepare_context("继续", lambda *_: None)
    info = session.summary_info()
    assert info
    # keep_recent=1000 covers ~3 rows of ~336 tokens; the cut moves forward
    # to the nearest user boundary, so a whole number of pairs is compressed
    assert info["source_message_count"] % 2 == 0  # always even


def test_structured_summary_has_sections(tmp_path):
    """The prompt produces a pi-style 6-section structured summary."""
    client = QueueClient(
        gate_skip(),
        ModelResponse(
            text="## Goal\n- 构建 LSM\n\n## Constraints & Preferences\n- 用 pnpm\n\n"
                 "## Progress\n### Done\n- 完成 loop 改造\n\n### In Progress\n- 上下文压缩\n\n"
                 "### Blocked\n- （无）\n\n## Key Decisions\n- 使用 SQLite\n\n"
                 "## Next Steps\n- 写测试\n\n## Critical Context\n- 不能丢失数据"
        ),
    )
    _, _, session = build_session(
        tmp_path, client,
        context_budget_tokens=2000,
        context_compression_tokens=1,
        context_keep_recent_tokens=400,
        summary_max_tokens=300,
    )
    seed(session, pairs=3)
    session.prepare_context("继续", lambda *_: None)
    summary = session.summary()
    # Pi's 6 sections, with Progress split into Done / In Progress / Blocked
    for section in ["Goal", "Constraints & Preferences", "Progress",
                     "Key Decisions", "Next Steps", "Critical Context"]:
        assert f"## {section}" in summary, f"Missing section: {section}"
    for sub in ["Done", "In Progress", "Blocked"]:
        assert f"### {sub}" in summary, f"Missing Progress sub-item: {sub}"


def test_incremental_merge_uses_update_prompt(tmp_path):
    """Second compression uses UPDATE_SUMMARY_PROMPT, not SUMMARY_PROMPT."""
    client = QueueClient(
        gate_skip(),
        ModelResponse(text="## Goal\n- 第一版\n\n## Progress\n- 初始化"),
        gate_skip(),
        ModelResponse(text="## Goal\n- 第一版（不变）\n\n## Progress\n- 初始化\n- 加新功能"),
    )
    _, _, session = build_session(
        tmp_path, client,
        context_budget_tokens=2000,
        context_compression_tokens=1,
        context_keep_recent_tokens=100,
    )
    seed(session, pairs=3, width=20)
    session.prepare_context("第一次", lambda *_: None)
    seed(session, pairs=2, width=20)
    session.prepare_context("第二次", lambda *_: None)
    info = session.summary_info()
    assert info and info["version"] == 2
    # The second call should have used UPDATE_SUMMARY_PROMPT which includes the
    # previous summary in the prompt.  Check the calls made.
    # call 0: gate_skip, call 1: first summary, call 2: gate_skip, call 3: second summary
    second_prompt = client.calls[3]["messages"][0]["content"]
    assert "第一版" in second_prompt  # previous summary included


# ── compaction-loop coupling ─────────────────────────────────────


def test_compact_and_rebuild_returns_fresh_context(tmp_path):
    """compact_and_rebuild forces compression and returns new context."""
    client = QueueClient(
        gate_skip(),
        ModelResponse(text="## Goal\n- 测试\n\n## Progress\n- 完成"),
    )
    _, _, session = build_session(
        tmp_path, client,
        context_budget_tokens=2000,
        context_compression_tokens=1,
        context_keep_recent_tokens=400,
        summary_max_tokens=200,
    )
    seed(session, pairs=4)

    events_dict = []
    system, messages = session.compact_and_rebuild(
        "溢出了", lambda k, d: events_dict.append((k, d)), []
    )
    kinds = [k for k, _ in events_dict]
    assert "context.compression.started" in kinds
    assert "context.compression.completed" in kinds
    # Should have a compression triggered by "overflow"
    started = next(d for k, d in events_dict if k == "context.compression.started")
    assert started["reason"] == "overflow"
    # summary injected as first message, not into the system prompt
    assert isinstance(messages[0], CustomMessage)
    assert messages[0].custom_type == "compaction_summary"


def test_on_truncation_callback_compacts_and_retries(tmp_path):
    """The loop calls on_truncation on length stop, then retries."""
    from lsm_harness.loop.agent import run_loop
    from lsm_harness.tools.registry import Tool, ToolRegistry
    from lsm_harness.types import ToolCall

    # Setup: session with compression capability
    client = QueueClient(
        gate_skip(),
        ModelResponse(text="## Goal\n- X"),
    )
    _, _, session = build_session(
        tmp_path, client,
        context_budget_tokens=2000,
        context_compression_tokens=1,
        context_keep_recent_tokens=400,
    )
    seed(session, pairs=2)

    # Create a loop client that first returns length, then recovers
    loop_client = QueueClient(
        ModelResponse(
            stop_reason="length",
            tool_calls=[ToolCall("1", "echo", {"value": "x"})],
        ),
        ModelResponse(text="recovered after compaction"),
    )
    tools = ToolRegistry()
    tools.register(
        Tool("echo", "", {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
             lambda value: f"ok:{value}")
    )

    system, messages = session.prepare_context("hello", lambda *_: None, [])
    events = []

    def on_trunc():
        return session.compact_and_rebuild("hello", lambda k, d: events.append((k, d)), [])

    result = run_loop(
        client=loop_client,
        model="test",
        system=system,
        messages=messages,
        tools=tools,
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        on_truncation=on_trunc,
    )
    assert result.reply == "recovered after compaction"
    kinds = [k for k, _ in events]
    assert "loop.overflow_recovery" in kinds
    assert "loop.truncation_rejected" in kinds


def test_on_truncation_error_is_surfaced(tmp_path):
    """If compaction fails, the trace ends as a structured length failure."""
    from lsm_harness.loop.agent import run_loop
    from lsm_harness.types import ToolCall

    loop_client = QueueClient(
        ModelResponse(
            stop_reason="length",
            tool_calls=[ToolCall("1", "echo", {"value": "x"})],
        ),
        ModelResponse(text="recovered without compaction"),
    )

    def bad_compact():
        raise RuntimeError("compaction failed")

    # Use a simple registry
    from lsm_harness.tools.registry import Tool, ToolRegistry
    tools = ToolRegistry()
    tools.register(
        Tool("echo", "", {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
             lambda value: f"ok:{value}")
    )

    events = []
    result = run_loop(
        client=loop_client,
        model="test",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        on_truncation=bad_compact,
    )
    assert result.status == "failed"
    assert result.stop_reason == "length"
    assert "compaction failed" in result.error
    assert len(loop_client.calls) == 1
    kinds = [k for k, _ in events]
    assert "loop.overflow_recovery_failed" in kinds


# ── error context estimation ─────────────────────────────────────


def test_estimate_from_messages_fallback(tmp_path):
    """_estimate_from_messages works when usage data is absent."""
    client = QueueClient(gate_skip())
    _, _, session = build_session(tmp_path, client, context_recent_turns=2)
    seed(session, pairs=3)
    rows = session._rows_after(0)
    estimated = session._estimate_from_messages(rows)
    assert estimated > 0


# ── session JSONL storage (batch B: recorder + deferred write) ────


def test_session_jsonl_defers_first_write_until_first_assistant(tmp_path):
    """A fresh session writes NOTHING until its first assistant message
    completes — a failed run never leaves half a session file (§5.6)."""
    _, _, session = build_session(tmp_path, QueueClient())
    assert not session.jsonl_path.exists()
    session.recorder.record(UserMessage(content="你好"), source="test")
    assert not session.jsonl_path.exists()  # a lone user question stays buffered
    session.recorder.record(AssistantMessage(text="你好！"), source="test")
    entries = read_session_entries(session.jsonl_path)
    assert entries[0].type == "session"  # header + buffer, one atomic write
    assert [e.type for e in entries[1:]] == ["message", "message"]


def test_recorder_abandon_drops_unflushed_buffer(tmp_path):
    """A failed first exchange is marked abandoned, never persisted (§5.6)."""
    _, _, session = build_session(tmp_path, QueueClient())
    events = []
    session.recorder.set_emit(lambda k, d: events.append((k, d)))
    session.recorder.record(UserMessage(content="这次会失败"), source="test")
    session.recorder.abandon("failed")
    assert not session.jsonl_path.exists()
    assert any(k == "session.jsonl_abandoned" for k, _ in events)


def test_add_exchange_projects_to_sqlite_only(tmp_path):
    """ChatProjector: add_exchange writes chat_log + sessions table; the
    JSONL tree is the recorder's job, so the status reports it honestly."""
    _, _, session = build_session(tmp_path, QueueClient())
    status = session.add_exchange(
        "你好", TurnResult(reply="你好！", iterations=1), "test"
    )
    assert status["sqlite"] == "ok"
    assert status["jsonl"] == "deferred"  # nothing recorded, nothing written
    rows = session.conn.execute(
        "SELECT role,content FROM chat_log WHERE session_id=? ORDER BY id",
        (session.session_id,),
    ).fetchall()
    assert [(r["role"], r["content"]) for r in rows] == [
        ("user", "你好"),
        ("assistant", "你好！"),
    ]
    # no [tools used: ...] fusion anymore — tool calls are tree entries
    session2_status = record_exchange(session, "用工具", "完成")
    assert session2_status["jsonl"] == "ok"
    last = session.conn.execute(
        "SELECT content FROM chat_log WHERE role='assistant' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert "[tools used:" not in last["content"]


def test_recorder_writes_per_message_entries(tmp_path):
    """One exchange lands in the tree as individual typed message entries."""
    _, _, session = build_session(tmp_path, QueueClient())
    record_exchange(session, "你好", "你好！")
    entries = read_session_entries(session.jsonl_path)
    messages = [e for e in entries if e.type == "message"]
    assert len(messages) == 2
    assert messages[0].message.role == "user"
    assert message_preview(messages[0].message, limit=1_000_000) == "你好"
    assert messages[1].message.role == "assistant"
    assert "你好！" in message_preview(messages[1].message, limit=1_000_000)
    # parent chain: user → assistant
    assert messages[1].parent_id == messages[0].id


def test_compaction_writes_jsonl_entry(tmp_path):
    """Compaction events are recorded in the session JSONL, with positional
    coverage (first_kept_entry_id) recorded on the entry."""
    client = QueueClient(
        gate_skip(),
        ModelResponse(text="## Goal\n- 测试"),
    )
    _, _, session = build_session(
        tmp_path, client,
        context_budget_tokens=2000,
        context_compression_tokens=1,
        context_keep_recent_tokens=400,
        summary_max_tokens=200,
    )
    seed(session, pairs=3)
    session.prepare_context("继续", lambda *_: None)
    entries = read_session_entries(session.jsonl_path)
    compactions = [e for e in entries if e.type == "compaction"]
    assert len(compactions) >= 1
    assert compactions[0].version == 1
    entry_ids = {e.id for e in entries}
    assert compactions[0].first_kept_entry_id in entry_ids


def test_export_jsonl_copies_file(tmp_path):
    """export_jsonl copies the session to an external file."""
    _, _, session = build_session(tmp_path, QueueClient())
    record_exchange(session, "hello", "hi")
    dest = tmp_path / "exported.jsonl"
    result = session.export_jsonl(dest)
    assert result == dest
    assert dest.exists()
    exported = read_session_entries(dest)
    assert len([e for e in exported if e.type == "message"]) == 2


def test_import_jsonl_restores_messages(tmp_path):
    """Importing a session JSONL restores messages into the session."""
    # First, create and export a session
    _, _, source = build_session(tmp_path, QueueClient(), session_id="source-session")
    record_exchange(source, "问题1", "答案1")
    record_exchange(source, "问题2", "答案2")
    exported_path = source.export_jsonl(tmp_path / "source.jsonl")

    # Then import into a new session
    _, _, target = build_session(tmp_path, QueueClient(), session_id="target-session")
    count = target.import_jsonl(exported_path)
    assert count == 2  # 2 user→assistant pairs
    # History should be loaded
    assert len(target.history) == 4


def test_replay_session_emits_events(tmp_path):
    """Replay walks the JSONL and emits events for each entry."""
    _, _, session = build_session(tmp_path, QueueClient())
    record_exchange(session, "hi", "hello")

    replay_events = []
    messages = session.replay(lambda k, d: replay_events.append((k, d)))

    kinds = [k for k, _ in replay_events]
    assert "session.replay.header" in kinds
    assert "session.replay.message" in kinds
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hi"
    assert messages[1]["role"] == "assistant"
    assert "hello" in messages[1]["content"]


def test_replay_respects_secret_redaction(tmp_path):
    """Secrets in replayed messages are redacted."""
    secret = "sk-replay-secret-123"
    client = QueueClient()
    _, _, session = build_session(tmp_path, client, api_key=secret)
    record_exchange(session, f"我的 key 是 {secret}", "收到")
    messages = session.replay(lambda *_: None)
    user_msg = messages[0]["content"]
    assert secret not in user_msg
    assert "[REDACTED]" in user_msg
