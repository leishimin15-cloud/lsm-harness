from __future__ import annotations

from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.memory.facade import Memory
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


def seed(memory, session_id, pairs=3, width=80):
    for index in range(pairs):
        memory.log_chat(
            f"用户第 {index} 轮：" + "项目背景" * width,
            f"助手第 {index} 轮：" + "执行结果" * width,
            session_id=session_id,
            source="test",
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
        context_recent_turns=1,
        summary_max_tokens=200,
    )
    seed(memory, session.session_id, pairs=3)
    events = []

    system, messages = session.prepare_context("继续开发", lambda kind, data: events.append(kind))

    info = session.summary_info()
    assert info and info["version"] == 1
    assert info["source_message_count"] == 4
    assert secret not in info["summary"]
    assert "[REDACTED]" in info["summary"]
    assert "当前会话历史摘要" in system
    assert len(messages) == 3  # 最近一轮原文，加当前用户消息
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
        context_recent_turns=1,
    )
    seed(memory, session.session_id, pairs=3)
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
        context_recent_turns=1,
    )
    seed(memory, session.session_id, pairs=3, width=20)
    session.prepare_context("第一次继续", lambda *_: None)
    first = session.summary_info()

    seed(memory, session.session_id, pairs=2, width=20)
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
        context_recent_turns=99,
    )
    seed(memory, session.session_id, pairs=4, width=100)
    events = []

    _, messages = session.prepare_context(
        "最后的问题", lambda kind, data: events.append((kind, data))
    )

    built = next(data for kind, data in events if kind == "context.built")
    assert built["dropped_messages"] > 0
    assert built["dropped_messages"] % 2 == 0
    assert messages[-1]["content"] == "最后的问题"
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
