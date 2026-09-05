"""批次 G-6: regression tests for the code-review findings (issues 一–七).

1. A compresses first, then B compresses — B's summary PROMPT contains
   neither A's messages nor A's summary (previous-summary chaining is
   path-scoped, JSONL authoritative).
2. After B compresses, branching back to A serves A's OWN summary.
3. Branching back past a model_change makes the next real Agent Loop
   call the OLD model — and writes NO new model_change entry
   (恢复 ≠ 变更).
4. Resume restores provider/model/small_model/thinking onto real calls.
5. The AgentLoopConfig actually used matches the tree state.
6. A failed JSONL write leaves no readable ghost summary in SQLite.
7. Legacy JSONL rows without small_model (and headers without the
   initial-state fields) still deserialize and build context.
"""

from __future__ import annotations

import json

from lsm_harness.agent.messages import AssistantMessage, UserMessage, message_preview
from lsm_harness.ai.registry import (
    ApiProvider,
    register_api_provider,
    unregister_api_provider,
)
from lsm_harness.ai.types import Model
from lsm_harness.app import Harness
from lsm_harness.coding_agent.session_recorder import SessionRecorder
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.memory.facade import Memory
from lsm_harness.ops.session_store import read_session_entries
from lsm_harness.runtime import Session
from lsm_harness.smoke import ScriptedClient
from lsm_harness.types import ModelResponse

from helpers import QueueClient


def build_session(tmp_path, client, session_id="session-g6", **overrides):
    settings = Settings(
        api_key=overrides.pop("api_key", "test-key"),
        home=tmp_path,
        consolidate_every=99,
        **overrides,
    )
    conn = connect(tmp_path)
    memory = Memory(conn, settings, client)
    return conn, memory, Session(settings, memory, session_id=session_id)


def _previews(context):
    return [message_preview(m, limit=1_000_000) for m in context.messages]


def _two_branch_compaction_session(tmp_path):
    """Shared scenario for tests 1 & 2: fork → branch A compresses →
    back to fork → branch B compresses.  Returns (session, client, ids).
    """
    client = QueueClient(
        ModelResponse(text="## Goal\n- A摘要"),
        ModelResponse(text="## Goal\n- B摘要"),
    )
    _, _, session = build_session(
        tmp_path,
        client,
        context_budget_tokens=100000,
        context_keep_recent_tokens=50,
    )
    recorder = session.recorder
    recorder.record(UserMessage(content="共同起点" + "共" * 300), source="test")
    recorder.record(AssistantMessage(text="共同回答" + "共" * 300), source="test")
    fork = recorder.last_entry_id
    # ── branch A: explores apples, then compacts ──────────────
    recorder.record(UserMessage(content="苹果方案" + "苹" * 300), source="test")
    recorder.record(AssistantMessage(text="苹果答复" + "苹" * 300), source="test")
    assert session._do_compress(lambda *_: None)
    leaf_a = session.recorder.last_entry_id  # A's CompactionEntry
    # ── back to the fork, branch B: explores bananas, compacts ─
    session.branch(fork, lambda *_: None)
    recorder = session.recorder
    recorder.record(UserMessage(content="香蕉方案" + "香" * 300), source="test")
    recorder.record(AssistantMessage(text="香蕉答复" + "香" * 300), source="test")
    assert session._do_compress(lambda *_: None)
    leaf_b = session.recorder.last_entry_id  # B's CompactionEntry
    return session, client, fork, leaf_a, leaf_b


# ── issue 一: cross-branch compaction summary pollution ──────────


def test_second_branch_compaction_prompt_excludes_first_branch(tmp_path):
    """A compresses, then B compresses: B's summary prompt must contain
    neither A's messages nor A's SUMMARY — the `previous` chaining is
    scoped to the current tree path, never the session-global latest."""
    session, client, _fork, _leaf_a, _leaf_b = _two_branch_compaction_session(tmp_path)

    prompt_b = client.calls[1]["messages"][0]["content"]
    assert "香蕉" in prompt_b and "共同" in prompt_b
    assert "苹果" not in prompt_b
    assert "A摘要" not in prompt_b  # the sibling branch's summary never leaks

    # Version stays session-global and monotonic (SQLite UNIQUE
    # constraint); correctness comes from the path-scoped `previous`.
    info = session.summary_info()
    assert info is not None and info["version"] == 2


def test_branch_back_to_a_serves_as_own_summary(tmp_path):
    """After B compressed (session-global latest = B), branching back to
    A makes summary()/context serve A's OWN compaction summary."""
    session, _client, _fork, leaf_a, leaf_b = _two_branch_compaction_session(tmp_path)

    # currently on B: B's summary is current
    assert "B摘要" in session.summary()

    session.branch(leaf_a, lambda *_: None)
    assert "A摘要" in session.summary()
    assert "B摘要" not in session.summary()
    previews = _previews(session.build_session_context())
    assert any("A摘要" in p for p in previews)
    assert not any("B摘要" in p for p in previews)

    session.branch(leaf_b, lambda *_: None)
    assert "B摘要" in session.summary()


# ── issue 五: JSONL first, SQLite projection second ──────────────


def test_jsonl_write_failure_leaves_no_ghost_summary(tmp_path, monkeypatch):
    """If the CompactionEntry JSONL write fails, the SQLite projection
    must NOT be written either — no readable ghost summary anywhere."""
    client = QueueClient(ModelResponse(text="## Goal\n- 幽灵摘要"))
    conn, _, session = build_session(
        tmp_path,
        client,
        context_budget_tokens=100000,
        context_keep_recent_tokens=50,
    )
    recorder = session.recorder
    recorder.record(UserMessage(content="旧问题" + "旧" * 300), source="test")
    recorder.record(AssistantMessage(text="旧回答" + "旧" * 300), source="test")
    recorder.record(UserMessage(content="新问题" + "新" * 300), source="test")
    recorder.record(AssistantMessage(text="新回答" + "新" * 300), source="test")

    def fail_write(self, rows, *, mode):
        self.error = "OSError: disk full"

    monkeypatch.setattr(SessionRecorder, "_write_lines", fail_write)

    assert not session._do_compress(lambda *_: None)
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM session_summaries WHERE session_id=?",
        (session.session_id,),
    ).fetchone()["c"]
    assert count == 0
    # nothing readable from either store
    assert session.summary() == ""
    assert not any(
        entry.type == "compaction"
        for entry in read_session_entries(session.jsonl_path)
    )


# ── issue 三: legacy rows without small_model stay readable ──────


def test_legacy_jsonl_without_small_model_still_loads(tmp_path):
    """Pre-G-3 files: header without the initial-state fields and a
    model_change row without small_model must deserialize with empty
    defaults and still build a context."""
    sid = "legacy7"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)
    rows = [
        {"id": sid, "type": "session", "version": 2, "cwd": "/tmp",
         "timestamp": "2026-01-01T00:00:00"},
        {"id": "legacy-u1", "type": "message", "parent_id": sid,
         "timestamp": "2026-01-01T00:00:01", "source": "cli",
         "message": {"role": "user", "content": "旧问题"}},
        {"id": "legacy-a1", "type": "message", "parent_id": "legacy-u1",
         "timestamp": "2026-01-01T00:00:02", "source": "cli",
         "message": {"role": "assistant", "content": "旧回答"}},
        # legacy model_change: provider/model only, NO small_model key
        {"id": "legacy-mc", "type": "model_change", "parent_id": "legacy-a1",
         "timestamp": "2026-01-01T00:00:03",
         "provider": "deepseek", "model": "legacy-m"},
        {"id": "legacy-u2", "type": "message", "parent_id": "legacy-mc",
         "timestamp": "2026-01-01T00:00:04", "source": "cli",
         "message": {"role": "user", "content": "新提问"}},
        {"id": "legacy-a2", "type": "message", "parent_id": "legacy-u2",
         "timestamp": "2026-01-01T00:00:05", "source": "cli",
         "message": {"role": "assistant", "content": "新回答"}},
    ]
    path = sessions_dir / f"{sid}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    entries = read_session_entries(path)
    change = next(entry for entry in entries if entry.type == "model_change")
    assert change.small_model == ""  # empty default, not a crash

    _, _, session = build_session(tmp_path, QueueClient(), session_id=sid)
    context = session.build_session_context()
    assert context is not None
    assert context.provider == "deepseek"
    assert context.model == "legacy-m"
    assert context.small_model in (None, "")
    assert context.thinking_level is None
    assert _previews(context) == ["旧问题", "旧回答", "新提问", "新回答"]


# ── issues 二/四: runtime state applied to the live Agent Loop ───


_API = "test-restore-api"


def _registry_events(reply: str):
    from lsm_harness.ai.api.common import snapshot
    from lsm_harness.ai.types import AssistantMessageEvent, ModelResponse

    final = ModelResponse(text=reply, stop_reason="stop")
    return [
        AssistantMessageEvent("start", snapshot(text="", thinking="", pending={})),
        AssistantMessageEvent("text_start", ModelResponse()),
        AssistantMessageEvent("text_delta", final, text_delta=reply),
        AssistantMessageEvent("text_end", final),
        AssistantMessageEvent("done", final),
    ]


def _capturing_stream(calls: list, reply: str = "ok"):
    def fake_stream(model, context, options):
        calls.append({"model": model, "options": options})
        yield from _registry_events(reply)

    return fake_stream


def _fake_get_client(**kw):
    """Stand-in provider client whose .model is a registry-backed Model,
    so the Harness re-resolves stream_fn through the fake API provider."""
    client = ScriptedClient()
    client.model = Model(
        id=kw.get("model") or "unknown",
        api=_API,
        provider=kw.get("provider_name") or "deepseek",
    )
    return client


def _harness_settings(tmp_path, **overrides):
    values = {
        "provider": "deepseek",
        "api_key": "scripted",
        "model": "old-main",
        "small_model": "old-small",
        "thinking": "disabled",
        "home": tmp_path / ".lsm",
        "sandbox_project_dir": str(tmp_path),
        "sandbox_enabled": False,
        "consolidate_every": 99,
    }
    values.update(overrides)
    return Settings(**values)


def _injected_client(model_id: str) -> ScriptedClient:
    client = ScriptedClient()
    client.model = Model(id=model_id, api=_API, provider="deepseek")
    return client


def _model_change_count(session) -> int:
    return sum(
        1
        for entry in read_session_entries(session.jsonl_path)
        if entry.type == "model_change"
    )


def test_branch_before_model_change_next_loop_uses_old_model(tmp_path, monkeypatch):
    """Issue 二: branch back past a model_change — the next respond runs
    the REAL Agent Loop with the old provider/model/small/thinking, and
    restoration writes no new model_change entry (恢复 ≠ 变更)."""
    calls: list[dict] = []
    register_api_provider(ApiProvider(_API, _capturing_stream(calls)))
    monkeypatch.setattr("lsm_harness.ai.providers.get_client", _fake_get_client)
    app = Harness(
        settings=_harness_settings(tmp_path),
        client=_injected_client("old-main"),
    )
    try:
        recorder = app.session.recorder
        recorder.record(UserMessage(content="阶段一"), source="test")
        recorder.record(AssistantMessage(text="回答一"), source="test")
        leaf_before_switch = recorder.last_entry_id

        app.switch_model("deepseek", model="new-main", small_model="new-small")
        app.session.record_thinking_change("enabled")
        assert app.settings.model == "new-main"
        app.session.recorder.record(UserMessage(content="阶段二"), source="test")
        app.session.recorder.record(AssistantMessage(text="回答二"), source="test")

        app.session.branch(leaf_before_switch, lambda *_: None)
        changes_before = _model_change_count(app.session)

        result = app.respond("又回来了")
        assert result.reply == "ok"
        assert calls[-1]["model"].id == "old-main"  # real loop call: OLD model
        assert app.settings.model == "old-main"
        assert app.settings.small_model == "old-small"
        assert app.settings.thinking == "disabled"
        assert _model_change_count(app.session) == changes_before
    finally:
        app.close()
        unregister_api_provider(_API)


def test_resume_restores_runtime_state_onto_real_calls(tmp_path, monkeypatch):
    """Issue 二/四: a fresh Harness resuming the session restores the
    recorded provider/model/small_model/thinking — not the global
    Settings the new process happens to have."""
    calls: list[dict] = []
    register_api_provider(ApiProvider(_API, _capturing_stream(calls)))
    monkeypatch.setattr("lsm_harness.ai.providers.get_client", _fake_get_client)

    app_a = Harness(
        settings=_harness_settings(tmp_path, model="m1", small_model="s1"),
        client=_injected_client("m1"),
    )
    sid = app_a.session.session_id
    app_a.session.recorder.record(UserMessage(content="第一轮"), source="test")
    app_a.session.recorder.record(AssistantMessage(text="回复一"), source="test")
    app_a.switch_model("deepseek", model="m2", small_model="s2")
    app_a.session.record_thinking_change("enabled")
    app_a.session.recorder.record(UserMessage(content="第二轮"), source="test")
    app_a.session.recorder.record(AssistantMessage(text="回复二"), source="test")
    app_a.close()

    # New process, different globals — same home → same session.
    app_b = Harness(
        settings=_harness_settings(
            tmp_path, model="global-m", small_model="global-s", thinking="disabled"
        ),
        client=_injected_client("global-m"),
    )
    try:
        assert app_b.session.session_id == sid
        result = app_b.respond("继续")
        assert result.reply == "ok"
        assert calls[-1]["model"].id == "m2"
        assert app_b.settings.provider == "deepseek"
        assert app_b.settings.model == "m2"
        assert app_b.settings.small_model == "s2"
        assert app_b.settings.thinking == "enabled"
    finally:
        app_b.close()
        unregister_api_provider(_API)


def test_loop_config_matches_tree_state_after_branch(tmp_path, monkeypatch):
    """Issue 二 acceptance: the AgentLoopConfig the loop actually ran
    with matches the tree's state at the current leaf."""
    calls: list[dict] = []
    register_api_provider(ApiProvider(_API, _capturing_stream(calls)))
    monkeypatch.setattr("lsm_harness.ai.providers.get_client", _fake_get_client)

    import lsm_harness.coding_agent.app as app_module

    real_run = app_module.run_agent_loop
    captured: dict = {}

    def spy_run(*, context, config, stream_fn, emit, interrupt=None):
        captured["config"] = config
        return real_run(
            context=context,
            config=config,
            stream_fn=stream_fn,
            emit=emit,
            interrupt=interrupt,
        )

    monkeypatch.setattr(app_module, "run_agent_loop", spy_run)

    app = Harness(
        settings=_harness_settings(tmp_path, thinking="disabled"),
        client=_injected_client("old-main"),
    )
    try:
        recorder = app.session.recorder
        recorder.record(UserMessage(content="阶段一"), source="test")
        recorder.record(AssistantMessage(text="回答一"), source="test")
        leaf_before_switch = recorder.last_entry_id
        app.switch_model("deepseek", model="new-main", small_model="new-small")
        app.session.record_thinking_change("enabled")
        app.session.branch(leaf_before_switch, lambda *_: None)

        result = app.respond("对照")
        assert result.reply == "ok"

        tree = app.session.build_session_context()
        config = captured["config"]
        assert tree is not None
        assert config.model.id == tree.model == "old-main"
        assert config.thinking == tree.thinking_level == "disabled"
        assert app.settings.small_model == tree.small_model == "old-small"
        assert calls[-1]["model"].id == tree.model
    finally:
        app.close()
        unregister_api_provider(_API)
