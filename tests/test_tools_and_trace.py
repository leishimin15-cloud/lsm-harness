import json
from pathlib import Path

from lsm_harness.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.memory.facade import Memory
from lsm_harness.loop.hooks import LoopHooks
from lsm_harness.smoke import ScriptedClient
from lsm_harness.tools import build_registry
from lsm_harness.tools.registry import Tool, ToolRegistry
from lsm_harness.types import ModelResponse, ToolCall

from helpers import QueueClient


def memory_and_tools(tmp_path):
    settings = Settings(api_key="x", home=tmp_path)
    conn = connect(tmp_path)
    client = QueueClient()
    memory = Memory(conn, settings, client)
    return conn, memory, build_registry(conn, settings, memory)


def test_calendar_is_local_and_idempotent(tmp_path):
    conn, _, tools = memory_and_tools(tmp_path)
    args = {"title": "测试会议", "start": "2026-08-06T09:00"}
    first = tools.execute("create_event", args).output
    second = tools.execute("create_event", args).output
    assert "not synced externally" in first
    assert "not duplicated" in second
    assert conn.execute("SELECT COUNT(*) FROM calendar_events").fetchone()[0] == 1
    assert (tmp_path / "calendar.ics").exists()


def test_save_update_delete_memory(tmp_path):
    _, memory, tools = memory_and_tools(tmp_path)
    tools.execute("save_note", {"subject": "项目", "content": "旧内容"})
    fact_id = memory.facts.list(1)[0]["id"]
    assert "Updated" in tools.execute(
        "manage_memory", {"action": "update", "id": fact_id, "content": "新内容"}
    ).output
    assert "新内容" in memory.facts.list(1)[0]["content"]
    assert "Deleted" in tools.execute(
        "manage_memory", {"action": "delete", "id": fact_id}
    ).output


def test_external_effect_is_blocked():
    registry = ToolRegistry({"read", "local_write"})
    registry.register(Tool("send", "send", {"type": "object"}, lambda: "sent", "external_write"))
    result = registry.execute("send", {})
    assert result.is_error
    assert "blocked" in result.output


class HarnessClient:
    def complete(self, *, model, system, messages, tools, max_tokens):
        first = str(messages[0].get("content", ""))
        if "长期记忆检索门" in first:
            return ModelResponse(text='{"retrieve":false,"query":"","reason":"测试"}')
        return ModelResponse(text="完成")


class FailingHarnessClient:
    def complete(self, *, model, system, messages, tools, max_tokens):
        first = str(messages[0].get("content", ""))
        if "长期记忆检索门" in first:
            return ModelResponse(text='{"retrieve":false,"query":"","reason":"测试"}')
        raise RuntimeError("main model unavailable")


def test_trace_order_and_secret_absence(tmp_path):
    secret = "sk-secret-must-not-appear"
    settings = Settings(api_key=secret, home=tmp_path, consolidate_every=99)
    app = Harness(settings=settings, client=HarnessClient())
    try:
        app.respond("你好", source="test")
    finally:
        app.close()
    path = next((tmp_path / "traces").glob("*.jsonl"))
    raw = path.read_text(encoding="utf-8")
    events = [json.loads(line)["type"] for line in raw.splitlines()]
    assert events[0] == "trace.started"
    assert events[-1] == "trace.completed"
    assert secret not in raw


def test_one_trace_contains_one_turn_per_model_call(tmp_path):
    settings = Settings(api_key="scripted", home=tmp_path, consolidate_every=99)
    observed = []
    app = Harness(settings=settings, client=ScriptedClient())
    try:
        result = app.respond("创建本地测试事件", observer=observed.append, source="test")
    finally:
        app.close()

    assert result.iterations == 2
    assert [event.type for event in observed].count("trace.started") == 1
    assert [event.type for event in observed].count("trace.completed") == 1

    turn_starts = [event for event in observed if event.type == "turn.started"]
    turn_ends = [event for event in observed if event.type == "turn.completed"]
    assert [event.data["turn_index"] for event in turn_starts] == [1, 2]
    assert [event.data["turn_index"] for event in turn_ends] == [1, 2]
    assert [event.data["status"] for event in turn_ends] == ["tool_use", "completed"]
    assert {event.trace_id for event in observed} == {observed[0].trace_id}


def test_failed_trace_is_not_persisted_and_ends_hook_once(tmp_path):
    observed = []
    hook_results = []
    hooks = LoopHooks(on_trace_end=lambda result, _emit: hook_results.append(result))
    settings = Settings(api_key="scripted", home=tmp_path, consolidate_every=99)
    app = Harness(settings=settings, client=FailingHarnessClient(), hooks=hooks)
    try:
        result = app.respond("不要保存失败请求", observer=observed.append, source="test")
        saved = app.conn.execute("SELECT COUNT(*) FROM chat_log").fetchone()[0]
    finally:
        app.close()

    assert result.status == "failed"
    assert result.stop_reason == "error"
    assert "main model unavailable" in result.error
    assert saved == 0
    assert hook_results == [result]
    event_types = [event.type for event in observed]
    assert "trace.failed" in event_types
    assert "trace.completed" not in event_types
    assert "persistence.started" not in event_types
    assert "memory.consolidation.started" not in event_types


def test_user_supplied_key_is_redacted_from_trace(tmp_path):
    secret = "sk-user-secret-123456789"
    settings = Settings(api_key="configured-key", home=tmp_path, consolidate_every=99)
    app = Harness(settings=settings, client=HarnessClient())
    try:
        app.respond(f"不要记录这个 Key：{secret}", source="test")
    finally:
        app.close()
    raw = next((tmp_path / "traces").glob("*.jsonl")).read_text(encoding="utf-8")
    assert secret not in raw
    assert "[REDACTED]" in raw


# ── multi-provider ────────────────────────────────────────────────


def test_get_client_returns_openai_compat():
    from lsm_harness.models import get_client
    client = get_client(
        provider_name="deepseek",
        api_key="sk-test",
        base_url="https://test.example.com",
        thinking="disabled",
    )
    assert hasattr(client, "complete")
    assert hasattr(client, "stream_complete")
    assert client._resolved_model == "deepseek-v4-pro"
    assert client._resolved_small_model == "deepseek-v4-flash"


def test_provider_fills_model_defaults():
    from lsm_harness.models import get_client
    # Use deepseek (openai format, always available)
    client = get_client(
        provider_name="deepseek",
        api_key="sk-test-ds",
        base_url="https://test.example.com",
    )
    assert "deepseek" in client._resolved_model.lower()


# ── usage tracking ────────────────────────────────────────────────


def test_usage_logging_on_completion(tmp_path):
    from lsm_harness.app import Harness
    from lsm_harness.config import Settings

    settings = Settings(
        provider="deepseek",
        api_key="sk-test",
        home=tmp_path,
        consolidate_every=99,
    )
    app = Harness(settings=settings, client=HarnessClient())
    try:
        app.respond("hi", source="test")
    finally:
        app.close()

    usage_path = tmp_path / "usage.jsonl"
    assert usage_path.exists()
    lines = usage_path.read_text().strip().splitlines()
    assert len(lines) >= 1
    import json
    entry = json.loads(lines[0])
    assert "model" in entry
    assert entry["input_tokens"] >= 0
    assert entry["output_tokens"] >= 0


def test_usage_summary_aggregates(tmp_path):
    from lsm_harness.ops.tracing import Tracer
    tracer = Tracer(tmp_path)
    tracer.log_usage("sess-1", "deepseek-v4-pro", 100, 50)
    tracer.log_usage("sess-1", "deepseek-v4-pro", 200, 80)
    tracer.log_usage("sess-2", "deepseek-v4-flash", 50, 30)

    summary = tracer.usage_summary()
    assert summary["total_input"] == 350
    assert summary["total_output"] == 160
    assert len(summary["by_model"]) == 2
