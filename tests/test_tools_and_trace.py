import json
from pathlib import Path

from lsm_harness.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.memory.facade import Memory
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
    first = tools.execute("create_event", args)
    second = tools.execute("create_event", args)
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
    )
    assert "新内容" in memory.facts.list(1)[0]["content"]
    assert "Deleted" in tools.execute(
        "manage_memory", {"action": "delete", "id": fact_id}
    )


def test_external_effect_is_blocked():
    registry = ToolRegistry({"read", "local_write"})
    registry.register(Tool("send", "send", {"type": "object"}, lambda: "sent", "external_write"))
    assert "blocked" in registry.execute("send", {})


class HarnessClient:
    def complete(self, *, model, system, messages, tools, max_tokens):
        first = str(messages[0].get("content", ""))
        if "长期记忆检索门" in first:
            return ModelResponse(text='{"retrieve":false,"query":"","reason":"测试"}')
        return ModelResponse(text="完成")


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
    assert events[0] == "turn.started"
    assert events[-1] == "turn.completed"
    assert secret not in raw

