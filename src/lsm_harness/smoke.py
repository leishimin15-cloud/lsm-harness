"""Deterministic, no-API end-to-end harness smoke test."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from lsm_harness.app import Harness
from lsm_harness.config import Settings
from lsm_harness.types import ModelResponse, ToolCall, Usage


class ScriptedClient:
    def complete(self, *, model, system, messages, tools, max_tokens):
        prompt = str(messages[0].get("content", "")) if messages else ""
        if "长期记忆检索门" in prompt:
            return ModelResponse(
                text='{"retrieve": false, "query": "", "reason": "自包含测试"}',
                usage=Usage(8, 8),
            )
        if "提炼为长期记忆" in prompt:
            return ModelResponse(
                text=(
                    '{"facts":[{"subject":"LSM Harness","content":'
                    '"本地核心闭环已通过确定性测试"}],'
                    '"episode":"完成了一次本地 Harness 冒烟测试"}'
                ),
                usage=Usage(12, 12),
            )
        if any(message.get("role") == "tool" for message in messages):
            return ModelResponse(text="本地测试事件已经创建。", usage=Usage(8, 8))
        return ModelResponse(
            tool_calls=[
                ToolCall(
                    "smoke-call-1",
                    "create_event",
                    {
                        "title": "LSM Harness Smoke Test",
                        "start": "2026-08-06T10:00",
                        "notes": "deterministic local validation",
                    },
                )
            ],
            stop_reason="tool_calls",
            usage=Usage(8, 8),
        )


def run() -> int:
    with tempfile.TemporaryDirectory(prefix="lsm-harness-smoke-") as directory:
        settings = Settings(
            api_key="scripted",
            model="scripted-main",
            small_model="scripted-small",
            home=Path(directory),
            consolidate_every=1,
        )
        events = []
        app = Harness(settings=settings, client=ScriptedClient())
        try:
            result = app.respond("创建本地测试事件", observer=events.append, source="smoke")
            counts = {
                table: app.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("calendar_events", "chat_log", "facts", "episodes")
            }
            trace_files = list((settings.home / "traces").glob("*.jsonl"))
            assert result.iterations == 2
            assert counts == {"calendar_events": 1, "chat_log": 2, "facts": 1, "episodes": 1}
            assert trace_files and (settings.home / "MEMORY.md").exists()
            event_types = [event.type for event in events]
            required = {
                "turn.started",
                "memory.gate.decided",
                "llm.completed",
                "tool.requested",
                "tool.completed",
                "memory.consolidated",
                "turn.completed",
            }
            assert required <= set(event_types)
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "iterations": result.iterations,
                        "counts": counts,
                        "events": event_types,
                        "reply": result.reply,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        finally:
            app.close()

