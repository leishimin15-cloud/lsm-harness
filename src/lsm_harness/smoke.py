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
            return ModelResponse(text="冒烟命令已执行。", usage=Usage(8, 8))
        return ModelResponse(
            tool_calls=[
                ToolCall(
                    "smoke-call-1",
                    "exec",
                    {"command": "echo lsm-smoke-ok"},
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
            sandbox_enabled=False,
        )
        events = []
        app = Harness(settings=settings, client=ScriptedClient())
        try:
            result = app.respond("执行一条本地冒烟命令", observer=events.append, source="smoke")
            counts = {
                table: app.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "chat_log",
                    "facts",
                    "episodes",
                    "sessions",
                    "session_summaries",
                )
            }
            trace_files = list((settings.home / "traces").glob("*.jsonl"))
            assert result.iterations == 2
            assert counts == {
                "chat_log": 2,
                "facts": 1,
                "episodes": 1,
                "sessions": 1,
                "session_summaries": 0,
            }
            assert trace_files and (settings.home / "MEMORY.md").exists()
            event_types = [event.type for event in events]
            required = {
                "trace.started",
                "turn.started",
                "memory.gate.decided",
                "context.measured",
                "context.built",
                "llm.completed",
                "tool.requested",
                "tool.completed",
                "memory.consolidated",
                "turn.completed",
                "trace.completed",
            }
            assert required <= set(event_types)
            assert event_types.count("trace.started") == 1
            assert event_types.count("turn.started") == 2
            assert event_types.count("turn.completed") == 2
            assert event_types.count("trace.completed") == 1
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
