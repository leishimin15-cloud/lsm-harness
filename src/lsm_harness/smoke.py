"""Deterministic, no-API end-to-end harness smoke test."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.ai.messages import message_to_wire
from lsm_harness.ai.stream import response_stream_function
from lsm_harness.ai.types import ModelResponse, ToolCall, Usage


class ScriptedClient:
    def complete(self, *, model, system, messages, tools, max_tokens):
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


def _scripted_stream_fn(client: ScriptedClient):
    """Canonical StreamFunction view of the scripted smoke client."""
    def respond(model, context, options):
        return client.complete(
            model=model.id,
            system=context.system_prompt,
            messages=[message_to_wire(m) for m in context.messages],
            tools=context.tools,
            max_tokens=options.max_tokens,
        )

    return response_stream_function(respond)


def run() -> int:
    with tempfile.TemporaryDirectory(prefix="lsm-harness-smoke-") as directory:
        settings = Settings(
            api_key="scripted",
            model="scripted-main",
            small_model="scripted-small",
            home=Path(directory),
        )
        events = []
        client = ScriptedClient()
        app = Harness(
            settings=settings, client=client, stream_fn=_scripted_stream_fn(client)
        )
        try:
            result = app.respond("执行一条本地冒烟命令", observer=events.append, source="smoke")
            counts = {
                table: app.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "chat_log",
                    "sessions",
                    "session_summaries",
                )
            }
            trace_files = list((settings.home / "traces").glob("*.jsonl"))
            assert result.iterations == 2
            assert counts == {
                "chat_log": 2,
                "sessions": 1,
                "session_summaries": 0,
            }
            assert trace_files
            event_types = [event.type for event in events]
            required = {
                "trace.started",
                "turn.started",
                "context.measured",
                "context.built",
                "llm.completed",
                "tool.requested",
                "tool.completed",
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
