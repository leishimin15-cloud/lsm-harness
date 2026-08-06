"""Composition root for one local LSM Harness instance."""

from __future__ import annotations

from uuid import uuid4

from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.events import HarnessEvent, Observer, make_event
from lsm_harness.loop import run_loop
from lsm_harness.memory import Memory
from lsm_harness.models import DeepSeekClient
from lsm_harness.ops.tracing import Tracer
from lsm_harness.runtime import Session
from lsm_harness.tools import build_registry
from lsm_harness.types import ModelClient, TurnResult


class Harness:
    def __init__(self, settings: Settings | None = None, client: ModelClient | None = None, conn=None):
        self.settings = settings or Settings()
        self.settings.ensure_home()
        self.conn = conn or connect(self.settings.home)
        self.client = client or DeepSeekClient(
            self.settings.api_key, self.settings.base_url, self.settings.thinking
        )
        self.memory = Memory(self.conn, self.settings, self.client)
        self.tools = build_registry(self.conn, self.settings, self.memory)
        self.session = Session(self.settings, self.memory)
        self.tracer = Tracer(self.settings.home)

    def respond(
        self,
        user_message: str,
        *,
        observer: Observer | None = None,
        source: str = "cli",
    ) -> TurnResult:
        turn_id = str(uuid4())

        def emit(event_type: str, data: dict) -> None:
            event = make_event(event_type, turn_id, data)
            self.tracer.write(event)
            if observer:
                observer(event)

        emit("turn.started", {"source": source, "message": user_message})
        try:
            system = self.session.build_system(user_message, emit)
            messages = [*self.session.window(), {"role": "user", "content": user_message}]
            result = run_loop(
                client=self.client,
                model=self.settings.model,
                system=system,
                messages=messages,
                tools=self.tools,
                max_iterations=self.settings.max_iterations,
                max_tokens=self.settings.max_tokens,
                emit=emit,
            )
            self.session.add_exchange(user_message, result, source)
            self.memory.consolidate(emit)
            self.memory.export_markdown()
            emit(
                "turn.completed",
                {
                    "reply": result.reply,
                    "iterations": result.iterations,
                    "tools": [item["tool"] for item in result.tool_calls],
                },
            )
            return result
        except Exception as exc:
            emit("turn.failed", {"error": f"{type(exc).__name__}: {exc}"})
            raise

    def close(self) -> None:
        self.conn.close()
