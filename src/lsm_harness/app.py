"""Composition root for one local LSM Harness instance."""

from __future__ import annotations

import queue
import threading
from uuid import uuid4

from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.events import HarnessEvent, Observer, make_event
from lsm_harness.loop import run_loop
from lsm_harness.memory import Memory
from lsm_harness.models import get_client, PROVIDERS
from lsm_harness.ops.tracing import Tracer
from lsm_harness.runtime import Session
from lsm_harness.security import redact_data
from lsm_harness.tools import build_registry
from lsm_harness.types import ModelClient, TurnResult


class Harness:
    def __init__(self, settings: Settings | None = None, client: ModelClient | None = None, conn=None):
        self.settings = settings or Settings()
        self.settings.ensure_home()
        self.conn = conn or connect(self.settings.home)
        self.client = client or get_client(
            provider_name=self.settings.provider,
            api_key=self.settings.api_key,
            base_url=self.settings.base_url or None,
            model=self.settings.model,
            small_model=self.settings.small_model,
            thinking=self.settings.thinking,
        )
        # Fill in defaults from provider if not explicitly set
        if hasattr(self.client, '_resolved_model'):
            if not self.settings.model:
                self.settings.model = self.client._resolved_model
            if not self.settings.small_model:
                self.settings.small_model = self.client._resolved_small_model
        self.memory = Memory(self.conn, self.settings, self.client)
        self.tools = build_registry(self.conn, self.settings, self.memory)
        self.session = Session(self.settings, self.memory)
        self.tracer = Tracer(self.settings.home)

        # ── interrupt / steering state ─────────────────────
        self._interrupt: threading.Event | None = None
        self._steering: queue.Queue[str] | None = None
        self._running = False

    # ── public API ───────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    def abort(self) -> None:
        """Cancel the currently running turn.

        Thread-safe — can be called from a signal handler or
        background thread.
        """
        if self._interrupt:
            self._interrupt.set()

    def steer(self, message: str) -> None:
        """Inject a message into the running loop.

        The message will be seen by the model at the next iteration
        boundary.  Call from any thread.
        """
        if self._steering:
            self._steering.put(message)

    def respond(
        self,
        user_message: str,
        *,
        observer: Observer | None = None,
        source: str = "cli",
    ) -> TurnResult:
        turn_id = str(uuid4())

        # ── fresh interrupt + steering for this turn ───────
        self._interrupt = threading.Event()
        self._steering = queue.Queue()
        self._running = True

        def emit(event_type: str, data: dict) -> None:
            safe_data = redact_data(data, (self.settings.api_key,))
            event = make_event(event_type, turn_id, safe_data)
            self.tracer.write(event)
            # ── usage tracking ────────────────────────────
            if event_type == "llm.completed" and "usage" in safe_data:
                u = safe_data["usage"]
                self.tracer.log_usage(
                    session_id=self.session.session_id,
                    model=safe_data.get("model", self.settings.model),
                    input_tokens=u.get("input_tokens", 0),
                    output_tokens=u.get("output_tokens", 0),
                    turn_id=turn_id,
                )
            if observer:
                observer(event)

        emit("turn.started", {"source": source, "message": user_message})
        try:
            system, messages = self.session.prepare_context(
                user_message, emit, self.tools.schemas()
            )

            # ── overflow recovery callback ──────────────────
            # When the loop hits a token limit, this compacts context
            # and returns a fresh (system, messages) so the retry
            # doesn't hit the same limit.
            def compact_and_retry():
                return self.session.compact_and_rebuild(
                    user_message, emit, self.tools.schemas()
                )

            result = run_loop(
                client=self.client,
                model=self.settings.model,
                system=system,
                messages=messages,
                tools=self.tools,
                max_iterations=self.settings.max_iterations,
                max_tokens=self.settings.max_tokens,
                emit=emit,
                interrupt=self._interrupt,
                steering_queue=self._steering,
                on_truncation=compact_and_retry,
            )
            if not result.aborted:
                self.session.add_exchange(user_message, result, source)
                self.memory.consolidate(emit)
                self.memory.export_markdown()
            # ── usage tracking ────────────────────────────
            if result.iterations > 0:
                # Log approximate usage (one main-model call per iteration)
                # The llm.completed events carry the real numbers;
                # this is a conservative estimate when events aren't available.
                pass  # usage is logged per-turn via llm.completed events
            emit(
                "turn.completed",
                {
                    "reply": result.reply,
                    "iterations": result.iterations,
                    "tools": [item["tool"] for item in result.tool_calls],
                    "aborted": result.aborted,
                },
            )
            return result
        except Exception as exc:
            emit("turn.failed", {"error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            self._running = False
            self._interrupt = None
            self._steering = None

    def close(self) -> None:
        self.conn.close()
