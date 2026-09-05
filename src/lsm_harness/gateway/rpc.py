"""RPC mode: ``lsm rpc`` — JSONL command protocol over stdin/stdout.

Pi/tau-style machine interface, v1 (plan Phase 8):

- **In**: one JSON object per line — ``{"id": "1", "type": "prompt",
  "message": "hi"}``.  Records larger than 16 MiB are rejected.
- **Out**: command answers as
  ``{"id": ..., "type": "response", "command": ..., "success": bool, ...}``
  and run events as ``{"type": "event", "event": {...}}`` — the same
  legacy string-event vocabulary the CLI/TUI/Tracer consume.
- **Threads**: the main thread blocks on stdin and dispatches; at most one
  worker thread runs ``harness.respond()`` (its observer fires on that
  thread).  A single lock serialises stdout writes.  stdin EOF aborts a
  running turn and closes the harness.

Sync design, stdlib only — no asyncio.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, TextIO

from lsm_harness.ai.providers import PROVIDERS
from lsm_harness.ops.session_store import _message_to_dict

MAX_RECORD_BYTES = 16 * 1024 * 1024  # 16 MiB, aligned with tau
THINKING_LEVELS = ["disabled", "auto", "enabled"]


class RpcServer:
    """Serves one Harness over stdin/stdout JSONL."""

    def __init__(self, app, stdin: TextIO, stdout: TextIO) -> None:
        self.app = app
        self.stdin = stdin
        self.stdout = stdout
        self._write_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        # Set synchronously before the worker starts — closes the race
        # where a second prompt arrives before respond() flips is_running.
        self._busy = False

    # ── wire ───────────────────────────────────────────────────

    def _write(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False, default=str)
        with self._write_lock:
            self.stdout.write(line + "\n")
            self.stdout.flush()

    def _answer(self, rid, command: str, success: bool, **fields) -> None:
        self._write({
            "id": rid,
            "type": "response",
            "command": command,
            "success": success,
            **fields,
        })

    def _on_event(self, event) -> None:
        # Fires on the worker thread during respond().
        self._write({"type": "event", "event": event.as_dict()})

    # ── main loop ──────────────────────────────────────────────

    def serve(self) -> int:
        try:
            while True:
                line = self.stdin.readline()
                if not line:  # EOF
                    break
                if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
                    self._write({
                        "id": None,
                        "type": "response",
                        "command": None,
                        "success": False,
                        "error": "record exceeds 16 MiB limit",
                    })
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._write({
                        "id": None,
                        "type": "response",
                        "command": None,
                        "success": False,
                        "error": f"invalid JSON: {exc}",
                    })
                    continue
                self._dispatch(request)
        finally:
            self.app.abort()
            worker = self._worker
            if worker is not None and worker.is_alive():
                worker.join(timeout=5)
            self.app.close()
        return 0

    def _dispatch(self, request: dict[str, Any]) -> None:
        rid = request.get("id")
        command = request.get("type")
        handler = self._COMMANDS.get(command)
        if handler is None:
            self._answer(rid, command, False, error=f"unknown command: {command!r}")
            return
        try:
            handler(self, rid, request)
        except Exception as exc:
            self._answer(rid, command, False, error=f"{type(exc).__name__}: {exc}")

    # ── commands ───────────────────────────────────────────────

    def _cmd_prompt(self, rid, request: dict[str, Any]) -> None:
        message = request.get("message")
        if not isinstance(message, str) or not message:
            self._answer(rid, "prompt", False, error="missing message")
            return
        if self._busy:
            # Wait (briefly) for the agent to actually enter the run so
            # steer/follow_up land on the live queues, not on the floor.
            deadline = time.monotonic() + 5
            while (
                not self.app.is_running
                and self._worker is not None
                and self._worker.is_alive()
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            behavior = request.get("streamingBehavior", "steer")
            if behavior == "followUp":
                queued = self.app.follow_up(message)
            else:
                queued = self.app.steer(message)
            self._answer(rid, "prompt", queued, queued_as=behavior)
            return
        self._start_worker(message)
        self._answer(rid, "prompt", True)

    def _start_worker(self, message: str) -> None:
        def run() -> None:
            try:
                self.app.respond(message, observer=self._on_event, source="rpc")
            except Exception as exc:
                # A crashed run must still terminate cleanly on the wire.
                self._write({
                    "type": "rpc_error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            finally:
                self._busy = False

        self._busy = True
        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _cmd_abort(self, rid, _request) -> None:
        self.app.abort()
        self._answer(rid, "abort", True)

    def _cmd_get_state(self, rid, _request) -> None:
        settings = self.app.settings
        pending = self.app.pending_messages()
        self._answer(
            rid,
            "get_state",
            True,
            is_running=self.app.is_running,
            provider=settings.provider,
            model=settings.model,
            small_model=settings.small_model,
            thinking=settings.thinking,
            session_id=self.app.session.session_id,
            pending={key: len(value) for key, value in pending.items()},
        )

    def _cmd_set_model(self, rid, request: dict[str, Any]) -> None:
        provider = request.get("provider")
        if provider not in PROVIDERS:
            self._answer(
                rid, "set_model", False,
                error=f"unknown provider: {provider!r}",
            )
            return
        self.app.switch_model(
            provider,
            model=request.get("model") or "",
            small_model=request.get("small_model") or "",
        )
        self._answer(
            rid, "set_model", True,
            provider=provider, model=self.app.settings.model,
        )

    def _set_thinking(self, level: str) -> None:
        self.app.settings.thinking = level
        self.app.session.record_thinking_change(level)

    def _cmd_set_thinking_level(self, rid, request: dict[str, Any]) -> None:
        level = request.get("level")
        if level not in THINKING_LEVELS:
            self._answer(
                rid, "set_thinking_level", False,
                error=f"level must be one of {THINKING_LEVELS}",
            )
            return
        self._set_thinking(level)
        self._answer(rid, "set_thinking_level", True, level=level)

    def _cmd_cycle_thinking_level(self, rid, _request) -> None:
        current = self.app.settings.thinking
        index = THINKING_LEVELS.index(current) if current in THINKING_LEVELS else 0
        level = THINKING_LEVELS[(index + 1) % len(THINKING_LEVELS)]
        self._set_thinking(level)
        self._answer(rid, "cycle_thinking_level", True, level=level)

    def _cmd_new_session(self, rid, _request) -> None:
        session_id = self.app.session.start_new()
        self._answer(rid, "new_session", True, session_id=session_id)

    def _cmd_switch_session(self, rid, request: dict[str, Any]) -> None:
        ref = str(request.get("session_id") or "")
        session_id = self.app.session.resume(ref)
        if session_id is None:
            self._answer(
                rid, "switch_session", False,
                error=f"no unique session matching {ref!r}",
            )
            return
        self._answer(rid, "switch_session", True, session_id=session_id)

    def _cmd_get_messages(self, rid, _request) -> None:
        worker = self._worker
        if worker is not None and worker.is_alive():
            # A sequential client asks after the run; waiting briefly keeps
            # the answer consistent instead of racing the recorder.
            worker.join(timeout=10)
        context = self.app.session.build_session_context()
        messages = (
            [_message_to_dict(m) for m in context.messages]
            if context is not None
            else []
        )
        self._answer(rid, "get_messages", True, messages=messages)

    def _cmd_get_available_models(self, rid, _request) -> None:
        models = [
            {
                "provider": name,
                "model": provider.model,
                "small_model": provider.small_model,
            }
            for name, provider in PROVIDERS.items()
        ]
        self._answer(rid, "get_available_models", True, models=models)

    def _cmd_compact(self, rid, _request) -> None:
        if self.app.is_running:
            self._answer(rid, "compact", False, error="cannot compact mid-run")
            return
        compacted = self.app.session.compact(lambda *_: None)
        self._answer(rid, "compact", compacted)

    _COMMANDS = {
        "prompt": _cmd_prompt,
        "abort": _cmd_abort,
        "get_state": _cmd_get_state,
        "set_model": _cmd_set_model,
        "set_thinking_level": _cmd_set_thinking_level,
        "cycle_thinking_level": _cmd_cycle_thinking_level,
        "new_session": _cmd_new_session,
        "switch_session": _cmd_switch_session,
        "get_messages": _cmd_get_messages,
        "get_available_models": _cmd_get_available_models,
        "compact": _cmd_compact,
    }


def run_rpc(*, app=None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """Entry point for ``lsm rpc``; streams are injectable for tests.

    Note: ``serve()`` aborts and closes the harness on stdin EOF — an
    injected app is closed too.  When building your own app for RPC,
    create its SQLite connection with ``check_same_thread=False`` because
    ``respond()`` runs on the worker thread.
    """
    if app is None:
        from lsm_harness.coding_agent.app import Harness
        from lsm_harness.config import Settings
        from lsm_harness.db import connect

        settings = Settings()
        settings.ensure_home()
        conn = connect(settings.home, check_same_thread=False)
        app = Harness(settings=settings, conn=conn)
    return RpcServer(app, stdin or sys.stdin, stdout or sys.stdout).serve()
