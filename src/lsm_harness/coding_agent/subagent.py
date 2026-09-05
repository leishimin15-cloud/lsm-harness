"""Coding-agent subagent manager with isolated product context.

Three modes:
  - ``spawn``: fire-and-forget, result announced via callback
  - ``run_inline``: block until done, return result directly
  - ``delegate``: (existing) shell out to pi/codex

Each subagent gets:
  - A fresh session (clean context window)
  - A filtered tool set (optional whitelist)
  - Shared model client and settings
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable
from uuid import uuid4


@dataclass
class SubagentStatus:
    """Real-time status of a running subagent."""

    task_id: str
    label: str
    task: str
    started_at: float = field(default_factory=time.monotonic)
    phase: str = "running"  # running | done | error
    iteration: int = 0
    tools_called: list[str] = field(default_factory=list)
    error: str | None = None
    reply: str = ""
    finished_at: float | None = None
    parent_turn_id: str = ""

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


ResultCallback = Callable[[str, str], None]  # (task_id, reply) -> None


class SubagentManager:
    """Manages background subagent execution with concurrency control.

    Usage::

        mgr = SubagentManager(max_concurrent=3)
        mgr.on_complete(lambda task_id, reply: print(f"{task_id} done"))
        task_id = mgr.spawn("search for X", harness=h, label="search-x")
        # or:
        reply = mgr.run_inline("search for X", harness=h)
    """

    def __init__(self, max_concurrent: int = 3):
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._futures: dict[str, Future] = {}
        self._statuses: dict[str, SubagentStatus] = {}
        self._lock = threading.Lock()
        self._callbacks: list[ResultCallback] = []
        self._children: dict[str, Any] = {}
        self.max_concurrent = max_concurrent

    # ── public API ─────────────────────────────────────────────

    def on_complete(self, callback: ResultCallback) -> None:
        """Register a callback invoked when any subagent finishes.

        ``callback(task_id, reply)`` is called from a worker thread.
        """
        self._callbacks.append(callback)

    def spawn(
        self,
        task: str,
        *,
        harness: Any,  # Harness (avoid circular import)
        label: str = "",
        tools_whitelist: list[str] | None = None,
        emit: Callable[[str, dict[str, Any]], None] | None = None,
        parent_turn_id: str = "",
    ) -> str:
        """Start a subagent in the background. Returns ``task_id`` immediately.

        The result is delivered via the ``on_complete`` callback when ready.

        Args:
            task: Clear description of what the subagent should do.
            harness: The active Harness instance (used to create a fresh session).
            label: Short human-readable label for status display.
            tools_whitelist: If set, only these tool names are available to the
                             subagent.  Useful for read-only subagents.
        """
        running = self.get_running_count()
        if running >= self.max_concurrent:
            return (
                f"无法启动子 Agent：并发已满 ({running}/{self.max_concurrent})。"
                f"请等待运行中的子 Agent 完成后再试。"
            )

        task_id = str(uuid4())[:8]
        display_label = label or task[:40] + ("…" if len(task) > 40 else "")

        status = SubagentStatus(
            task_id=task_id,
            label=display_label,
            task=task,
            parent_turn_id=parent_turn_id,
        )
        with self._lock:
            self._statuses[task_id] = status

        future = self._executor.submit(
            self._run,
            task_id=task_id,
            task=task,
            harness=harness,
            tools_whitelist=tools_whitelist,
            emit=emit,
        )
        with self._lock:
            self._futures[task_id] = future

        def _done_callback(f: Future) -> None:
            with self._lock:
                self._futures.pop(task_id, None)
            try:
                reply = f.result()
            except Exception as exc:
                reply = f"子 Agent 异常: {type(exc).__name__}: {exc}"
                if task_id in self._statuses:
                    self._statuses[task_id].phase = "error"
                    self._statuses[task_id].error = str(exc)
            with self._lock:
                status = self._statuses.get(task_id)
                if status:
                    status.reply = reply
                    status.finished_at = time.monotonic()
            if emit:
                emit("subagent.completed", {
                    "task_id": task_id,
                    "label": display_label,
                    "status": self._statuses.get(task_id).phase if task_id in self._statuses else "done",
                    "reply": reply,
                })
            for cb in self._callbacks:
                try:
                    cb(task_id, reply)
                except Exception:
                    pass

        future.add_done_callback(_done_callback)
        if emit:
            emit("subagent.started", {
                "task_id": task_id,
                "label": display_label,
                "task": task,
            })
        return f"子 Agent [{display_label}] 已启动 (id: {task_id})。"

    def run_inline(
        self,
        task: str,
        *,
        harness: Any,
        tools_whitelist: list[str] | None = None,
        emit: Callable[[str, dict[str, Any]], None] | None = None,
        parent_turn_id: str = "",
    ) -> str:
        """Run a subagent synchronously and return its reply.

        Blocks until the subagent completes.  The subagent's tool calls
        are invisible to the caller; only the final reply is returned.
        """
        task_id = str(uuid4())[:8]
        with self._lock:
            self._statuses[task_id] = SubagentStatus(
                task_id=task_id,
                label=task[:40],
                task=task,
                parent_turn_id=parent_turn_id,
            )
        if emit:
            emit("subagent.started", {"task_id": task_id, "label": task[:40], "task": task})
        reply = self._run(
            task_id=task_id,
            task=task,
            harness=harness,
            tools_whitelist=tools_whitelist,
            emit=emit,
        )
        status = self.get_status(task_id)
        if emit:
            emit("subagent.completed", {
                "task_id": task_id,
                "label": task[:40],
                "status": status.phase if status else "done",
                "reply": reply,
            })
        return reply

    def get_status(self, task_id: str) -> SubagentStatus | None:
        with self._lock:
            return self._statuses.get(task_id)

    def get_running_count(self) -> int:
        with self._lock:
            return sum(
                1 for f in self._futures.values()
                if not f.done()
            )

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            statuses = list(self._statuses.values())
        return [
            {
                "task_id": item.task_id,
                "label": item.label,
                "task": item.task,
                "phase": item.phase,
                "iteration": item.iteration,
                "tools_called": list(item.tools_called),
                "error": item.error,
                "reply": item.reply,
                "elapsed": round(item.elapsed, 3),
                "parent_turn_id": item.parent_turn_id,
            }
            for item in statuses
        ]

    def cancel(self, task_id: str) -> bool:
        """Request cancellation of a running subagent.

        Returns True if the task was found and not already done.
        """
        with self._lock:
            future = self._futures.get(task_id)
            if future and not future.done():
                child = self._children.get(task_id)
                if child is not None:
                    child.abort()
                    return True
                return future.cancel()
        return False

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the executor, optionally waiting for running tasks."""
        self._executor.shutdown(wait=wait)

    # ── internal ───────────────────────────────────────────────

    def _run(
        self,
        *,
        task_id: str,
        task: str,
        harness: Any,
        tools_whitelist: list[str] | None,
        emit: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> str:
        """Core subagent execution — runs in a worker thread."""
        status = self._statuses.get(task_id)
        child = None

        try:
            if harness is None:
                raise RuntimeError("subagent manager is not bound to a Harness")

            # A subagent gets its own mutable runtime. It may share the
            # thread-safe model client and durable home, but never the parent
            # Session, ToolRegistry, interrupt state, or SQLite connection.
            child_settings = replace(
                harness.settings,
                sandbox_enabled=False,
                subagent_max_concurrent=1,
            )
            child = type(harness)(
                settings=child_settings,
                client=harness.client,
                stream_fn=harness.stream_fn,
            )
            child.session.start_new()
            with self._lock:
                self._children[task_id] = child

            # ── optional tool filtering ──────────────────────
            if tools_whitelist:
                child.tools = child.tools.filter(tools_whitelist)

            # ── execute ──────────────────────────────────────
            result = child.respond(
                task,
                source="subagent",
                observer=(lambda event: emit("subagent.progress", {
                    "task_id": task_id,
                    "event": event.as_dict(),
                })) if emit else None,
            )

            if status:
                status.phase = "done"
                status.iteration = result.iterations
                status.tools_called = [
                    t["tool"] for t in result.tool_calls
                ]
                status.reply = result.reply
                status.finished_at = time.monotonic()

            return result.reply or "任务完成但未生成回复。"

        except Exception as exc:
            if status:
                status.phase = "error"
                status.error = str(exc)
                status.finished_at = time.monotonic()
            return f"子 Agent 执行失败: {type(exc).__name__}: {exc}"

        finally:
            with self._lock:
                self._children.pop(task_id, None)
            if child is not None:
                child.close()
