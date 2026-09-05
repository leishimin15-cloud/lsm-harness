"""Stateful Agent wrapper around the stateless Agent Loop."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from lsm_harness.agent.events import AgentEventListener
from lsm_harness.agent.hooks import PrepareNextTurn, ShouldStopAfterTurn
from lsm_harness.agent.pending import PendingMessage, PendingMessageQueue, QueueMode
from lsm_harness.agent.types import AfterToolCall, BeforeToolCall, ToolExecutionMode


@dataclass(frozen=True)
class ActiveRun:
    interrupt: threading.Event


def _wrap_listener(listener: AgentEventListener) -> AgentEventListener:
    """Isolate an untrusted listener so it cannot take down a run."""

    def wrapped(event):  # type: ignore[no-untyped-def]
        try:
            listener(event)
        except Exception:
            pass

    return wrapped


class Agent:
    """Own active-run state, cancellation, and Pi-style message queues."""

    def __init__(
        self,
        *,
        prepare_next_turn: PrepareNextTurn | None = None,
        should_stop_after_turn: ShouldStopAfterTurn | None = None,
        before_tool_call: BeforeToolCall | None = None,
        after_tool_call: AfterToolCall | None = None,
        tool_execution: ToolExecutionMode = "parallel",
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
    ) -> None:
        self.prepare_next_turn = prepare_next_turn
        self.should_stop_after_turn = should_stop_after_turn
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.tool_execution = tool_execution
        self.steering_queue = PendingMessageQueue(steering_mode)
        self.follow_up_queue = PendingMessageQueue(follow_up_mode)
        self._active_run: ActiveRun | None = None
        self._listeners: list[tuple[AgentEventListener, bool]] = []
        self._lock = threading.Lock()

    # ── typed kernel event subscription (Chapter 7) ──────────────

    def subscribe(
        self,
        listener: AgentEventListener,
        *,
        wrap: bool = False,
    ) -> Callable[[], None]:
        """Register a kernel event listener; returns an unsubscribe function.

        Listeners are injected into every run's ``AgentEventSink`` in
        subscription order. ``wrap=True`` isolates untrusted listeners with
        try/except (the third-party extension tier); unwrapped listeners
        fail fast — a listener bug fails the run, by design.
        """
        entry = (listener, wrap)
        with self._lock:
            self._listeners.append(entry)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._listeners.remove(entry)
                except ValueError:
                    pass

        return unsubscribe

    @property
    def listeners(self) -> list[AgentEventListener]:
        """Current listeners, wrapped on demand per their tier."""
        with self._lock:
            entries = list(self._listeners)
        return [
            _wrap_listener(listener) if wrapped else listener
            for listener, wrapped in entries
        ]

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._active_run is not None

    def begin(self) -> ActiveRun:
        with self._lock:
            if self._active_run is not None:
                raise RuntimeError("another trace is already running")
            active = ActiveRun(interrupt=threading.Event())
            self._active_run = active
            return active

    def finish(self, active: ActiveRun) -> None:
        with self._lock:
            if self._active_run is active:
                self._active_run = None

    def abort(self) -> bool:
        with self._lock:
            active = self._active_run
        if active is None:
            return False
        active.interrupt.set()
        return True

    def steer(self, message: PendingMessage) -> bool:
        with self._lock:
            if self._active_run is None:
                return False
            self.steering_queue.enqueue(message)
            return True

    def follow_up(self, message: PendingMessage) -> bool:
        with self._lock:
            if self._active_run is None:
                return False
            self.follow_up_queue.enqueue(message)
            return True

    def has_queued_messages(self) -> bool:
        return (
            self.steering_queue.has_items()
            or self.follow_up_queue.has_items()
        )

    def pending_messages(self) -> dict[str, list[PendingMessage]]:
        return {
            "steering": self.steering_queue.snapshot(),
            "follow_up": self.follow_up_queue.snapshot(),
        }

    def clear_queue(self) -> dict[str, list[PendingMessage]]:
        """Compatibility alias for clearing both Pi-style queues."""
        return self.clear_all_queues()

    def clear_steering_queue(self) -> list[PendingMessage]:
        return self.steering_queue.clear()

    def clear_follow_up_queue(self) -> list[PendingMessage]:
        return self.follow_up_queue.clear()

    def clear_all_queues(self) -> dict[str, list[PendingMessage]]:
        return {
            "steering": self.clear_steering_queue(),
            "follow_up": self.clear_follow_up_queue(),
        }
