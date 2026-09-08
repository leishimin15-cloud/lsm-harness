"""Stateful Agent wrapper around the stateless Agent Loop."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

from lsm_harness.agent.agent_loop import Emit, run_agent_loop
from lsm_harness.agent.events import AgentEventListener
from lsm_harness.agent.hooks import PrepareNextTurn, ShouldStopAfterTurn
from lsm_harness.agent.pending import PendingMessage, PendingMessageQueue, QueueMode
from lsm_harness.agent.types import (
    AfterToolCall,
    AgentContext,
    AgentLoopConfig,
    BeforeToolCall,
    ToolExecutionMode,
    TraceResult,
)
from lsm_harness.ai.types import StreamFunction


@dataclass
class ActiveRun:
    """State of one in-flight run, owned by the Agent.

    Not frozen: ``result`` is written by ``finish()`` at teardown.  The
    happens-before rule is ``result`` → clear ``_active_run`` →
    ``done.set()``; readers must ``done.wait()`` before reading ``result``.
    """

    interrupt: threading.Event
    trace_id: str = ""
    session_id: str = ""
    started_at: float = 0.0
    done: threading.Event = field(default_factory=threading.Event)
    result: TraceResult | None = None


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

    def begin(self, *, trace_id: str = "", session_id: str = "") -> ActiveRun:
        """Accept a new run; the Agent is busy from this instant.

        Called on the ACCEPTING thread (before any worker starts), so
        ``is_running`` covers the whole startup window.  Raises
        ``RuntimeError`` when another run is already active.
        """
        with self._lock:
            if self._active_run is not None:
                raise RuntimeError("another trace is already running")
            active = ActiveRun(
                interrupt=threading.Event(),
                trace_id=trace_id,
                session_id=session_id,
                started_at=time.monotonic(),
            )
            self._active_run = active
            return active

    def finish(self, active: ActiveRun, result: TraceResult | None = None) -> None:
        """End a run: record its result, then signal idle — never before.

        Write order is ``result`` → clear ``_active_run`` → ``done.set()``;
        a woken ``wait_for_idle`` reader can therefore trust ``result``.
        """
        with self._lock:
            if self._active_run is active:
                active.result = result
                self._active_run = None
                active.done.set()

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Wait until no run is active.  True when already idle.

        Takes the run reference under the lock but waits OUTSIDE it —
        holding the lock across ``wait`` would deadlock ``finish()``.
        """
        with self._lock:
            active = self._active_run
        if active is None:
            return True
        return active.done.wait(timeout)

    # ── loop invocation (Phase 1: run lifecycle lives on the Agent) ──

    def _full_config(self, config: AgentLoopConfig) -> AgentLoopConfig:
        """Inject the Agent-owned policy fields into a run's config.

        Callers pass only the product/per-run fields (model, budgets,
        truncation hook, approval, trace/session ids, extra listeners);
        the queues, turn hooks, execution mode and subscribed listeners
        belong to the Agent — nobody reads them back out of it.
        """
        return replace(
            config,
            get_steering_messages=self.steering_queue.drain,
            get_follow_up_messages=self.follow_up_queue.drain,
            prepare_next_turn=self.prepare_next_turn,
            should_stop_after_turn=self.should_stop_after_turn,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            tool_execution=self.tool_execution,
            # Subscription order matters: the Agent's own listeners first,
            # then run-scoped ones (e.g. the session recorder) last.
            listeners=[*self.listeners, *(config.listeners or [])],
        )

    def run(
        self,
        active: ActiveRun,
        context: AgentContext,
        config: AgentLoopConfig,
        *,
        stream_fn: StreamFunction,
        emit: Emit,
    ) -> TraceResult:
        """Run the loop inside an ALREADY-begun run.

        Lifecycle stays with the caller: this never calls ``begin()`` or
        ``finish()`` — teardown (persistence, terminal events) happens
        between the loop's return and the caller's ``finish()``.
        """
        return run_agent_loop(
            context=context,
            config=self._full_config(config),
            stream_fn=stream_fn,
            emit=emit,
            interrupt=active.interrupt,
        )

    def prompt(
        self,
        context: AgentContext,
        config: AgentLoopConfig,
        *,
        stream_fn: StreamFunction,
        emit: Emit,
    ) -> TraceResult:
        """Convenience entry: begin → run → finish, for bare-Agent use.

        A standalone Agent + fake model runs with no CLI/TUI/SQLite.
        Products that need post-loop work before idle (persistence,
        terminal events) use ``begin()``/``run()``/``finish()`` instead.
        """
        active = self.begin()
        result: TraceResult | None = None
        try:
            result = self.run(active, context, config, stream_fn=stream_fn, emit=emit)
            return result
        finally:
            self.finish(active, result)

    def continue_(
        self,
        context: AgentContext,
        config: AgentLoopConfig,
        *,
        stream_fn: StreamFunction,
        emit: Emit,
    ) -> TraceResult:
        """Re-run the loop on the EXISTING context (Pi ``continue()``).

        Unlike ``prompt()`` this implies no fresh user turn: nothing is
        appended — the context's own tail drives the loop.  Pi legality:
        the last message must be one the model can answer (user or tool
        result).  An assistant-tipped context is legal ONLY with queued
        messages: ONE drained batch becomes the run's input — steering
        first, else follow-ups (Pi's drain order, honoring each queue's
        one-at-a-time mode) — delivered via the loop's initial-pending
        channel with its source intact.  The batches stay in their own
        queues until drained; nothing is re-labelled or moved wholesale.
        """
        messages = context.messages
        if not messages:
            raise ValueError("nothing to continue: context is empty")
        if getattr(messages[-1], "role", None) == "assistant":
            steering = self.steering_queue.drain()
            if steering:
                config = replace(
                    config,
                    initial_pending_messages=steering,
                    initial_pending_source="steering",
                )
            else:
                follow_ups = self.follow_up_queue.drain()
                if not follow_ups:
                    raise ValueError(
                        "nothing to continue: context ends on an assistant "
                        "message and no steering/follow-up is queued"
                    )
                config = replace(
                    config,
                    initial_pending_messages=follow_ups,
                    initial_pending_source="follow_up",
                )
        return self.prompt(context, config, stream_fn=stream_fn, emit=emit)


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
