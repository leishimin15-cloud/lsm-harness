"""Stateful Agent wrapper around the stateless Agent Loop."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

from lsm_harness.agent.agent_loop import Emit, run_agent_loop
from lsm_harness.agent.events import (
    AgentEndEvent,
    AgentEventListener,
    AgentEventSink,
    MessageEndEvent,
    MessageStartEvent,
    TurnEndEvent,
    make_legacy_adapter,
)
from lsm_harness.agent.hooks import PrepareNextTurn, ShouldStopAfterTurn
from lsm_harness.agent.messages import (
    AgentMessage,
    ConvertToLlm,
    TransformContext,
    assistant_message,
    default_convert_to_llm,
    user_message,
)
from lsm_harness.agent.pending import PendingMessage, PendingMessageQueue, QueueMode
from lsm_harness.agent.state import AgentState
from lsm_harness.agent.tool_history import repair_tool_history
from lsm_harness.agent.types import (
    AfterToolCall,
    AgentContext,
    AgentLoopConfig,
    BeforeToolCall,
    ToolExecutionMode,
    TraceResult,
)
from lsm_harness.ai.types import (
    PayloadHook,
    ResponseHook,
    StreamFunction,
    ThinkingBudgets,
    Transport,
)

# config=None 时的预算兜底(与 Settings 默认值一致)。
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_TOKENS = 8192


def _noop_emit(_kind: str, _data: dict) -> None:
    """Default legacy-event channel: drop everything."""


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


class Agent:
    """Own active-run state, cancellation, Pi-style message queues —
    and, since 阶段 A 批 2, the full :class:`AgentState` (Pi AgentState).

    The Agent holds a PERSISTENT :class:`AgentEventSink` that owns the
    state; ``agent.state`` is its transcript + identity + runtime
    projections.  ``prompt()`` ingests the user message through kernel
    events (the sink's reducer appends it), then runs the loop on
    ``state.messages`` — callers never build an ``AgentContext``.
    """

    def __init__(
        self,
        *,
        initial_state: AgentState | None = None,
        stream_fn: StreamFunction | None = None,
        prepare_next_turn: PrepareNextTurn | None = None,
        should_stop_after_turn: ShouldStopAfterTurn | None = None,
        before_tool_call: BeforeToolCall | None = None,
        after_tool_call: AfterToolCall | None = None,
        convert_to_llm: ConvertToLlm = default_convert_to_llm,
        transform_context: TransformContext | None = None,
        on_payload: PayloadHook | None = None,
        on_response: ResponseHook | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
        transport: Transport = "auto",
        max_retry_delay_ms: int | None = None,
        tool_execution: ToolExecutionMode = "parallel",
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        on_queue_change: Callable[[], None] | None = None,
    ) -> None:
        self._sink = AgentEventSink(state=initial_state)
        # Plain attribute, synced per-run by products (tests reassign
        # ``harness.stream_fn`` between runs).
        self.stream_fn = stream_fn
        self.prepare_next_turn = prepare_next_turn
        self.should_stop_after_turn = should_stop_after_turn
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.convert_to_llm = convert_to_llm
        self.transform_context = transform_context
        self.on_payload = on_payload
        self.on_response = on_response
        self.thinking_budgets = thinking_budgets
        self.transport = transport
        self.max_retry_delay_ms = max_retry_delay_ms
        self.tool_execution = tool_execution
        self.steering_queue = PendingMessageQueue(
            steering_mode, on_change=on_queue_change
        )
        self.follow_up_queue = PendingMessageQueue(
            follow_up_mode, on_change=on_queue_change
        )
        self._active_run: ActiveRun | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> AgentState:
        """The Agent's full observable state (Pi ``agent.state``)."""
        return self._sink.state

    # ── typed kernel event subscription (Chapter 7) ──────────────

    def subscribe(
        self,
        listener: AgentEventListener,
        *,
        wrap: bool = False,
    ) -> Callable[[], None]:
        """Register a PERSISTENT kernel event listener; returns an
        unsubscribe function.

        Persistent listeners sit in the sink's middle tier — after the
        run-scoped legacy adapter, before run-scoped listeners — on
        every run. ``wrap=True`` isolates untrusted listeners with
        try/except (the third-party extension tier); unwrapped listeners
        fail fast — a listener bug fails the run, by design.
        """
        return self._sink.subscribe(listener, wrap=wrap)

    @property
    def listeners(self) -> list[AgentEventListener]:
        """Current persistent listeners, wrapped on demand per their tier."""
        return self._sink.persistent_listeners

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._active_run is not None

    @property
    def signal(self) -> threading.Event | None:
        """Interrupt token for the active run, mirroring Pi ``signal``."""
        with self._lock:
            active = self._active_run
        return active.interrupt if active is not None else None

    @property
    def steering_mode(self) -> QueueMode:
        return self.steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self.steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self.follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self.follow_up_queue.mode = mode

    def reset(self) -> None:
        """Clear transcript, runtime projections, error, and both queues."""
        state = self.state
        state.messages = []
        state._is_streaming = False
        state._streaming_message = None
        state._pending_tool_calls = frozenset()
        state._error_message = None
        state._terminal_seen = False
        self.clear_all_queues()

    def begin(self, *, trace_id: str = "", session_id: str = "") -> ActiveRun:
        """Accept a new run; the Agent is busy from this instant.

        Called on the ACCEPTING thread (before any worker starts), so
        ``is_running`` covers the whole startup window.  Raises
        ``RuntimeError`` when another run is already active.

        Pi ``runWithLifecycle``: before anything runs, flip the state
        into streaming mode — ``is_streaming=True``, and clear
        ``streaming_message`` / ``error_message`` / the terminal guard.
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
        state = self._sink.state
        state._is_streaming = True
        state._streaming_message = None
        state._error_message = None
        state._terminal_seen = False
        return active

    def finish(self, active: ActiveRun, result: TraceResult | None = None) -> None:
        """End a run: record its result, then signal idle — never before.

        Write order is ``result`` → clear ``_active_run`` → ``done.set()``;
        a woken ``wait_for_idle`` reader can therefore trust ``result``.
        Also the lifecycle fallback: if ``run()`` never happened (startup
        failure between begin and run), this still drops the streaming
        projections ``begin()`` raised — idempotent with run's teardown.
        """
        with self._lock:
            if self._active_run is active:
                active.result = result
                self._active_run = None
                active.done.set()
                state = self._sink.state
                state._is_streaming = False
                state._streaming_message = None
                state._pending_tool_calls = frozenset()

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

    def _resolve_config(self, config: AgentLoopConfig | None) -> AgentLoopConfig:
        """Build a config when the caller passed none.

        Model comes from ``state.model`` (may be None — the single
        validation point is ``_full_config``); budgets fall back to the
        module defaults.  Everything else uses AgentLoopConfig's own
        defaults.
        """
        if config is not None:
            return config
        return AgentLoopConfig(
            model=self._sink.state.model,
            max_iterations=DEFAULT_MAX_ITERATIONS,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

    def _full_config(self, config: AgentLoopConfig) -> AgentLoopConfig:
        """Inject the Agent-owned policy fields into a run's config.

        Callers pass only the product/per-run fields (budgets, truncation
        hook, approval, trace/session ids, extra listeners, and — as
        per-run overrides — model/thinking); the queues, turn hooks and
        execution mode belong to the Agent — nobody reads them back out
        of it.  Listeners are NOT merged here: the Agent's persistent
        listeners live in the sink's middle tier, and ``config.listeners``
        ride the run tier (批 2).

        批 4 resolution order: explicit config > ``agent.state`` >
        built-in default.  A run with no model anywhere fails HERE with
        a clear ValueError, before the loop starts.
        """
        state = self._sink.state
        model = config.model if config.model is not None else state.model
        if model is None:
            raise ValueError(
                "no model configured: set agent.state.model "
                "or pass an explicit AgentLoopConfig"
            )
        thinking = (
            config.thinking
            if config.thinking is not None
            else (state.thinking_level or "off")
        )
        return replace(
            config,
            model=model,
            thinking=thinking,
            convert_to_llm=(
                self.convert_to_llm
                if config.convert_to_llm is default_convert_to_llm
                else config.convert_to_llm
            ),
            transform_context=(
                config.transform_context
                if config.transform_context is not None
                else self.transform_context
            ),
            on_payload=config.on_payload or self.on_payload,
            on_response=config.on_response or self.on_response,
            thinking_budgets=config.thinking_budgets or self.thinking_budgets,
            transport=(
                config.transport
                if config.transport != "auto"
                else self.transport
            ),
            max_retry_delay_ms=(
                config.max_retry_delay_ms
                if config.max_retry_delay_ms is not None
                else self.max_retry_delay_ms
            ),
            get_steering_messages=self.steering_queue.drain,
            get_follow_up_messages=self.follow_up_queue.drain,
            prepare_next_turn=self.prepare_next_turn,
            should_stop_after_turn=self.should_stop_after_turn,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            tool_execution=self.tool_execution,
        )

    def run(
        self,
        active: ActiveRun,
        config: AgentLoopConfig | None = None,
        *,
        emit: Emit | None = None,
    ) -> TraceResult:
        """Run the loop inside an ALREADY-begun run, on ``state``.

        The context is derived from ``self.state`` (system_prompt /
        messages / tools) — no ``AgentContext`` argument.  Lifecycle
        stays with the caller: this never calls ``begin()`` or
        ``finish()`` — teardown (persistence, terminal events) happens
        between the loop's return and the caller's ``finish()``.

        Pi ``finishRun``: after the loop returned (its ``agent_end``
        was already reduced and dispatched), clear the streaming
        projections.  Run-scoped dispatch is disarmed in the same
        finally — both are idempotent with ``finish()``'s fallback.
        """
        full = self._full_config(self._resolve_config(config))
        stream_fn = self.stream_fn
        if stream_fn is None:
            raise ValueError(
                "no stream_fn configured: pass one to Agent(...) "
                "or set agent.stream_fn before run"
            )
        emit_fn: Emit = emit if emit is not None else _noop_emit
        state = self._sink.state
        sink = self._sink
        sink.begin_run_dispatch(
            make_legacy_adapter(emit_fn), full.listeners or []
        )
        try:
            repair = repair_tool_history(state.messages)
            if repair.changed:
                sink.replace_messages(list(repair.messages))
            context = AgentContext(
                system_prompt=state.system_prompt,
                messages=state.messages,
                tools=state.tools,
            )
            return run_agent_loop(
                context=context,
                config=full,
                stream_fn=stream_fn,
                emit=emit_fn,
                interrupt=active.interrupt,
                sink=sink,
            )
        except Exception as exc:
            # Pi handleRunFailure: a crash inside the loop resolves the
            # run as FAILED through the normal event chain instead of
            # propagating.  Guard: if agent_end was already reduced
            # (state-first — the reducer ran before the listener that
            # raised), never emit a second one; re-raise as-is.
            if state._terminal_seen:
                raise
            return self._synthesize_run_failure(sink, exc)
        finally:
            sink.end_run_dispatch()
            # After agent_end listeners (Pi finishRun): drop the runtime
            # projections.  error_message survives until the next begin.
            state._is_streaming = False
            state._streaming_message = None
            state._pending_tool_calls = frozenset()

    def _synthesize_run_failure(
        self, sink: AgentEventSink, exc: Exception
    ) -> TraceResult:
        """Synthesize a failed assistant message and drive it through the
        NORMAL event chain (message_start/end → agent_end), so reducers
        and listeners observe the failure exactly like a loop-produced
        one (Pi ``handleRunFailure``), including the synthetic
        ``turn_end``.  The assistant message itself carries stop/error
        and model metadata so JSONL replay does not depend on AgentEnd.
        """
        error_text = f"{type(exc).__name__}: {exc}"
        model = sink.state.model
        failure = assistant_message(
            "",
            api=model.api if model is not None else "",
            provider=model.provider if model is not None else "",
            model=model.id if model is not None else "",
            usage={"input_tokens": 0, "output_tokens": 0},
            stop_reason="error",
            error_message=error_text,
            timestamp=int(time.time() * 1000),
        )
        sink.process_event(
            MessageStartEvent(message=failure, source="assistant")
        )
        sink.process_event(
            MessageEndEvent(message=failure, source="assistant")
        )
        sink.process_event(TurnEndEvent(
            turn_index=sink.current_turn,
            model=model.id if model is not None else "",
            stop_reason="error",
            status="error",
            usage={"input_tokens": 0, "output_tokens": 0},
            tool_count=0,
            tool_error_count=0,
            message=failure,
        ))
        sink.process_event(
            AgentEndEvent(status="failed", stop_reason="error", error=error_text)
        )
        return TraceResult(
            reply=error_text,
            status="failed",
            stop_reason="error",
            error=error_text,
        )

    def prompt(
        self,
        input: str | AgentMessage | list[AgentMessage],
        config: AgentLoopConfig | None = None,
        *,
        emit: Emit | None = None,
    ) -> TraceResult:
        """Convenience entry: begin → ingest → run → finish.

        ``input`` is a string, one message, or a list of messages; each
        is ingested through kernel events (``message_start/end`` with
        ``source="user"``) so the sink's reducer — not this method —
        appends them to ``state.messages``, and persistent listeners
        observe the ingestion in state-first order.  The legacy string
        channel stays silent for source="user" (the adapter only
        forwards steering/follow_up) — zero string-contract change.

        A standalone Agent + fake model runs with no CLI/TUI/SQLite.
        Products that need post-loop work before idle (persistence,
        terminal events) use ``begin()``/``run()``/``finish()`` instead.
        """
        if isinstance(input, str):
            new_messages = [user_message(input)]
        elif isinstance(input, (list, tuple)):
            new_messages = list(input)
        else:
            new_messages = [input]
        config = self._resolve_config(config)
        active = self.begin()
        result: TraceResult | None = None
        try:
            try:
                # Repair the old tail before appending a fresh user prompt;
                # afterwards the interrupted assistant would no longer be the
                # transcript tip and could not be recognized safely.
                repair = repair_tool_history(self.state.messages)
                if repair.changed:
                    self._sink.replace_messages(list(repair.messages))
                for message in new_messages:
                    self._sink.process_event(
                        MessageStartEvent(message=message, source="user")
                    )
                    self._sink.process_event(
                        MessageEndEvent(message=message, source="user")
                    )
            except Exception as exc:
                # Prompt ingestion belongs to the same lifecycle as the loop
                # in Pi. Normalize listener failures through the terminal
                # event chain. Configuration errors from run() remain caller
                # errors and are deliberately not swallowed here.
                if self.state._terminal_seen:
                    raise
                result = self._synthesize_run_failure(self._sink, exc)
                return result
            result = self.run(active, config, emit=emit)
            return result
        finally:
            self.finish(active, result)

    def continue_(
        self,
        config: AgentLoopConfig | None = None,
        *,
        emit: Emit | None = None,
    ) -> TraceResult:
        """Re-run the loop on the EXISTING state (Pi ``continue()``).

        Unlike ``prompt()`` this implies no fresh user turn: nothing is
        appended — ``state.messages``' own tail drives the loop.  Pi
        legality: the last message must be one the model can answer
        (user or tool result).  An assistant-tipped state is legal ONLY
        with queued messages: ONE drained batch becomes the run's input
        — steering first, else follow-ups (Pi's drain order, honoring
        each queue's one-at-a-time mode) — delivered via the loop's
        initial-pending channel with its source intact.  The batches
        stay in their own queues until drained; nothing is re-labelled
        or moved wholesale.
        """
        messages = self._sink.state.messages
        if not messages:
            raise ValueError("nothing to continue: context is empty")
        config = self._resolve_config(config)
        if getattr(messages[-1], "role", None) == "assistant":
            repair = repair_tool_history(messages)
            steering = [] if repair.changed else self.steering_queue.drain()
            if repair.changed:
                self._sink.replace_messages(list(repair.messages))
            elif steering:
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
        active = self.begin()
        result: TraceResult | None = None
        try:
            result = self.run(active, config, emit=emit)
            return result
        finally:
            self.finish(active, result)


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
