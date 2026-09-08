"""Coding-agent composition root for one local LSM Harness instance."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from lsm_harness.agent import Agent, AgentContext, AgentLoopConfig
from lsm_harness.agent.runtime import ActiveRun
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.events import HarnessEvent, Observer, make_event
from lsm_harness.agent.governance import ContextGovernor, GovernanceConfig
from lsm_harness.agent.hooks import (
    LoopHooks,
    PrepareNextTurn,
    ShouldStopAfterTurn,
    invoke_trace_end,
    invoke_trace_start,
)
from lsm_harness.agent.pending import PendingMessage
from lsm_harness.ai.providers import get_client, get_model, PROVIDERS
from lsm_harness.ai.registry import registered_api_providers
from lsm_harness.ai.stream import stream_simple
from lsm_harness.ops.file_state import FileState
from lsm_harness.ops.tracing import Tracer
from lsm_harness.coding_agent.messages import register_coding_agent_messages
from lsm_harness.coding_agent.session import Session
from lsm_harness.coding_agent.skills import SkillLoader
from lsm_harness.coding_agent.subagent import SubagentManager
from lsm_harness.security import redact_data
from lsm_harness.tools import build_registry
from lsm_harness.tools.multimodal import has_image
from lsm_harness.ai.types import Model, ModelClient, StreamFunction
from lsm_harness.agent.types import (
    AfterToolCall,
    BeforeToolCall,
    ToolExecutionMode,
    TraceResult,
)


def _safe_message_text(msg: str | dict | None) -> str:
    """Extract safe text from a message for tracing (no image data)."""
    if msg is None:
        return "[continue]"  # respond_continue(): no fresh user message
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    parts.append("[image]")
            return "\n".join(parts)
        return str(content)
    return str(msg)


class RunBusyError(RuntimeError):
    """A state-changing command was refused while a run is active.

    Shared product control entry: every frontend (CLI/TUI/RPC) gets the
    same busy rule from the Harness instead of each inventing its own.
    """


THINKING_LEVELS = ("disabled", "auto", "enabled")


class Harness:
    def __init__(
        self,
        settings: Settings | None = None,
        client: ModelClient | None = None,
        conn=None,
        hooks: LoopHooks | None = None,
        prepare_next_turn: PrepareNextTurn | None = None,
        should_stop_after_turn: ShouldStopAfterTurn | None = None,
        before_tool_call: BeforeToolCall | None = None,
        after_tool_call: AfterToolCall | None = None,
        tool_execution: ToolExecutionMode = "parallel",
        stream_fn: StreamFunction | None = None,
    ):
        self.settings = settings or Settings()
        self.settings.ensure_home()
        # Chapter 6: fill the core package's empty custom-message slot.
        register_coding_agent_messages()
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
        resolved_model = getattr(self.client, "model", None)
        self.model = (
            resolved_model
            if isinstance(resolved_model, Model)
            else Model(
                id=self.settings.model or "injected-model",
                api="legacy-client",
                provider=self.settings.provider or "injected",
            )
        )
        # Batch E (plan §8): the main chain resolves through the provider
        # registry (stream_simple); tests may inject a StreamFunction
        # directly via ``stream_fn``.
        self.stream_fn = stream_fn or self._resolve_stream_fn()
        self.workspace_root = Path(os.getcwd()).expanduser().resolve()
        self.hooks = hooks

        # ── context governor ────────────────────────────────
        self.governor = ContextGovernor(
            config=GovernanceConfig(
                max_result_chars=self.settings.governance_max_result_chars,
                offload_threshold_chars=self.settings.governance_offload_threshold,
            ),
            home=self.settings.home,
        )

        # ── file state tracker ──────────────────────────────
        self.file_state = FileState(home=self.workspace_root)

        # ── subagent manager ───────────────────────────────
        self.subagents = SubagentManager(
            max_concurrent=self.settings.subagent_max_concurrent,
        )
        self.subagents._harness = self

        self.tool_prompt_snippets: list[str] = []
        # §9.3: product renderers survive the AgentTool wrap via this
        # name → (render_call, render_result) map; the CLI/TUI listener
        # consults it with priority custom renderer → label → tool name.
        self.tool_renderers: dict = {}
        self.tools = build_registry(
            self.conn, self.settings,
            subagent_manager=self.subagents,
            file_state=self.file_state,
            workspace_root=self.workspace_root,
            prompt_snippets=self.tool_prompt_snippets,
            renderers=self.tool_renderers,
        )
        self.session = Session(
            self.settings,
            conn=self.conn,
            client=self.client,
            skills=SkillLoader([self.settings.home / "skills"]),
        )
        self.tracer = Tracer(self.settings.home)

        # ── stateful Agent shell around the stateless loop ──
        self.agent = Agent(
            prepare_next_turn=prepare_next_turn,
            should_stop_after_turn=should_stop_after_turn,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
            tool_execution=tool_execution,
        )
        # Startup is a session open (Pi createAgentSession): restore the
        # selected session's recorded model/thinking immediately, so the
        # first status line and the first run agree.
        self._restore_runtime_state_safely()

    # ── public API ───────────────────────────────────────────

    def _resolve_stream_fn(self) -> StreamFunction:
        """Resolve the canonical model-call chain for the current model.

        ``Model → stream_simple → resolve_api_provider(Model.api)`` is the
        one entry point; the registry path injects the API key from
        settings into every request's StreamOptions.  A model whose
        ``api`` dialect is NOT registered is an error — tests and smoke
        inject a ``StreamFunction`` directly instead.
        """
        return self._build_stream_fn(self.model, self.settings.api_key)

    @staticmethod
    def _build_stream_fn(model: Model, api_key: str) -> StreamFunction:
        """Bind (model, api_key) into a registry-backed stream function.

        The key arrives as an explicit argument rather than being read off
        ``self.settings`` inside the closure, so the closure can never
        capture a stale credential from a previous provider.
        """
        if model.api not in registered_api_providers():
            raise ValueError(
                f"model api {model.api!r} is not registered; "
                "pass stream_fn=... to Harness for scripted clients"
            )

        def registry_stream(model, context, options):
            return stream_simple(
                model, context, replace(options, api_key=api_key)
            )

        return registry_stream

    @property
    def is_running(self) -> bool:
        return self.agent.is_running

    def begin_run(self) -> ActiveRun:
        """Accept a run on the CALLING thread: ``is_running`` is true from
        this instant, so there is no accepted-but-not-running window.

        Frontends that run ``respond()`` on a worker thread call this first
        and pass the returned run as ``active_run``; a second prompt while
        busy gets ``RunBusyError`` instead of racing the worker's start.
        """
        try:
            return self.agent.begin(session_id=self.session.session_id)
        except RuntimeError as exc:
            raise RunBusyError(
                "cannot start a run while a run is active"
            ) from exc

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """True once no run is active — including its teardown.

        ``finish()`` runs after persistence and the terminal events, so a
        True here means the run is fully done, never reported early.
        """
        return self.agent.wait_for_idle(timeout)

    def abort(self) -> bool:
        """Cancel the currently running trace.

        Thread-safe — can be called from a signal handler or
        background thread.  Returns True when a run was actually
        interrupted (False when idle).
        """
        return self.agent.abort()

    def steer(self, message: PendingMessage) -> bool:
        """Inject a message into the running loop.

        The message will be seen by the model at the next turn
        boundary.  Call from any thread.
        """
        return self.agent.steer(message)

    def follow_up(self, message: PendingMessage) -> bool:
        """Queue work that starts only after the inner loop would stop."""
        return self.agent.follow_up(message)

    def pending_messages(self) -> dict[str, list[PendingMessage]]:
        """Return read-only snapshots of the two control queues."""
        return self.agent.pending_messages()

    def switch_model(
        self,
        provider_name: str,
        *,
        model: str = "",
        small_model: str = "",
    ) -> None:
        """Switch the main loop and summarization clients as one operation."""
        self._ensure_idle("set_model")
        self._apply_model_state(provider_name, model, small_model)
        # State-change entry: branching back past this point restores the
        # model that was in effect then (plan §5.9).  small_model rides
        # along — compaction and branch summaries consume
        # it (code-review issue 三).
        self.session.record_model_change(
            provider_name, self.settings.model, small_model=self.settings.small_model
        )

    # ── shared product control entries (busy-guarded) ────────────

    def _ensure_idle(self, action: str) -> None:
        """State changes are refused while a run is active.

        One rule for every frontend: a run's events and persistence stay
        with the session/model/thinking state it started with.
        """
        if self.agent.is_running:
            raise RunBusyError(f"cannot {action} while a run is active")

    def new_session(self) -> str:
        """Start a fresh session (idle only); returns the new id."""
        self._ensure_idle("new_session")
        return self.session.start_new()

    def switch_session(self, session_ref: str) -> str | None:
        """Resume another session (idle only); None when no unique match.

        Pi parity (createAgentSession): opening a session EAGERLY restores
        the model/thinking recorded on its current path, so the status bar
        and the next run agree from this instant — not only after the
        first respond.
        """
        self._ensure_idle("switch_session")
        switched = self.session.resume(session_ref)
        if switched is not None:
            self._restore_runtime_state_safely()
        return switched

    def set_thinking(self, level: str) -> None:
        """Set the thinking level (idle only) and record it on the tree.

        Collapses the ``settings.thinking = ...; record_thinking_change``
        pair the CLI/TUI/RPC frontends each used to duplicate.
        """
        self._ensure_idle("set_thinking_level")
        if level not in THINKING_LEVELS:
            raise ValueError(
                f"thinking level must be one of {list(THINKING_LEVELS)}"
            )
        self.settings.thinking = level
        self.session.record_thinking_change(level)

    def cycle_thinking(self) -> str:
        """Cycle disabled → auto → enabled (idle only); returns the level."""
        self._ensure_idle("cycle_thinking_level")
        current = self.settings.thinking
        index = (
            THINKING_LEVELS.index(current)
            if current in THINKING_LEVELS
            else 0
        )
        level = THINKING_LEVELS[(index + 1) % len(THINKING_LEVELS)]
        self.settings.thinking = level
        self.session.record_thinking_change(level)
        return level

    def compact(self) -> bool:
        """Manually compact the current session path (idle only)."""
        self._ensure_idle("compact")
        return self.session.compact(lambda *_: None)

    def _apply_model_state(
        self, provider_name: str, model: str, small_model: str
    ) -> None:
        """Rebuild client/stream_fn/settings for a provider+model combo.

        Shared by switch_model (a real change — records a model_change
        entry) and _restore_runtime_state (a restoration — records
        nothing, 恢复 ≠ 变更).

        Candidate-first: the client, Model and stream_fn are fully built
        and validated against the NEW provider config before a single
        attribute is swapped, so a failure leaves the previous
        provider/model/key untouched — no half-updated mix of old
        credentials with a new model.
        """
        provider = PROVIDERS[provider_name]
        new_model = model or provider.model
        new_small = small_model or provider.small_model
        # Credential rule: an explicit global override
        # (LSM_API_KEY/WAKU_API_KEY) always wins; otherwise the TARGET
        # provider's own env var.  The old provider's key is never
        # inherited.  base_url likewise only ever comes from the user's
        # explicit settings — a provider switch neither invents nor
        # removes a custom endpoint.
        explicit_key = os.getenv("LSM_API_KEY") or os.getenv("WAKU_API_KEY") or ""
        resolved_key = explicit_key or os.getenv(provider.key_env, "")
        # Late import path: tests monkeypatch lsm_harness.ai.providers.get_client
        from lsm_harness.ai import providers as _providers

        candidate_client = _providers.get_client(
            provider_name=provider_name,
            api_key=resolved_key,
            base_url=self.settings.base_url or None,
            model=new_model,
            small_model=new_small,
            thinking=self.settings.thinking,
        )
        resolved_model = getattr(candidate_client, "model", None)
        candidate_model = (
            resolved_model
            if isinstance(resolved_model, Model)
            else get_model(
                provider_name,
                new_model,
                base_url=self.settings.base_url or None,
            )
        )
        candidate_stream_fn = self._build_stream_fn(candidate_model, resolved_key)

        # Every candidate piece is built and validated — swap atomically.
        self.client = candidate_client
        self.model = candidate_model
        self.stream_fn = candidate_stream_fn
        self.session.client = candidate_client
        self.settings.provider = provider_name
        self.settings.api_key = resolved_key
        self.settings.model = new_model
        self.settings.small_model = new_small

    def _restore_emit(self, kind: str, data: dict) -> None:
        """Emit a restore event outside any run (switch/startup).

        Best-effort: a tracing hiccup must never block a session switch,
        so unlike the in-run emit this swallows tracer failures.
        """
        try:
            self.tracer.write(make_event(
                kind, "session-restore", data,
                session_id=self.session.session_id,
            ))
        except Exception:
            pass

    def _restore_runtime_state_safely(self) -> None:
        """Eager restore at session open/switch, Pi-modelFallbackMessage
        style: a restore failure keeps the current model and is recorded,
        never blocks the switch."""
        try:
            self._restore_runtime_state(self._restore_emit)
        except Exception as exc:
            self._restore_emit("session.runtime_state_restore_failed", {
                "error": f"{type(exc).__name__}: {exc}",
            })

    def _restore_runtime_state(self, emit) -> None:
        """Apply the session tree's runtime state at open/switch time.

        The current path's header baseline plus model_change /
        thinking_level_change entries describe the model in effect at
        that point of the tree.  Opening or switching to a session makes
        THAT state live: provider/model/small_model rebuild the client
        and stream_fn exactly like switch_model — but WITHOUT writing
        model_change entries (恢复 ≠ 变更) — and thinking syncs into
        settings.

        Pi parity (阶段 4 批 2): this runs eagerly when a session is
        opened/switched (Pi createAgentSession), NOT lazily before each
        run.  In-session branching does NOT re-derive model state
        (Pi navigateTree only replaces messages) — a deliberate model
        switch survives a trip back to an older node; after a restart,
        open-time restore reads whichever path the file's last line is on.

        No-op when the tree carries no state yet (deferred first write)
        or the recorded state already matches the live settings.
        """
        try:
            tree = self.session.build_session_context()
        except Exception:
            return  # a state-read hiccup must never block the run
        if tree is None:
            return
        if tree.thinking_level:
            self.settings.thinking = tree.thinking_level
        target_provider = tree.provider or ""
        if not target_provider or target_provider not in PROVIDERS:
            return
        if (
            target_provider == self.settings.provider
            and (not tree.model or tree.model == self.settings.model)
            and (not tree.small_model or tree.small_model == self.settings.small_model)
        ):
            return
        self._apply_model_state(
            target_provider,
            tree.model or "",
            tree.small_model or "",
        )
        emit("session.runtime_state_restored", {
            "provider": self.settings.provider,
            "model": self.settings.model,
            "small_model": self.settings.small_model,
            "thinking": self.settings.thinking,
        })

    def respond(
        self,
        user_message: str | dict,
        *,
        observer: Observer | None = None,
        source: str = "cli",
        trace_id: str | None = None,
        turn_id: str | None = None,
        approval_broker=None,
        active_run: ActiveRun | None = None,
    ) -> TraceResult:
        return self._respond_impl(
            user_message,
            observer=observer,
            source=source,
            trace_id=trace_id,
            turn_id=turn_id,
            approval_broker=approval_broker,
            active_run=active_run,
        )

    def respond_continue(
        self,
        *,
        observer: Observer | None = None,
        source: str = "cli",
        trace_id: str | None = None,
        approval_broker=None,
    ) -> TraceResult:
        """Re-run the loop WITHOUT appending a user message (Pi continue()).

        Legal inputs: queued follow-ups, or a tree whose current path
        ENDS on a legal continuation point (Pi: an unanswered user
        message, or a tool result whose next model call never happened).
        The judgment comes from the persistent session tree, not
        in-memory state, so it works identically after a restart.  A
        completed run with empty queues has nothing to continue (the
        context would end on an assistant message, which providers
        reject).
        """
        if (
            not self.agent.has_queued_messages()
            and not self.session.tree_tip_allows_continue()
        ):
            raise ValueError(
                "nothing to continue: no queued follow-up messages "
                "and the last run was not aborted"
            )
        return self._respond_impl(
            None,
            observer=observer,
            source=source,
            trace_id=trace_id,
            approval_broker=approval_broker,
        )

    def _respond_impl(
        self,
        user_message: str | dict | None,
        *,
        observer: Observer | None = None,
        source: str = "cli",
        trace_id: str | None = None,
        turn_id: str | None = None,
        approval_broker=None,
        active_run: ActiveRun | None = None,
    ) -> TraceResult:
        if trace_id and turn_id and trace_id != turn_id:
            raise ValueError("trace_id and legacy turn_id must match")
        trace_id = trace_id or turn_id or str(uuid4())
        started_at = time.monotonic()
        sequence = 0
        trace_hook_ended = False
        # Last MAIN-model context size this run (pi ch9 agent-end
        # auto-compaction measures the real usage, not an estimate).
        last_context_tokens = 0

        # Run acceptance: begun by the caller on the accepting thread via
        # begin_run() (``is_running`` true from that instant), or right
        # here for direct callers.  EVERYTHING after begin() — including
        # the startup events below — lives inside the outermost
        # try/finally, so finish() is unconditional and an accepted run
        # can never wedge busy, whatever stage fails.
        active = active_run if active_run is not None else self.agent.begin()
        # Bind everything the emit closure and the except/finally clauses
        # need BEFORE the try: a startup failure must still be able to
        # report and release the run.  ``run_session_id`` also pins this
        # run's session ownership: every event, usage record and the final
        # persistence belong to the session the run STARTED in, never to
        # whatever the mutable current session might become.
        recorder = self.session.recorder
        run_session_id = self.session.session_id
        result: TraceResult | None = None

        def emit(event_type: str, data: dict) -> None:
            nonlocal sequence, last_context_tokens
            sequence += 1
            safe_data = redact_data(data, (self.settings.api_key,))
            event = make_event(
                event_type,
                trace_id,
                safe_data,
                session_id=run_session_id,
                sequence=sequence,
                duration_ms=int((time.monotonic() - started_at) * 1000),
            )
            self.tracer.write(event)
            # ── usage tracking ────────────────────────────
            if event_type == "llm.completed" and "usage" in safe_data:
                u = safe_data["usage"]
                self.tracer.log_usage(
                    session_id=run_session_id,
                    model=safe_data.get("model", self.settings.model),
                    input_tokens=u.get("input_tokens", 0),
                    output_tokens=u.get("output_tokens", 0),
                    turn_id=trace_id,
                )
                if safe_data.get("role") == "main":
                    # input + output ≈ the context the NEXT turn starts with
                    last_context_tokens = u.get("input_tokens", 0) + u.get(
                        "output_tokens", 0
                    )
            if observer:
                # 订阅者失败契约:UI 观察者(前端渲染)的异常必须隔离——
                # 它不能杀死运行、不能破坏持久化;记录层(tracer.write、
                # recorder listener)的异常不在这里兜底,必须使运行失败。
                try:
                    observer(event)
                except Exception as exc:
                    try:
                        # 直连 tracer(绕过本 emit),避免失败处理自身递归。
                        self.tracer.write(make_event(
                            "observer.failed",
                            trace_id,
                            {"error": f"{type(exc).__name__}: {exc}"},
                            session_id=run_session_id,
                        ))
                    except Exception:
                        pass

        def end_trace_hooks(result: TraceResult) -> None:
            nonlocal trace_hook_ended
            if trace_hook_ended:
                return
            trace_hook_ended = True
            invoke_trace_end(self.hooks, result, emit)

        try:
            active.trace_id = trace_id
            active.session_id = active.session_id or run_session_id
            # 启动动作也在 try 内:tracer/hook 在这里失败同样到达 finally。
            emit("trace.started", {"source": source, "message": _safe_message_text(user_message)})
            invoke_trace_start(self.hooks, _safe_message_text(user_message), emit)
            emit("trace.accepted", {"source": source, "message": _safe_message_text(user_message)})

            # Model/thinking restore happens at session open/switch
            # (Pi createAgentSession parity), not here — see
            # _restore_runtime_state's docstring.
            emit("context.build.started", {"session_id": self.session.session_id})
            system, messages = self.session.prepare_context(
                user_message, emit, self.tools.schemas()
            )
            system = self._with_tool_prompt_snippets(system)
            emit("context.build.completed", {
                "session_id": self.session.session_id,
                "message_count": len(messages),
            })

            # ── overflow recovery callback ──────────────────
            # When the loop hits a token limit, this compacts context
            # and returns a fresh (system, messages) so the retry
            # doesn't hit the same limit.
            def compact_and_retry():
                compacted_system, compacted_messages = self.session.compact_and_rebuild(
                    user_message, emit, self.tools.schemas()
                )
                return self._with_tool_prompt_snippets(compacted_system), compacted_messages

            context = AgentContext(
                system_prompt=system,
                messages=messages,
                tools=self.tools,
            )
            # ── session recorder wiring (batch B) ────────────
            # The initial user message is recorded explicitly (it is
            # context, not a kernel event); assistant / tool / steering /
            # follow_up messages arrive via the listener below, one event
            # = one entry.
            if recorder is not None:
                recorder.set_emit(emit)
                # The initial user message is recorded explicitly (it is
                # context, not a kernel event); a continue() run adds none.
                if user_message is not None:
                    try:
                        recorder.record(
                            self.session.persistable_user_message(user_message),
                            source=source,
                        )
                    except Exception as exc:
                        emit("session.jsonl_write_failed", {
                            "session_id": self.session.session_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
            listeners = [recorder.listener()] if recorder is not None else []
            # Pi continue(): a continue run whose tree tip is ASSISTANT
            # can only be legal because of queued messages — drain ONE
            # batch (steering first, else follow-ups, each queue's own
            # drain so one-at-a-time is honored) and hand it to the loop's
            # initial-pending channel.  Without this the first model call
            # would carry an assistant-tipped context (providers reject).
            initial_pending: list | None = None
            initial_pending_source = "follow_up"
            if (
                user_message is None
                and messages
                and getattr(messages[-1], "role", None) == "assistant"
            ):
                steering = self.agent.steering_queue.drain()
                if steering:
                    initial_pending = steering
                    initial_pending_source = "steering"
                else:
                    initial_pending = self.agent.follow_up_queue.drain() or None
            loop_config = AgentLoopConfig(
                model=self.model,
                max_iterations=self.settings.max_iterations,
                max_tokens=self.settings.max_tokens,
                # Agent-owned policy fields (queues, turn hooks, execution
                # mode, subscribed listeners) are injected by Agent.run —
                # respond() no longer reads them back out of the Agent.
                on_truncation=compact_and_retry,
                governor=self.governor,
                thinking=self.settings.thinking,
                cache_retention=self.settings.cache_retention,
                hooks=self.hooks,
                max_model_retries=self.settings.max_model_retries,
                max_empty_retries=self.settings.max_empty_retries,
                max_length_recoveries=self.settings.max_length_recoveries,
                approval_broker=approval_broker,
                trace_id=trace_id,
                session_id=run_session_id,
                initial_pending_messages=initial_pending,
                initial_pending_source=initial_pending_source,
                # Run-scoped listeners come LAST, after the Agent's own
                # subscriptions — Agent.run preserves that order.
                listeners=listeners,
            )
            result = self.agent.run(
                active,
                context,
                loop_config,
                stream_fn=self.stream_fn,
                emit=emit,
            )
            # A failed/aborted FIRST exchange still ASKED the question —
            # flush the recorder's buffer so the tree (the fact source)
            # keeps it and a later continue() can answer it, even after a
            # restart.  (Dropping it made continue() run on empty context.)
            if result.status != "completed" and recorder is not None and recorder.deferred:
                recorder.flush()
            if result.status == "completed":
                emit("persistence.started", {"stores": ["sqlite", "jsonl"]})
                persistence = self.session.add_exchange(
                    user_message, result, source,
                    session_id=run_session_id, recorder=recorder,
                )
                # Record file changes at trace end
                file_summary = self.file_state.summary()
                if self.file_state.modified_files:
                    emit("trace.file_changes", {
                        "files": self.file_state.modified_files,
                        "summary": file_summary,
                    })
                self.file_state.end_turn()
                emit("persistence.completed", persistence)
                # ── agent-end auto-compaction (pi ch9) ──────
                # The agent is idle now: if the LAST measured context
                # size crossed the red line, compact the current tree
                # path immediately instead of waiting for the next
                # turn's prepare_context.  Only when the run's session is
                # STILL the live one — after a mid-run switch (only
                # possible by bypassing the RPC busy guard) compacting
                # would operate on the wrong session.
                if self.session.session_id == run_session_id:
                    self.session.compact_if_due(last_context_tokens, emit)
            elif self.session.session_id == run_session_id:
                # 中断/失败的运行:add_exchange 没跑,但用户消息已随内核
                # 事件写进树(事实来源)。展示历史从树重新同步,否则
                # /tree 会丢掉被中断的问题——而 respond_continue() 恰恰
                # 依赖树梢那个问题(阶段 4 批 3)。
                self.session.resync_history()
            # ── usage tracking ────────────────────────────
            if result.iterations > 0:
                # Log approximate usage (one main-model call per iteration)
                # The llm.completed events carry the real numbers;
                # this is a conservative estimate when events aren't available.
                pass  # usage is logged per model turn via llm.completed events
            final_data = {
                "reply": result.reply,
                "iterations": result.iterations,
                "turns": result.turn_count,
                "tools": [item["tool"] for item in result.tool_calls],
                "aborted": result.aborted,
                "status": result.status,
                "stop_reason": result.stop_reason,
                "error": result.error,
            }
            if result.status == "aborted":
                emit("trace.aborted", final_data)
                end_trace_hooks(result)
                emit("trace.completed", final_data)
            elif result.status == "failed":
                emit("trace.error", {"error": result.error or result.reply})
                end_trace_hooks(result)
                emit("trace.failed", final_data)
            else:
                emit("trace.done", final_data)
                end_trace_hooks(result)
                emit("trace.completed", final_data)
            return result
        except Exception as exc:
            if recorder is not None and recorder.deferred:
                recorder.flush()
            error_message = f"{type(exc).__name__}: {exc}"
            failed_result = TraceResult(
                reply=error_message,
                status="failed",
                stop_reason="error",
                error=error_message,
            )
            result = failed_result
            emit("trace.error", {"error": error_message})
            end_trace_hooks(failed_result)
            emit("trace.failed", {"error": error_message, "status": "failed"})
            raise
        finally:
            if approval_broker is not None:
                approval_broker.reject_all("trace_finished")
            # Idle is reported ONLY here — after persistence, terminal
            # events and teardown — so wait_for_idle never fires early.
            self.agent.finish(active, result)

    def _with_tool_prompt_snippets(self, system: str) -> str:
        """Add product-only tool guidance without leaking it into Agent/AI."""
        if not self.tool_prompt_snippets:
            return system
        return system + "\n\n工具使用指南：\n" + "\n".join(
            f"- {snippet}" for snippet in self.tool_prompt_snippets
        )

    def close(self) -> None:
        self.subagents.shutdown(wait=False)
        self.conn.close()
