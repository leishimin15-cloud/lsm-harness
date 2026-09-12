"""Coding-session composition root for one local LSM instance."""

from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4

from lsm_harness.agent import Agent, AgentLoopConfig, AgentState
from lsm_harness.agent.messages import message_preview
from lsm_harness.agent.runtime import ActiveRun
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.events import Observer, make_event
from lsm_harness.agent.governance import ContextGovernor, GovernanceConfig
from lsm_harness.agent.hooks import (
    LoopHooks,
    PrepareNextTurn,
    ShouldStopAfterTurn,
    invoke_trace_end,
    invoke_trace_start,
)
from lsm_harness.agent.pending import PendingMessage
from lsm_harness.agent.tool_history import repair_tool_history
from lsm_harness.ai.providers import canonical_provider_name
from lsm_harness.ops.file_state import FileState
from lsm_harness.ops.tracing import Tracer
from lsm_harness.coding_agent.messages import register_coding_agent_messages
from lsm_harness.coding_agent.model_runtime import ModelRuntime
from lsm_harness.coding_agent.events import (
    AgentSettledEvent,
    AutoRetryEndEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    CodingSessionEventListener,
    CodingSessionEventSink,
    ErrorEvent,
    EntryAppendedEvent,
    ModelChangedEvent,
    QueueUpdateEvent,
    RetryEvent,
    SessionChangedEvent,
    ThinkingLevelChangedEvent,
    ToolHistoryRepairedEvent,
)
from lsm_harness.coding_agent.session import Session
from lsm_harness.coding_agent.skills import SkillLoader
from lsm_harness.coding_agent.subagent import SubagentManager
from lsm_harness.security import redact_data
from lsm_harness.tools import build_registry
from lsm_harness.ai.types import ModelClient, StreamFunction
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


# 七档 thinking 唯一定义在 AI 层;这里 re-export 保持既有
# ``from ...coding_agent.app import THINKING_LEVELS`` 的调用方(RPC
# 校验等)不用改。旧三档(disabled/auto/enabled)只在读取旧配置/
# 旧 JSONL/旧 API 入参时经 normalize_thinking_level 转换。
from lsm_harness.ai.models import (
    LEGACY_THINKING_LEVELS,
    THINKING_LEVELS,
    available_thinking_levels,
    clamp_thinking_level,
    normalize_thinking_level,
)


class CodingSession:
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
        workspace_root: str | Path | None = None,
    ):
        self.settings = settings or Settings()
        self._events = CodingSessionEventSink()
        self.settings.ensure_home()
        # Chapter 6: fill the core package's empty custom-message slot.
        register_coding_agent_messages()
        self.conn = conn or connect(self.settings.home)
        self.model_runtime = ModelRuntime(
            self.settings, client=client, stream_fn=stream_fn
        )
        self.client = self.model_runtime.client
        self.model = self.model_runtime.model
        self.stream_fn = self.model_runtime.stream_fn
        self.workspace_root = Path(
            workspace_root if workspace_root is not None else os.getcwd()
        ).expanduser().resolve()
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
            workspace_root=self.workspace_root,
        )
        self.session.set_entry_listener(
            lambda entry: self._events.publish(EntryAppendedEvent(entry))
        )
        self.session.set_compaction_listener(self._on_compaction_event)
        # 压缩红线跟随当前模型窗口(lambda 动态读 self.model,
        # switch_model 换模型后自动生效,无需重接线)。
        self.session.context_window_getter = lambda: self.model.context_window
        self.tracer = Tracer(self.settings.home)

        # ── stateful Agent shell around the stateless loop ──
        # 批 2:Agent 持有持久 sink + AgentState;身份字段(model /
        # thinking / tools)在此落进 state,transcript(messages)每 run
        # 由 _respond_impl wholesale 赋值(Pi agent-session 模式)。
        # 七档 thinking:旧配置值(disabled/auto/enabled)先归一,再按
        # 当前模型能力 clamp(如 K3 的 off:null → minimal)。
        self.settings.thinking = clamp_thinking_level(
            self.model, normalize_thinking_level(self.settings.thinking)
        )
        self.agent = Agent(
            initial_state=AgentState(
                model=self.model,
                thinking_level=self.settings.thinking,
                tools=self.tools,
            ),
            stream_fn=self.stream_fn,
            prepare_next_turn=prepare_next_turn,
            should_stop_after_turn=should_stop_after_turn,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
            tool_execution=tool_execution,
            on_queue_change=lambda: self._events.publish(self._queue_event()),
        )
        self.agent.subscribe(self._events.publish)
        # Startup is a session open (Pi createAgentSession): restore the
        # selected session's recorded model/thinking immediately, so the
        # first status line and the first run agree.
        self._restore_runtime_state_safely()

    # ── public API ───────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self.agent.is_running

    def subscribe(
        self,
        listener: CodingSessionEventListener,
        *,
        wrap: bool = False,
    ):
        """Subscribe to Agent and coding-session events as one typed stream."""
        return self._events.subscribe(listener, wrap=wrap)

    @staticmethod
    def _pending_text(message: PendingMessage) -> str:
        return message if isinstance(message, str) else message_preview(message)

    def _queue_event(self) -> QueueUpdateEvent:
        pending = self.agent.pending_messages()
        return QueueUpdateEvent(
            steering=tuple(
                self._pending_text(message) for message in pending["steering"]
            ),
            follow_up=tuple(
                self._pending_text(message) for message in pending["follow_up"]
            ),
        )

    def _on_compaction_event(
        self,
        phase: str,
        reason: str,
        result,
        error: str | None,
    ) -> None:
        if phase == "start":
            self._events.publish(CompactionStartEvent(reason=reason))
            return
        self._events.publish(CompactionEndEvent(
            reason=reason,
            result=result,
            error_message=error,
        ))

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
        accepted = self.agent.steer(message)
        return accepted

    def follow_up(self, message: PendingMessage) -> bool:
        """Queue work that starts only after the inner loop would stop."""
        accepted = self.agent.follow_up(message)
        return accepted

    def pending_messages(self) -> dict[str, list[PendingMessage]]:
        """Return read-only snapshots of the two control queues."""
        return self.agent.pending_messages()

    def clear_steering_queue(self) -> list[PendingMessage]:
        removed = self.agent.clear_steering_queue()
        return removed

    def clear_follow_up_queue(self) -> list[PendingMessage]:
        removed = self.agent.clear_follow_up_queue()
        return removed

    def clear_all_queues(self) -> dict[str, list[PendingMessage]]:
        removed = self.agent.clear_all_queues()
        return removed

    def switch_model(
        self,
        provider_name: str,
        *,
        model: str = "",
        small_model: str = "",
    ) -> None:
        """Switch the main loop and summarization clients as one operation."""
        self._ensure_idle("set_model")
        provider_name = canonical_provider_name(provider_name)
        self._apply_model_state(provider_name, model, small_model)
        # State-change entry: branching back past this point restores the
        # model that was in effect then (plan §5.9).  small_model rides
        # along — compaction and branch summaries consume
        # it (code-review issue 三).
        self.session.record_model_change(
            provider_name, self.settings.model, small_model=self.settings.small_model
        )
        self._events.publish(ModelChangedEvent(
            provider=provider_name,
            model=self.settings.model,
        ))

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
        session_id = self.session.start_new()
        self._events.publish(SessionChangedEvent(session_id=session_id))
        return session_id

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
            self._events.publish(SessionChangedEvent(session_id=switched))
        return switched

    def set_thinking(self, level: str) -> str:
        """Set the thinking level (idle only) and record it on the tree.

        入参先归一(旧三档 disabled/auto/enabled 仍被接受并转换为
        七档),再按当前模型能力 clamp——写到树上的永远是七档合法值,
        与实际请求一致。返回生效档位(归一 + clamp 之后)。
        """
        self._ensure_idle("set_thinking_level")
        if level not in THINKING_LEVELS and level not in LEGACY_THINKING_LEVELS:
            raise ValueError(
                f"thinking level must be one of {list(THINKING_LEVELS)}"
            )
        clamped = clamp_thinking_level(
            self.model, normalize_thinking_level(level)
        )
        self.settings.thinking = clamped
        self.agent.state.thinking_level = clamped  # 批 4:thinking 入 state
        self.session.record_thinking_change(clamped)
        self._events.publish(ThinkingLevelChangedEvent(level=clamped))
        return clamped

    def cycle_thinking(self) -> str:
        """Shift+Tab:只循环当前模型实际支持的档位(idle only)。"""
        self._ensure_idle("cycle_thinking_level")
        available = available_thinking_levels(self.model)
        current = normalize_thinking_level(self.settings.thinking)
        index = available.index(current) if current in available else -1
        level = available[(index + 1) % len(available)]
        self.settings.thinking = level
        self.agent.state.thinking_level = level  # 批 4:thinking 入 state
        self.session.record_thinking_change(level)
        self._events.publish(ThinkingLevelChangedEvent(level=level))
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
        self.model_runtime.switch(
            provider_name, model=model, small_model=small_model
        )
        self.client = self.model_runtime.client
        self.model = self.model_runtime.model
        self.stream_fn = self.model_runtime.stream_fn
        self.session.client = self.client
        # 批 4:model 已迁入 agent.state——switch 与 restore 共用此路径,
        # 一处同步;agent 在 __init__ 里早于本方法的任何调用点构造。
        self.agent.state.model = self.model
        # 换模型后按新模型能力重新 clamp thinking(K3 的 off:null →
        # minimal;切到非 reasoning 模型 → off),保证显示与实际请求一致。
        clamped = clamp_thinking_level(
            self.model, normalize_thinking_level(self.settings.thinking)
        )
        if clamped != self.settings.thinking:
            self.settings.thinking = clamped
            self.agent.state.thinking_level = clamped
            self._events.publish(ThinkingLevelChangedEvent(level=clamped))

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
            # 旧 JSONL 可能写着三档(disabled/auto/enabled)——读取点归一,
            # 并按当前模型 clamp(模型随后若切换,_apply_model_state
            # 会按新模型再 clamp 一次)。
            clamped = clamp_thinking_level(
                self.model, normalize_thinking_level(tree.thinking_level)
            )
            self.settings.thinking = clamped
            self.agent.state.thinking_level = clamped  # 批 4
        target_provider = canonical_provider_name(tree.provider or "")
        if (
            not target_provider
            or target_provider not in self.model_runtime.catalog.providers
        ):
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
        retry_attempts = 0
        retry_closed = False
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
            # Typed coding-session events are the public product stream.
            # The legacy trace vocabulary remains as a compatibility
            # transport for existing CLI/RPC/TUI consumers.
            if event_type in {
                "context.build.failed",
                "session.jsonl_write_failed",
            }:
                self._events.publish(ErrorEvent(
                    phase=event_type,
                    message=str(
                        safe_data.get("error") or safe_data.get("message") or ""
                    ),
                ))
            elif event_type in {"loop.steered", "loop.followed_up"}:
                self._events.publish(self._queue_event())
            # ── usage tracking ────────────────────────────
            if event_type == "llm.completed" and "usage" in safe_data:
                u = safe_data["usage"]
                self.tracer.log_usage(
                    session_id=run_session_id,
                    model=safe_data.get("model", self.settings.model),
                    input_tokens=u.get("input_tokens", 0),
                    output_tokens=u.get("output_tokens", 0),
                    cache_read_tokens=u.get("cache_read_tokens", 0),
                    cache_write_tokens=u.get("cache_write_tokens", 0),
                    cost_total=float(u.get("cost_total", 0.0)),
                    turn_id=trace_id,
                )
                if safe_data.get("role") == "main":
                    # input + output ≈ the context the NEXT turn starts with
                    last_context_tokens = u.get(
                        "total_tokens",
                        u.get("input_tokens", 0) + u.get("output_tokens", 0),
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

        def on_model_retry(attempt: int, category: str, message: str) -> None:
            nonlocal retry_attempts
            retry_attempts = max(retry_attempts, attempt)
            self._events.publish(RetryEvent(
                attempt=attempt,
                category=category,
                message=message,
                max_attempts=self.settings.max_model_retries,
            ))

        def end_retry(success: bool, final_error: str | None = None) -> None:
            nonlocal retry_closed
            if retry_closed or retry_attempts == 0:
                return
            retry_closed = True
            self._events.publish(AutoRetryEndEvent(
                success=success,
                attempt=retry_attempts,
                final_error=final_error,
            ))

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
            # 批 3:三元组——history 进 state.messages,current 经 kernel
            # 事件摄入(initial_pending, source="user")。
            system, history, current = self.session.prepare_context(
                user_message, emit, self.tools.schemas()
            )
            history_repair = repair_tool_history(history)
            if history_repair.changed:
                history = list(history_repair.messages)
                diagnostics = history_repair.diagnostic_data()
                emit("session.tool_history_repaired", diagnostics)
                self._events.publish(ToolHistoryRepairedEvent(**diagnostics))
            system = self._with_tool_prompt_snippets(system)
            emit("context.build.completed", {
                "session_id": self.session.session_id,
                "message_count": len(history) + (1 if current is not None else 0),
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

            # ── state wholesale assignment (Pi agent-session) ──
            # The Agent owns the transcript: the freshly built HISTORY
            # replaces state.messages outright; the current user message
            # joins it through kernel events (initial-pending channel
            # below).  The run derives its AgentContext from state inside
            # Agent.run.  (批 2/3;model / thinking still ride the config
            # until 批 4.)
            self.agent.state.system_prompt = system
            self.agent.state.messages = history
            # 批 4 兜底:live settings 是 model/thinking 的事实来源,
            # 每 run 前同步进 state(config 不再携带它们,run 时
            # Agent._full_config 从 state 解析)。
            self.agent.state.model = self.model
            self.agent.state.thinking_level = self.settings.thinking
            # Tests reassign harness.stream_fn between runs — sync the
            # Agent's plain attribute with the Harness's current one.
            self.agent.stream_fn = self.stream_fn
            # ── session recorder wiring (batch B / 批 3) ─────────
            # EVERY message — the prompt's user message included — now
            # arrives via the listener below: one event = one entry.
            # No explicit record call remains.
            if recorder is not None:
                recorder.set_emit(emit)
            listeners = [recorder.listener()] if recorder is not None else []
            # 批 3:the prompt's own user message is the run's initial
            # batch, ingested through kernel events (source="user") before
            # the first turn — the sink's reducer appends it to
            # state.messages and the recorder listener persists it.
            #
            # Pi continue(): a continue run whose tree tip is ASSISTANT
            # can only be legal because of queued messages — drain ONE
            # batch (steering first, else follow-ups, each queue's own
            # drain so one-at-a-time is honored) and hand it to the loop's
            # initial-pending channel.  Without this the first model call
            # would carry an assistant-tipped context (providers reject).
            initial_pending: list | None = None
            initial_pending_source = "follow_up"
            if current is not None:
                initial_pending = [current]
                initial_pending_source = "user"
            elif (
                history
                and getattr(history[-1], "role", None) == "assistant"
            ):
                steering = self.agent.steering_queue.drain()
                if steering:
                    initial_pending = steering
                    initial_pending_source = "steering"
                else:
                    initial_pending = self.agent.follow_up_queue.drain() or None
            # 批 4:model/thinking 已迁入 agent.state(上方兜底赋值),
            # config 不再携带它们——Agent._full_config 从 state 解析。
            loop_config = AgentLoopConfig(
                max_iterations=self.settings.max_iterations,
                max_tokens=self.settings.max_tokens,
                # Agent-owned policy fields (queues, turn hooks, execution
                # mode, subscribed listeners) are injected by Agent.run —
                # respond() no longer reads them back out of the Agent.
                on_truncation=compact_and_retry,
                governor=self.governor,
                cache_retention=self.settings.cache_retention,
                hooks=self.hooks,
                max_model_retries=self.settings.max_model_retries,
                on_model_retry=on_model_retry,
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
                loop_config,
                emit=emit,
            )
            end_retry(
                result.status == "completed",
                result.error or None,
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
            end_retry(False, f"{type(exc).__name__}: {exc}")
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
            self._events.publish(AgentSettledEvent(
                status=result.status if result is not None else "failed"
            ))

    def _with_tool_prompt_snippets(self, system: str) -> str:
        """Add product-only tool guidance without leaking it into Agent/AI."""
        if not self.tool_prompt_snippets:
            return system
        return system + "\n\n工具使用指南：\n" + "\n".join(
            f"- {snippet}" for snippet in self.tool_prompt_snippets
        )

    # ── 前端查询/操作 API ─────────────────────────────────
    # TUI/RPC 等前端只依赖这些公开入口,不触碰 session.recorder、
    # JSONL 文件或 tracer 内部。

    def list_sessions(self, limit: int = 15) -> list[dict]:
        """会话列表(供选择器)。"""
        return self.session.list_sessions(limit)

    def usage_summary(self) -> dict:
        """token 用量汇总(/usage)。"""
        return self.tracer.usage_summary()

    def summary_info(self) -> dict | None:
        """当前会话的滚动摘要信息(/summary)。"""
        return self.session.summary_info()

    def footer_snapshot(self) -> "FooterSnapshot":
        """Pi 风格 footer 快照:身份/环境/累计用量/context 一次给出。

        累计用量从当前路径历史消息现算——恢复/切换会话后立即可用,
        前端不维护自己的计数器。
        """
        from lsm_harness.coding_agent.footer import footer_snapshot

        return footer_snapshot(self)

    def current_path_messages(self) -> list:
        """当前分支路径的 typed 消息(TUI 界面重建用)。

        与 ``session.history`` 同源(tree path 的 message entries),但保留
        完整 typed 结构(text / thinking / tool_call_id / is_error)。
        TUI 不应自行读取 JSONL——这是 CodingSession 的公开查询边界。
        """
        from lsm_harness.ops.session_store import (
            path_to_leaf,
            read_session_entries,
        )

        recorder = self.session.recorder
        if recorder is None:
            return []
        entries = read_session_entries(self.session.jsonl_path)
        if not entries:
            return []
        path = path_to_leaf(entries, recorder.last_entry_id)
        return [entry.message for entry in path if entry.type == "message"]

    def current_path_entries(self) -> tuple[list, str | None]:
        """(当前路径条目列表, 当前 leaf entry id)——tree picker 用。"""
        from lsm_harness.ops.session_store import (
            path_to_leaf,
            read_session_entries,
        )

        recorder = self.session.recorder
        if recorder is None:
            return [], None
        entries = read_session_entries(self.session.jsonl_path)
        if not entries:
            return [], None
        leaf_id = recorder.last_entry_id
        return path_to_leaf(entries, leaf_id), leaf_id

    def branch_to(self, entry_ref: str) -> str | None:
        """切换当前分支到历史节点(idle only);emit 由内部提供。

        返回解析后的 entry id;节点不存在或已是当前 leaf 返回 None。
        """
        self._ensure_idle("branch")
        return self.session.branch(entry_ref, self._restore_emit)

    def close(self) -> None:
        self.subagents.shutdown(wait=False)
        self.conn.close()


# Public compatibility name used by the existing CLI/RPC/TUI and third-party
# callers.  The concrete owner is CodingSession; no duplicate wrapper state.
Harness = CodingSession


__all__ = ["CodingSession", "Harness", "RunBusyError"]
