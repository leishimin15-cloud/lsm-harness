"""Coding-agent composition root for one local LSM Harness instance."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from lsm_harness.agent import Agent, AgentContext, AgentLoopConfig
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.events import HarnessEvent, Observer, make_event
from lsm_harness.agent.agent_loop import run_agent_loop
from lsm_harness.loop.governance import ContextGovernor, GovernanceConfig
from lsm_harness.agent.hooks import (
    LoopHooks,
    PrepareNextTurn,
    ShouldStopAfterTurn,
    invoke_trace_end,
    invoke_trace_start,
)
from lsm_harness.agent.pending import PendingMessage
from lsm_harness.mcp import MCPClient
from lsm_harness.memory import Memory
from lsm_harness.ai.providers import get_client, get_model, PROVIDERS
from lsm_harness.ai.registry import registered_api_providers
from lsm_harness.ai.stream import client_stream_function, stream_simple
from lsm_harness.ops.file_state import FileState
from lsm_harness.ops.sandbox import SandboxManager
from lsm_harness.ops.tracing import Tracer
from lsm_harness.rag import RAGEngine
from lsm_harness.coding_agent.messages import register_coding_agent_messages
from lsm_harness.coding_agent.session import Session
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


def _safe_message_text(msg: str | dict) -> str:
    """Extract safe text from a message for tracing (no image data)."""
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
        # registry (stream_simple) whenever the model's API dialect is
        # registered; injected legacy clients keep the adapter, and tests
        # may inject a StreamFunction directly.
        self.stream_fn = stream_fn or self._resolve_stream_fn()
        self.memory = Memory(self.conn, self.settings, self.client)
        self.workspace_root = Path(
            self.settings.sandbox_project_dir or os.getcwd()
        ).expanduser().resolve()
        self.hooks = hooks

        # ── RAG engine ─────────────────────────────────────
        self.rag: RAGEngine | None = None
        if self.settings.rag_enabled:
            self.rag = RAGEngine(
                conn=self.conn,
                client=self.client,
                home=self.settings.home,
                small_model=self.settings.small_model,
                chunk_size=self.settings.rag_chunk_size,
            )

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

        # ── sandbox (optional) ──────────────────────────────
        self.sandbox: SandboxManager | None = None
        if self.settings.sandbox_enabled:
            self.sandbox = SandboxManager(
                project_dir=self.settings.sandbox_project_dir or os.getcwd()
            )
            try:
                self.sandbox.ensure_image()
            except Exception as exc:
                print(f"[sandbox] Image build failed: {exc}", file=sys.stderr)
                self.sandbox = None

        # ── subagent manager ───────────────────────────────
        self.subagents = SubagentManager(
            max_concurrent=self.settings.subagent_max_concurrent,
        )
        self.subagents._harness = self

        # ── MCP client ──────────────────────────────────────
        self.mcp: MCPClient | None = None
        if self.settings.mcp_enabled:
            self.mcp = MCPClient()
            try:
                self.mcp.start()
            except Exception as exc:
                print(f"[MCP] Failed to start: {exc}", file=sys.stderr)
                self.mcp = None

        self.tool_prompt_snippets: list[str] = []
        # §9.3: product renderers survive the AgentTool wrap via this
        # name → (render_call, render_result) map; the CLI/TUI listener
        # consults it with priority custom renderer → label → tool name.
        self.tool_renderers: dict = {}
        self.tools = build_registry(
            self.conn, self.settings, self.memory,
            subagent_manager=self.subagents,
            mcp_client=self.mcp,
            rag_engine=self.rag,
            sandbox=self.sandbox,
            file_state=self.file_state,
            workspace_root=self.workspace_root,
            prompt_snippets=self.tool_prompt_snippets,
            renderers=self.tool_renderers,
        )
        self.session = Session(self.settings, self.memory)
        self.tracer = Tracer(self.settings.home)

        # ── stateful Agent shell around the stateless loop ──
        self.agent = Agent(
            prepare_next_turn=prepare_next_turn,
            should_stop_after_turn=should_stop_after_turn,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
            tool_execution=tool_execution,
        )

    # ── public API ───────────────────────────────────────────

    def _resolve_stream_fn(self) -> StreamFunction:
        """Pick the canonical model-call chain for the current model.

        Plan §8: ``Model → stream_simple → resolve_api_provider(Model.api)
        → translator`` is the one entry point.  The registry path injects
        the API key from settings into every request's StreamOptions; a
        model whose ``api`` dialect is NOT registered (an injected legacy
        client) keeps the ``client_stream_function`` adapter — the
        ModelClient facade still serves memory gate / consolidation /
        RAG / compaction regardless.
        """
        if self.model.api in registered_api_providers():
            api_key = self.settings.api_key

            def registry_stream(model, context, options):
                return stream_simple(
                    model, context, replace(options, api_key=api_key)
                )

            return registry_stream
        return client_stream_function(self.client)

    @property
    def is_running(self) -> bool:
        return self.agent.is_running

    def abort(self) -> None:
        """Cancel the currently running trace.

        Thread-safe — can be called from a signal handler or
        background thread.
        """
        self.agent.abort()

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
        """Switch the main loop, memory, and RAG clients as one operation."""
        self._apply_model_state(provider_name, model, small_model)
        # State-change entry: branching back past this point restores the
        # model that was in effect then (plan §5.9).  small_model rides
        # along — memory gate / RAG rerank / compaction summaries consume
        # it (code-review issue 三).
        self.session.record_model_change(
            provider_name, self.settings.model, small_model=self.settings.small_model
        )

    def _apply_model_state(
        self, provider_name: str, model: str, small_model: str
    ) -> None:
        """Rebuild client/stream_fn/settings for a provider+model combo.

        Shared by switch_model (a real change — records a model_change
        entry) and _restore_runtime_state (a restoration — records
        nothing, 恢复 ≠ 变更).
        """
        provider = PROVIDERS[provider_name]
        new_model = model or provider.model
        new_small = small_model or provider.small_model
        explicit_key = os.getenv("LSM_API_KEY") or os.getenv("WAKU_API_KEY") or ""
        resolved_key = explicit_key or os.getenv(provider.key_env, "")
        # Late import path: tests monkeypatch lsm_harness.ai.providers.get_client
        from lsm_harness.ai import providers as _providers

        new_client = _providers.get_client(
            provider_name=provider_name,
            api_key=resolved_key,
            base_url=self.settings.base_url or None,
            model=new_model,
            small_model=new_small,
            thinking=self.settings.thinking,
        )
        self.client = new_client
        resolved_model = getattr(new_client, "model", None)
        self.model = (
            resolved_model
            if isinstance(resolved_model, Model)
            else get_model(
                provider_name,
                new_model,
                base_url=self.settings.base_url or None,
            )
        )
        self.stream_fn = self._resolve_stream_fn()
        self.memory.client = new_client
        if self.rag is not None:
            self.rag.client = new_client
            self.rag.small_model = new_small
        self.settings.provider = provider_name
        self.settings.api_key = resolved_key
        self.settings.model = new_model
        self.settings.small_model = new_small

    def _restore_runtime_state(self, emit) -> None:
        """Apply the session tree's runtime state before the run (issue 二).

        The current path's header baseline plus model_change /
        thinking_level_change entries describe the model in effect at
        that point of the tree.  Branching or resuming changes which path
        is current, so the next respond must run with THAT state:
        provider/model/small_model rebuild the client and stream_fn
        exactly like switch_model — but WITHOUT writing model_change
        entries (恢复 ≠ 变更) — and thinking syncs into settings.

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

        active_run = self.agent.begin()
        # The session recorder owns this run's JSONL tree writes (batch B).
        recorder = self.session.recorder

        def emit(event_type: str, data: dict) -> None:
            nonlocal sequence, last_context_tokens
            sequence += 1
            safe_data = redact_data(data, (self.settings.api_key,))
            event = make_event(
                event_type,
                trace_id,
                safe_data,
                session_id=self.session.session_id,
                sequence=sequence,
                duration_ms=int((time.monotonic() - started_at) * 1000),
            )
            self.tracer.write(event)
            # ── usage tracking ────────────────────────────
            if event_type == "llm.completed" and "usage" in safe_data:
                u = safe_data["usage"]
                self.tracer.log_usage(
                    session_id=self.session.session_id,
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
                observer(event)

        def end_trace_hooks(result: TraceResult) -> None:
            nonlocal trace_hook_ended
            if trace_hook_ended:
                return
            trace_hook_ended = True
            invoke_trace_end(self.hooks, result, emit)

        emit("trace.started", {"source": source, "message": _safe_message_text(user_message)})
        invoke_trace_start(self.hooks, _safe_message_text(user_message), emit)
        emit("trace.accepted", {"source": source, "message": _safe_message_text(user_message)})

        # ── ensure sandbox container for this session ──────
        if self.sandbox is not None:
            self.sandbox.set_session(self.session.session_id)
            if not self.sandbox.is_running(self.session.session_id):
                try:
                    self.sandbox.create(self.session.session_id)
                except Exception as exc:
                    emit("sandbox.create_failed", {"error": str(exc)})

        try:
            # ── tree runtime state restore (code-review issue 二) ──
            # Branch / resume changes the current path; the next run must
            # use the model state recorded on THAT path.  Restoration
            # never writes model_change entries.
            self._restore_runtime_state(emit)
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
            listeners = list(self.agent.listeners or [])
            if recorder is not None:
                listeners.append(recorder.listener())
            loop_config = AgentLoopConfig(
                model=self.model,
                max_iterations=self.settings.max_iterations,
                max_tokens=self.settings.max_tokens,
                get_steering_messages=self.agent.steering_queue.drain,
                get_follow_up_messages=self.agent.follow_up_queue.drain,
                prepare_next_turn=self.agent.prepare_next_turn,
                should_stop_after_turn=self.agent.should_stop_after_turn,
                before_tool_call=self.agent.before_tool_call,
                after_tool_call=self.agent.after_tool_call,
                tool_execution=self.agent.tool_execution,
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
                session_id=self.session.session_id,
                sandboxed=self.sandbox is not None,
                listeners=listeners,
            )
            result = run_agent_loop(
                context=context,
                config=loop_config,
                stream_fn=self.stream_fn,
                emit=emit,
                interrupt=active_run.interrupt,
            )
            # A failed/aborted FIRST exchange leaves the recorder's buffer
            # unflushed — drop it so a dead run never persists half a
            # session (plan §5.6).
            if result.status != "completed" and recorder is not None and recorder.deferred:
                recorder.abandon(result.status or "failed")
            if result.status == "completed":
                emit("persistence.started", {"stores": ["sqlite", "jsonl"]})
                persistence = self.session.add_exchange(user_message, result, source)
                emit("memory.consolidation.started", {})
                facts_saved, episode_saved = self.memory.consolidate(emit)
                emit("memory.consolidation.completed", {
                    "facts_saved": facts_saved,
                    "episode_saved": episode_saved,
                    "ran": bool(facts_saved or episode_saved),
                })
                self.memory.export_markdown()
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
                # turn's prepare_context.
                self.session.compact_if_due(last_context_tokens, emit)
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
                recorder.abandon("exception")
            error_message = f"{type(exc).__name__}: {exc}"
            failed_result = TraceResult(
                reply=error_message,
                status="failed",
                stop_reason="error",
                error=error_message,
            )
            emit("trace.error", {"error": error_message})
            end_trace_hooks(failed_result)
            emit("trace.failed", {"error": error_message, "status": "failed"})
            raise
        finally:
            if approval_broker is not None:
                approval_broker.reject_all("trace_finished")
            self.agent.finish(active_run)

    def _with_tool_prompt_snippets(self, system: str) -> str:
        """Add product-only tool guidance without leaking it into Agent/AI."""
        if not self.tool_prompt_snippets:
            return system
        return system + "\n\n工具使用指南：\n" + "\n".join(
            f"- {snippet}" for snippet in self.tool_prompt_snippets
        )

    def close(self) -> None:
        # Destroy sandbox container if running
        if self.sandbox and self.sandbox.current_session:
            self.sandbox.destroy(self.sandbox.current_session)

        self.subagents.shutdown(wait=False)
        if self.mcp:
            self.mcp.stop()
        self.conn.close()
