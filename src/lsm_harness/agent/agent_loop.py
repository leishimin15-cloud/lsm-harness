"""Stateless provider-neutral reason → act → observe Agent Loop.

Ten guardrails protect every turn:
  1. **streaming** – text & tool calls arrive as fine-grained events
  2. **truncation protection** – stop_reason="length" rejects all tool calls
  3. **interrupt** – threading.Event lets the caller cancel mid-turn
  4. **steering** – a queue lets the caller inject messages at turn boundaries
  5. **overflow recovery** – compacts context and retries after a length stop
  6. **empty response retry** – blank model output triggers up to 2 retries
  7. **length recovery cap** – at most 3 compaction-based retries per trace
  8. **repeated tool errors** – same tool failing twice → early exit
  9. **error categorisation** – arrearage / rate-limit / transient / permanent
 10. **lifecycle hooks** – callbacks at every phase (trace, turn, model, tool)

Parallel-safe tools execute concurrently via a thread pool.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from lsm_harness.agent.hooks import (
    LoopHooks,
    NextTurnUpdate,
    TurnControlContext,
    TurnContext,
    invoke_model_call,
    invoke_model_response,
    invoke_tool_error,
    invoke_tool_request,
    invoke_turn_end,
    invoke_turn_start,
)
from lsm_harness.agent.pending import PendingMessage
from lsm_harness.agent.tools import (
    ExecutionContext,
    ToolRegistry,
    ToolResultMessage as ToolResultEnvelope,
)
from lsm_harness.agent.events import (
    AgentEndEvent,
    AgentEventSink,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
    TurnEndEvent,
    TurnStartEvent,
    make_legacy_adapter,
)
from lsm_harness.agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
    assistant_message,
    message_preview,
    tool_result_message,
    user_message,
)
from lsm_harness.agent.types import (
    AfterToolCall,
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    BeforeToolCall,
    BeforeToolCallContext,
    ToolExecutionMode,
    TraceResult,
    TraceStatus,
)
from lsm_harness.ai.types import (
    AIContext,
    CacheRetention,
    ErrorCategory,
    Model,
    ModelResponse,
    StopReason,
    StreamFunction,
    StreamOptions,
    ThinkingLevel,
    normalize_stop_reason,
)
from lsm_harness.ai.errors import is_context_overflow
from lsm_harness.ai.models import with_model_id


Emit = Callable[[str, dict], None]
PendingMessageGetter = Callable[[], list[PendingMessage]]


@dataclass
class ExecutedToolBatch:
    messages: list[AgentMessage]
    terminate: bool = False


def _complete_turn(
    hooks: LoopHooks | None,
    ctx: TurnContext,
    status: str,
    emit: Emit,
    sink: AgentEventSink,
    *,
    message: AssistantMessage | None = None,
    tool_results: list[AgentMessage] | tuple[AgentMessage, ...] = (),
) -> None:
    """Close one model-call + caused-tool-batch lifecycle."""
    ctx.status = status
    invoke_turn_end(hooks, ctx, emit)
    sink.process_event(TurnEndEvent(
        turn_index=ctx.turn_index,
        model=ctx.model,
        stop_reason=ctx.stop_reason,
        status=status,
        usage=ctx.usage,
        tool_count=ctx.tool_count,
        tool_error_count=ctx.tool_error_count,
        message=message,
        tool_results=tuple(tool_results),
    ))


def _finish_trace(
    result: TraceResult,
    *,
    status: TraceStatus,
    stop_reason: StopReason,
    reply: str | None = None,
    error: str = "",
    sink: AgentEventSink | None = None,
) -> TraceResult:
    result.status = status
    result.stop_reason = stop_reason
    result.aborted = status == "aborted"
    if reply is not None:
        result.reply = reply
    result.error = error
    if sink is not None:
        sink.process_event(AgentEndEvent(
            status=status,
            stop_reason=stop_reason,
            error=error,
        ))
    return result


def run_agent_loop(
    *,
    context: AgentContext,
    config: AgentLoopConfig,
    stream_fn: StreamFunction,
    emit: Emit,
    interrupt: threading.Event | None = None,
    sink: AgentEventSink | None = None,
) -> TraceResult:
    """Execute the reason→act→observe loop.

    The loop runs on typed AgentMessages held by ``context.messages``;
    callers observe the run's appends through that same list.

    transform_context: Optionally prepares a request-only context snapshot.
    before_tool_call: May block a validated tool request before execution.
    after_tool_call: May override a finalized tool result field by field.
    tool_execution: Executes eligible tool batches sequentially or in parallel.
    get_steering_messages: Polls urgent messages before the first Turn and
        after every completed Turn.
    get_follow_up_messages: Polls queued work after the inner loop would stop.
    prepare_next_turn: May replace context/model/thinking state after a Turn.
    should_stop_after_turn: Gracefully stops before polling either queue.
    on_truncation: Called when stop_reason == "length".
    governor: ContextGovernor for tool result size management.
    hooks: LoopHooks with lifecycle callbacks.
    thinking: "disabled" | "enabled" | "auto" — when "auto",
        thinking is enabled dynamically based on task complexity.
    """
    system = context.system_prompt
    messages = context.messages
    tools = context.tools

    if config.model is None:
        # Direct callers must pass a model explicitly; via Agent.run the
        # _full_config resolution (config > state) guarantees one.
        raise ValueError(
            "AgentLoopConfig.model is required for run_agent_loop; "
            "via Agent.run it is resolved from agent.state.model"
        )
    current_model = (
        config.model
        if isinstance(config.model, Model)
        else Model(id=config.model, api="legacy-client", provider="legacy")
    )
    max_iterations = config.max_iterations
    max_tokens = config.max_tokens
    convert_to_llm = config.convert_to_llm
    transform_context = config.transform_context
    before_tool_call = config.before_tool_call
    after_tool_call = config.after_tool_call
    tool_execution = config.tool_execution
    get_steering_messages = config.get_steering_messages
    get_follow_up_messages = config.get_follow_up_messages
    initial_pending = list(config.initial_pending_messages or [])
    initial_pending_source = config.initial_pending_source
    prepare_next_turn = config.prepare_next_turn
    should_stop_after_turn = config.should_stop_after_turn
    on_truncation = config.on_truncation
    governor = config.governor
    hooks = config.hooks
    listeners = config.listeners
    thinking = config.thinking or "off"  # 批 4:None 等价 off
    cache_retention = config.cache_retention
    max_model_retries = config.max_model_retries
    on_model_retry = config.on_model_retry
    on_payload = config.on_payload
    on_response = config.on_response
    thinking_budgets = config.thinking_budgets
    transport = config.transport
    max_retry_delay_ms = config.max_retry_delay_ms
    max_empty_retries = config.max_empty_retries
    max_length_recoveries = config.max_length_recoveries
    approval_broker = config.approval_broker
    trace_id = config.trace_id
    session_id = config.session_id
    active_stream_fn = stream_fn

    # Chapter 7: typed kernel events. The legacy string channel is
    # reproduced by an adapter subscribed FIRST, so existing consumers
    # (tracer / flow / CLI / tests) observe a byte-identical stream.
    if sink is None:
        # Direct-call path (run_test_loop and friends): build a fresh
        # per-run sink exactly as before — persistent tier holds the
        # adapter first, then config.listeners.
        sink = AgentEventSink(messages=messages)
        sink.subscribe(make_legacy_adapter(emit))
        for listener in listeners or []:
            sink.subscribe(listener)
    else:
        # Agent.run path: the caller's PERSISTENT sink drives this run;
        # its adapter + run listeners were armed by begin_run_dispatch.
        # Guard the single-source invariant: the loop aliases the sink's
        # message list — never a second copy.
        assert messages is sink.state.messages, (
            "context.messages must be sink.state.messages (same object)"
        )
    sink.process_event(AgentStartEvent(model=current_model.id))

    result = TraceResult(reply="")
    empty_retries = 0
    length_recoveries = 0
    _max_empty = max(0, int(max_empty_retries))
    _max_length_recoveries = max(0, int(max_length_recoveries))
    _tool_errors: dict[str, int] = {}
    # 连续失败熔断的 streak 状态:[当前签名, 连续次数](跨 turn 持续,
    # 签名一变即重置)。
    _error_streak: list = [None, 0]

    current_thinking = _resolve_thinking(thinking, messages)
    iteration = 0
    final_stop_reason: StopReason = "stop"
    # Pi skipInitialSteeringPoll: when the initial batch came from a
    # steering drain (Agent.continue_), that queue was just polled by the
    # caller — don't poll it twice.  A follow-up-sourced batch leaves the
    # start poll intact; polled steering lands AFTER the initial batch
    # (Pi: prompts are the run's input, steering arrives after).
    if initial_pending and initial_pending_source == "steering":
        pending_messages = []
    else:
        pending_messages = _poll_pending_messages(
            get_steering_messages,
            source="steering",
            emit=emit,
        )
    # Pi prompt-ingestion: the caller-drained initial batch (批 3: the
    # prompt's own user message, source="user"; 批 7: continue()'s drained
    # queue batch) is ingested via kernel events BEFORE the first turn —
    # ahead of the abort check, so even an instantly-aborted run keeps
    # its question in state.messages and the session tree.
    if initial_pending:
        _inject_pending_messages(
            initial_pending,
            source=initial_pending_source,
            emit=emit,
            sink=sink,
        )
        initial_pending = []
    pending_source = "steering"

    # Pi-style outer loop: follow-up messages can revive the same Trace.
    while True:
        has_more_tool_calls = True

        # Pi-style inner loop: tool calls or steering keep producing Turns.
        while has_more_tool_calls or pending_messages:
            # Pi parity: the reusable Agent Loop has no fixed Turn cap by
            # default.  Eval/automation products may opt into one.
            if (
                max_iterations is not None
                and iteration >= max_iterations
            ):
                error_message = _termination_message(
                    result.tool_calls, max_iterations
                )
                emit("loop.limit_reached", {"max_iterations": max_iterations})
                return _finish_trace(
                    result,
                sink=sink,
                    status="failed",
                    stop_reason="error",
                    reply=error_message,
                    error=error_message,
                )

            if _should_abort(interrupt, emit):
                result.iterations = iteration
                return _finish_trace(
                    result,
                sink=sink,
                    status="aborted",
                    stop_reason="aborted",
                    reply="任务已中断。",
                )

            iteration += 1
            result.iterations = iteration
            sink.process_event(TurnStartEvent(
                turn_index=iteration,
                model=current_model.id,
            ))
            invoke_turn_start(hooks, iteration, emit)

            if pending_messages:
                _inject_pending_messages(
                    pending_messages,
                    source=pending_source,
                    emit=emit,
                    sink=sink,
                )
                pending_messages = []

            invoke_model_call(hooks, current_model.id, iteration, emit)
            emit("llm.started", {
                "model": current_model.id,
                "iteration": iteration,
            })
            try:
                request_messages = list(messages)
                if transform_context is not None:
                    request_messages = transform_context(request_messages)
                request_messages = convert_to_llm(request_messages)
            except Exception as exc:
                error_message = (
                    "模型上下文转换失败："
                    f"{type(exc).__name__}: {exc}"
                )
                emit("loop.context_transform_failed", {
                    "iteration": iteration,
                    "error": str(exc),
                })
                assistant = assistant_message(
                    error_message,
                    api=current_model.api,
                    provider=current_model.provider,
                    model=current_model.id,
                    usage={"input_tokens": 0, "output_tokens": 0},
                    stop_reason="error",
                    error_message=error_message,
                    timestamp=int(time.time() * 1000),
                )
                sink.process_event(MessageStartEvent(
                    message=assistant,
                    source="assistant",
                    turn_index=iteration,
                ))
                raw_stop_reason = "error"
                usage = {"input_tokens": 0, "output_tokens": 0}
            else:
                # Listener/stream failures must reach Agent.run's
                # handleRunFailure path. Only context conversion belongs to
                # the conversion-error branch above.
                assistant, raw_stop_reason, usage = _consume_assistant_stream(
                    stream_fn=active_stream_fn,
                    model=current_model,
                    system=system,
                    messages=request_messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    emit=emit,
                    sink=sink,
                    iteration=iteration,
                    interrupt=interrupt,
                    thinking=current_thinking,
                    cache_retention=cache_retention,
                    session_id=session_id,
                    max_model_retries=max_model_retries,
                    on_model_retry=on_model_retry,
                    on_payload=on_payload,
                    on_response=on_response,
                    thinking_budgets=thinking_budgets,
                    transport=transport,
                    max_retry_delay_ms=max_retry_delay_ms,
                )

            call_list = assistant.tool_calls
            normalized_stop_reason = normalize_stop_reason(str(raw_stop_reason))
            if normalized_stop_reason in {"error", "aborted", "length"}:
                stop_reason = normalized_stop_reason
            elif call_list:
                stop_reason = "tool_calls"
            elif normalized_stop_reason == "tool_calls":
                stop_reason = "error"
                emit("loop.stop_reason_mismatch", {
                    "raw_stop_reason": str(raw_stop_reason),
                    "message": "provider reported tool calls without any parsed calls",
                })
            else:
                stop_reason = "stop"
            final_stop_reason = stop_reason

            raw_text = (assistant.text or "")[:200]
            turn_ctx = TurnContext(
                turn_index=iteration,
                model=current_model.id,
                stop_reason=stop_reason,
                usage=usage,
                tool_count=len(call_list),
                tool_error_count=0,
                accumulated_text=raw_text,
            )
            if interrupt is not None and interrupt.is_set():
                stop_reason = "aborted"
                turn_ctx.stop_reason = "aborted"
                assistant = replace(assistant, stop_reason="aborted")

            sink.process_event(MessageEndEvent(
                message=assistant,
                source="assistant",
                turn_index=iteration,
            ))
            invoke_model_response(hooks, stop_reason, raw_text, usage, emit)

            if stop_reason == "aborted":
                _complete_turn(
                    hooks, turn_ctx, "aborted", emit, sink, message=assistant
                )
                return _finish_trace(
                    result,
                sink=sink,
                    status="aborted",
                    stop_reason="aborted",
                    reply="任务已中断。",
                )

            if stop_reason == "error":
                error_message = assistant.error_message or assistant.text or "模型调用失败。"
                emit("llm.error", {
                    "iteration": iteration,
                    "attempt": "final",
                    "category": "final",
                    "error": "ModelCallError",
                    "message": error_message[:200],
                })
                _complete_turn(
                    hooks, turn_ctx, "error", emit, sink, message=assistant
                )
                return _finish_trace(
                    result,
                sink=sink,
                    status="failed",
                    stop_reason="error",
                    reply=error_message,
                    error=error_message,
                )

            emit("llm.completed", {
                "role": "main",
                "model": current_model.id,
                "iteration": iteration,
                "stop_reason": stop_reason,
                "usage": dict(usage),
            })

            tool_results: list[AgentMessage] = []
            turn_status = "completed"
            has_more_tool_calls = False

            if stop_reason == "length":
                tool_results = _reject_truncated_calls(assistant, emit, sink, iteration)
                turn_ctx.tool_error_count = len(tool_results)
                if on_truncation and length_recoveries < _max_length_recoveries:
                    length_recoveries += 1
                    emit("loop.overflow_recovery", {
                        "iteration": iteration,
                        "recovery_count": length_recoveries,
                    })
                    try:
                        system, messages = on_truncation()
                    except Exception as exc:
                        error_message = (
                            f"上下文恢复失败：{type(exc).__name__}: {exc}"
                        )
                        emit("loop.overflow_recovery_failed", {"error": str(exc)})
                        _complete_turn(
                            hooks, turn_ctx, "error", emit, sink,
                            message=assistant,
                            tool_results=tool_results,
                        )
                        return _finish_trace(
                            result,
                sink=sink,
                            status="failed",
                            stop_reason="length",
                            reply=error_message,
                            error=error_message,
                        )
                    sink.replace_messages(messages)
                    has_more_tool_calls = True
                    turn_status = "length_recovery"
                else:
                    emit("loop.length_recovery_exhausted", {
                        "recoveries": length_recoveries,
                        "max": _max_length_recoveries,
                        "recovery_available": on_truncation is not None,
                    })
                    error_message = (
                        "模型输出达到长度限制，且无法继续恢复。"
                        "请开启新会话、缩小请求范围或提高输出上限后重试。"
                    )
                    _complete_turn(
                        hooks, turn_ctx, "length_exhausted", emit, sink,
                        message=assistant,
                        tool_results=tool_results,
                    )
                    return _finish_trace(
                        result,
                sink=sink,
                        status="failed",
                        stop_reason="length",
                        reply=error_message,
                        error=error_message,
                    )
            elif not call_list:
                text = (assistant.text or "").strip()
                if not text and empty_retries < _max_empty:
                    empty_retries += 1
                    emit("loop.empty_response_retry", {"attempt": empty_retries})
                    # Prompt-path nudge: no kernel event carries it (Pi
                    # injects such prompts outside the event stream too).
                    messages.append(user_message("Please provide a response."))
                    has_more_tool_calls = True
                    turn_status = "empty_response_retry"
                elif not text:
                    error_message = "模型在重试后仍未生成回复。"
                    _complete_turn(
                        hooks, turn_ctx, "error", emit, sink,
                        message=assistant,
                    )
                    return _finish_trace(
                        result,
                sink=sink,
                        status="failed",
                        stop_reason="error",
                        reply=error_message,
                        error=error_message,
                    )
                else:
                    result.reply = assistant.text
            else:
                prior_tool_errors = sum(_tool_errors.values())
                executed_tool_batch = _execute_tool_calls(
                    call_list=call_list,
                    tools=tools,
                    messages=messages,
                    emit=emit,
                    sink=sink,
                    turn_index=iteration,
                    result=result,
                    interrupt=interrupt,
                    _tool_errors=_tool_errors,
                    _error_streak=_error_streak,
                    governor=governor,
                    hooks=hooks,
                    before_tool_call=before_tool_call,
                    after_tool_call=after_tool_call,
                    tool_execution=tool_execution,
                    agent_context=AgentContext(
                        system_prompt=system,
                        messages=messages,
                        tools=tools,
                    ),
                    approval_broker=approval_broker,
                    trace_id=trace_id,
                    session_id=session_id,
                )
                tool_results = executed_tool_batch.messages
                turn_ctx.tool_error_count = (
                    sum(_tool_errors.values()) - prior_tool_errors
                )
                has_more_tool_calls = not executed_tool_batch.terminate
                turn_status = (
                    "terminated" if executed_tool_batch.terminate else "tool_use"
                )
                if executed_tool_batch.terminate:
                    result.reply = "任务已完成。"

            if _should_abort(interrupt, emit):
                turn_ctx.stop_reason = "aborted"
                _complete_turn(
                    hooks, turn_ctx, "aborted", emit, sink,
                    message=assistant,
                    tool_results=tool_results,
                )
                return _finish_trace(
                    result,
                sink=sink,
                    status="aborted",
                    stop_reason="aborted",
                    reply="任务已中断。",
                )

            if result.status == "failed":
                _complete_turn(
                    hooks, turn_ctx, "error", emit, sink,
                    message=assistant,
                    tool_results=tool_results,
                )
                sink.process_event(AgentEndEvent(
                    status="failed",
                    stop_reason=final_stop_reason,
                    error=result.error,
                ))
                return result

            _complete_turn(
                hooks, turn_ctx, turn_status, emit, sink,
                message=assistant,
                tool_results=tool_results,
            )
            control_context = TurnControlContext(
                message=assistant,
                tool_results=tool_results,
                system=system,
                messages=messages,
                result=result,
                turn=turn_ctx,
            )

            if prepare_next_turn is not None:
                try:
                    update = prepare_next_turn(control_context)
                except Exception as exc:
                    error_message = (
                        f"prepareNextTurn 失败：{type(exc).__name__}: {exc}"
                    )
                    emit("loop.prepare_next_turn_failed", {"error": str(exc)})
                    return _finish_trace(
                        result,
                sink=sink,
                        status="failed",
                        stop_reason="error",
                        reply=error_message,
                        error=error_message,
                    )
                if update is not None:
                    if not isinstance(update, NextTurnUpdate):
                        error_message = "prepareNextTurn 必须返回 NextTurnUpdate 或 None。"
                        emit("loop.prepare_next_turn_failed", {
                            "error": error_message,
                        })
                        return _finish_trace(
                            result,
                sink=sink,
                            status="failed",
                            stop_reason="error",
                            reply=error_message,
                            error=error_message,
                        )
                    system = update.system if update.system is not None else system
                    messages = (
                        update.messages
                        if update.messages is not None
                        else messages
                    )
                    sink.replace_messages(messages)
                    if isinstance(update.model, Model):
                        current_model = update.model
                    elif update.model is not None:
                        current_model = with_model_id(current_model, update.model)
                    if update.thinking is not None:
                        current_thinking = _resolve_thinking(
                            update.thinking, messages
                        )
                    emit("loop.next_turn_prepared", {
                        "model": current_model.id,
                        "thinking": current_thinking,
                        "context_replaced": update.messages is not None,
                    })
                    control_context.system = system
                    control_context.messages = messages

            if should_stop_after_turn is not None:
                try:
                    should_stop = should_stop_after_turn(control_context)
                except Exception as exc:
                    error_message = (
                        f"shouldStopAfterTurn 失败：{type(exc).__name__}: {exc}"
                    )
                    emit("loop.should_stop_after_turn_failed", {
                        "error": str(exc),
                    })
                    return _finish_trace(
                        result,
                sink=sink,
                        status="failed",
                        stop_reason="error",
                        reply=error_message,
                        error=error_message,
                    )
                if should_stop:
                    emit("loop.stopped_after_turn", {"turn_index": iteration})
                    if not result.reply:
                        result.reply = raw_text or "已在当前 Turn 后停止。"
                    return _finish_trace(
                        result,
                sink=sink,
                        status="completed",
                        stop_reason=final_stop_reason,
                    )

            pending_messages = _poll_pending_messages(
                get_steering_messages,
                source="steering",
                emit=emit,
            )
            pending_source = "steering"

        follow_up_messages = _poll_pending_messages(
            get_follow_up_messages,
            source="follow_up",
            emit=emit,
        )
        if follow_up_messages:
            pending_messages = follow_up_messages
            pending_source = "follow_up"
            continue
        break

    return _finish_trace(
        result,
                sink=sink,
        status="completed",
        stop_reason=final_stop_reason,
    )


# ── thinking auto-detection ──────────────────────────────────────


def _resolve_thinking(
    thinking: str,
    messages: list[AgentMessage],
) -> ThinkingLevel:
    """Resolve the effective thinking mode.

    When thinking="auto", enable thinking if the task looks complex.
    """
    if thinking in {
        "off", "minimal", "low", "medium", "high", "xhigh", "max"
    }:
        return thinking
    if thinking == "enabled":
        return "high"
    if thinking == "disabled":
        return "off"
    if thinking == "auto":
        if _task_is_complex(messages):
            return "high"
        return "off"
    return "off"


def _task_is_complex(messages: list[AgentMessage]) -> bool:
    """Heuristic: is this task complex enough to warrant thinking mode?

    Checks:
      - User message > 200 chars
      - Multiple turns of history (the context already has tool calls)
      - Contains planning keywords (设计, 实现, 重构, 架构, 分析)
    """
    # Find the last user message
    last_user = ""
    for msg in reversed(messages):
        if isinstance(msg, UserMessage):
            last_user = message_preview(msg, limit=10000)
            break

    if len(last_user) > 200:
        return True

    # If there are tool calls in the history, the task is multi-step
    tool_count = sum(1 for m in messages if isinstance(m, ToolResultMessage))
    if tool_count > 3:
        return True

    # Keyword check
    complexity_keywords = [
        "设计", "实现", "重构", "架构", "分析", "优化",
        "implement", "design", "refactor", "architecture",
        "debug", "diagnose", "investigate",
    ]
    lower = last_user.lower()
    if any(kw in lower for kw in complexity_keywords):
        return True

    return False


# ── internal helpers ──────────────────────────────────────────────


def _should_abort(interrupt: threading.Event | None, emit: Emit) -> bool:
    if interrupt and interrupt.is_set():
        emit("loop.aborted", {})
        return True
    return False


def _poll_pending_messages(
    getter: PendingMessageGetter | None,
    *,
    source: str,
    emit: Emit,
) -> list[PendingMessage]:
    pending: list[PendingMessage] = []
    if getter is not None:
        try:
            pending.extend(getter())
        except Exception as exc:
            emit("loop.pending_messages_failed", {
                "source": source,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return pending


def _inject_pending_messages(
    pending_messages: list[PendingMessage],
    *,
    source: str,
    emit: Emit,
    sink: AgentEventSink,
) -> None:
    event_type = "loop.steered" if source == "steering" else "loop.followed_up"
    for pending in pending_messages:
        if isinstance(pending, str):
            message = user_message(pending)
        else:
            message = pending  # already a typed AgentMessage
        preview = message_preview(message)
        # The sink appends on message_end (Pi: append ownership is the
        # event processor's, not the loop's).
        sink.process_event(MessageStartEvent(message=message, source=source))
        sink.process_event(MessageEndEvent(message=message, source=source))
        # The string channel carries QUEUE provenance only; the prompt's
        # own user message (source="user", 批 3) stays silent — no
        # loop.user event, zero string-contract change.
        if source in ("steering", "follow_up"):
            emit(event_type, {"message": preview})


def _consume_assistant_stream(
    *,
    stream_fn: StreamFunction,
    model: Model,
    system: str,
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    max_tokens: int,
    emit: Emit,
    sink: AgentEventSink,
    iteration: int,
    interrupt: threading.Event | None,
    thinking: ThinkingLevel,
    cache_retention: CacheRetention,
    session_id: str,
    max_model_retries: int,
    on_model_retry: Callable[[int, ErrorCategory, str], None] | None = None,
    on_payload=None,
    on_response=None,
    thinking_budgets=None,
    transport="auto",
    max_retry_delay_ms: int | None = None,
) -> tuple[AssistantMessage, StopReason, dict[str, int | float]]:
    """Translate the canonical AI event stream into one Agent Turn message."""

    def on_retry(
        attempt: int,
        category: ErrorCategory,
        message: str,
    ) -> None:
        if on_model_retry is not None:
            on_model_retry(attempt, category, message)
        emit("llm.error", {
            "iteration": iteration,
            "attempt": attempt,
            "category": category,
            "error": "ModelStreamRetry",
            "message": message[:200],
        })

    context = AIContext(
        system_prompt=system,
        messages=messages,
        tools=tools.ai_tools(),
    )
    options = StreamOptions(
        max_tokens=max_tokens,
        reasoning=thinking,
        cache_retention=cache_retention,
        session_id=session_id,
        interrupt=interrupt,
        max_retries=max(0, int(max_model_retries)),
        on_retry=on_retry,
        on_payload=on_payload,
        on_response=on_response,
        thinking_budgets=thinking_budgets,
        transport=transport,
        max_retry_delay_ms=max_retry_delay_ms,
    )
    final: ModelResponse | None = None
    error_category: ErrorCategory | None = None
    terminal_seen = False
    message_started = False

    for event in stream_fn(model, context, options):
        if interrupt is not None and interrupt.is_set():
            emit("loop.stream_aborted", {"iteration": iteration})
            final = replace(
                final or event.partial,
                stop_reason="aborted",
                error_message="model request aborted",
            )
            terminal_seen = True
            break

        if not message_started:
            sink.process_event(MessageStartEvent(
                message=AssistantMessage(
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                ),
                source="assistant",
                turn_index=iteration,
            ))
            message_started = True

        final = event.partial
        # Every ai-layer event becomes one kernel message_update carrying
        # the cumulative snapshot plus the original event passed through
        # (the adapter folds it back into legacy llm.* names).
        sink.process_event(MessageUpdateEvent(
            message=assistant_message(
                event.partial.text or None,
                thinking=event.partial.thinking,
                thinking_signature=event.partial.thinking_signature,
                tool_calls=event.partial.tool_calls,
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=event.partial.usage.as_dict(),
                stop_reason=event.partial.stop_reason,
                error_message=event.partial.error_message,
            ),
            assistant_message_event=event,
            turn_index=iteration,
        ))

        if event.kind in {"done", "error"}:
            error_category = event.error_category
            terminal_seen = True
            break

    if final is None:
        final = ModelResponse(
            stop_reason="error",
            error_message="model stream ended without a terminal event",
        )
    elif not terminal_seen:
        final = replace(
            final,
            stop_reason="error",
            error_message="model stream ended without a terminal event",
        )
    stop_reason = final.stop_reason
    if is_context_overflow(final):
        stop_reason = "length"
    final = replace(final, usage=final.usage.with_model_cost(model))

    error_message = final.error_message
    if stop_reason == "error" and error_category == "arrearage":
        error_message = (
            "API 调用失败：账户欠费或配额用尽。"
            "请检查 API key 的余额/账单状态后重试。"
        )
    assistant = assistant_message(
        final.text or error_message or None,
        thinking=final.thinking,
        thinking_signature=final.thinking_signature,
        tool_calls=final.tool_calls or None,
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=final.usage.as_dict(),
        stop_reason=stop_reason,
        error_message=error_message,
        timestamp=int(time.time() * 1000),
    )
    if not message_started:
        sink.process_event(MessageStartEvent(
            message=assistant,
            source="assistant",
            turn_index=iteration,
        ))
    usage = final.usage.as_dict()
    return assistant, stop_reason, usage


def _execute_tool_calls(
    *,
    call_list: tuple[ToolCallContent, ...],
    tools: ToolRegistry,
    messages: list[AgentMessage],
    emit: Emit,
    sink: AgentEventSink,
    turn_index: int,
    result: TraceResult,
    interrupt: threading.Event | None,
    agent_context: AgentContext,
    _tool_errors: dict[str, int] | None = None,
    _error_streak: list | None = None,
    governor: Any = None,
    hooks: LoopHooks | None = None,
    before_tool_call: BeforeToolCall | None = None,
    after_tool_call: AfterToolCall | None = None,
    tool_execution: ToolExecutionMode = "parallel",
    approval_broker: Any = None,
    trace_id: str = "",
    session_id: str = "",
) -> ExecutedToolBatch:
    from lsm_harness.agent.tools import AbortHandle

    # Bridge the loop's interrupt signal into the tool-level AbortHandle so a
    # tool polling ``_abort.aborted`` mid-run sees a user interrupt, not only
    # its own timeout.
    abort_handle = AbortHandle(external=interrupt)

    ctx = ExecutionContext(
        abort=abort_handle,
        turn_id=trace_id,
        session_id=session_id,
        interrupt=interrupt,
        approval_broker=approval_broker,
        emit=emit,
        event_sink=sink,
    )

    parsed: list[tuple[str, str, dict[str, Any]]] = [
        (call.id, call.name, dict(call.arguments))
        for call in call_list
    ]
    # Hooks still receive the OpenAI-shaped raw call (legacy contract of
    # BeforeToolCallContext.tool_call); rebuild it from the typed call.
    raw_calls: dict[str, dict[str, Any]] = {
        call.id: {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False),
            },
        }
        for call in call_list
    }

    for call_id, name, args in parsed:
        tool = tools._tools.get(name)
        emit("tool.requested", {
            "tool": name,
            "label": tool.display_label if tool is not None else name,
            "tool_call_id": call_id,
            "args": args,
        })
        invoke_tool_request(hooks, name, args, emit)

    if interrupt and interrupt.is_set():
        emit("loop.aborted", {})
        return ExecutedToolBatch(messages=[])

    def invoke_before_tool_call(
        _name: str,
        args: dict[str, Any],
        tool_call: dict[str, Any],
    ) -> str | None:
        if before_tool_call is None:
            return None
        decision = before_tool_call(BeforeToolCallContext(
            assistant_message=messages[-1],
            tool_call=tool_call,
            args=args,
            context=agent_context,
        ))
        if decision is None or not decision.block:
            return None
        return decision.reason or "Tool call blocked by beforeToolCall."

    def invoke_after_tool_call(
        _name: str,
        args: dict[str, Any],
        tool_call: dict[str, Any],
        tool_result,
    ):
        if after_tool_call is None:
            return tool_result
        update = after_tool_call(AfterToolCallContext(
            assistant_message=messages[-1],
            tool_call=tool_call,
            args=args,
            result=tool_result,
            is_error=tool_result.is_error,
            context=agent_context,
        ))
        if update is None:
            return tool_result
        if not isinstance(update, AfterToolCallResult):
            raise TypeError(
                "afterToolCall must return AfterToolCallResult or None"
            )
        return type(tool_result)(
            output=update.output if update.output is not None else tool_result.output,
            is_error=(
                update.is_error
                if update.is_error is not None
                else tool_result.is_error
            ),
            terminate=(
                update.terminate
                if update.terminate is not None
                else tool_result.terminate
            ),
            details=(
                update.details
                if update.details is not None
                else tool_result.details
            ),
        )

    emit("loop.tools.batch", {
        "count": len(parsed),
        "execution": tool_execution,
    })
    batch_results = tools.execute_batch(
        parsed,
        ctx=ctx,
        mode=tool_execution,
        before_tool_call=invoke_before_tool_call,
        after_tool_call=invoke_after_tool_call,
        raw_calls=raw_calls,
    )
    tool_messages: list[AgentMessage] = []

    for (call_id, name, args), (_, _, tool_result) in zip(parsed, batch_results):
        # Pi parity: finalized results are NEVER discarded here.  An
        # interrupt during collection is handled after the batch (the loop's
        # post-turn abort check); dropping results would orphan executions
        # whose side effects already happened and leave dangling tool_calls.
        output = tool_result.output
        status = "error" if tool_result.is_error else "ok"
        details = dict(tool_result.details or {})

        if governor is not None and not tool_result.is_error:
            if governor.is_self_truncating(name):
                # read/exec/grep already applied their Pi-style truncation
                # inside the tool — the governor never cuts them again;
                # details record WHICH layer governed the result (§7.3).
                details["governed"] = "tool"
            else:
                output = governor.manage_result(name, output)
                details["governed"] = "governor"

        registered_tool = tools._tools.get(name)
        label = registered_tool.display_label if registered_tool is not None else name
        record: dict[str, Any] = {
            "tool": name,
            "label": label,
            "tool_call_id": call_id,
            "args": args,
            "output": output,
            "is_error": tool_result.is_error,
        }
        if details:
            record["details"] = details
        result.tool_calls.append(record)
        emit("tool.completed", {**record, "status": status})
        tool_message = tool_result_message(
            replace(
                tool_result,
                output=output,
                tool_call_id=call_id,
                tool_name=name,
                details=details or tool_result.details,
            )
        )
        # The sink appends on message_end (Pi: append ownership is the
        # event processor's). tool_messages collects the Turn's results.
        sink.process_event(MessageStartEvent(
            message=tool_message, source="tool", turn_index=turn_index,
        ))
        sink.process_event(MessageEndEvent(
            message=tool_message, source="tool", turn_index=turn_index,
        ))
        tool_messages.append(tool_message)

        # ── hook: tool result ──────────────────────────
        if tool_result.is_error:
            invoke_tool_error(hooks, name, output[:200], emit)
        else:
            from lsm_harness.agent.hooks import invoke_tool_result
            invoke_tool_result(hooks, name, output[:200], False, emit)
            # 工具成功即重置连续失败 streak:streak 只衡量"连续"。
            if _error_streak is not None:
                _error_streak[0] = None
                _error_streak[1] = 0

        if tool_result.is_error and _tool_errors is not None:
            _tool_errors[name] = _tool_errors.get(name, 0) + 1
            # 熔断按「同工具 + 同错误签名」的连续 streak:不同原因/不同
            # 命令的失败互不累计(2026-09-12 实机:grep 参数错、sqlite3
            # 策略拒绝等三件不同的事曾烧断同一个按工具计数的熔断器)。
            if _error_streak is not None:
                signature = _tool_error_signature(name, args, output)
                if _error_streak[0] == signature:
                    _error_streak[1] += 1
                else:
                    _error_streak[0] = signature
                    _error_streak[1] = 1
                if _error_streak[1] >= 3:
                    kind = _classify_tool_error(output)
                    emit("loop.repeated_tool_error", {
                        "tool": name, "count": _error_streak[1],
                        "error_kind": kind,
                    })
                    result.reply = (
                        f"工具 '{name}' 连续 {_error_streak[1]} 次以相同方式失败"
                        f"({kind})。最后一个错误：{output[:200]}"
                    )
                    result.status = "failed"
                    result.stop_reason = "error"
                    result.error = result.reply
                    # Keep recording the rest of this already-executed batch.
                    # Stopping here loses sibling results and their side effects
                    # from the session tree. The loop fails after closing the Turn.

    # Pi shouldTerminateToolBatch: the inner loop stops only when the batch
    # is non-empty and EVERY finalized result asks to terminate.  A mixed
    # batch (terminate + normal/error result) continues — the model still
    # has work to reconcile.
    terminate = bool(batch_results) and all(
        tool_result.terminate for _, _, tool_result in batch_results
    )
    return ExecutedToolBatch(messages=tool_messages, terminate=terminate)


# ── termination helpers ───────────────────────────────────────────

def _classify_tool_error(output: str) -> str:
    """工具失败的粗分类(熔断签名与诊断用):policy / args /
    exit_code / timeout / not_found / error。"""
    first = output.split("\n", 1)[0]
    if "not allowed by the shell policy" in output:
        return "policy"
    if "invalid arguments" in first or "preparing arguments" in first:
        return "args"
    if "exited with code" in first:
        return "exit_code"
    if "timed out" in first:
        return "timeout"
    if "not found" in first:
        return "not_found"
    return "error"


def _tool_error_signature(name: str, args: dict, output: str) -> tuple:
    """连续失败熔断签名:同工具 + 同错误类别 + 同错误行(+ 同参数)。

    策略拒绝不带参数(被禁的是命令本身,错误行已含命令名)——换
    查询语句重试同一条被禁命令仍算同一签名,盲试会被熔断;普通
    非零退出带参数,不同的命令/参数组合互不累计。
    """
    kind = _classify_tool_error(output)
    first = output.split("\n", 1)[0][:80]
    if kind == "policy":
        return (name, kind, first)
    try:
        args_key = json.dumps(
            args, sort_keys=True, ensure_ascii=False, default=str
        )[:120]
    except (TypeError, ValueError):
        args_key = repr(args)[:120]
    return (name, kind, first, args_key)


def _termination_message(
    tool_calls: list[dict[str, Any]],
    max_iterations: int,
) -> str:
    """到达显式 max_iterations 的固定报告。

    这是迭代上限,与工具健康无关——"反复失败"只能来自 _error_streak
    熔断路径(同工具同签名连续 ≥3,那里有专属消息);整次 Trace 的
    累计错误数会把"不同原因、且随后成功"的调用误判成连续失败。
    """
    tools_used = [t["tool"] for t in tool_calls]
    if tools_used:
        return (
            f"达到最大迭代次数 ({max_iterations})，已执行工具："
            f"{', '.join(tools_used[-5:])}。请缩小请求范围后重试。"
        )
    return f"达到最大迭代次数 ({max_iterations})，任务尚未完成。请缩小请求范围后重试。"


def _reject_truncated_calls(
    assistant: AssistantMessage,
    emit: Emit,
    sink: AgentEventSink,
    turn_index: int,
) -> list[AgentMessage]:
    messages: list[AgentMessage] = []
    call_count = len(assistant.tool_calls)
    if call_count:
        for call in assistant.tool_calls:
            sink.process_event(ToolExecutionStartEvent(
                tool_call_id=call.id, tool_name=call.name,
                label=call.name, args=dict(call.arguments),
            ))
            result = ToolResultEnvelope(
                output=(
                    "这个工具调用没有被执行：模型输出达到 token 上限，"
                    "参数可能被截断。请将任务拆分为更小的步骤后重新发起调用。"
                ),
                is_error=True,
                tool_call_id=call.id,
                tool_name=call.name,
            )
            sink.process_event(ToolExecutionEndEvent(
                tool_call_id=call.id, tool_name=call.name,
                label=call.name, is_error=True, result=result,
            ))
            message = tool_result_message(result)
            sink.process_event(MessageStartEvent(
                message=message, source="tool", turn_index=turn_index,
            ))
            sink.process_event(MessageEndEvent(
                message=message, source="tool", turn_index=turn_index,
            ))
            messages.append(message)
        emit("loop.truncation_rejected", {"rejected_tool_calls": call_count})
    else:
        emit("loop.truncation_warning", {"message": "模型输出因 token 限制被截断"})
    return messages
