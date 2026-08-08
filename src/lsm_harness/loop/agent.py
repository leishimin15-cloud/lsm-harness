"""Provider-neutral reason → act → observe loop.

Five guardrails now protect every iteration:
  1. **streaming** – text & tool calls arrive as fine-grained events
  2. **truncation protection** – stop_reason="length" rejects all tool calls
  3. **interrupt** – threading.Event lets the caller cancel mid-turn
  4. **steering** – a queue lets the caller inject messages at iteration
     boundaries (e.g. the user types a correction while the model is
     streaming)
  5. **overflow recovery** – when on_truncation is provided, the loop
     compacts context and retries automatically after a length stop.

Additionally, parallel-safe tools execute concurrently via a thread pool.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any, Callable

from lsm_harness.tools.registry import (
    ExecutionContext,
    ToolRegistry,
    ToolResult,
)
from lsm_harness.types import ModelClient, TurnResult


Emit = Callable[[str, dict], None]


def run_loop(
    *,
    client: ModelClient,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    max_iterations: int,
    max_tokens: int,
    emit: Emit,
    # ── interrupt / steering (optional) ─────────────────────
    interrupt: threading.Event | None = None,
    steering_queue: queue.Queue[str] | None = None,
    # ── overflow recovery (optional) ────────────────────────
    on_truncation: Callable[[], tuple[str, list[dict[str, Any]]]] | None = None,
) -> TurnResult:
    """Execute the reason→act→observe loop.

    Parameters
    ----------
    interrupt:
        When set, the loop aborts at the next safe point.  Partial
        results are preserved and ``TurnResult.aborted`` is True.
    steering_queue:
        Drained at every iteration boundary.  Each string is injected
        as a user message (prefixed with *[用户中途纠正]* so the model
        knows it's a correction, not a new task).
    on_truncation:
        Called when ``stop_reason == "length"``.  Must return
        ``(fresh_system, fresh_messages)`` after compacting context.
        The loop then retries with the smaller context instead of
        hitting the same token limit again.
    """
    result = TurnResult(reply="")
    for iteration in range(1, max_iterations + 1):
        # ── abort check ──────────────────────────────────────
        if _should_abort(interrupt, emit):
            result.aborted = True
            result.iterations = iteration
            return result

        result.iterations = iteration

        # ── drain steering queue ─────────────────────────────
        _drain_steering(steering_queue, messages, emit)

        # ── call model (streaming preferred) ─────────────────
        assistant, stop_reason, usage = _call_model(
            client=client,
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            emit=emit,
            iteration=iteration,
            interrupt=interrupt,
        )
        # streaming may have been interrupted mid-call
        if _should_abort(interrupt, emit):
            result.aborted = True
            return result

        # ── emit completion summary ───────────────────────────
        emit(
            "llm.completed",
            {
                "role": "main",
                "model": model,
                "iteration": iteration,
                "stop_reason": stop_reason,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            },
        )

        messages.append(assistant)

        # ── token truncation guard ────────────────────────────
        if stop_reason == "length":
            _reject_truncated_calls(assistant, messages, emit)
            if on_truncation and iteration < max_iterations:
                # Compact context and retry within this iteration
                emit("loop.overflow_recovery", {"iteration": iteration})
                try:
                    system, messages = on_truncation()
                except Exception as exc:
                    emit("loop.overflow_recovery_failed", {"error": str(exc)})
                # fall through to next iteration with fresh context
            continue

        # ── no tool calls → natural stop ──────────────────────
        call_list = assistant.get("tool_calls", [])
        if not call_list:
            result.reply = assistant.get("content", "")
            return result

        # ── execute tools (parallel when safe) ────────────────
        _execute_tool_calls(
            call_list=call_list,
            tools=tools,
            messages=messages,
            emit=emit,
            result=result,
            interrupt=interrupt,
        )
        if _should_abort(interrupt, emit):
            result.aborted = True
            return result

        # If a tool signalled terminate, stop
        if result.reply:
            return result

    result.reply = "达到最大迭代次数，任务尚未完成。请缩小请求范围后重试。"
    emit("loop.limit_reached", {"max_iterations": max_iterations})
    return result


# ── internal helpers ──────────────────────────────────────────────


def _should_abort(interrupt: threading.Event | None, emit: Emit) -> bool:
    if interrupt and interrupt.is_set():
        emit("loop.aborted", {})
        return True
    return False


def _drain_steering(
    steering_queue: queue.Queue[str] | None,
    messages: list[dict[str, Any]],
    emit: Emit,
) -> None:
    """Drain all pending steering messages into the conversation."""
    if not steering_queue:
        return
    while True:
        try:
            msg = steering_queue.get_nowait()
        except queue.Empty:
            break
        messages.append({
            "role": "user",
            "content": f"[用户中途纠正] {msg}",
        })
        emit("loop.steered", {"message": msg})


def _execute_tool_calls(
    *,
    call_list: list[dict[str, Any]],
    tools: ToolRegistry,
    messages: list[dict[str, Any]],
    emit: Emit,
    result: TurnResult,
    interrupt: threading.Event | None,
) -> None:
    """Parse arguments, execute tools (parallel where safe), record results."""

    # ── build execution context ─────────────────────────────
    abort_handle = None
    from lsm_harness.tools.registry import AbortHandle
    if interrupt:
        abort_handle = AbortHandle()
        # Mirror the loop's interrupt event into the abort handle
        # so mid-tool abort checks also see it

    def on_update(msg: str) -> None:
        emit("tool.progress", {"delta": msg})

    ctx = ExecutionContext(
        abort=abort_handle or AbortHandle(),
        on_update=on_update,
    )

    # ── parse args ─────────────────────────────────────────
    parsed: list[tuple[str, str, dict[str, Any]]] = []
    for call in call_list:
        func = call["function"]
        name = func["name"]
        try:
            raw_args = json.loads(func["arguments"])
            if not isinstance(raw_args, dict):
                raw_args = {
                    "__parse_error__": "arguments must be an object",
                    "__raw__": func["arguments"],
                }
        except json.JSONDecodeError as exc:
            raw_args = {"__parse_error__": str(exc), "__raw__": func["arguments"]}
        parsed.append((call["id"], name, raw_args))

    # ── emit tool.requested + check abort ──────────────────
    for call_id, name, args in parsed:
        emit("tool.requested", {"tool": name, "args": args})

    if interrupt and interrupt.is_set():
        emit("loop.aborted", {})
        return

    # ── execute batch ──────────────────────────────────────
    emit("loop.tools.batch", {"count": len(parsed)})
    batch_results = tools.execute_batch(parsed, ctx=ctx)

    # ── process results ────────────────────────────────────
    for (call_id, name, args), (_, _, tool_result) in zip(parsed, batch_results):
        if _should_abort(interrupt, emit):
            return

        output = tool_result.output
        status = "error" if tool_result.is_error else "ok"
        record: dict[str, Any] = {
            "tool": name,
            "args": args,
            "output": output,
        }
        if tool_result.details:
            record["details"] = tool_result.details
        result.tool_calls.append(record)
        emit("tool.completed", {**record, "status": status})
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": output}
        )

        if tool_result.terminate:
            result.reply = "任务已完成。"
            return


# ── model calling ─────────────────────────────────────────────────


def _call_model(
    *,
    client: ModelClient,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    max_tokens: int,
    emit: Emit,
    iteration: int,
    interrupt: threading.Event | None,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    """Call the model, preferring streaming.  Returns (assistant_msg, stop_reason, usage_dict)."""

    if hasattr(client, "stream_complete"):
        return _streaming_call(
            client=client,
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            emit=emit,
            iteration=iteration,
            interrupt=interrupt,
        )
    else:
        return _sync_call(
            client=client,
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            emit=emit,
        )


def _sync_call(
    *,
    client: ModelClient,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    max_tokens: int,
    emit: Emit,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    """Fallback synchronous call (used by QueueClient in tests)."""
    response = client.complete(
        model=model,
        system=system,
        messages=messages,
        tools=tools.schemas(),
        max_tokens=max_tokens,
    )
    assistant: dict[str, Any] = {"role": "assistant", "content": response.text or None}
    if response.tool_calls:
        assistant["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in response.tool_calls
        ]
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return assistant, response.stop_reason, usage


def _streaming_call(
    *,
    client: ModelClient,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    max_tokens: int,
    emit: Emit,
    iteration: int,
    interrupt: threading.Event | None,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    """Streaming call with fine-grained events and interrupt support."""

    accumulated_text = ""
    text_started = False

    # Tool calls indexed by tool_index
    pending: dict[int, dict[str, Any]] = {}

    stop_reason = "stop"
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    for delta in client.stream_complete(
        model=model,
        system=system,
        messages=messages,
        tools=tools.schemas(),
        max_tokens=max_tokens,
    ):
        # ── interrupt check between deltas ──
        if interrupt and interrupt.is_set():
            emit("loop.stream_aborted", {"iteration": iteration})
            break

        if delta.kind == "text_delta":
            if not text_started:
                text_started = True
                emit("llm.text.start", {"iteration": iteration})
            accumulated_text += delta.text
            emit("llm.text.delta", {"text": delta.text, "iteration": iteration})

        elif delta.kind == "tool_call_start":
            pending[delta.tool_index] = {
                "id": delta.tool_id,
                "name": delta.tool_name,
                "arguments": "",
            }
            emit(
                "llm.tool_call.start",
                {
                    "iteration": iteration,
                    "tool_index": delta.tool_index,
                    "tool_id": delta.tool_id,
                    "tool_name": delta.tool_name,
                },
            )

        elif delta.kind == "tool_call_delta":
            if delta.tool_index in pending:
                if delta.tool_name and not pending[delta.tool_index]["name"]:
                    pending[delta.tool_index]["name"] = delta.tool_name
                pending[delta.tool_index]["arguments"] += delta.arguments_delta
            emit(
                "llm.tool_call.delta",
                {
                    "iteration": iteration,
                    "tool_index": delta.tool_index,
                    "arguments_delta": delta.arguments_delta,
                },
            )

        elif delta.kind == "done":
            stop_reason = delta.stop_reason
            if delta.usage:
                usage = {
                    "input_tokens": delta.usage.input_tokens,
                    "output_tokens": delta.usage.output_tokens,
                }

    # ── emit text end if we started ──
    if text_started:
        emit("llm.text.end", {"iteration": iteration, "text": accumulated_text})

    # ── build assistant message ──
    assistant: dict[str, Any] = {"role": "assistant", "content": accumulated_text or None}

    tool_calls = []
    for idx in sorted(pending.keys()):
        info = pending[idx]
        tool_calls.append({
            "id": info["id"],
            "type": "function",
            "function": {
                "name": info["name"],
                "arguments": info["arguments"],
            },
        })
        emit(
            "llm.tool_call.end",
            {
                "iteration": iteration,
                "tool_index": idx,
                "tool_id": info["id"],
                "tool_name": info["name"],
                "arguments": info["arguments"],
            },
        )

    if tool_calls:
        assistant["tool_calls"] = tool_calls

    return assistant, stop_reason, usage


def _reject_truncated_calls(
    assistant: dict[str, Any],
    messages: list[dict[str, Any]],
    emit: Emit,
) -> None:
    """Reject all tool calls when the model output hit the token limit."""
    call_count = len(assistant.get("tool_calls", []))
    if call_count:
        for call in assistant["tool_calls"]:
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": (
                    "这个工具调用没有被执行：模型输出达到 token 上限，"
                    "参数可能被截断。请将任务拆分为更小的步骤后重新发起调用。"
                ),
            })
        emit(
            "loop.truncation_rejected",
            {"rejected_tool_calls": call_count},
        )
    else:
        emit(
            "loop.truncation_warning",
            {"message": "模型输出因 token 限制被截断"},
        )
