"""Unified ``stream`` and convenience ``stream_simple`` entry points."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Iterator

from lsm_harness.ai.api.common import PendingToolCall, is_aborted, snapshot
from lsm_harness.ai.errors import categorize_error, is_retryable
from lsm_harness.ai.messages import message_to_wire
from lsm_harness.ai.models import (
    clamp_thinking_level,
    resolve_cache_retention,
    resolve_max_tokens,
)
from lsm_harness.ai.registry import resolve_api_provider
from lsm_harness.ai.types import (
    AIContext,
    AssistantMessageEvent,
    ErrorCategory,
    Model,
    ModelClient,
    ModelResponse,
    StopReason,
    StreamFunction,
    StreamOptions,
    Usage,
)


_SEMANTIC_EVENTS = {
    "text_start",
    "text_delta",
    "thinking_start",
    "thinking_delta",
    "toolcall_start",
    "toolcall_delta",
}


def stream(
    model: Model,
    context: AIContext,
    options: StreamOptions,
) -> Iterator[AssistantMessageEvent]:
    provider = resolve_api_provider(model.api)
    return provider.stream(model, context, options)


def stream_simple(
    model: Model,
    context: AIContext,
    options: StreamOptions,
) -> Iterator[AssistantMessageEvent]:
    """Apply capability fallback and safe pre-output retries."""
    reasoning = clamp_thinking_level(model, options.reasoning)
    effective = replace(
        options,
        max_tokens=resolve_max_tokens(model, options.max_tokens, reasoning),
        reasoning=reasoning,
        cache_retention=resolve_cache_retention(
            model,
            options.cache_retention,
        ),
    )

    provider = resolve_api_provider(model.api)
    return _stream_with_retries(provider.stream, model, context, effective)


def _stream_with_retries(
    stream_fn: StreamFunction,
    model: Model,
    context: AIContext,
    options: StreamOptions,
) -> Iterator[AssistantMessageEvent]:
    for attempt in range(options.max_retries + 1):
        buffered: list[AssistantMessageEvent] = []
        visible_output = False
        retry = False
        for event in stream_fn(model, context, options):
            if not visible_output and event.kind == "start":
                buffered.append(event)
                continue
            if event.kind in _SEMANTIC_EVENTS:
                visible_output = True
                yield from buffered
                buffered.clear()
                yield event
                continue
            if (
                event.kind == "error"
                and not visible_output
                and event.error_category is not None
                and is_retryable(event.error_category)
                and attempt < options.max_retries
            ):
                retry = True
                if options.on_retry is not None:
                    options.on_retry(
                        attempt + 1,
                        event.error_category,
                        event.partial.error_message,
                    )
                delay = 2.0 if event.error_category == "rate_limit" else 0.5 * (attempt + 1)
                time.sleep(delay)
                break
            yield from buffered
            buffered.clear()
            yield event
        if retry:
            continue
        yield from buffered
        return


def client_stream_function(client: ModelClient) -> StreamFunction:
    """Adapt the legacy ``ModelClient`` protocol to the canonical stream."""
    canonical = getattr(client, "stream", None)
    if callable(canonical):
        base: StreamFunction = canonical
    else:
        base = _legacy_client_stream(client)

    def run(
        model: Model,
        context: AIContext,
        options: StreamOptions,
    ) -> Iterator[AssistantMessageEvent]:
        reasoning = clamp_thinking_level(model, options.reasoning)
        effective = replace(
            options,
            max_tokens=resolve_max_tokens(
                model,
                options.max_tokens,
                reasoning,
            ),
            reasoning=reasoning,
            cache_retention=resolve_cache_retention(
                model,
                options.cache_retention,
            ),
        )
        yield from _stream_with_retries(base, model, context, effective)

    return run


def _legacy_client_stream(client: ModelClient) -> StreamFunction:
    def run(
        model: Model,
        context: AIContext,
        options: StreamOptions,
    ) -> Iterator[AssistantMessageEvent]:
        text = ""
        pending: dict[int, PendingToolCall] = {}
        usage = Usage()
        text_active = False
        try:
            yield AssistantMessageEvent(
                "start",
                snapshot(text="", thinking="", pending={}),
            )
            stream_complete = getattr(client, "stream_complete", None)
            if not callable(stream_complete):
                response = client.complete(
                    model=model.id,
                    system=context.system_prompt,
                    messages=[message_to_wire(m) for m in context.messages],
                    tools=context.tools,
                    max_tokens=options.max_tokens,
                )
                if response.text:
                    yield AssistantMessageEvent("text_start", ModelResponse())
                    yield AssistantMessageEvent(
                        "text_delta",
                        response,
                        text_delta=response.text,
                    )
                    yield AssistantMessageEvent("text_end", response)
                for index, call in enumerate(response.tool_calls):
                    raw = json.dumps(call.arguments, ensure_ascii=False)
                    pending[index] = PendingToolCall(call.id, call.name, raw)
                    partial = snapshot(
                        text=response.text,
                        thinking=response.thinking,
                        pending=pending,
                        stop_reason=response.stop_reason,
                        usage=response.usage,
                    )
                    yield AssistantMessageEvent(
                        "toolcall_start",
                        partial,
                        tool_index=index,
                        tool_id=call.id,
                        tool_name=call.name,
                    )
                    yield AssistantMessageEvent(
                        "toolcall_delta",
                        partial,
                        tool_index=index,
                        tool_id=call.id,
                        tool_name=call.name,
                        arguments_delta=raw,
                    )
                    yield AssistantMessageEvent(
                        "toolcall_end",
                        partial,
                        tool_index=index,
                        tool_id=call.id,
                        tool_name=call.name,
                    )
                terminal = "error" if response.stop_reason in {"error", "aborted"} else "done"
                category: ErrorCategory | None = None
                if response.stop_reason == "aborted":
                    category = "aborted"
                elif response.stop_reason == "error":
                    category = "permanent"
                yield AssistantMessageEvent(terminal, response, error_category=category)
                return

            yield from _consume_legacy_deltas(
                stream_complete(
                    model=model.id,
                    system=context.system_prompt,
                    messages=[message_to_wire(m) for m in context.messages],
                    tools=context.tools,
                    max_tokens=options.max_tokens,
                ),
                options,
            )
        except Exception as exc:
            reason: StopReason = "aborted" if is_aborted(options.interrupt) else "error"
            category: ErrorCategory = "aborted" if reason == "aborted" else categorize_error(exc)
            yield AssistantMessageEvent(
                "error",
                snapshot(
                    text=text,
                    thinking="",
                    pending=pending,
                    stop_reason=reason,
                    usage=usage,
                    error_message=str(exc),
                ),
                error_category=category,
            )

    return run


def _consume_legacy_deltas(
    deltas,
    options: StreamOptions,
) -> Iterator[AssistantMessageEvent]:
    text = ""
    text_active = False
    pending: dict[int, PendingToolCall] = {}
    usage = Usage()
    stop_reason: StopReason = "stop"
    for delta in deltas:
        if is_aborted(options.interrupt):
            partial = snapshot(
                text=text,
                thinking="",
                pending=pending,
                stop_reason="aborted",
                usage=usage,
                error_message="model request aborted",
            )
            yield AssistantMessageEvent("error", partial, error_category="aborted")
            return
        if delta.kind == "text_delta":
            if not text_active:
                text_active = True
                yield AssistantMessageEvent(
                    "text_start",
                    snapshot(text=text, thinking="", pending=pending),
                )
            text += delta.text
            yield AssistantMessageEvent(
                "text_delta",
                snapshot(text=text, thinking="", pending=pending),
                text_delta=delta.text,
            )
        elif delta.kind == "tool_call_start":
            pending[delta.tool_index] = PendingToolCall(
                delta.tool_id,
                delta.tool_name,
            )
            yield AssistantMessageEvent(
                "toolcall_start",
                snapshot(text=text, thinking="", pending=pending),
                tool_index=delta.tool_index,
                tool_id=delta.tool_id,
                tool_name=delta.tool_name,
            )
        elif delta.kind == "tool_call_delta":
            call = pending[delta.tool_index]
            if delta.tool_name and not call.name:
                call.name = delta.tool_name
            call.arguments += delta.arguments_delta
            yield AssistantMessageEvent(
                "toolcall_delta",
                snapshot(text=text, thinking="", pending=pending),
                tool_index=delta.tool_index,
                tool_id=call.id,
                tool_name=call.name,
                arguments_delta=delta.arguments_delta,
            )
        elif delta.kind == "done":
            stop_reason = delta.stop_reason or "stop"
            usage = delta.usage or Usage()
    if text_active:
        yield AssistantMessageEvent(
            "text_end",
            snapshot(text=text, thinking="", pending=pending),
        )
    for index, call in sorted(pending.items()):
        yield AssistantMessageEvent(
            "toolcall_end",
            snapshot(text=text, thinking="", pending=pending),
            tool_index=index,
            tool_id=call.id,
            tool_name=call.name,
        )
    if pending and stop_reason == "stop":
        stop_reason = "tool_calls"
    partial = snapshot(
        text=text,
        thinking="",
        pending=pending,
        stop_reason=stop_reason,
        usage=usage,
    )
    terminal = "error" if stop_reason in {"error", "aborted"} else "done"
    category = "aborted" if stop_reason == "aborted" else ("permanent" if stop_reason == "error" else None)
    yield AssistantMessageEvent(terminal, partial, error_category=category)


def collect_stream(events: Iterator[AssistantMessageEvent]) -> ModelResponse:
    final = ModelResponse(error_message="model stream ended without a terminal event")
    for event in events:
        final = event.partial
        if event.kind in {"done", "error"}:
            return final
    return replace(
        final,
        stop_reason="error",
        error_message="model stream ended without a terminal event",
    )


def fixed_stream_function(events: list[AssistantMessageEvent]) -> StreamFunction:
    """Small deterministic helper for protocol-level tests and examples."""
    def run(
        _model: Model,
        _context: AIContext,
        _options: StreamOptions,
    ) -> Iterator[AssistantMessageEvent]:
        yield from events

    return run
