"""Unified ``stream`` and convenience ``stream_simple`` entry points."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Iterator

from lsm_harness.ai.api.common import PendingToolCall, is_aborted, snapshot
from lsm_harness.ai.errors import categorize_error, is_retryable
from lsm_harness.ai.models import (
    clamp_thinking_level,
    resolve_cache_retention,
    resolve_max_tokens,
)
from lsm_harness.ai.registry import resolve_api_provider
from lsm_harness.ai.types import (
    AIContext,
    AssistantMessageEvent,
    Model,
    ModelResponse,
    StreamFunction,
    StreamOptions,
)


_SEMANTIC_EVENTS = {
    "text_start",
    "text_delta",
    "thinking_start",
    "thinking_delta",
    "toolcall_start",
    "toolcall_delta",
    "text_end",
    "thinking_end",
    "toolcall_end",
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
        max_tokens=resolve_max_tokens(
            model,
            options.max_tokens,
            reasoning,
            options.thinking_budgets,
        ),
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
        if is_aborted(options.interrupt):
            yield AssistantMessageEvent(
                "error",
                ModelResponse(stop_reason="aborted", error_message="model request aborted"),
                error_category="aborted",
            )
            return
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
                if options.max_retry_delay_ms is not None:
                    delay = min(delay, max(0, options.max_retry_delay_ms) / 1000)
                # A cancelled backoff must not start another paid request.
                if options.interrupt is not None:
                    options.interrupt.wait(delay)
                else:
                    time.sleep(delay)
                break
            yield from buffered
            buffered.clear()
            yield event
            if event.kind in {"done", "error"}:
                return
        if retry:
            continue
        yield from buffered
        return


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


def response_stream_function(responder) -> StreamFunction:
    """Build a StreamFunction from a synchronous ``ModelResponse`` callable.

    Deterministic counterpart of :func:`fixed_stream_function` for scripted
    clients (smoke / eval / tests): ``responder(model, context, options)``
    returns one ModelResponse, which is expanded into the canonical event
    sequence.  Exceptions become categorized error events; pre-output
    retryable errors are retried via ``options.max_retries`` / ``on_retry``,
    same as :func:`stream_simple`.
    """

    def run(
        model: Model,
        context: AIContext,
        options: StreamOptions,
    ) -> Iterator[AssistantMessageEvent]:
        text = ""
        pending: dict[int, PendingToolCall] = {}
        try:
            yield AssistantMessageEvent(
                "start", snapshot(text="", thinking="", pending={})
            )
            response = responder(model, context, options)
            if response.text:
                text = response.text
                yield AssistantMessageEvent("text_start", response)
                yield AssistantMessageEvent(
                    "text_delta", response, text_delta=text
                )
            for index, call in enumerate(response.tool_calls):
                raw = json.dumps(call.arguments, ensure_ascii=False)
                pending[index] = PendingToolCall(call.id, call.name, raw)
                partial = snapshot(
                    text=text,
                    thinking=response.thinking,
                    pending=pending,
                    stop_reason=response.stop_reason,
                    usage=response.usage,
                )
                yield AssistantMessageEvent(
                    "toolcall_start", partial,
                    tool_index=index, tool_id=call.id, tool_name=call.name,
                )
                yield AssistantMessageEvent(
                    "toolcall_delta", partial,
                    tool_index=index, tool_id=call.id, tool_name=call.name,
                    arguments_delta=raw,
                )
                yield AssistantMessageEvent(
                    "toolcall_end", partial,
                    tool_index=index, tool_id=call.id, tool_name=call.name,
                )
            if text:
                yield AssistantMessageEvent("text_end", response)
            terminal = (
                "error" if response.stop_reason in {"error", "aborted"} else "done"
            )
            category = (
                "aborted" if response.stop_reason == "aborted"
                else "permanent" if response.stop_reason == "error"
                else None
            )
            yield AssistantMessageEvent(terminal, response, error_category=category)
        except Exception as exc:
            reason = "aborted" if is_aborted(options.interrupt) else "error"
            yield AssistantMessageEvent(
                "error",
                snapshot(
                    text=text,
                    thinking="",
                    pending=pending,
                    stop_reason=reason,
                    error_message=str(exc),
                ),
                error_category=(
                    "aborted" if reason == "aborted" else categorize_error(exc)
                ),
            )

    def streaming(
        model: Model,
        context: AIContext,
        options: StreamOptions,
    ) -> Iterator[AssistantMessageEvent]:
        yield from _stream_with_retries(run, model, context, options)

    return streaming
