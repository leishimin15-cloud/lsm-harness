"""Anthropic Messages request and stream translator."""

from __future__ import annotations

import json
from typing import Any, Iterator

from lsm_harness.ai.api.common import (
    PendingToolCall,
    is_aborted,
    sdk_http_client,
    snapshot,
)
from lsm_harness.ai.api.transform_messages import transform_messages
from lsm_harness.ai.errors import categorize_error
from lsm_harness.ai.messages import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from lsm_harness.ai.types import (
    AIContext,
    AssistantMessageEvent,
    CacheRetention,
    Model,
    StopReason,
    StreamOptions,
    Usage,
    normalize_stop_reason,
)


def _cache_control(retention: CacheRetention) -> dict[str, str] | None:
    if retention == "none":
        return None
    value = {"type": "ephemeral"}
    if retention == "long":
        value["ttl"] = "1h"
    return value


def _user_blocks(message: UserMessage) -> list[dict[str, Any]]:
    """UserMessage → Anthropic content blocks (text / image)."""
    content = message.content
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
            continue
        if isinstance(block, ImageContent):
            url = block.url
            if url.startswith("data:") and ";base64," in url:
                header, data = url.split(",", 1)
                media_type = block.media_type or header[5:].split(";", 1)[0]
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                })
            elif url:
                blocks.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })
            continue
        raise TypeError(f"unsupported user content block: {type(block).__name__}")
    return blocks


def _append_message(
    messages: list[dict[str, Any]],
    role: str,
    blocks: list[dict[str, Any]],
) -> None:
    if not blocks:
        return
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
        return
    messages.append({"role": role, "content": blocks})


def translate_anthropic_messages(
    messages: list[Message],
    cache_retention: CacheRetention = "none",
    *,
    allow_empty_thinking_signature: bool = False,
) -> list[dict[str, Any]]:
    """Exhaustive Message → Anthropic wire conversion.

    Anything that is not a standard Message is a programming error at the
    Agent/AI boundary — fail loudly instead of guessing a shape.
    """
    translated: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserMessage):
            _append_message(translated, "user", _user_blocks(message))
            continue
        if isinstance(message, AssistantMessage):
            blocks: list[dict[str, Any]] = []
            if message.thinking:
                if message.thinking_signature:
                    blocks.append({
                        "type": "thinking",
                        "thinking": message.thinking,
                        "signature": message.thinking_signature,
                    })
                elif allow_empty_thinking_signature:
                    blocks.append({
                        "type": "thinking",
                        "thinking": message.thinking,
                        "signature": "",
                    })
                else:
                    # Pi preserves an unsigned/aborted thinking block as
                    # ordinary text for providers that reject empty
                    # signatures; it does not silently lose the history.
                    blocks.append({"type": "text", "text": message.thinking})
            if message.text:
                blocks.append({"type": "text", "text": message.text})
            for call in message.tool_calls:
                blocks.append({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                })
            _append_message(translated, "assistant", blocks)
            continue
        if isinstance(message, ToolResultMessage):
            _append_message(translated, "user", [{
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
                "is_error": message.is_error,
            }])
            continue
        raise TypeError(
            f"unsupported message for Anthropic translator: "
            f"{type(message).__name__}"
        )

    cache = _cache_control(cache_retention)
    if cache:
        for message in reversed(translated):
            if message["role"] != "user":
                continue
            content = message["content"]
            if isinstance(content, list) and content:
                content[-1] = {**content[-1], "cache_control": cache}
            break
    return translated


def build_anthropic_request(
    model: Model,
    context: AIContext,
    options: StreamOptions,
) -> dict[str, Any]:
    cache = _cache_control(options.cache_retention)
    request: dict[str, Any] = {
        "model": model.id,
        "messages": translate_anthropic_messages(
            transform_messages(context.messages, model),
            options.cache_retention,
            allow_empty_thinking_signature=(
                model.allow_empty_thinking_signature
            ),
        ),
        "max_tokens": options.max_tokens,
    }
    if context.system_prompt:
        system_block: dict[str, Any] = {
            "type": "text",
            "text": context.system_prompt,
        }
        if cache:
            system_block["cache_control"] = cache
        request["system"] = [system_block]
    if context.tools:
        request["tools"] = [
            {
                "name": _tool_field(tool, "name"),
                "description": _tool_field(tool, "description"),
                "input_schema": _tool_parameters(tool),
            }
            for tool in context.tools
        ]
        if cache:
            request["tools"][-1]["cache_control"] = cache
    reasoning = model.thinking_level_map.get(options.reasoning)
    if model.force_adaptive_thinking and options.reasoning != "off":
        effort = reasoning
        if not isinstance(effort, str):
            effort = {
                "minimal": "low",
                "low": "low",
                "medium": "medium",
                "high": "high",
            }.get(options.reasoning, "high")
        request["thinking"] = {"type": "adaptive"}
        request["output_config"] = {"effort": effort}
        return request
    if reasoning:
        configured_budget = (
            options.thinking_budgets.get(options.reasoning)
            if options.thinking_budgets is not None
            else None
        )
        try:
            budget = configured_budget if configured_budget is not None else int(reasoning)
        except ValueError:
            request["thinking"] = {"type": "adaptive"}
        else:
            request["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
    return request


def _tool_field(tool: Any, field: str) -> Any:
    return tool.get(field) if isinstance(tool, dict) else getattr(tool, field)


def _tool_parameters(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        return tool.get("parameters") or tool.get("input_schema") or {}
    return tool.parameters


def stream_anthropic_messages(
    model: Model,
    context: AIContext,
    options: StreamOptions,
) -> Iterator[AssistantMessageEvent]:
    try:
        import anthropic

        kwargs: dict[str, Any] = {
            "timeout": options.timeout,
            "max_retries": 0,
        }
        kwargs[
            "auth_token" if model.auth_mode == "bearer" else "api_key"
        ] = options.api_key
        if model.base_url:
            kwargs["base_url"] = model.base_url
        if model.headers:
            kwargs["default_headers"] = dict(model.headers)
        # 系统/环境代理不可达时直连兜底(否则全量 Connection refused)
        http_client = sdk_http_client(anthropic.DefaultHttpxClient)
        if http_client is not None:
            kwargs["http_client"] = http_client
        client = anthropic.Anthropic(**kwargs)
        try:
            yield from stream_anthropic_client(client, model, context, options)
        finally:
            client.close()
    except Exception as exc:
        reason: StopReason = "aborted" if is_aborted(options.interrupt) else "error"
        category = "aborted" if reason == "aborted" else categorize_error(exc)
        partial = snapshot(
            text="",
            thinking="",
            pending={},
            stop_reason=reason,
            error_message=str(exc),
        )
        yield AssistantMessageEvent("error", partial, error_category=category)


def stream_anthropic_client(
    client: Any,
    model: Model,
    context: AIContext,
    options: StreamOptions,
) -> Iterator[AssistantMessageEvent]:
    text = ""
    thinking = ""
    thinking_signature = ""
    pending: dict[int, PendingToolCall] = {}
    active: dict[int, str] = {}
    stop_reason: StopReason = "stop"
    usage = Usage()
    started = False
    try:
        request = build_anthropic_request(model, context, options)
        if options.on_payload is not None:
            replacement = options.on_payload(dict(request), model)
            if replacement is not None:
                if not isinstance(replacement, dict):
                    raise TypeError("on_payload must return a dict or None")
                request = replacement
        with client.messages.stream(
            **request
        ) as response:
            if options.on_response is not None:
                raw_headers = getattr(response, "headers", None)
                options.on_response({
                    "status": getattr(response, "status_code", 200),
                    "headers": dict(raw_headers) if raw_headers is not None else {},
                }, model)
            started = True
            yield AssistantMessageEvent(
                "start",
                snapshot(text=text, thinking=thinking, pending=pending),
            )
            for event in response:
                if is_aborted(options.interrupt):
                    raise InterruptedError("model request aborted")
                event_type = getattr(event, "type", "")
                if event_type == "message_start":
                    raw_usage = getattr(getattr(event, "message", None), "usage", None)
                    if raw_usage is not None:
                        usage = Usage(
                            input_tokens=getattr(raw_usage, "input_tokens", 0),
                            cache_read_tokens=getattr(
                                raw_usage, "cache_read_input_tokens", 0
                            ),
                            cache_write_tokens=getattr(
                                raw_usage, "cache_creation_input_tokens", 0
                            ),
                        )
                elif event_type == "content_block_start":
                    index = event.index
                    block = event.content_block
                    block_type = getattr(block, "type", "")
                    active[index] = block_type
                    if block_type == "text":
                        yield AssistantMessageEvent(
                            "text_start",
                            snapshot(text=text, thinking=thinking, pending=pending),
                        )
                        initial = getattr(block, "text", "") or ""
                        if initial:
                            text += initial
                            yield AssistantMessageEvent(
                                "text_delta",
                                snapshot(text=text, thinking=thinking, pending=pending),
                                text_delta=initial,
                            )
                    elif block_type == "thinking":
                        yield AssistantMessageEvent(
                            "thinking_start",
                            snapshot(text=text, thinking=thinking, pending=pending),
                        )
                    elif block_type == "tool_use":
                        initial_input = getattr(block, "input", None) or {}
                        initial_json = json.dumps(initial_input, ensure_ascii=False) if initial_input else ""
                        pending[index] = PendingToolCall(
                            id=getattr(block, "id", ""),
                            name=getattr(block, "name", ""),
                            arguments=initial_json,
                        )
                        yield AssistantMessageEvent(
                            "toolcall_start",
                            snapshot(text=text, thinking=thinking, pending=pending),
                            tool_index=index,
                            tool_id=pending[index].id,
                            tool_name=pending[index].name,
                        )
                elif event_type == "content_block_delta":
                    index = event.index
                    delta = event.delta
                    delta_type = getattr(delta, "type", "")
                    if delta_type == "text_delta":
                        value = getattr(delta, "text", "") or ""
                        text += value
                        yield AssistantMessageEvent(
                            "text_delta",
                            snapshot(text=text, thinking=thinking, pending=pending),
                            text_delta=value,
                        )
                    elif delta_type == "thinking_delta":
                        value = getattr(delta, "thinking", "") or ""
                        thinking += value
                        yield AssistantMessageEvent(
                            "thinking_delta",
                            snapshot(text=text, thinking=thinking, pending=pending),
                            thinking_delta=value,
                        )
                    elif delta_type == "signature_delta":
                        thinking_signature += getattr(delta, "signature", "") or ""
                        yield AssistantMessageEvent(
                            "thinking_signature_delta",
                            snapshot(text=text, thinking=thinking, pending=pending),
                        )
                    elif delta_type == "input_json_delta":
                        value = getattr(delta, "partial_json", "") or ""
                        pending[index].arguments += value
                        yield AssistantMessageEvent(
                            "toolcall_delta",
                            snapshot(text=text, thinking=thinking, pending=pending),
                            tool_index=index,
                            tool_id=pending[index].id,
                            tool_name=pending[index].name,
                            arguments_delta=value,
                        )
                elif event_type == "content_block_stop":
                    index = event.index
                    block_type = active.pop(index, "")
                    if block_type == "text":
                        yield AssistantMessageEvent(
                            "text_end",
                            snapshot(text=text, thinking=thinking, pending=pending),
                        )
                    elif block_type == "thinking":
                        yield AssistantMessageEvent(
                            "thinking_end",
                            snapshot(text=text, thinking=thinking, pending=pending),
                        )
                    elif block_type == "tool_use":
                        call = pending[index]
                        yield AssistantMessageEvent(
                            "toolcall_end",
                            snapshot(text=text, thinking=thinking, pending=pending),
                            tool_index=index,
                            tool_id=call.id,
                            tool_name=call.name,
                        )
                elif event_type == "message_delta":
                    raw_delta = getattr(event, "delta", None)
                    raw_reason = getattr(raw_delta, "stop_reason", None)
                    if raw_reason:
                        stop_reason = normalize_stop_reason(raw_reason)
                    raw_usage = getattr(event, "usage", None)
                    if raw_usage is not None:
                        usage = Usage(
                            input_tokens=usage.input_tokens,
                            output_tokens=getattr(raw_usage, "output_tokens", 0),
                            cache_read_tokens=usage.cache_read_tokens,
                            cache_write_tokens=usage.cache_write_tokens,
                        )
        if pending and stop_reason == "stop":
            stop_reason = "tool_calls"
        partial = snapshot(
            text=text,
            thinking=thinking,
            pending=pending,
            stop_reason=stop_reason,
            usage=usage,
            thinking_signature=thinking_signature,
        )
        yield AssistantMessageEvent("done", partial)
    except Exception as exc:
        reason: StopReason = "aborted" if is_aborted(options.interrupt) else "error"
        if isinstance(exc, InterruptedError):
            reason = "aborted"
        category = "aborted" if reason == "aborted" else categorize_error(exc)
        partial = snapshot(
            text=text,
            thinking=thinking,
            pending=pending,
            stop_reason=reason,
            usage=usage,
            error_message=str(exc),
            thinking_signature=thinking_signature,
        )
        if not started:
            yield AssistantMessageEvent("start", snapshot(text="", thinking="", pending={}))
        yield AssistantMessageEvent("error", partial, error_category=category)
