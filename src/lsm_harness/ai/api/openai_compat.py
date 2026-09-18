"""OpenAI Chat Completions compatible request and stream translator."""

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
    Model,
    StopReason,
    StreamOptions,
    Usage,
    normalize_stop_reason,
)


def build_openai_request(
    model: Model,
    context: AIContext,
    options: StreamOptions,
) -> dict[str, Any]:
    messages = (
        [{"role": "system", "content": context.system_prompt}]
        if context.system_prompt
        else []
    )
    messages.extend(
        _translate_openai_message(message)
        for message in transform_messages(context.messages, model)
    )
    request: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "max_tokens": options.max_tokens,
        "stream": True,
        # Ask the API to attach usage to the final chunk; without this,
        # OpenAI-family streams omit usage and token accounting reads zero.
        "stream_options": {"include_usage": True},
    }
    if context.tools:
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": _tool_field(tool, "name"),
                    "description": _tool_field(tool, "description"),
                    "parameters": _tool_parameters(tool),
                },
            }
            for tool in context.tools
        ]
    reasoning = model.thinking_level_map.get(options.reasoning)
    if reasoning and model.thinking_format == "deepseek":
        request["extra_body"] = {
            "thinking": {"type": "disabled" if options.reasoning == "off" else "enabled"}
        }
    elif reasoning and model.thinking_format == "reasoning_effort":
        request["reasoning_effort"] = reasoning
    if model.cache_control_format == "openai" and options.session_id:
        request["prompt_cache_key"] = options.session_id
        if options.cache_retention == "long":
            request["prompt_cache_retention"] = "24h"
    return request


def _tool_field(tool: Any, field: str) -> Any:
    """Read canonical descriptors while accepting persisted legacy dicts."""
    return tool.get(field) if isinstance(tool, dict) else getattr(tool, field)


def _tool_parameters(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        return tool.get("parameters") or tool.get("input_schema") or {}
    return tool.parameters


def _translate_openai_message(message: Message) -> dict[str, Any]:
    """Exhaustive Message → OpenAI wire dict conversion.

    Standard messages only: Agent-only fields (``details``/``terminate``/
    ``thinking``) have already been stripped by ``convert_to_llm`` at the
    Agent/AI boundary, and anything that is not a standard Message is a
    programming error — fail loudly instead of guessing a shape.

    Reasoning text is not echoed back: OpenAI-family APIs treat
    ``reasoning_content`` as output-only.
    """
    if isinstance(message, UserMessage):
        content = message.content
        if isinstance(content, str):
            return {"role": "user", "content": content}
        blocks: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, TextContent):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContent):
                blocks.append({"type": "image_url", "image_url": {"url": block.url}})
            else:
                raise TypeError(
                    f"unsupported user content block: {type(block).__name__}"
                )
        return {"role": "user", "content": blocks}
    if isinstance(message, AssistantMessage):
        translated: dict[str, Any] = {
            "role": "assistant",
            "content": message.text or None,
        }
        if message.tool_calls:
            translated["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments, ensure_ascii=False
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        return translated
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    raise TypeError(
        f"unsupported message for OpenAI translator: {type(message).__name__}"
    )


def stream_openai_compat(
    model: Model,
    context: AIContext,
    options: StreamOptions,
) -> Iterator[AssistantMessageEvent]:
    try:
        from openai import OpenAI

        kwargs: dict[str, Any] = {
            "api_key": options.api_key,
            "timeout": options.timeout,
            "max_retries": 0,
        }
        if model.base_url:
            kwargs["base_url"] = model.base_url
        # 系统/环境代理不可达时直连兜底(否则全量 Connection refused)
        http_client = sdk_http_client()
        if http_client is not None:
            kwargs["http_client"] = http_client
        client = OpenAI(**kwargs)
        try:
            yield from stream_openai_client(client, model, context, options)
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


def stream_openai_client(
    client: Any,
    model: Model,
    context: AIContext,
    options: StreamOptions,
) -> Iterator[AssistantMessageEvent]:
    text = ""
    thinking = ""
    pending: dict[int, PendingToolCall] = {}
    text_active = False
    thinking_active = False
    stop_reason: StopReason = "stop"
    usage = Usage()
    started = False
    try:
        request = build_openai_request(model, context, options)
        if options.on_payload is not None:
            replacement = options.on_payload(dict(request), model)
            if replacement is not None:
                if not isinstance(replacement, dict):
                    raise TypeError("on_payload must return a dict or None")
                request = replacement
        response = client.chat.completions.create(**request)
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
        for chunk in response:
            if is_aborted(options.interrupt):
                raise InterruptedError("model request aborted")
            choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
            if choice is not None:
                delta = choice.delta
                reasoning_delta = (
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                    or ""
                )
                if reasoning_delta:
                    if not thinking_active:
                        thinking_active = True
                        yield AssistantMessageEvent(
                            "thinking_start",
                            snapshot(text=text, thinking=thinking, pending=pending),
                        )
                    thinking += reasoning_delta
                    yield AssistantMessageEvent(
                        "thinking_delta",
                        snapshot(text=text, thinking=thinking, pending=pending),
                        thinking_delta=reasoning_delta,
                    )
                content = getattr(delta, "content", None) or ""
                if content:
                    if not text_active:
                        text_active = True
                        yield AssistantMessageEvent(
                            "text_start",
                            snapshot(text=text, thinking=thinking, pending=pending),
                        )
                    text += content
                    yield AssistantMessageEvent(
                        "text_delta",
                        snapshot(text=text, thinking=thinking, pending=pending),
                        text_delta=content,
                    )
                for tool_call in getattr(delta, "tool_calls", None) or []:
                    index = tool_call.index
                    is_new = index not in pending
                    function = getattr(tool_call, "function", None)
                    name_delta = (
                        getattr(function, "name", None) or ""
                        if function is not None
                        else ""
                    )
                    arguments_delta = (
                        getattr(function, "arguments", None) or ""
                        if function is not None
                        else ""
                    )
                    if is_new:
                        pending[index] = PendingToolCall(
                            id=tool_call.id or "",
                            name=name_delta,
                        )
                        yield AssistantMessageEvent(
                            "toolcall_start",
                            snapshot(text=text, thinking=thinking, pending=pending),
                            tool_index=index,
                            tool_id=tool_call.id or "",
                            tool_name=name_delta,
                        )
                    else:
                        pending[index].name += name_delta
                    pending[index].arguments += arguments_delta
                    if arguments_delta or (name_delta and not is_new):
                        yield AssistantMessageEvent(
                            "toolcall_delta",
                            snapshot(text=text, thinking=thinking, pending=pending),
                            tool_index=index,
                            tool_id=pending[index].id,
                            tool_name=pending[index].name,
                            arguments_delta=arguments_delta,
                        )
                if getattr(choice, "finish_reason", None):
                    stop_reason = normalize_stop_reason(choice.finish_reason)
            raw_usage = getattr(chunk, "usage", None)
            if raw_usage is not None:
                prompt_details = getattr(
                    raw_usage, "prompt_tokens_details", None
                )
                cache_read = getattr(
                    prompt_details, "cached_tokens", 0
                ) if prompt_details is not None else 0
                cache_write = getattr(
                    prompt_details, "cache_write_tokens", 0
                ) if prompt_details is not None else 0
                prompt_tokens = getattr(raw_usage, "prompt_tokens", 0)
                usage = Usage(
                    input_tokens=max(
                        0, prompt_tokens - cache_read - cache_write
                    ),
                    output_tokens=getattr(raw_usage, "completion_tokens", 0),
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                )
        if thinking_active:
            yield AssistantMessageEvent(
                "thinking_end",
                snapshot(text=text, thinking=thinking, pending=pending),
            )
        if text_active:
            yield AssistantMessageEvent(
                "text_end",
                snapshot(text=text, thinking=thinking, pending=pending),
            )
        for index, call in sorted(pending.items()):
            yield AssistantMessageEvent(
                "toolcall_end",
                snapshot(text=text, thinking=thinking, pending=pending),
                tool_index=index,
                tool_id=call.id,
                tool_name=call.name,
            )
        if pending and stop_reason == "stop":
            stop_reason = "tool_calls"
        partial = snapshot(
            text=text,
            thinking=thinking,
            pending=pending,
            stop_reason=stop_reason,
            usage=usage,
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
        )
        if not started:
            yield AssistantMessageEvent("start", snapshot(text="", thinking="", pending={}))
        yield AssistantMessageEvent("error", partial, error_category=category)
