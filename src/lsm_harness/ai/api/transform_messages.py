"""Request-only replay normalization, modelled on Pi's transformMessages.

Durable history keeps failures and original provider metadata. Only the
request snapshot is changed when replaying it to a different model.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace

from lsm_harness.ai.messages import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
)
from lsm_harness.ai.types import Model


def transform_messages(messages: list[Message], model: Model) -> list[Message]:
    transformed: list[Message] = []
    call_ids: dict[str, str] = {}
    for message in messages:
        if isinstance(message, UserMessage):
            if "image" not in model.input_modalities and isinstance(message.content, tuple):
                blocks = []
                previous_image = False
                for block in message.content:
                    if isinstance(block, ImageContent):
                        if not previous_image:
                            blocks.append(TextContent("(image omitted: model does not support images)"))
                        previous_image = True
                    else:
                        blocks.append(block)
                        previous_image = (
                            isinstance(block, TextContent)
                            and block.text == "(image omitted: model does not support images)"
                        )
                message = replace(message, content=tuple(blocks))
            transformed.append(message)
        elif isinstance(message, AssistantMessage):
            same_model = (
                message.provider == model.provider
                and message.api == model.api
                and message.model == model.id
            )
            # Legacy hand-built messages without source metadata remain valid.
            if not same_model and (message.provider or message.api or message.model):
                message = replace(
                    message,
                    text="\n".join(part for part in (message.thinking, message.text) if part),
                    thinking="", thinking_signature="",
                )
            calls = []
            for call in message.tool_calls:
                normalized = call.id
                if (
                    not same_model
                    and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", call.id)
                ):
                    normalized = "call_" + hashlib.sha256(call.id.encode()).hexdigest()[:48]
                call_ids[call.id] = normalized
                calls.append(replace(call, id=normalized))
            transformed.append(replace(message, tool_calls=tuple(calls)))
        elif isinstance(message, ToolResultMessage):
            transformed.append(replace(
                message,
                tool_call_id=call_ids.get(message.tool_call_id, message.tool_call_id),
            ))
        else:
            raise TypeError(f"unsupported message for replay: {type(message).__name__}")

    # Provider requests must never contain an assistant tool call without a
    # matching result. Durable history remains unchanged; only this snapshot
    # receives synthetic error results. Failed/aborted assistant messages are
    # incomplete turns and are omitted entirely.
    result: list[Message] = []
    pending_calls: tuple[ToolCallContent, ...] = ()
    existing_result_ids: set[str] = set()

    def append_missing_results() -> None:
        nonlocal pending_calls, existing_result_ids
        for call in pending_calls:
            if call.id not in existing_result_ids:
                result.append(ToolResultMessage(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    content="No result provided",
                    is_error=True,
                ))
        pending_calls = ()
        existing_result_ids = set()

    for message in transformed:
        if isinstance(message, AssistantMessage):
            append_missing_results()
            if message.stop_reason in {"error", "aborted"}:
                continue
            pending_calls = message.tool_calls
            result.append(message)
        elif isinstance(message, ToolResultMessage):
            existing_result_ids.add(message.tool_call_id)
            result.append(message)
        elif isinstance(message, UserMessage):
            append_missing_results()
            result.append(message)
    append_missing_results()
    return result
