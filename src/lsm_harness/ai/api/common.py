"""Shared helpers for provider event translators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from lsm_harness.ai.types import ModelResponse, StopReason, ToolCall, Usage


@dataclass
class PendingToolCall:
    id: str
    name: str = ""
    arguments: str = ""


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be an object")
        return parsed
    except (json.JSONDecodeError, ValueError) as exc:
        return {"__parse_error__": str(exc), "__raw__": raw}


def snapshot(
    *,
    text: str,
    thinking: str,
    pending: dict[int, PendingToolCall],
    stop_reason: StopReason = "stop",
    usage: Usage | None = None,
    error_message: str = "",
    thinking_signature: str = "",
) -> ModelResponse:
    return ModelResponse(
        text=text,
        thinking=thinking,
        thinking_signature=thinking_signature,
        tool_calls=[
            ToolCall(
                id=item.id,
                name=item.name,
                arguments=parse_tool_arguments(item.arguments),
            )
            for _, item in sorted(pending.items())
        ],
        stop_reason=stop_reason,
        usage=usage or Usage(),
        error_message=error_message,
    )


def is_aborted(interrupt: Any) -> bool:
    return bool(interrupt is not None and interrupt.is_set())
