"""Provider-neutral contracts shared by the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)


@dataclass
class TurnResult:
    reply: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0


class ModelClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ModelResponse: ...

