"""AI-layer standard messages (Chapter 6, refactor batch A).

Provider-neutral, model-visible, field-strict. Three message types are the
ONLY things an LLM API ever sees::

    Message = UserMessage | AssistantMessage | ToolResultMessage

Agent-only data (``details``, ``terminate``, renderer payloads, custom
roles) lives one tier up in ``agent/messages.py`` and is stripped or
translated by ``default_convert_to_llm`` before anything reaches this
layer. Provider translators dispatch on these types exhaustively — an
unknown type is an error, never a silent pass-through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union


# ---------------------------------------------------------------------------
# Content blocks (Pi: TextContent / ImageContent / ThinkingContent / ToolCall)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextContent:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ImageContent:
    url: str
    media_type: str = ""
    type: Literal["image"] = "image"


@dataclass(frozen=True)
class ThinkingContent:
    thinking: str
    signature: str = ""
    type: Literal["thinking"] = "thinking"


@dataclass(frozen=True)
class ToolCallContent:
    """One model-requested tool call, arguments already parsed to an object."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    type: Literal["tool_call"] = "tool_call"


UserContentBlock = Union[TextContent, ImageContent]


# ---------------------------------------------------------------------------
# The three standard messages.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserMessage:
    """End-user (or tool-result-carrying) input; text or content blocks."""

    content: str | tuple[UserContentBlock, ...]
    role: Literal["user"] = "user"


@dataclass(frozen=True)
class AssistantMessage:
    """Model output: text, reasoning (with provider signature), tool calls.

    ``thinking_signature`` lets providers that sign reasoning blocks
    (Anthropic) receive them back verbatim on the next turn.
    """

    text: str = ""
    thinking: str = ""
    thinking_signature: str = ""
    tool_calls: tuple[ToolCallContent, ...] = ()
    # Pi keeps terminal/model metadata on the assistant message itself so a
    # persisted transcript is sufficient to reconstruct failures and usage.
    # Provider translators deliberately read only the model-visible fields.
    api: str = ""
    provider: str = ""
    model: str = ""
    usage: dict[str, int | float] = field(default_factory=dict)
    stop_reason: str = ""
    error_message: str = ""
    timestamp: int = 0
    role: Literal["assistant"] = "assistant"


@dataclass(frozen=True)
class ToolResultMessage:
    """Model-visible tool output. Agent execution metadata (``details`` /
    ``terminate``) is added one tier up by subclassing — never here."""

    tool_call_id: str
    tool_name: str = ""
    content: str = ""
    is_error: bool = False
    role: Literal["tool"] = "tool"


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]

STANDARD_MESSAGE_TYPES = (UserMessage, AssistantMessage, ToolResultMessage)


def message_text(message: Message) -> str:
    """Plain-text view of any standard message (previews, heuristics)."""
    if isinstance(message, AssistantMessage):
        return message.text
    if isinstance(message, ToolResultMessage):
        return message.content
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(block.text for block in content if isinstance(block, TextContent))


def message_to_wire(message: Message) -> dict[str, Any]:
    """OpenAI-flavoured dict form of a standard message.

    Used only by the legacy ``ModelClient`` facade (memory gate,
    consolidation, …): its ``complete()`` predates typed messages and
    still consumes dicts. Provider translators have their own exhaustive
    converters and never use this.
    """
    import json

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
        return {"role": "user", "content": blocks}
    if isinstance(message, AssistantMessage):
        wire: dict[str, Any] = {
            "role": "assistant",
            "content": message.text or None,
        }
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        return wire
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    raise TypeError(f"unknown standard message type: {type(message).__name__}")


__all__ = [
    "AssistantMessage",
    "ImageContent",
    "Message",
    "STANDARD_MESSAGE_TYPES",
    "TextContent",
    "ThinkingContent",
    "ToolCallContent",
    "ToolResultMessage",
    "UserContentBlock",
    "UserMessage",
    "message_text",
    "message_to_wire",
]
