"""Two-tier message system: typed inside, strict at the LLM boundary.

Agent code works with :data:`AgentMessage` — a typed union of the AI-layer
standard messages plus agent extensions::

    AgentMessage = UserMessage
                 | AssistantMessage
                 | ToolResultMessage        (may carry agent extras)
                 | CustomMessage            (registered app types)

Agent-only data is expressed through explicit types, never ad-hoc dict
keys: :class:`AgentToolResultMessage` adds ``details`` / ``terminate`` to
the standard tool result; :class:`CustomMessage` carries application
payloads in ``fields``.  At the LLM boundary,
:func:`default_convert_to_llm` performs an EXHAUSTIVE whitelist
conversion to the strict :data:`ai.messages.Message` union — unknown or
unregistered shapes raise instead of leaking to the provider.

``message_from_legacy`` / ``message_to_legacy`` adapt the dict-shaped
edges that batch A deliberately keeps (session storage, chat_log, TUI);
batch B removes them as the Session Tree takes over persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Union

from lsm_harness.ai.api.common import parse_tool_arguments
from lsm_harness.ai.messages import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ToolCallContent,
    ToolResultMessage,
    UserContentBlock,
    UserMessage,
)

if TYPE_CHECKING:  # avoid a runtime import cycle: tools -> events -> messages
    from lsm_harness.agent.tools import ToolResultMessage as ToolResultEnvelope

CUSTOM_ROLE = "custom"


# ---------------------------------------------------------------------------
# Agent extensions on top of the standard messages.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentToolResultMessage(ToolResultMessage):
    """Standard tool result + agent-only execution metadata.

    ``details`` (UI/trace payload) and ``terminate`` (loop control) never
    cross the LLM boundary: ``default_convert_to_llm`` rebuilds a plain
    :class:`ToolResultMessage` from this.
    """

    details: Any = None
    terminate: bool = False


@dataclass(frozen=True)
class CustomMessage:
    """Application-defined message (inner tier only).

    Structured payload lives in ``fields``; at the LLM boundary the
    registered ``to_llm`` translator turns it into a standard Message
    (or it is filtered out when excluded from context).

    ``exclude_from_context=None`` leaves the flag unset so the type-level
    registry default applies; an explicit bool overrides it.
    """

    custom_type: str
    content: Any = ""
    fields: Mapping[str, Any] = field(default_factory=dict)
    exclude_from_context: bool | None = None
    role: Literal["custom"] = CUSTOM_ROLE


AgentMessage = Union[
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    CustomMessage,
]

# Same-tier vs cross-tier transform protocols (pi ch6 §五).
TransformContext = Callable[[list[AgentMessage]], list[AgentMessage]]
ConvertToLlm = Callable[[list[AgentMessage]], list[Message]]


# ---------------------------------------------------------------------------
# Typed constructors — the single construction points for context messages.
# ---------------------------------------------------------------------------


def _user_content(content: Any) -> str | tuple[UserContentBlock, ...]:
    """Normalize user content: str passes through, block lists become typed."""
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return str(content)
    blocks: list[UserContentBlock] = []
    for item in content:
        if isinstance(item, (TextContent, ImageContent)):
            blocks.append(item)
        elif isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "text":
                blocks.append(TextContent(text=str(item.get("text", ""))))
            elif item_type == "image_url":
                image_url = item.get("image_url", {})
                url = (
                    image_url.get("url", "")
                    if isinstance(image_url, dict)
                    else str(image_url)
                )
                blocks.append(ImageContent(url=url))
            else:
                blocks.append(TextContent(text=str(item)))
        else:
            blocks.append(TextContent(text=str(item)))
    return tuple(blocks)


def _tool_call_content(call: Any) -> ToolCallContent:
    """Normalize a tool call: typed blocks pass through, legacy OpenAI
    dicts (``function.name`` / JSON-string ``arguments``) are parsed."""
    if isinstance(call, ToolCallContent):
        return call
    if isinstance(call, dict):
        function = call.get("function", {})
        if function:
            return ToolCallContent(
                id=str(call.get("id", "")),
                name=str(function.get("name", "")),
                arguments=parse_tool_arguments(str(function.get("arguments", ""))),
            )
        return ToolCallContent(
            id=str(call.get("id", "")),
            name=str(call.get("name", "")),
            arguments=dict(call.get("arguments") or {}),
        )
    # ai.types.ToolCall (id/name/arguments dataclass)
    return ToolCallContent(
        id=str(call.id),
        name=str(call.name),
        arguments=dict(call.arguments),
    )


def user_message(content: Any) -> UserMessage:
    """Build a standard user message."""
    return UserMessage(content=_user_content(content))


def assistant_message(
    content: str | None = None,
    *,
    thinking: str | None = None,
    thinking_signature: str = "",
    tool_calls: list[Any] | tuple[Any, ...] | None = None,
) -> AssistantMessage:
    """Build a standard assistant message.

    ``thinking`` (and its provider ``thinking_signature``) stay on the
    message so the next turn can echo signed reasoning blocks back to
    providers that require them.
    """
    return AssistantMessage(
        text=content or "",
        thinking=thinking or "",
        thinking_signature=thinking_signature,
        tool_calls=tuple(_tool_call_content(c) for c in tool_calls or ()),
    )


def tool_result_message(result: ToolResultEnvelope) -> AgentToolResultMessage:
    """Fold the Chapter-5 ToolResultMessage envelope into a context message.

    The only construction point for tool messages. ``details`` /
    ``terminate`` are inner-tier fields: they serve trace and UI, and are
    stripped at the LLM boundary.
    """
    return AgentToolResultMessage(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        content=result.output,
        is_error=result.is_error,
        details=result.details,
        terminate=result.terminate,
    )


def custom_message(
    custom_type: str,
    content: Any,
    *,
    exclude_from_context: bool | None = None,
    **fields: Any,
) -> CustomMessage:
    """Build an application-defined message (inner tier only)."""
    return CustomMessage(
        custom_type=custom_type,
        content=content,
        fields=fields,
        exclude_from_context=exclude_from_context,
    )


# ---------------------------------------------------------------------------
# Custom message registry — Python's counterpart of TS declaration merging.
# The core package ships an empty slot; the app layer registers its types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CustomMessageType:
    """Registration spec for one custom message kind."""

    name: str
    to_llm: Callable[[CustomMessage], Message] | None = None
    exclude_from_context: bool = False
    render: Callable[[CustomMessage], str] | None = None


_registry: dict[str, CustomMessageType] = {}


def register_custom_message_type(spec: CustomMessageType) -> None:
    """Register an application message type into the core slot."""
    if spec.name in _registry:
        raise ValueError(f"custom message type already registered: {spec.name}")
    _registry[spec.name] = spec


def get_custom_message_type(name: str) -> CustomMessageType | None:
    """Look up a registered custom message type."""
    return _registry.get(name)


def clear_custom_message_registry() -> None:
    """Reset the registry. Test fixture only."""
    _registry.clear()


def is_custom_message(message: AgentMessage) -> bool:
    """Whether a message uses the application extension slot."""
    return isinstance(message, CustomMessage)


def is_excluded_from_context(message: AgentMessage) -> bool:
    """Message-level exclude flag wins over the type-level default."""
    if not isinstance(message, CustomMessage):
        return False
    if message.exclude_from_context is not None:
        return message.exclude_from_context
    spec = get_custom_message_type(message.custom_type)
    return bool(spec and spec.exclude_from_context)


def message_preview(message: AgentMessage, limit: int = 200) -> str:
    """Plain-text preview of any agent message (legacy event adapter, UI)."""
    if isinstance(message, AssistantMessage):
        return message.text[:limit]
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content[:limit]
    if isinstance(content, tuple):
        text = " ".join(b.text for b in content if isinstance(b, TextContent))
        return text[:limit]
    return str(content)[:limit]


# ---------------------------------------------------------------------------
# Cross-tier translation (provider-neutral). Provider protocol translation
# stays in the ai/ layer — these are two independent translation layers.
# ---------------------------------------------------------------------------


def default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Translate inner-tier messages to the strict LLM union.

    Runs at every LLM call, after ``transform_context``.  Exhaustive
    whitelist — there is no silent fallback:

    1. standard messages pass through (agent extras on tool results are
       stripped by rebuilding the base :class:`ToolResultMessage`);
    2. ``exclude_from_context=True`` custom messages are dropped (they
       stay in agent state for UI/persistence, the LLM never sees them);
    3. custom messages translate via their registered ``to_llm`` —
       an unregistered type or an illegal return value is an ERROR at
       this boundary, never a best-effort pass-through.
    """
    llm_messages: list[Message] = []
    for message in messages:
        if isinstance(message, CustomMessage):
            if is_excluded_from_context(message):
                continue
            spec = get_custom_message_type(message.custom_type)
            if spec is None or spec.to_llm is None:
                raise ValueError(
                    "no LLM translator registered for custom message type: "
                    f"{message.custom_type}"
                )
            converted = spec.to_llm(message)
            if not isinstance(
                converted, (UserMessage, AssistantMessage, ToolResultMessage)
            ):
                raise TypeError(
                    f"custom translator for '{message.custom_type}' returned "
                    f"illegal message type: {type(converted).__name__}"
                )
            llm_messages.append(converted)
            continue
        if isinstance(message, ToolResultMessage):
            # Rebuild the base type: strips AgentToolResultMessage extras
            # (details/terminate) even for subclasses defined elsewhere.
            llm_messages.append(
                ToolResultMessage(
                    tool_call_id=message.tool_call_id,
                    tool_name=message.tool_name,
                    content=message.content,
                    is_error=message.is_error,
                )
            )
            continue
        if isinstance(message, (UserMessage, AssistantMessage)):
            llm_messages.append(message)
            continue
        raise TypeError(
            f"unknown AgentMessage type: {type(message).__name__}"
        )
    return llm_messages


# ---------------------------------------------------------------------------
# Legacy dict adapters — the dict-shaped edges batch A deliberately keeps
# (session storage / chat_log / TUI). Batch B removes them as the Session
# Tree becomes the persistence layer.
# ---------------------------------------------------------------------------


def message_from_legacy(data: Mapping[str, Any]) -> AgentMessage:
    """Convert a legacy dict message into the typed union."""
    role = data.get("role")
    if role == "user":
        return UserMessage(content=_user_content(data.get("content", "")))
    if role == "assistant":
        return AssistantMessage(
            text=data.get("content") or "",
            thinking=data.get("thinking") or "",
            thinking_signature=data.get("thinking_signature") or "",
            tool_calls=tuple(
                _tool_call_content(c) for c in data.get("tool_calls") or ()
            ),
        )
    if role == "tool":
        return AgentToolResultMessage(
            tool_call_id=str(data.get("tool_call_id", "")),
            tool_name=str(data.get("tool_name", "")),
            content=str(data.get("content", "")),
            is_error=bool(data.get("is_error", False)),
            details=data.get("details"),
            terminate=bool(data.get("terminate", False)),
        )
    if role == CUSTOM_ROLE:
        # Fields may arrive nested (message_to_legacy's "fields" key) or
        # spread at the top level (older stored dicts) — accept both.
        extras = {
            key: value
            for key, value in data.items()
            if key
            not in ("role", "custom_type", "content", "exclude_from_context", "fields")
        }
        nested = data.get("fields")
        if isinstance(nested, Mapping):
            extras = {**nested, **extras}
        return CustomMessage(
            custom_type=str(data.get("custom_type", "")),
            content=data.get("content"),
            fields=extras,
            exclude_from_context=data.get("exclude_from_context"),
        )
    raise ValueError(f"cannot convert legacy message with role: {role!r}")


def message_to_legacy(message: AgentMessage) -> dict[str, Any]:
    """Convert a typed agent message back to the legacy dict shape."""
    if isinstance(message, CustomMessage):
        legacy: dict[str, Any] = {
            "role": CUSTOM_ROLE,
            "custom_type": message.custom_type,
            "content": message.content,
            **message.fields,
        }
        if message.exclude_from_context is not None:
            legacy["exclude_from_context"] = message.exclude_from_context
        return legacy
    if isinstance(message, ToolResultMessage):
        legacy = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": message.content,
            "is_error": message.is_error,
        }
        if isinstance(message, AgentToolResultMessage):
            legacy["details"] = message.details
            legacy["terminate"] = message.terminate
        return legacy
    if isinstance(message, AssistantMessage):
        legacy = {"role": "assistant", "content": message.text or None}
        if message.thinking:
            legacy["thinking"] = message.thinking
        if message.thinking_signature:
            legacy["thinking_signature"] = message.thinking_signature
        if message.tool_calls:
            legacy["tool_calls"] = [
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
        return legacy
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
    raise TypeError(f"unknown AgentMessage type: {type(message).__name__}")


def messages_from_legacy(messages: list[Mapping[str, Any] | AgentMessage]) -> list[AgentMessage]:
    """Boundary helper: dicts become typed, typed messages pass through."""
    return [
        message_from_legacy(m) if isinstance(m, Mapping) else m
        for m in messages
    ]


__all__ = [
    "AgentMessage",
    "AgentToolResultMessage",
    "AssistantMessage",
    "CUSTOM_ROLE",
    "ConvertToLlm",
    "CustomMessage",
    "CustomMessageType",
    "ImageContent",
    "Message",
    "TextContent",
    "ToolCallContent",
    "ToolResultMessage",
    "TransformContext",
    "UserMessage",
    "assistant_message",
    "clear_custom_message_registry",
    "custom_message",
    "default_convert_to_llm",
    "get_custom_message_type",
    "is_custom_message",
    "is_excluded_from_context",
    "message_from_legacy",
    "message_preview",
    "message_to_legacy",
    "messages_from_legacy",
    "register_custom_message_type",
    "tool_result_message",
    "user_message",
]
