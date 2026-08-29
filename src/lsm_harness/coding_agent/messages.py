"""Product-layer custom message registration (Chapter 6).

The core package ships an empty extension slot; this module is where the
coding-agent app fills it — the Python counterpart of Pi's
``declare module`` declaration merging.

Two custom types are registered:

- ``compaction_summary`` — replaces compacted history in context (ch9);
- ``branch_summary`` — the "last words" of an abandoned conversation
  branch, injected at the fork point (ch10).
"""

from __future__ import annotations

from lsm_harness.agent.messages import (
    AgentMessage,
    CustomMessage,
    CustomMessageType,
    UserMessage,
    custom_message,
    get_custom_message_type,
    register_custom_message_type,
)

COMPACTION_SUMMARY = "compaction_summary"
BRANCH_SUMMARY = "branch_summary"

# Pi's exact preambles: they frame how the LLM should treat each summary —
# compacted mainline history vs. a side exploration it merely returned from.
COMPACTION_PREAMBLE = (
    "The conversation history before this point was compacted "
    "into the following summary:"
)
BRANCH_SUMMARY_PREAMBLE = (
    "The user explored a different conversation branch before returning here.\n"
    "Summary of that exploration:"
)


def compaction_summary_message(
    summary: str,
    *,
    through_chat_id: int,
    version: int,
) -> AgentMessage:
    """Build a compaction-summary custom message.

    ``exclude_from_context`` stays unset: a summary exists precisely so the
    LLM can see it.
    """
    return custom_message(
        COMPACTION_SUMMARY,
        summary,
        through_chat_id=through_chat_id,
        version=version,
    )


def branch_summary_message(summary: str, *, from_id: str) -> AgentMessage:
    """Build a branch-summary custom message (an abandoned branch's gist)."""
    return custom_message(BRANCH_SUMMARY, summary, from_id=from_id)


def _compaction_summary_to_llm(message: CustomMessage) -> UserMessage:
    return UserMessage(
        content=(
            f"{COMPACTION_PREAMBLE}\n\n"
            f"<summary>\n{message.content}\n</summary>"
        )
    )


def _branch_summary_to_llm(message: CustomMessage) -> UserMessage:
    return UserMessage(
        content=(
            "The following is a summary of a branch that this conversation "
            "came back from:\n\n"
            f"<summary>\n{BRANCH_SUMMARY_PREAMBLE}\n\n"
            f"{message.content}\n</summary>"
        )
    )


def _compaction_summary_render(message: CustomMessage) -> str:
    return f"🗜 摘要（覆盖至 #{message.fields.get('through_chat_id')}）"


def _branch_summary_render(message: CustomMessage) -> str:
    return f"🌿 分支摘要（来自 {message.fields.get('from_id')}）"


def register_coding_agent_messages() -> None:
    """Register the app's custom message types into the core slot.

    Idempotent: ``Harness`` may be constructed many times per process.
    """
    if get_custom_message_type(COMPACTION_SUMMARY) is None:
        register_custom_message_type(
            CustomMessageType(
                name=COMPACTION_SUMMARY,
                to_llm=_compaction_summary_to_llm,
                render=_compaction_summary_render,
            )
        )
    if get_custom_message_type(BRANCH_SUMMARY) is None:
        register_custom_message_type(
            CustomMessageType(
                name=BRANCH_SUMMARY,
                to_llm=_branch_summary_to_llm,
                render=_branch_summary_render,
            )
        )


__all__ = [
    "BRANCH_SUMMARY",
    "BRANCH_SUMMARY_PREAMBLE",
    "COMPACTION_PREAMBLE",
    "COMPACTION_SUMMARY",
    "branch_summary_message",
    "compaction_summary_message",
    "register_coding_agent_messages",
]
