"""Provider-safe repair for an interrupted trailing tool batch.

Pi writes an assistant tool-call message before tool execution finishes.  A
process crash can therefore leave the durable transcript ending after only a
subset of the batch's tool results.  Before the transcript is reused, append a
deterministic error result for every missing call in that trailing batch.

The function intentionally repairs only the active tail.  Earlier malformed
history requires a branch rewrite rather than silently moving persisted
messages; accepting that broader corruption would hide a damaged session.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from lsm_harness.agent.messages import (
    AgentMessage,
    AgentToolResultMessage,
    AssistantMessage,
    ToolCallContent,
    ToolResultMessage,
)

INTERRUPTED_TOOL_RESULT = "Tool call interrupted before completion"


@dataclass(frozen=True)
class ToolHistoryRepair:
    """Provider-safe transcript plus deterministic repair diagnostics."""

    messages: tuple[AgentMessage, ...]
    changed: bool = False
    synthesized_results: int = 0
    dropped_orphan_results: int = 0
    dropped_duplicate_results: int = 0
    reordered_results: int = 0

    def diagnostic_data(self) -> dict[str, int]:
        return {
            "synthesized_results": self.synthesized_results,
            "dropped_orphan_results": self.dropped_orphan_results,
            "dropped_duplicate_results": self.dropped_duplicate_results,
            "reordered_results": self.reordered_results,
        }


def repair_tool_history(messages: Sequence[AgentMessage]) -> ToolHistoryRepair:
    """Ensure every tool call has exactly one adjacent result.

    Valid adjacent pairs are reserved first, which also makes repeated call IDs
    deterministic. Existing non-adjacent results are moved beside their call;
    missing results are synthesized; orphan and duplicate results are dropped.
    """
    source = tuple(messages)
    calls: list[tuple[tuple[int, int], ToolCallContent, int]] = []
    for message_index, message in enumerate(source):
        if not isinstance(message, AssistantMessage):
            continue
        for call_offset, call in enumerate(message.tool_calls, start=1):
            calls.append(
                ((message_index, call_offset), call, message_index + call_offset)
            )

    results_by_id: dict[str, list[tuple[int, ToolResultMessage]]] = defaultdict(list)
    for message_index, message in enumerate(source):
        if isinstance(message, ToolResultMessage):
            results_by_id[message.tool_call_id].append((message_index, message))

    selected: dict[
        tuple[int, int], tuple[int | None, ToolResultMessage]
    ] = {}
    used_positions: set[int] = set()

    for occurrence, call, expected_position in calls:
        if expected_position >= len(source):
            continue
        candidate = source[expected_position]
        if (
            isinstance(candidate, ToolResultMessage)
            and candidate.tool_call_id == call.id
            and expected_position not in used_positions
        ):
            selected[occurrence] = (expected_position, candidate)
            used_positions.add(expected_position)

    synthesized = 0
    for occurrence, call, _expected_position in calls:
        if occurrence in selected:
            continue
        candidates = [
            candidate
            for candidate in results_by_id.get(call.id, ())
            if candidate[0] not in used_positions
        ]
        if candidates:
            after_call = [item for item in candidates if item[0] > occurrence[0]]
            pool = after_call or candidates
            chosen = next(
                (
                    item
                    for item in pool
                    if not _is_interruption_result(item[1])
                ),
                pool[0],
            )
            selected[occurrence] = chosen
            used_positions.add(chosen[0])
            continue
        selected[occurrence] = (
            None,
            AgentToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=INTERRUPTED_TOOL_RESULT,
                is_error=True,
                details={"repair": "interrupted_tool_call"},
            ),
        )
        synthesized += 1

    # Prefer a real duplicate over a previously selected synthetic result.
    for occurrence, call, _expected_position in calls:
        selected_position, selected_result = selected[occurrence]
        if selected_position is None or not _is_interruption_result(selected_result):
            continue
        replacement = next(
            (
                item
                for item in results_by_id.get(call.id, ())
                if item[0] not in used_positions
                and not _is_interruption_result(item[1])
            ),
            None,
        )
        if replacement is not None:
            used_positions.remove(selected_position)
            used_positions.add(replacement[0])
            selected[occurrence] = replacement

    repaired: list[AgentMessage] = []
    reordered = 0
    for message_index, message in enumerate(source):
        if isinstance(message, ToolResultMessage):
            continue
        repaired.append(message)
        if not isinstance(message, AssistantMessage):
            continue
        for call_offset, _call in enumerate(message.tool_calls, start=1):
            position, result = selected[(message_index, call_offset)]
            repaired.append(result)
            if position is not None and position != message_index + call_offset:
                reordered += 1

    call_ids = {call.id for _occurrence, call, _expected in calls}
    unused = [
        result
        for results in results_by_id.values()
        for position, result in results
        if position not in used_positions
    ]
    orphans = sum(result.tool_call_id not in call_ids for result in unused)
    duplicates = sum(result.tool_call_id in call_ids for result in unused)
    repaired_messages = tuple(repaired)
    return ToolHistoryRepair(
        messages=repaired_messages,
        changed=repaired_messages != source,
        synthesized_results=synthesized,
        dropped_orphan_results=orphans,
        dropped_duplicate_results=duplicates,
        reordered_results=reordered,
    )


def _is_interruption_result(message: ToolResultMessage) -> bool:
    return message.is_error and message.content == INTERRUPTED_TOOL_RESULT


def interrupted_tool_results(
    messages: Sequence[AgentMessage],
) -> list[AgentToolResultMessage]:
    """Return missing results for the final, partially completed tool batch."""
    if not messages:
        return []

    index = len(messages) - 1
    returned_ids: set[str] = set()
    while index >= 0 and isinstance(messages[index], ToolResultMessage):
        returned_ids.add(messages[index].tool_call_id)
        index -= 1

    if index < 0 or not isinstance(messages[index], AssistantMessage):
        return []
    assistant = messages[index]
    if not assistant.tool_calls:
        return []

    return [
        AgentToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            content=INTERRUPTED_TOOL_RESULT,
            is_error=True,
            details={"repair": "interrupted_tool_call"},
        )
        for call in assistant.tool_calls
        if call.id not in returned_ids
    ]


__all__ = [
    "INTERRUPTED_TOOL_RESULT",
    "ToolHistoryRepair",
    "interrupted_tool_results",
    "repair_tool_history",
]
