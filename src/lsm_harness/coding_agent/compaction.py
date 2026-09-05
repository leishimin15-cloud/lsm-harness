"""Compaction algorithms (pi chapter 9).

The Session class owns persistence and LLM calls; this module owns the pure
decisions:

- :func:`should_compact` — the red line: ``tokens > budget - reserve``
- :func:`find_cut_point` — walk backwards accumulating tokens, cut at the
  nearest *valid* cut point (recent context matters most)
- :func:`find_turn_start` — locate the user message that opened a turn,
  for split-turn (turnPrefix) handling
- file tracking — extract ``<read-files>`` / ``<modified-files>`` from tool
  calls and merge them across successive compactions

Since batch B the session tree records every message individually
(user / assistant / toolResult), so assistant cut points are possible and
a cut may split a turn — Pi's split-turn / turnPrefix case applies here.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

# Token reserve behind the red line: space kept free for the model's reply
# and the summary it is about to write (Pi: reserveTokens, default 16384 —
# our budget is an order of magnitude smaller, so is the reserve).
DEFAULT_RESERVE_TOKENS = 2048

BRANCH_SUMMARY_MAX_TOKENS = 2048

_READ_TOOLS = {"read_file"}
_MODIFIED_TOOLS = {"write_file"}

_FILE_TAG_RE = {
    "read": re.compile(r"<read-files>\s*(.*?)\s*</read-files>", re.DOTALL),
    "modified": re.compile(r"<modified-files>\s*(.*?)\s*</modified-files>", re.DOTALL),
}

# A cut point is the FIRST KEPT row.  Valid cut points are user and
# assistant rows — never a tool result, which must stay with the assistant
# message whose call produced it (Pi: findValidCutPoints).
VALID_CUT_ROLES = frozenset({"user", "assistant"})


def should_compact(
    context_tokens: int,
    budget_tokens: int,
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
) -> bool:
    """The red line: context filled past ``budget - reserve`` must compact."""
    return context_tokens > max(1, budget_tokens - reserve_tokens)


def find_cut_point(
    rows: Sequence[Any],
    keep_recent_tokens: int,
    estimate,
) -> int:
    """Index of the first row that *survives* compression.

    Walk from the newest row backwards, accumulating tokens until the keep
    budget is covered, then take the first VALID cut point at or after the
    stop position (user / assistant rows only — a tool result is never a
    cut point).  ``0`` means "everything is kept" (nothing to compress).

    An assistant cut point may split a turn; the caller detects that with
    :func:`find_turn_start` and handles the prefix separately (turnPrefix).
    """
    if not rows:
        return 0
    valid = [i for i, row in enumerate(rows) if row["role"] in VALID_CUT_ROLES]
    if not valid:
        return 0
    accumulated = 0
    for i in range(len(rows) - 1, -1, -1):
        accumulated += 4 + estimate(str(rows[i]["content"]))
        if accumulated >= keep_recent_tokens:
            # First valid cut point at or after the stop position; if the
            # stop fell past the last valid point (trailing tool rows),
            # fall back to the last valid one.
            return next((cp for cp in valid if cp >= i), valid[-1])
    return 0  # the whole history fits the keep budget: nothing to compress


def find_turn_start(rows: Sequence[Any], cut_index: int) -> int:
    """Index of the user row that opened the turn containing ``cut_index``.

    Returns ``-1`` when the cut row IS a user row (no split) or when no
    user row precedes it in range (the turn start is already compacted).
    """
    if cut_index <= 0 or rows[cut_index]["role"] == "user":
        return -1
    for i in range(cut_index - 1, -1, -1):
        if rows[i]["role"] == "user":
            return i
    return -1


def _iter_file_calls(entry: Any):
    """Yield (tool_name, path) pairs from one JSONL entry.

    Assistant entries carry typed ``ToolCallContent``.  Fused tool-call
    records written before batch B (``meta["_v1_tool_calls"]``) are no
    longer migrated — old sessions lose file tracking, not readability.
    """
    message = getattr(entry, "message", None)
    for call in getattr(message, "tool_calls", None) or ():
        yield str(getattr(call, "name", "")), (getattr(call, "arguments", None) or {})


def extract_file_operations(
    entries: Sequence[Any],
    from_chat_id: int,
    to_chat_id: int,
) -> tuple[list[str], list[str]]:
    """Collect file paths touched by tool calls in a chat-id range.

    Reads structured ``tool_calls`` off JSONL message entries — read tools
    feed ``<read-files>``, write tools feed ``<modified-files>``.
    """
    read_files: list[str] = []
    modified_files: list[str] = []
    for entry in entries:
        if getattr(entry, "type", None) != "message":
            continue
        meta = getattr(entry, "meta", None) or {}
        chat_id = meta.get("chat_id")
        # Entries written live by the recorder carry no chat_id yet (the
        # chat_log projection is written after the run) — range filters only
        # apply to entries that DO carry one (backfilled / pre-batch-B).
        if chat_id is not None and not (from_chat_id < int(chat_id) <= to_chat_id):
            continue
        for tool, args in _iter_file_calls(entry):
            path = args.get("path")
            if not isinstance(path, str) or not path:
                continue
            if tool in _READ_TOOLS and path not in read_files:
                read_files.append(path)
            if tool in _MODIFIED_TOOLS and path not in modified_files:
                modified_files.append(path)
    return read_files, modified_files


def parse_file_tags(summary: str) -> tuple[list[str], list[str]]:
    """Recover accumulated file lists from a previous summary's tags."""
    lists: list[list[str]] = []
    for kind in ("read", "modified"):
        match = _FILE_TAG_RE[kind].search(summary or "")
        files = [line.strip() for line in match.group(1).splitlines()] if match else []
        lists.append([f for f in files if f])
    return lists[0], lists[1]


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """Render the trailing file-tracking tags appended to a summary."""
    parts: list[str] = []
    if read_files:
        parts.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        parts.append(
            "<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>"
        )
    return "\n\n".join(parts)


def merge_file_lists(
    previous: tuple[list[str], list[str]],
    new: tuple[list[str], list[str]],
) -> tuple[list[str], list[str]]:
    """Union two (read, modified) list pairs, preserving order."""
    merged: list[list[str]] = []
    for prev_list, new_list in zip(previous, new):
        combined = list(prev_list)
        combined.extend(path for path in new_list if path not in combined)
        merged.append(combined)
    return merged[0], merged[1]


__all__ = [
    "BRANCH_SUMMARY_MAX_TOKENS",
    "DEFAULT_RESERVE_TOKENS",
    "VALID_CUT_ROLES",
    "extract_file_operations",
    "find_cut_point",
    "find_turn_start",
    "format_file_operations",
    "merge_file_lists",
    "parse_file_tags",
    "should_compact",
]
