"""Shared tool-output truncation (pi chapter 8, mechanism ①).

Dual limits — lines OR bytes, whichever hits first — plus two directions:

- ``truncate_head`` keeps the *beginning* (read_file: imports / signatures
  carry the most information density at the top of a file).
- ``truncate_tail`` keeps the *end* (shell output: error traces and final
  results carry the most signal at the bottom).

Byte counting is UTF-8 aware: we accumulate per-character encoded length, so
a multi-byte character (emoji, CJK) is never split.  In Python ``str``
slicing is already code-point based, but the *budget* must be measured in
bytes to match what the LLM actually pays for.

Every function returns a :class:`TruncationResult` carrying full metadata so
callers can render an honest notice ("[Showing lines X–Y of Z …]") instead of
silently dropping content.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
GREP_MAX_LINE_LENGTH = 500


@dataclass(frozen=True)
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: str | None          # "lines" | "bytes" | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    first_line_partial: bool = False  # truncate_head: last kept line was cut
    last_line_partial: bool = False   # truncate_tail: first kept line was cut


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _take_from_end(line: str, max_bytes: int) -> str:
    """Keep the tail of a single line within a byte budget, never splitting
    a multi-byte character."""
    kept: list[str] = []
    total = 0
    for char in reversed(line):
        char_bytes = _byte_len(char)
        if total + char_bytes > max_bytes:
            break
        kept.append(char)
        total += char_bytes
    return "".join(reversed(kept))


def _take_from_start(line: str, max_bytes: int) -> str:
    """Keep the head of a single line within a byte budget."""
    kept: list[str] = []
    total = 0
    for char in line:
        char_bytes = _byte_len(char)
        if total + char_bytes > max_bytes:
            break
        kept.append(char)
        total += char_bytes
    return "".join(kept)


def truncate_tail(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the *end* of ``content`` within both limits (shell output)."""
    lines = content.split("\n")
    total_bytes = _byte_len(content)

    kept: list[str] = []
    bytes_so_far = 0
    truncated_by: str | None = None
    for line in reversed(lines):
        line_bytes = _byte_len(line) + 1  # +1 for the newline
        if len(kept) >= max_lines:
            truncated_by = "lines"
            break
        if bytes_so_far + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        kept.insert(0, line)
        bytes_so_far += line_bytes

    last_line_partial = False
    if not kept and lines:
        # A single line alone exceeds the byte budget: returning empty would
        # mislead the model, so keep the tail of that line and flag it.
        kept = [_take_from_end(lines[-1], max_bytes)]
        truncated_by = "bytes"
        last_line_partial = True

    truncated = len(kept) < len(lines) or last_line_partial
    output = "\n".join(kept)
    return TruncationResult(
        content=output,
        truncated=truncated,
        truncated_by=truncated_by if truncated else None,
        total_lines=len(lines),
        total_bytes=total_bytes,
        output_lines=len(kept),
        output_bytes=_byte_len(output),
        last_line_partial=last_line_partial,
    )


def truncate_head(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the *beginning* of ``content`` within both limits (file reads)."""
    lines = content.split("\n")
    total_bytes = _byte_len(content)

    kept: list[str] = []
    bytes_so_far = 0
    truncated_by: str | None = None
    for line in lines:
        line_bytes = _byte_len(line) + 1
        if len(kept) >= max_lines:
            truncated_by = "lines"
            break
        if bytes_so_far + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        kept.append(line)
        bytes_so_far += line_bytes

    first_line_partial = False
    if not kept and lines:
        kept = [_take_from_start(lines[0], max_bytes)]
        truncated_by = "bytes"
        first_line_partial = True

    truncated = len(kept) < len(lines) or first_line_partial
    output = "\n".join(kept)
    return TruncationResult(
        content=output,
        truncated=truncated,
        truncated_by=truncated_by if truncated else None,
        total_lines=len(lines),
        total_bytes=total_bytes,
        output_lines=len(kept),
        output_bytes=_byte_len(output),
        first_line_partial=first_line_partial,
    )


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> str:
    """Cap a single line (grep hits on minified files can be huge)."""
    if len(line) <= max_chars:
        return line
    return line[:max_chars] + "... [truncated]"


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "GREP_MAX_LINE_LENGTH",
    "TruncationResult",
    "format_size",
    "truncate_head",
    "truncate_line",
    "truncate_tail",
]
