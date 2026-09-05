"""Context governance: tool result size management.

Two responsibilities:
  1. Tool result truncation    — cap individual tool outputs
  2. Large result offload      — persist oversized results to temp files
"""

from __future__ import annotations

import tempfile
from pathlib import Path


# ── configuration ──────────────────────────────────────────────────

# Tools that apply their own Pi-style truncation INSIDE the tool
# (truncate_head / truncate_tail / truncate_line with an escape-hatch
# notice).  The governor must not cut their output a second time —
# exactly one primary truncation per tool result (plan §7.3).
SELF_TRUNCATING_TOOLS = frozenset({"read_file", "exec", "grep"})


class GovernanceConfig:
    """Per-tool governance settings."""

    def __init__(
        self,
        max_result_chars: int = 4000,
        offload_threshold_chars: int = 12000,
        offload_dir: Path | None = None,
    ):
        self.max_result_chars = max_result_chars
        self.offload_threshold_chars = offload_threshold_chars
        self.offload_dir = offload_dir or Path(tempfile.gettempdir()) / "lsm-tool-results"

    def effective_max(self, tool_name: str) -> int:
        """Return the max chars to keep in context for a given tool.

        Tools that produce structured, reusable output (like read_file)
        get a larger budget.  Voluminous tools (like grep) get a
        smaller one.
        """
        generous = {"read_file", "web_search"}
        if tool_name in generous:
            return self.max_result_chars * 2
        return self.max_result_chars


# ── governance ─────────────────────────────────────────────────────


class ContextGovernor:
    """Manages tool result sizes before they enter the conversation.

    Usage in the agent loop::

        governor = ContextGovernor(home)

        # After each tool call:
        managed = governor.manage_result(tool_name, raw_output)
    """

    def __init__(self, config: GovernanceConfig | None = None, home: Path | None = None):
        self.config = config or GovernanceConfig()
        if home:
            self.config.offload_dir = home / "tool-results"
        self._offload_count = 0

    # ── tool result management ─────────────────────────────────

    @staticmethod
    def is_self_truncating(tool_name: str) -> bool:
        """Tools whose output is already Pi-truncated at the source."""
        return tool_name in SELF_TRUNCATING_TOOLS

    def manage_result(self, tool_name: str, raw_output: str) -> str:
        """Process a tool result before it enters the conversation.

        Strategy:
          - result <= max_result_chars        → pass through unchanged
          - max_result_chars < result <= offload_threshold
            → truncate with a note
          - result > offload_threshold
            → offload to temp file, return a placeholder
        """
        limit = self.config.effective_max(tool_name)
        if len(raw_output) <= limit:
            return raw_output

        if len(raw_output) <= self.config.offload_threshold_chars:
            return self._truncate(raw_output, limit, tool_name)

        return self._offload(raw_output, tool_name)

    def _truncate(self, output: str, limit: int, tool_name: str) -> str:
        """Truncate to limit and append a summary note."""
        head = output[:limit]
        tail = output[-200:] if len(output) > limit + 200 else ""
        omitted = len(output) - limit - (200 if tail else 0)

        note = (
            f"\n\n[... 已截断 {omitted} 个字符。"
            f"使用 read_file 或更精确的查询以获取完整内容。]"
        )
        if tail:
            return head + note + f"\n\n[... 末尾内容 ...]\n{tail}"
        return head + note

    def _offload(self, output: str, tool_name: str) -> str:
        """Write the full output to a temp file, return a placeholder."""
        self.config.offload_dir.mkdir(parents=True, exist_ok=True)
        self._offload_count += 1
        path = self.config.offload_dir / f"{tool_name}-{self._offload_count}.txt"

        path.write_text(output, encoding="utf-8")

        return (
            f"[工具输出已保存到 {path}，共 {len(output)} 个字符。]\n"
            f"[首 1000 字符预览]\n"
            f"{output[:1000]}\n"
            f"[... 剩余 {len(output) - 1000} 个字符。"
            f"使用 read_file('{path}') 读取完整内容。]"
        )
