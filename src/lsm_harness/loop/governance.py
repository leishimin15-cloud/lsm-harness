"""Context governance: tool result management and safe truncation.

Three responsibilities:
  1. Tool result truncation    — cap individual tool outputs
  2. Large result offload      — persist oversized results to temp files
  3. Safe truncation boundary  — never cut in the middle of a tool-call chain
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


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
        generous = {"read_file", "web_search", "search_documents"}
        if tool_name in generous:
            return self.max_result_chars * 2
        return self.max_result_chars


# ── governance ─────────────────────────────────────────────────────


class ContextGovernor:
    """Manages tool result sizes and ensures safe truncation boundaries.

    Usage in the agent loop::

        governor = ContextGovernor(home)

        # After each tool call:
        managed = governor.manage_result(tool_name, raw_output)
        messages.append({"role": "tool", ..., "content": managed})

        # When trimming context:
        safe_index = governor.find_safe_truncation_point(messages)
        trimmed = messages[safe_index:]
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

    # ── safe truncation boundary ───────────────────────────────

    @staticmethod
    def find_safe_truncation_point(messages: list[dict[str, Any]]) -> int:
        """Find the oldest index that is safe to start from.

        "Safe" means we start at a ``user`` message — never in the
        middle of an assistant → tool_calls → tool_results chain.
        """
        if not messages:
            return 0

        # Walk back from the end until we find a user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                return i

        return 0  # fallback: keep everything

    @staticmethod
    def is_legal_start(messages: list[dict[str, Any]], index: int) -> bool:
        """Check whether starting at ``index`` produces a valid API request.

        A valid request must start with a user message (or system, but
        system is handled separately).  It must not start with a stray
        tool result whose preceding tool call has been truncated away.
        """
        if index < 0 or index >= len(messages):
            return False

        role = messages[index].get("role", "")
        if role == "tool":
            return False  # can't start with a tool result
        if role == "assistant" and messages[index].get("tool_calls"):
            return False  # tool calls without preceding user message = invalid
        return role in ("user", "system")

    @staticmethod
    def compactable_tools() -> frozenset[str]:
        """Tools whose results are safe to offload or truncate aggressively.

        Tools NOT in this set (like read_file, search_documents) get
        more conservative treatment.
        """
        return frozenset({
            "exec", "grep", "list_dir", "web_fetch",
            "list_events", "list_rag_documents",
        })

    # ── context budget helpers ─────────────────────────────────

    @staticmethod
    def estimate_message_tokens(msg: dict[str, Any]) -> int:
        """Estimate tokens for a single message."""
        from lsm_harness.runtime import estimate_tokens
        content = str(msg.get("content", ""))
        tokens = 4 + estimate_tokens(content)  # 4 tokens overhead per message

        # Tool calls add overhead
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            tokens += estimate_tokens(
                fn.get("name", "") + fn.get("arguments", "")
            )
        return tokens

    @staticmethod
    def estimate_context_tokens(
        system: str,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Estimate total tokens for a full context window."""
        import json
        from lsm_harness.runtime import estimate_tokens

        total = estimate_tokens(system) + 4
        for msg in messages:
            total += ContextGovernor.estimate_message_tokens(msg)
        if tool_schemas:
            total += estimate_tokens(
                json.dumps(tool_schemas, ensure_ascii=False, default=str)
            )
        return total
