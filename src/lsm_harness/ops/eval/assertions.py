"""Deterministic assertions over one scenario run.

An :class:`Assertion` is a small, data-only check evaluated against the
:class:`AssertionContext` the runner collects (tool calls, replies, files,
session tree, commands).  Keeping them data-driven means a scenario is plain
data — serializable, diffable, and independent of any model-derived
self-fulfilling output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Assertion:
    kind: str
    value: Any = ""
    target: str = ""

    def describe(self) -> str:
        if self.target:
            return f"{self.kind}({self.target!r}, {self.value!r})"
        return f"{self.kind}({self.value!r})"


@dataclass
class CommandOutcome:
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class AssertionContext:
    """Everything an assertion may inspect from a finished run."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    final_reply: str = ""
    workspace: Path = field(default_factory=Path)
    session_entries: list[Any] = field(default_factory=list)
    session_tree: list[Any] = field(default_factory=list)
    compaction_summary: str = ""
    command_outcomes: list[CommandOutcome] = field(default_factory=list)
    user_messages: list[str] = field(default_factory=list)
    # trace 事件流 + 会话 JSONL 的拼接文本,供 trace_not_contains 检查
    # (密钥等敏感串不应出现在任何持久化记录里)。
    trace_text: str = ""
    # 每个 prompt step 完成后快照的 system prompt(每 run 重建),
    # 供 system_prompt_contains 检查注入链(project memory 等)。
    system_prompts: list[str] = field(default_factory=list)


def _tool_calls(ctx: AssertionContext, tool: str) -> list[dict[str, Any]]:
    return [c for c in ctx.tool_calls if c.get("tool") == tool]


def evaluate(assertion: Assertion, ctx: AssertionContext) -> tuple[bool, str]:
    """Evaluate one assertion; return ``(passed, message)``."""
    kind = assertion.kind

    if kind == "expect_tool":
        called = {c.get("tool") for c in ctx.tool_calls}
        if assertion.value in called:
            return True, f"tool {assertion.value!r} called"
        return False, f"expected tool {assertion.value!r}, called {sorted(called)}"

    if kind == "tool_not_called":
        called = {c.get("tool") for c in ctx.tool_calls}
        if assertion.value not in called:
            return True, f"tool {assertion.value!r} never called"
        return False, f"forbidden tool {assertion.value!r} was called"

    if kind == "tool_args":
        for call in _tool_calls(ctx, assertion.target):
            args = call.get("args") or {}
            if _subset(args, assertion.value):
                return True, f"{assertion.target} args ⊇ {assertion.value!r}"
        return False, (
            f"no {assertion.target!r} call with args ⊇ {assertion.value!r}; "
            f"got {[c.get('args') for c in _tool_calls(ctx, assertion.target)]}"
        )

    if kind == "tool_output_contains":
        needle = assertion.value
        for call in _tool_calls(ctx, assertion.target):
            if needle in str(call.get("output", "")):
                return True, f"{assertion.target} output contains {needle!r}"
        return False, f"no {assertion.target!r} output contains {needle!r}"

    if kind == "reply_contains":
        for reply in ctx.replies:
            if assertion.value in reply:
                return True, f"reply contains {assertion.value!r}"
        return False, f"expected {assertion.value!r} in replies {ctx.replies!r}"

    if kind == "reply_not_contains":
        for reply in ctx.replies:
            if assertion.value in reply:
                return False, f"forbidden {assertion.value!r} found in {reply!r}"
        return True, f"no reply contains {assertion.value!r}"

    if kind == "file_contains":
        content = _read_workspace_file(ctx.workspace, assertion.target)
        if content is None:
            return False, f"file {assertion.target!r} not found in workspace"
        if assertion.value in content:
            return True, f"{assertion.target} contains {assertion.value!r}"
        return False, f"{assertion.target} does not contain {assertion.value!r}"

    if kind == "file_not_contains":
        content = _read_workspace_file(ctx.workspace, assertion.target)
        if content is None:
            return True, f"file {assertion.target!r} not found (nothing to forbid)"
        if assertion.value in content:
            return False, f"{assertion.target} contains forbidden {assertion.value!r}"
        return True, f"{assertion.target} does not contain {assertion.value!r}"

    if kind == "file_equals":
        content = _read_workspace_file(ctx.workspace, assertion.target)
        if content is None:
            return False, f"file {assertion.target!r} not found in workspace"
        if content == assertion.value:
            return True, f"{assertion.target} matches expected content"
        return False, f"{assertion.target} differs:\n{_short_diff(assertion.value, content)}"

    if kind == "command_exit":
        expected = int(assertion.value)
        if not ctx.command_outcomes:
            return False, "no command ran"
        actual = ctx.command_outcomes[-1].exit_code
        if actual == expected:
            return True, f"command exit code == {expected}"
        return False, f"command exit code {actual} != {expected}"

    if kind == "session_tree_has":
        found = [e for e in ctx.session_entries if getattr(e, "type", None) == assertion.value]
        if found:
            return True, f"session tree has {assertion.value!r} entry"
        return False, f"session tree has no {assertion.value!r} entry"

    if kind == "summary_contains":
        if assertion.value in ctx.compaction_summary:
            return True, f"summary contains {assertion.value!r}"
        return False, (
            f"summary does not contain {assertion.value!r}; "
            f"summary={ctx.compaction_summary!r}"
        )

    if kind == "context_user_once":
        count = sum(1 for m in ctx.user_messages if m == assertion.value)
        if count == 1:
            return True, f"user message {assertion.value!r} appears exactly once"
        return False, f"user message {assertion.value!r} appears {count} times"

    if kind == "trace_contains":
        if assertion.value in ctx.trace_text:
            return True, f"trace/session contains {assertion.value!r}"
        return False, f"trace/session missing {assertion.value!r}"

    if kind == "trace_not_contains":
        if assertion.value in ctx.trace_text:
            return False, f"trace/session contains forbidden {assertion.value!r}"
        return True, f"trace/session does not contain {assertion.value!r}"

    if kind == "system_prompt_contains":
        for prompt in ctx.system_prompts:
            if assertion.value in prompt:
                return True, f"system prompt contains {assertion.value!r}"
        return False, (
            f"no system prompt contains {assertion.value!r} "
            f"({len(ctx.system_prompts)} captured)"
        )

    raise ValueError(f"unknown assertion kind: {kind!r}")


def _subset(actual: dict, expected: dict) -> bool:
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def _read_workspace_file(workspace: Path, path: str) -> str | None:
    target = workspace / path
    if not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _short_diff(expected: str, actual: str) -> str:
    import difflib

    diff = list(
        difflib.unified_diff(
            expected.splitlines(), actual.splitlines(),
            fromfile="expected", tofile="actual", lineterm="",
        )
    )
    return "\n".join(diff[:12]) if diff else "(contents equal?)"


# ── constructors ──────────────────────────────────────────────────


def expect_tool(tool: str) -> Assertion:
    return Assertion(kind="expect_tool", value=tool)


def tool_not_called(tool: str) -> Assertion:
    return Assertion(kind="tool_not_called", value=tool)


def tool_args(tool: str, expected_args: dict[str, Any]) -> Assertion:
    return Assertion(kind="tool_args", target=tool, value=expected_args)


def tool_output_contains(tool: str, substring: str) -> Assertion:
    return Assertion(kind="tool_output_contains", target=tool, value=substring)


def reply_contains(substring: str) -> Assertion:
    return Assertion(kind="reply_contains", value=substring)


def reply_not_contains(substring: str) -> Assertion:
    return Assertion(kind="reply_not_contains", value=substring)


def file_contains(path: str, substring: str) -> Assertion:
    return Assertion(kind="file_contains", target=path, value=substring)


def file_not_contains(path: str, substring: str) -> Assertion:
    return Assertion(kind="file_not_contains", target=path, value=substring)


def file_equals(path: str, content: str) -> Assertion:
    return Assertion(kind="file_equals", target=path, value=content)


def command_exit(code: int) -> Assertion:
    return Assertion(kind="command_exit", value=code)


def session_tree_has(entry_type: str) -> Assertion:
    return Assertion(kind="session_tree_has", value=entry_type)


def summary_contains(substring: str) -> Assertion:
    return Assertion(kind="summary_contains", value=substring)


def context_user_once(text: str) -> Assertion:
    return Assertion(kind="context_user_once", value=text)


def trace_contains(substring: str) -> Assertion:
    return Assertion(kind="trace_contains", value=substring)


def trace_not_contains(substring: str) -> Assertion:
    return Assertion(kind="trace_not_contains", value=substring)


def system_prompt_contains(substring: str) -> Assertion:
    return Assertion(kind="system_prompt_contains", value=substring)
