"""Shell ToolDefinition with policy separated from execution Operations."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from uuid import uuid4

from lsm_harness.coding_agent.operations import (
    LocalShellOperations,
    ShellOperations,
    ShellResult,
)
from lsm_harness.coding_agent.tools import ToolDefinition
from lsm_harness.tools.truncate import format_size, truncate_tail


_SAFE_COMMANDS: frozenset[str] = frozenset({
    "ls", "cat", "head", "tail", "wc", "find", "file", "stat",
    "tree", "du", "df", "grep", "rg", "awk", "sed", "sort",
    "uniq", "cut", "tr", "git", "python", "python3", "pip", "pip3",
    "node", "npm", "npx", "cargo", "rustc", "go", "make", "cmake",
    "curl", "wget", "uname", "whoami", "date", "pwd", "env", "echo",
    "which", "where", "brew", "apt", "apt-get", "dnf", "yum", "pacman",
    "diff", "patch", "xxd", "hexdump", "tar", "gzip", "zip", "unzip",
    "ps", "top", "htop", "kill", "killall", "ping", "nslookup", "dig",
    "netstat", "ss", "ifconfig",
})

_ALWAYS_DENY: frozenset[str] = frozenset({
    "rm", "mv", "cp", "dd", "shutdown", "reboot", "su", "sudo",
    "chmod", "chown", "mkfs", "mount", "umount",
})

# 这些命令的退出码 1 是"未找到匹配",不是执行失败(GNU/BSD grep 与
# rg 一致;退出码 2 才是真正的用法/IO 错误)。不标 Error 前缀 →
# 不计入连续失败熔断。
_NO_MATCH_EXIT1: frozenset[str] = frozenset({"grep", "rg"})


def _parse_command(command: str) -> tuple[str, list[str]]:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    if not parts:
        raise ValueError("empty command")
    return parts[0], parts


def _resolve_command(name: str) -> str:
    basename = os.path.basename(name)
    return basename if basename in _SAFE_COMMANDS or basename in _ALWAYS_DENY else name


def _is_allowed(command: str, allow_extra: set[str], deny_extra: set[str]) -> bool:
    basename = _resolve_command(command)
    if basename in _ALWAYS_DENY or basename in deny_extra:
        return False
    return basename in allow_extra or basename in _SAFE_COMMANDS


def _write_full_output(content: str, workspace: Path) -> str:
    """Spill the untruncated output INSIDE the workspace; return the
    workspace-relative path.

    The escape hatch only works if the model can actually read the file:
    read_file is workspace-bounded, so the log lives under
    ``<workspace>/.lsm/tool-results/`` — never a system temp dir that
    read_file would refuse (plan §7.2).
    """
    directory = workspace / ".lsm" / "tool-results"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"exec-{uuid4().hex[:8]}.log"
    path.write_text(content, encoding="utf-8")
    return str(path.relative_to(workspace))


def _format_output(
    stdout: str,
    stderr: str,
    exit_code: int = 0,
    workspace: Path | None = None,
) -> str:
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    combined = "\n".join(parts) if parts else "(no output)"

    # Dual limit (2000 lines / 50KB, whichever hits first), keeping the
    # *tail* — error traces and final results live at the end of shell output.
    result = truncate_tail(combined)
    body = result.content
    if result.truncated:
        full_path = _write_full_output(
            combined, (workspace or Path.cwd()).resolve()
        )
        if result.last_line_partial:
            notice = (
                f"[Showing last {format_size(result.output_bytes)} of line 1 "
                f"(line is {format_size(result.total_bytes)}). "
                f"Full output: {full_path}]"
            )
        else:
            first = result.total_lines - result.output_lines + 1
            notice = (
                f"[Showing lines {first}–{result.total_lines} of "
                f"{result.total_lines}. Full output: {full_path}]"
            )
        body = f"{body}\n{notice}" if body else notice

    if exit_code != 0:
        body = f"Error: command exited with code {exit_code}.\n{body}"
    return body


def _exec_shell(
    command: str,
    cwd: str = "",
    timeout: int = 60,
    home: Path | None = None,
    allow: str | None = None,
    deny: str | None = None,
    _session_id: str = "",
    operations: ShellOperations | None = None,
) -> str:
    """Validate command policy, delegate execution, then format the result.

    Execution is host-only (Pi-style): the process runs with the current
    user's permissions, its cwd bounded to the workspace.  No isolation
    is promised beyond the command allow/deny policy.
    """
    del _session_id
    allow_extra = set(
        (allow if allow is not None else os.getenv("LSM_SHELL_ALLOW", "")).split(",")
    ) - {""}
    deny_extra = set(
        (deny if deny is not None else os.getenv("LSM_SHELL_DENY", "")).split(",")
    ) - {""}
    try:
        base, parts = _parse_command(command)
    except ValueError as exc:
        return f"Error: {exc}"
    if not _is_allowed(base, allow_extra, deny_extra):
        return (
            f"Error: command '{base}' is not allowed by the shell policy.\n"
            "该命令被安全策略拒绝——不要重试它(换查询条件也会被拒)。"
            "请换用允许的方式:读文件用 read_file 工具;查 SQLite 数据库用 "
            "python3 -c 'import sqlite3; ...'(python3 已放行)。\n"
            f"Allowed commands include: {', '.join(sorted(_SAFE_COMMANDS)[:20])}...\n"
            "Set LSM_SHELL_ALLOW=cmd1,cmd2 to allow more."
        )

    operation = operations or LocalShellOperations((home or Path.cwd()).resolve())
    effective_timeout = max(1, min(int(timeout), 300))
    try:
        result = operation.run(parts, cwd=cwd, timeout=effective_timeout)
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {effective_timeout}s."
    except FileNotFoundError as exc:
        message = str(exc)
        if message.startswith("working directory"):
            return f"Error: {message}"
        return f"Error: command '{base}' not found."
    except PermissionError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"
    # grep/rg 退出码 1 = 未找到匹配,是正常结果而非执行失败
    if result.exit_code == 1 and base in _NO_MATCH_EXIT1:
        if result.stdout or result.stderr:
            body = _format_output(
                result.stdout, result.stderr, 0,
                workspace=(home or Path.cwd()).resolve(),
            )
            return f"(no matches found)\n{body}"
        return "(no matches found)"
    return _format_output(
        result.stdout,
        result.stderr,
        result.exit_code,
        workspace=(home or Path.cwd()).resolve(),
    )


def make_tool(
    home: Path,
    *,
    default_timeout: int = 60,
    allow: str | None = None,
    deny: str | None = None,
    operations: ShellOperations | None = None,
) -> ToolDefinition:
    shell_operations = operations or LocalShellOperations(home.resolve())
    return ToolDefinition(
        name="exec",
        label="执行命令",
        description=(
            "执行一个 shell 命令并返回 stdout 和 stderr。"
            "输出截断为最后 2000 行或 50KB（先到为准）；"
            "截断时完整输出会保存到 .lsm/tool-results/，可用 read_file 查看。"
            "超时时间默认 60 秒，最长 300 秒。"
            "危险命令（rm、sudo 等）被禁止。"
            "在运行之前，优先使用非破坏性命令了解状态。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 shell 命令。如 'ls -la' 或 'python script.py'。",
                },
                "cwd": {"type": "string", "description": "工作目录，默认当前目录。"},
                "timeout": {
                    "type": "integer",
                    "description": "超时秒数，默认 60，上限 300。",
                },
            },
            "required": ["command"],
        },
        execute=lambda command, cwd="", timeout=default_timeout: _exec_shell(
            command,
            cwd,
            timeout,
            home=home,
            allow=allow,
            deny=deny,
            operations=shell_operations,
        ),
        effect="external_write",
        execution_mode="sequential",
        timeout=300.0,
        prompt_snippet="执行命令使用 exec；先用只读命令确认状态，危险命令会被策略拒绝。",
    )


__all__ = ["_exec_shell", "_format_output", "make_tool", "ShellResult"]
