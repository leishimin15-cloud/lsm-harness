"""Shell ToolDefinition with policy separated from execution Operations."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from uuid import uuid4

from lsm_harness.coding_agent.operations import (
    LocalShellOperations,
    ShellAbortedError,
    ShellOperations,
    ShellResult,
)
from lsm_harness.agent.tools import AbortedError
from lsm_harness.coding_agent.tools import ToolDefinition
from lsm_harness.tools.truncate import format_size, truncate_tail


_SAFE_COMMANDS: frozenset[str] = frozenset({
    "ls", "cat", "head", "tail", "wc", "find", "file", "stat",
    "tree", "du", "df", "grep", "rg", "awk", "sed", "sort",
    "uniq", "cut", "tr", "git", "python", "python3", "pip", "pip3",
    "node", "npm", "npx", "cargo", "rustc", "go", "make", "cmake",
    "curl", "wget", "uname", "whoami", "date", "pwd", "env", "echo",
    "printf", "sleep",
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

# ── 分段安全策略(2026-09-12,bash -c 化后的防绕过)────────────────
# bash -c 语义下 `ls; rm -rf x` 的 rm 是第二段命令——只查第一个 token
# 等于没有策略。这里按 shell 的 list/pipeline 操作符分段,逐段检查
# 命令名;无法安全解析的复杂结构(命令替换/进程替换/引号不配对)
# 直接返回 policy error,不放行。

_LIST_OPS: frozenset[str] = frozenset({";", "&&", "||", "|", "|&"})
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

_POLICY_GUIDANCE = (
    "该命令被安全策略拒绝——不要重试它(换查询条件也会被拒)。"
    "请换用允许的方式:读文件用 read_file 工具;查 SQLite 数据库用 "
    "python3 -c 'import sqlite3; ...'(python3 已放行)。"
)


def _find_command_substitution(command: str) -> str | None:
    """检测单引号之外的 ``$( )`` / 反引号 / ``<( )`` / ``>( )``。

    命令替换可以藏任意命令(``echo "$(rm x)"``),分段器看不到它——
    宁可误报 ``$((算术))`` 之外的复杂结构,也不放行。
    """
    in_single = False
    for i, ch in enumerate(command):
        if ch == "'":
            in_single = not in_single
        elif not in_single:
            if ch == "`":
                return "backtick command substitution"
            if (
                ch == "$"
                and command[i + 1 : i + 2] == "("
                and command[i + 2 : i + 3] != "("  # $(( 是算术展开,放行
            ):
                return "$(...) command substitution"
            if ch in "<>" and command[i + 1 : i + 2] == "(":
                return "process substitution"
    return None


def _split_segments(command: str) -> list[list[str]] | None:
    """按 ; && || | 把命令分成若干段(shlex 处理引号;失败返回 None)。"""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _LIST_OPS:
            segments.append(current)
            current = []
        else:
            current.append(token)
    segments.append(current)
    return segments


def _segment_command(segment: list[str]) -> tuple[str | None, str | None]:
    """取一段的命令名:跳过前导 ``VAR=val`` 赋值;``env`` 前缀解包到
    内层命令(``env X=1 python -V`` 查的是 python)。返回
    (命令名 | None, 复杂结构错误 | None)。"""
    index = 0
    while index < len(segment) and _ASSIGNMENT_RE.match(segment[index]):
        index += 1
    if index >= len(segment):
        return None, None  # 纯赋值段,不执行命令
    name = segment[index]
    if name == "env":
        index += 1
        while index < len(segment):
            token = segment[index]
            if _ASSIGNMENT_RE.match(token) or token == "--":
                index += 1
                continue
            if token.startswith("-"):
                # env 带选项(-i/-u/-C…)超出可安全分析范围
                return None, (
                    "Error: command rejected by the shell policy: "
                    "'env' with options is too complex to analyze safely. "
                    "请用 VAR=值 命令 或 env VAR=值 命令 的形式。"
                )
            break
        if index >= len(segment):
            return None, None
        name = segment[index]
    return name, None


def _check_policy(
    command: str,
    allow_extra: set[str],
    deny_extra: set[str],
    *,
    workspace: Path | None = None,
    approved: bool = False,
) -> str | None:
    """对完整命令做分段策略检查;返回 None 表示放行,否则为 policy
    error 文本(以 Error: 开头,进 is_error 与熔断的 policy 分类)。

    ``approved``:前端配置了 approval broker 时,registry 已在执行前
    就该调用征求过用户批准(external_write 工具必经审批)——此时
    不在白名单的命令放行(Pi 的 approval-first 模型);``_ALWAYS_DENY``
    与无法解析的复杂结构仍硬拒(深度防御,审批也不放行)。
    """
    substitution = _find_command_substitution(command)
    if substitution is not None:
        return (
            "Error: command rejected by the shell policy: "
            f"{substitution} is too complex to analyze safely.\n"
            "请拆成多条简单命令执行。"
        )
    segments = _split_segments(command)
    if segments is None:
        return (
            "Error: command rejected by the shell policy: "
            "could not parse the command (unbalanced quotes?)."
        )
    for segment in segments:
        redirect_error = _check_redirects(segment, workspace)
        if redirect_error is not None:
            return redirect_error
        name, complex_error = _segment_command(segment)
        if complex_error is not None:
            return complex_error
        if name is None:
            continue
        base = os.path.basename(name)
        if base in _ALWAYS_DENY or base in deny_extra or name in deny_extra:
            return (
                f"Error: command '{base}' is denied by the shell policy.\n"
                "策略对 ;、&&、||、| 分段的每个命令都生效,"
                "把危险命令放在后段不能绕过。"
            )
        if (
            base in _SAFE_COMMANDS
            or name in _SAFE_COMMANDS
            or base in allow_extra
            or name in allow_extra
        ):
            continue
        if approved:
            continue  # approval broker 已在执行前问过用户(Pi 同款)
        return (
            f"Error: command '{base}' is not allowed by the shell policy.\n"
            f"{_POLICY_GUIDANCE}\n"
            f"Allowed commands include: {', '.join(sorted(_SAFE_COMMANDS)[:20])}...\n"
            "Set LSM_SHELL_ALLOW=cmd1,cmd2 to allow more."
        )
    return None


# 写重定向目标必须落在 workspace 内(echo x > /tmp/y 不该逃出去);
# 2>&1 这类 fd 复制与 < 输入重定向不受限。
_REDIRECT_OP_ONLY = re.compile(r"(?:\d*|&)>{1,2}$")
_REDIRECT_ATTACHED = re.compile(r"(?:\d*|&)>{1,2}(.+)$")
_FD_DUP = re.compile(r"\d*>&\d*$")


def _check_redirects(segment: list[str], workspace: Path | None) -> str | None:
    if workspace is None:
        return None
    for index, token in enumerate(segment):
        if _FD_DUP.match(token):
            continue
        target = None
        if _REDIRECT_OP_ONLY.fullmatch(token):
            if index + 1 < len(segment) and not segment[index + 1].startswith("&"):
                target = segment[index + 1]
        else:
            match = _REDIRECT_ATTACHED.match(token)
            if match and not match.group(1).startswith("&"):
                target = match.group(1)
        if not target:
            continue
        resolved = Path(os.path.expanduser(target))
        resolved = resolved if resolved.is_absolute() else (workspace / resolved)
        try:
            resolved.resolve().relative_to(workspace.resolve())
        except (ValueError, OSError):
            return (
                f"Error: redirect target '{target}' is outside the allowed "
                f"workspace ({workspace}).\n写文件请重定向到工作区内,"
                "或用 write_file 工具。"
            )
    return None


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
    _abort=None,
    _ctx=None,
) -> str:
    """Validate command policy, delegate execution, then format the result.

    Execution is host-only (Pi-style): the process runs with the current
    user's permissions, its cwd bounded to the workspace.  No isolation
    is promised beyond the command allow/deny policy.

    ``_abort`` / ``_ctx`` 由 registry 按签名注入(tools 层 opt-in):
    ``_abort`` 让中断(Esc/批次取消)能立刻杀死进程组;``_ctx`` 携带
    approval broker 时,不在白名单的命令已在执行前问过用户。
    """
    del _session_id
    allow_extra = set(
        (allow if allow is not None else os.getenv("LSM_SHELL_ALLOW", "")).split(",")
    ) - {""}
    deny_extra = set(
        (deny if deny is not None else os.getenv("LSM_SHELL_DENY", "")).split(",")
    ) - {""}
    if not command.strip():
        return "Error: empty command"
    workspace = (home or Path.cwd()).resolve()
    approved = _ctx is not None and getattr(_ctx, "approval_broker", None) is not None
    policy_error = _check_policy(
        command, allow_extra, deny_extra, workspace=workspace, approved=approved
    )
    if policy_error is not None:
        return policy_error

    operation = operations or LocalShellOperations(workspace)
    effective_timeout = max(1, min(int(timeout), 300))
    should_stop = (lambda: _abort.aborted) if _abort is not None else None
    try:
        result = operation.run(
            command, cwd=cwd, timeout=effective_timeout, should_stop=should_stop
        )
    except ShellAbortedError:
        raise AbortedError("shell command aborted") from None
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {effective_timeout}s."
    except FileNotFoundError as exc:
        return f"Error: {exc}"
    except PermissionError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"
    # grep/rg 退出码 1 = 未找到匹配,是正常结果而非执行失败。
    # 只对单段命令判定(管道里 grep 的退出码会被后段掩盖)。
    segments = _split_segments(command) or []
    single = None
    if len(segments) == 1:
        single, _ = _segment_command(segments[0])
    if result.exit_code == 1 and single in _NO_MATCH_EXIT1:
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

    def _execute(command, cwd="", timeout=default_timeout, _abort=None, _ctx=None):
        # _abort/_ctx 由 registry 按签名注入:中断可杀进程组;approval
        # broker 存在时白名单外命令已在执行前问过用户。
        return _exec_shell(
            command,
            cwd,
            timeout,
            home=home,
            allow=allow,
            deny=deny,
            operations=shell_operations,
            _abort=_abort,
            _ctx=_ctx,
        )

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
        execute=_execute,
        effect="external_write",
        execution_mode="sequential",
        timeout=300.0,
        prompt_snippet="执行命令使用 exec；先用只读命令确认状态，危险命令会被策略拒绝。",
    )


__all__ = ["_exec_shell", "_format_output", "make_tool", "ShellResult"]
