"""exec 的 bash -c 执行链 + 分段安全策略 + 子进程环境(2026-09-12)。

对齐 Pi(bash.ts / utils/shell.ts):完整命令经显式 argv
``[shell, "-c", command]`` 执行(/bin/bash → PATH bash → /bin/sh),
不再 shlex.split 成参数列表;安全策略对 ;、&&、||、| 分段的每个
命令逐一检查;子进程环境只移除精确密钥集合,不再按 TOKEN 子串
误伤 LSM_MAX_TOKENS 等配置变量。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from lsm_harness.coding_agent.operations import (
    LocalShellOperations,
    MockShellOperations,
    ShellResult,
    _resolve_shell,
    _sanitized_environment,
)
from lsm_harness.tools.shell import _exec_shell

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── shell 选择(Pi getShellConfig Unix 分支)─────────────────────


def test_resolve_shell_prefers_bin_bash():
    shell = _resolve_shell()
    assert shell in {"/bin/bash", "/bin/sh"} or shell.endswith("/bash")
    assert os.access(shell, os.X_OK)


# ── 目标一:真实 shell 语法 ─────────────────────────────────────


def test_pipe_semicolon_and_and_basics(tmp_path):
    ops = LocalShellOperations(tmp_path)
    result = ops.run(
        "printf 'alpha\nbeta\n' | grep beta && echo SHELL_OK",
        cwd="", timeout=10,
    )
    assert result.exit_code == 0
    assert "beta" in result.stdout and "SHELL_OK" in result.stdout


def test_semicolon_and_stderr_redirect(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\n", encoding="utf-8")
    ops = LocalShellOperations(tmp_path)
    result = ops.run("head -1 a.py 2>&1; ls .", cwd="", timeout=10)
    assert result.exit_code == 0
    assert "print(1)" in result.stdout and "a.py" in result.stdout


def test_env_assignment_prefix_and_or_operator(tmp_path):
    ops = LocalShellOperations(tmp_path)
    # env VAR=值 前缀
    result = ops.run(
        "env LSM_TEST_MARKER=hello python3 -c \"import os; print(os.environ['LSM_TEST_MARKER'])\"",
        cwd="", timeout=10,
    )
    assert result.exit_code == 0 and "hello" in result.stdout
    # || 短路
    result = ops.run("ls /nonexistent-path-xyz || echo FALLBACK", cwd="", timeout=10)
    assert result.exit_code == 0 and "FALLBACK" in result.stdout


def test_output_redirection_inside_workspace(tmp_path):
    ops = LocalShellOperations(tmp_path)
    result = ops.run("echo hi > out.txt && cat out.txt", cwd="", timeout=10)
    assert result.exit_code == 0
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hi\n"


def test_mock_shell_operations_records_full_command_string(tmp_path):
    mock = MockShellOperations(ShellResult(0, "ok", ""))
    out = _exec_shell("ls | head -1", home=tmp_path, operations=mock)
    assert out == "ok"
    assert mock.calls == [("ls | head -1", "", 60)]  # 完整字符串,不是参数列表


# ── 目标二:分段策略防绕过 ──────────────────────────────────────


def test_denied_command_in_second_segment_rejected_before_any_execution(tmp_path):
    """ls; rm file:rm 在第二段也要拒;且整条命令在策略检查阶段拒绝,
    第一段 ls 也不会执行。"""
    (tmp_path / "victim.txt").write_text("keep me", encoding="utf-8")
    mock = MockShellOperations()
    out = _exec_shell("ls; rm victim.txt", home=tmp_path, operations=mock)
    assert "denied by the shell policy" in out
    assert "'rm'" in out
    assert mock.calls == []  # 第一段也没执行
    assert (tmp_path / "victim.txt").exists()


def test_sudo_after_and_operator_rejected(tmp_path):
    mock = MockShellOperations()
    out = _exec_shell("echo ok && sudo -n true", home=tmp_path, operations=mock)
    assert "denied by the shell policy" in out and "'sudo'" in out
    assert mock.calls == []


def test_pipe_into_shell_rejected(tmp_path):
    mock = MockShellOperations()
    out = _exec_shell("cat x | sh", home=tmp_path, operations=mock)
    assert "not allowed by the shell policy" in out and "'sh'" in out
    assert mock.calls == []


def test_env_prefix_unwraps_to_inner_command(tmp_path):
    """env X=1 python -V 放行(env 解包后 python 在白名单);
    env X=1 sudo x 仍拒(内层是 sudo)。"""
    mock = MockShellOperations()
    out = _exec_shell("env X=1 python -V", home=tmp_path, operations=mock)
    assert not out.startswith("Error")
    assert mock.calls == [("env X=1 python -V", "", 60)]

    mock2 = MockShellOperations()
    out2 = _exec_shell("env X=1 sudo whoami", home=tmp_path, operations=mock2)
    assert "denied by the shell policy" in out2
    assert mock2.calls == []


def test_command_substitution_rejected_as_unanalyzable(tmp_path):
    for command in ('echo "$(rm victim.txt)"', "cat `rm victim.txt`",
                    "cat <(rm victim.txt)"):
        mock = MockShellOperations()
        out = _exec_shell(command, home=tmp_path, operations=mock)
        assert "rejected by the shell policy" in out, command
        assert mock.calls == []


def test_cwd_outside_workspace_still_rejected(tmp_path):
    ops = LocalShellOperations(tmp_path)
    try:
        ops.run("pwd", cwd="/", timeout=5)
    except PermissionError as exc:
        assert "outside the allowed workspace" in str(exc)
    else:
        raise AssertionError("cwd 逃逸未被拒绝")


# ── 目标三:子进程环境精确密钥集合 ──────────────────────────────


def test_sanitized_environment_keeps_config_vars_drops_exact_keys(monkeypatch):
    monkeypatch.setenv("LSM_MAX_TOKENS", "8191")
    monkeypatch.setenv("LSM_CONTEXT_BUDGET_TOKENS", "50000")
    monkeypatch.setenv("LSM_SUMMARY_MAX_TOKENS", "1200")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-should-not-leak")
    env = _sanitized_environment()
    # TOKEN 子串不再误伤配置变量
    assert env["LSM_MAX_TOKENS"] == "8191"
    assert env["LSM_CONTEXT_BUDGET_TOKENS"] == "50000"
    assert env["LSM_SUMMARY_MAX_TOKENS"] == "1200"
    # 精确密钥集合被移除
    assert "DEEPSEEK_API_KEY" not in env


def test_exec_python_reads_real_config_value(tmp_path, monkeypatch):
    """exec 里的 python 能正常读 Settings(旧实现会把 LSM_MAX_TOKENS
    洗成 [REDACTED],int() 解析直接炸)。"""
    monkeypatch.setenv("LSM_MAX_TOKENS", "8191")
    # 未激活 venv 时，系统 python3 可能缺少项目依赖。把当前
    # 测试环境放到 PATH 最前，同时保留产品的 python3 调用形式。
    monkeypatch.setenv(
        "PATH",
        f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
    )
    out = _exec_shell(
        "python3 -c \"import sys; sys.path.insert(0, 'src'); "
        "from lsm_harness.config import Settings; print(Settings().max_tokens)\"",
        home=REPO_ROOT,
    )
    assert "8191" in out
    assert "REDACTED" not in out


def test_exec_does_not_leak_api_key_into_subprocess(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-secret-xyz")
    out = _exec_shell(
        "python3 -c \"import os; print(os.environ.get('DEEPSEEK_API_KEY'))\"",
        home=tmp_path,
    )
    assert "sk-test-secret-xyz" not in out
    assert "None" in out


# ── 进程组管理:超时/中断整组杀死(Pi killProcessTree)────────────

def test_timeout_kills_whole_process_group(tmp_path):
    """bash -c 派生的整组进程都被杀死,不是只杀 shell 自己。"""
    import subprocess as sp
    import time as _time

    ops = LocalShellOperations(tmp_path)
    start = _time.monotonic()
    try:
        ops.run("sleep 30", cwd="", timeout=1)
        raise AssertionError("应当超时")
    except sp.TimeoutExpired:
        pass
    assert _time.monotonic() - start < 5  # 杀死而不是等满 30s


def test_should_stop_aborts_running_command(tmp_path):
    import threading

    from lsm_harness.coding_agent.operations import ShellAbortedError

    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    ops = LocalShellOperations(tmp_path)
    start = __import__("time").monotonic()
    try:
        ops.run("sleep 30", cwd="", timeout=60, should_stop=stop.is_set)
        raise AssertionError("应当被中断")
    except ShellAbortedError:
        pass
    assert __import__("time").monotonic() - start < 5


def test_exec_abort_translates_to_registry_abort(tmp_path):
    """_abort(中断)经 registry 的标准 AbortedError 路径收尾。"""
    from lsm_harness.agent.tools import AbortedError, AbortHandle

    import pytest

    handle = AbortHandle()
    handle.abort()  # 已中断:命令立即被杀
    with pytest.raises(AbortedError):
        _exec_shell("sleep 30", home=tmp_path, _abort=handle)


# ── 重定向目标限定 workspace ────────────────────────────────────

def test_redirect_outside_workspace_rejected(tmp_path):
    mock = MockShellOperations()
    out = _exec_shell("echo x > /tmp/lsm-evil-xyz.txt", home=tmp_path, operations=mock)
    assert "outside the allowed workspace" in out
    assert mock.calls == []
    out2 = _exec_shell("echo x > ../escape.txt", home=tmp_path, operations=mock)
    assert "outside the allowed workspace" in out2


def test_redirect_inside_workspace_allowed(tmp_path):
    out = _exec_shell("echo hi > ok.txt && cat ok.txt", home=tmp_path)
    assert "hi" in out
    # fd 复制不受限
    out2 = _exec_shell("ls missing 2>&1 | cat", home=tmp_path)
    assert "outside the allowed workspace" not in out2


# ── approval:白名单外命令在配置了 broker 时已审批放行 ────────────

def test_unlisted_command_allowed_when_approved(tmp_path):
    from types import SimpleNamespace

    mock = MockShellOperations()
    ctx = SimpleNamespace(approval_broker=object())  # registry 已批准
    out = _exec_shell("sqlite3 x.db '.tables'", home=tmp_path,
                      operations=mock, _ctx=ctx)
    assert out == "ok"
    assert mock.calls  # 审批后确实执行
    # 无 broker:仍硬拒
    out2 = _exec_shell("sqlite3 x.db '.tables'", home=tmp_path, operations=MockShellOperations())
    assert "not allowed by the shell policy" in out2
    # ALWAYS_DENY 审批也不放行
    mock3 = MockShellOperations()
    out3 = _exec_shell("sudo true", home=tmp_path, operations=mock3, _ctx=ctx)
    assert "denied by the shell policy" in out3
    assert mock3.calls == []


# ── 代理不可达直连兜底 ──────────────────────────────────────────

def test_proxy_fallback_direct_client_when_proxy_down(monkeypatch):
    from lsm_harness.ai.api import common

    monkeypatch.setattr(common, "_PROXY_REACHABLE", None)
    monkeypatch.setattr(
        "urllib.request.getproxies",
        lambda: {"https": "http://127.0.0.1:9"},  # 9 端口无服务
    )
    assert common.proxy_reachable() is False
    client = common.sdk_http_client()
    assert client is not None and client._trust_env is False
    client.close()


def test_proxy_default_when_none_or_reachable(monkeypatch):
    from lsm_harness.ai.api import common

    monkeypatch.setattr(common, "_PROXY_REACHABLE", None)
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})
    assert common.proxy_reachable() is True
    assert common.sdk_http_client() is None  # SDK 默认行为
