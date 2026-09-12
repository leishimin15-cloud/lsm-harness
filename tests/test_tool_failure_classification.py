"""工具失败分类 + 熔断签名(2026-09-12 实机问题)。

实机场景:模型先写错 grep 参数(退出码 2),再正常 grep 无匹配
(退出码 1),再试图用 sqlite3 被策略拒绝——三件不同原因的事被
同一个「按工具累计」的熔断器算成"连续 3 次失败",loop 被误杀。

新契约:
- grep/rg 退出码 1 = 未找到匹配,不是错误,不进熔断;
- 熔断按「同工具 + 同错误类别 + 同错误行 + 同参数」的连续 streak,
  不同原因/不同命令互不累计;连续 3 次同签名才熔断;
- 策略拒绝不换参数签名(被禁的是命令本身),并明确指引模型换用
  允许的方式,不能盲试。
"""

from __future__ import annotations

from lsm_harness.agent.tools import ToolRegistry
from lsm_harness.ai.types import ModelResponse, ToolCall
from lsm_harness.tools.shell import _exec_shell

from helpers import QueueClient, Tool, run_test_loop


def _loop(client, tools, maximum=10):
    events = []
    result = run_test_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        max_iterations=maximum,
        max_tokens=100,
        emit=lambda kind, data: events.append((kind, data)),
    )
    return result, events


# ── shell:grep 退出码 1 与策略拒绝 ──────────────────────────────


def test_grep_exit1_is_no_match_not_error(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello\n", encoding="utf-8")
    out = _exec_shell(f"grep zzz {target}", home=tmp_path)
    assert out == "(no matches found)"
    # 不带 Error 前缀 → _normalize 不标 is_error → 不进熔断
    assert not out.lower().startswith("error")

    # 退出码 2(用法/IO 错误)仍然是错误
    bad = _exec_shell("grep --no-such-flag x", home=tmp_path)
    assert bad.startswith("Error: command exited with code 2.")


def test_policy_rejection_guides_to_allowed_alternatives(tmp_path):
    out = _exec_shell("sqlite3 x.db '.tables'", home=tmp_path)
    assert "not allowed by the shell policy" in out
    # 明确告诉模型:不要重试;换允许的方式(read_file / python3)
    assert "不要重试" in out
    assert "python3" in out and "read_file" in out


# ── loop:熔断按错误签名累计 ─────────────────────────────────────


def _failing_tool(name: str, message: str):
    def fail():
        raise RuntimeError(message)

    return Tool(name, "fails", {"type": "object", "properties": {}}, fail)


def test_distinct_failures_do_not_trip_breaker():
    """两个不同工具交替各失败两次:不同签名互不累计,不熔断。
    (旧的按工具累计会在 bad1 第二次失败时误杀本轮——反向变异点。)"""
    tools = ToolRegistry()
    tools.register(_failing_tool("bad1", "boom-1"))
    tools.register(_failing_tool("bad2", "boom-2"))
    client = QueueClient(
        ModelResponse(tool_calls=[
            ToolCall("1", "bad1", {}),
            ToolCall("2", "bad2", {}),
        ]),
        ModelResponse(tool_calls=[
            ToolCall("3", "bad1", {}),
            ToolCall("4", "bad2", {}),
        ]),
        ModelResponse(text="恢复了"),
    )
    result, events = _loop(client, tools)
    assert not [
        kind for kind, _ in events if kind == "loop.repeated_tool_error"
    ]
    assert result.status == "completed"
    assert result.reply == "恢复了"


def test_same_signature_thrice_trips_breaker():
    """同一工具 + 同一错误签名连续 3 次:熔断,错误信息带分类。"""
    tools = ToolRegistry()
    tools.register(_failing_tool("bad", "boom"))
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "bad", {"q": "x"})]),
        ModelResponse(tool_calls=[ToolCall("2", "bad", {"q": "x"})]),
        ModelResponse(tool_calls=[ToolCall("3", "bad", {"q": "x"})]),
        ModelResponse(text="不应到达"),
    )
    result, events = _loop(client, tools)
    repeated = [
        data for kind, data in events if kind == "loop.repeated_tool_error"
    ]
    assert repeated and repeated[0]["count"] == 3
    assert repeated[0]["error_kind"]
    assert result.status == "failed"
    assert "连续 3 次以相同方式失败" in result.reply


def test_same_tool_different_args_do_not_accumulate():
    """同一工具但参数不同的失败(模型在换着法子试)不是同一签名,
    不累计熔断。"""
    tools = ToolRegistry()
    tools.register(_failing_tool("bad", "boom"))
    client = QueueClient(
        *(
            ModelResponse(tool_calls=[ToolCall(str(i), "bad", {"q": f"v{i}"})])
            for i in range(4)
        ),
        ModelResponse(text="恢复了"),
    )
    result, events = _loop(client, tools)
    assert not [
        kind for kind, _ in events if kind == "loop.repeated_tool_error"
    ]
    assert result.status == "completed"
