"""Agent 运行状态与 prompt/run/wait_for_idle（阶段 1 · Batch 1）。

验收 (a)：脱离 CLI/TUI/SQLite，裸 Agent + 假模型即可跑完一次运行。
验收 (c) 的 Agent 半边：idle 信号（done）只在 finish() 置位，绝不提前。
"""

from __future__ import annotations

import threading
import time

import pytest

from lsm_harness.agent import Agent, AgentContext, AgentLoopConfig
from lsm_harness.agent.messages import (
    AgentToolResultMessage,
    AssistantMessage,
    ToolCallContent,
    UserMessage,
    user_message,
)
from lsm_harness.agent.tools import ToolRegistry
from lsm_harness.agent.types import TraceResult
from lsm_harness.ai.types import Model, ModelResponse, Usage

from helpers import QueueClient

MODEL = Model(id="test-model", api="legacy-client", provider="legacy")


def _config(**overrides) -> AgentLoopConfig:
    return AgentLoopConfig(
        model=MODEL,
        max_iterations=overrides.pop("max_iterations", 10),
        max_tokens=overrides.pop("max_tokens", 100),
        **overrides,
    )


def _context(*messages) -> AgentContext:
    return AgentContext(
        system_prompt="system", messages=list(messages), tools=ToolRegistry()
    )


def test_agent_prompt_runs_standalone_with_fake_model():
    """无 CLI/TUI/SQLite：Agent + 假模型即可完成一次运行并回到 idle。"""
    agent = Agent()
    context = _context(user_message("你好"))
    client = QueueClient(ModelResponse(text="你好呀", usage=Usage(4, 4)))

    result = agent.prompt(
        context,
        _config(),
        stream_fn=client.as_stream_fn(),
        emit=lambda _kind, _data: None,
    )

    assert result.status == "completed"
    assert result.reply == "你好呀"
    # 生命周期已收尾：不再 running，idle 立即可得，结果可从 run 状态读取。
    assert agent.is_running is False
    assert agent.wait_for_idle(0) is True
    assert len(client.calls) == 1


def test_prompt_injects_agent_owned_config_fields():
    """queues/hooks/listeners 由 Agent 注入 config——调用方不再反读 Agent 内部。"""
    agent = Agent()
    seen_events = []
    agent.subscribe(seen_events.append)

    # steering 队列在首个 Turn 前被轮询：若注入生效，loop 会消费这条消息。
    agent.steering_queue.enqueue("插队一句")

    context = _context(user_message("正事"))
    client = QueueClient(
        ModelResponse(text="先回正事", usage=Usage(2, 2)),
    )
    result = agent.prompt(
        context,
        _config(),
        stream_fn=client.as_stream_fn(),
        emit=lambda _kind, _data: None,
    )
    assert result.status == "completed"
    # steering 消息进入了上下文（由 Agent 的队列注入，而非调用方塞 config）。
    contents = [
        getattr(m, "content", "") for m in context.messages
        if isinstance(m, UserMessage)
    ]
    assert any("插队一句" in str(c) for c in contents)
    # 订阅者收到了 kernel 事件。
    assert seen_events


def test_agent_run_leaves_lifecycle_to_caller():
    """run() 不调 begin/finish：idle 信号由调用方的 finish() 决定。"""
    agent = Agent()
    active = agent.begin(trace_id="t-1", session_id="s-1")
    assert active.trace_id == "t-1"
    assert active.session_id == "s-1"
    assert active.started_at <= time.monotonic()

    client = QueueClient(ModelResponse(text="好", usage=Usage(1, 1)))
    result = agent.run(
        active,
        _context(user_message("问")),
        _config(),
        stream_fn=client.as_stream_fn(),
        emit=lambda _kind, _data: None,
    )

    # run 已返回，但生命周期未结束：仍 running，wait_for_idle 会等到超时。
    assert agent.is_running is True
    assert agent.wait_for_idle(0.05) is False
    assert active.result is None

    agent.finish(active, result)
    assert agent.is_running is False
    assert agent.wait_for_idle(0) is True
    assert active.done.is_set()
    assert active.result is result


def test_begin_twice_raises():
    agent = Agent()
    agent.begin()
    with pytest.raises(RuntimeError, match="already running"):
        agent.begin()


def test_wait_for_idle_waits_for_finish_on_another_thread():
    """运行停在流里时 wait_for_idle 阻塞；finish 后才返回 True。"""
    entered = threading.Event()
    release = threading.Event()

    def parked_stream(_model, _context, options):
        from lsm_harness.ai.api.common import snapshot
        from lsm_harness.ai.types import AssistantMessageEvent

        entered.set()
        release.wait(5)
        final = snapshot(text="醒来了", thinking="", pending={})
        yield AssistantMessageEvent("start", snapshot(text="", thinking="", pending={}))
        yield AssistantMessageEvent("text_start", final)
        yield AssistantMessageEvent("text_delta", final, text_delta="醒来了")
        yield AssistantMessageEvent("text_end", final)
        yield AssistantMessageEvent("done", final)

    agent = Agent()
    outcome: list[TraceResult] = []

    def run():
        outcome.append(agent.prompt(
            _context(user_message("等")),
            _config(),
            stream_fn=parked_stream,
            emit=lambda _kind, _data: None,
        ))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert entered.wait(5)

    # 流被按住：短暂等待必须超时（不提前报闲）。
    assert agent.wait_for_idle(0.05) is False
    release.set()
    assert agent.wait_for_idle(5) is True
    worker.join(5)
    assert outcome and outcome[0].reply == "醒来了"


def test_continue_rejects_empty_context():
    """Pi continue() 契约：空上下文没有可继续的东西——直接拒绝。"""
    agent = Agent()
    client = QueueClient(ModelResponse(text="不应出现", usage=Usage(1, 1)))
    with pytest.raises(ValueError, match="nothing to continue"):
        agent.continue_(
            _context(), _config(),
            stream_fn=client.as_stream_fn(), emit=lambda _k, _d: None,
        )
    assert client.calls == []  # 模型从未被调用
    assert agent.is_running is False


def test_continue_rejects_assistant_tip_without_queues():
    """assistant 已答完且无队列：continue 无条件调模型与 Pi 相反——拒绝。"""
    agent = Agent()
    context = _context(user_message("唯一的问题"))
    client = QueueClient(ModelResponse(text="第一答", usage=Usage(2, 2)))
    emit = lambda _kind, _data: None  # noqa: E731

    first = agent.prompt(context, _config(), stream_fn=client.as_stream_fn(), emit=emit)
    assert first.status == "completed"
    assert context.messages[-1].role == "assistant"

    with pytest.raises(ValueError, match="nothing to continue"):
        agent.continue_(context, _config(), stream_fn=client.as_stream_fn(), emit=emit)
    assert len(client.calls) == 1  # 第二次没有调用模型
    assert agent.is_running is False


def test_continue_allows_user_tip_without_appending():
    """user 结尾（如被中断的问题）：continue 直接回答它,不追加任何
    user 消息。"""
    agent = Agent()
    context = _context(user_message("唯一的问题"))
    client = QueueClient(ModelResponse(text="答", usage=Usage(2, 2)))

    result = agent.continue_(
        context, _config(),
        stream_fn=client.as_stream_fn(), emit=lambda _k, _d: None,
    )
    assert result.status == "completed"
    assert result.reply == "答"
    user_count = sum(isinstance(m, UserMessage) for m in context.messages)
    assert user_count == 1


def test_continue_allows_tool_result_tip():
    """tool 结尾(Pi 的 toolResult:工具结果已落盘、模型还没看到):
    continue 合法,模型接着工具结果继续。"""
    agent = Agent()
    context = _context(
        user_message("查一下"),
        AssistantMessage(
            text="", tool_calls=(ToolCallContent(id="c1", name="list_dir"),)
        ),
        AgentToolResultMessage(tool_call_id="c1", tool_name="list_dir", content="a.py"),
    )
    assert context.messages[-1].role == "tool"
    client = QueueClient(ModelResponse(text="查完了", usage=Usage(2, 2)))

    result = agent.continue_(
        context, _config(),
        stream_fn=client.as_stream_fn(), emit=lambda _k, _d: None,
    )
    assert result.status == "completed"
    assert result.reply == "查完了"


def test_continue_with_queued_follow_up_injects_it_first():
    """assistant 结尾但有排队 follow-up:continue 合法——队列消息经
    initial 通道在首次模型调用前注入(source 保留为 follow_up),
    再驱动下一轮。"""
    agent = Agent()
    context = _context(user_message("问题"))
    client = QueueClient(
        ModelResponse(text="第一答", usage=Usage(2, 2)),
        ModelResponse(text="答后续", usage=Usage(2, 2)),
    )
    emit = lambda _kind, _data: None  # noqa: E731

    first = agent.prompt(context, _config(), stream_fn=client.as_stream_fn(), emit=emit)
    assert first.status == "completed"
    assert context.messages[-1].role == "assistant"

    agent.follow_up_queue.enqueue("顺便做这个")
    user_count_before = sum(isinstance(m, UserMessage) for m in context.messages)

    second = agent.continue_(context, _config(), stream_fn=client.as_stream_fn(), emit=emit)
    assert second.status == "completed"
    assert second.reply == "答后续"
    # follow-up 被注入为恰好一条新 user 消息,且是模型看到的最后一条。
    user_messages = [m for m in context.messages if isinstance(m, UserMessage)]
    assert len(user_messages) == user_count_before + 1
    assert user_messages[-1].content == "顺便做这个"
    # 首次(也是唯一一次)模型调用的末条消息就是它——先注入,再运行。
    assert client.calls[-1]["messages"][-1]["content"] == "顺便做这个"


def _echo_tools(calls: list):
    """最小真实工具:记录调用,返回 ok:<value>。"""
    from lsm_harness.agent.tools import Tool

    tools = ToolRegistry()
    tools.register(
        Tool(
            "echo",
            "echo a value",
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
            lambda value="": calls.append(value) or f"ok:{value}",
            "local_write",
        )
    )
    return tools


def test_continue_runs_queued_follow_ups_in_order_across_tool_chain():
    """两条 follow-up,第一条含工具调用——Pi 顺序:

    follow-up 1 → 工具调用 → 工具结果 → 模型完成 follow-up 1 → follow-up 2

    关键断言:第二次模型调用的末条消息是**工具结果**,而不是 follow-up 2
    (one-at-a-time:follow-up 2 必须等第一条完整结束才注入);两条都以
    source=follow_up 注入,不产生 loop.steered 事件。"""
    from lsm_harness.ai.types import ToolCall

    tool_calls_seen: list[str] = []
    tools = _echo_tools(tool_calls_seen)
    agent = Agent()
    context = AgentContext(
        system_prompt="system", messages=[user_message("问题")], tools=tools
    )
    client = QueueClient(
        ModelResponse(text="第一答", usage=Usage(2, 2)),
        ModelResponse(
            text="", tool_calls=[ToolCall("t1", "echo", {"value": "一"})],
            stop_reason="tool_calls", usage=Usage(2, 2),
        ),
        ModelResponse(text="第一件事完成", usage=Usage(2, 2)),
        ModelResponse(text="第二件完成", usage=Usage(2, 2)),
    )
    event_kinds: list[str] = []
    emit = lambda kind, _data: event_kinds.append(kind)  # noqa: E731

    first = agent.prompt(context, _config(), stream_fn=client.as_stream_fn(), emit=emit)
    assert first.status == "completed"
    assert context.messages[-1].role == "assistant"

    agent.follow_up_queue.enqueue("第一件事")
    agent.follow_up_queue.enqueue("第二件事")
    result = agent.continue_(
        context, _config(), stream_fn=client.as_stream_fn(), emit=emit
    )

    assert result.status == "completed"
    assert result.reply == "第二件完成"
    assert tool_calls_seen == ["一"]  # 工具真的执行了
    assert len(client.calls) == 4  # prompt 1 + continue 3
    # 第二次模型调用(continue 的工具链中)末条消息是工具结果……
    assert client.calls[2]["messages"][-1]["role"] == "tool"
    # ……follow-up 2 只在第一条完整结束后才注入,驱动最后一次调用。
    assert client.calls[3]["messages"][-1]["content"] == "第二件事"
    # source 保留:两条都是 follow_up,没有被错误搬成 steering。
    assert event_kinds.count("loop.followed_up") == 2
    assert "loop.steered" not in event_kinds


def test_continue_prefers_queued_steering_over_follow_up():
    """assistant 结尾、两种队列都有货:先 drain 一批 steering 作为
    initial(Pi 的 drain 顺序),follow-up 等 steering 轮结束后再注入;
    各自 source 正确。"""
    agent = Agent()
    context = _context(user_message("问题"))
    client = QueueClient(
        ModelResponse(text="第一答", usage=Usage(2, 2)),
        ModelResponse(text="答插队", usage=Usage(2, 2)),
        ModelResponse(text="答后续", usage=Usage(2, 2)),
    )
    event_kinds: list[str] = []
    emit = lambda kind, _data: event_kinds.append(kind)  # noqa: E731

    first = agent.prompt(context, _config(), stream_fn=client.as_stream_fn(), emit=emit)
    assert first.status == "completed"

    agent.follow_up_queue.enqueue("后续的事")
    agent.steering_queue.enqueue("插队的事")
    result = agent.continue_(
        context, _config(), stream_fn=client.as_stream_fn(), emit=emit
    )

    assert result.status == "completed"
    assert result.reply == "答后续"
    assert len(client.calls) == 3
    # steering 先行:continue 的首次模型调用末条消息是插队内容……
    assert client.calls[1]["messages"][-1]["content"] == "插队的事"
    # ……follow-up 在其后。
    assert client.calls[2]["messages"][-1]["content"] == "后续的事"
    assert event_kinds.count("loop.steered") == 1
    assert event_kinds.count("loop.followed_up") == 1
