"""阶段 A 批 5:handleRunFailure —— loop 崩溃走合成失败链(Pi)。

钉住:
- loop 内异常(fail-fast 监听者、适配器、未知 bug)不再抛出,而是
  合成一条失败 assistant 消息并走**正常事件链**(message_start/end
  → agent_end),返回 failed TraceResult;错误进 state.error_message;
- 守卫:agent_end 已归约(_terminal_seen)就不再驱动合成链——
  agent_end 上的监听者崩溃不会二次发射 agent_end,异常如实上抛;
- Harness 零改动:respond 对 loop 内崩溃走既有 failed 分支。

可观察行为变化(批 5 交付说明单列):loop 崩溃从**抛出**变为**返回
failed**;loop 外(启动动作/prepare_context)失败仍抛出。
"""

from __future__ import annotations

import pytest

from lsm_harness.agent import Agent, AgentLoopConfig, AgentState
from lsm_harness.agent.events import (
    AgentEndEvent,
    MessageEndEvent,
    MessageUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from lsm_harness.agent.tools import ToolRegistry
from lsm_harness.ai.types import Model, ModelResponse, Usage
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect

from helpers import QueueClient

MODEL = Model(id="test-model", api="legacy-client", provider="legacy")


def _agent(stream_fn) -> tuple[Agent, AgentLoopConfig]:
    state = AgentState(
        system_prompt="system",
        model=MODEL,
        tools=ToolRegistry(),
    )
    agent = Agent(initial_state=state, stream_fn=stream_fn)
    return agent, AgentLoopConfig(model=MODEL, max_iterations=10, max_tokens=100)


def _boom_on(event_type, message="listener bug"):
    def boom(event):
        if isinstance(event, event_type):
            raise RuntimeError(message)

    return boom


def test_loop_crash_synthesizes_failure_chain_via_events():
    """turn_start 上 fail-fast 监听者崩溃 → 不抛出;合成链走正常事件:
    持久监听者依次看到合成 assistant 的 message_end 与恰好一次
    agent_end;error 进 state.error_message;投影清空。"""
    client = QueueClient(ModelResponse(text="不应出现", usage=Usage(1, 1)))
    agent, config = _agent(client.as_stream_fn())
    seen: list = []
    agent.subscribe(seen.append)  # 先订阅计数者,再订阅崩溃者
    agent.subscribe(_boom_on(TurnStartEvent))

    result = agent.prompt("问", config)

    assert result.status == "failed"
    assert result.error == "RuntimeError: listener bug"
    assert agent.state.error_message == "RuntimeError: listener bug"
    # 合成 assistant 消息经 message_end 事件进入 state(非直塞 append)。
    assistant_ends = [
        e for e in seen
        if isinstance(e, MessageEndEvent) and e.source == "assistant"
    ]
    assert len(assistant_ends) == 1
    assert agent.state.messages[-1] is assistant_ends[0].message
    # 恰好一次 agent_end,且带 error。
    agent_ends = [e for e in seen if isinstance(e, AgentEndEvent)]
    assert len(agent_ends) == 1
    assert agent_ends[0].error == "RuntimeError: listener bug"
    assert agent_ends[0].status == "failed"
    turn_ends = [e for e in seen if isinstance(e, TurnEndEvent)]
    assert len(turn_ends) == 1
    assert turn_ends[0].message is assistant_ends[0].message
    assert turn_ends[0].message.stop_reason == "error"
    assert turn_ends[0].message.error_message == "RuntimeError: listener bug"
    assert turn_ends[0].message.model == "test-model"
    # 生命周期收尾干净。
    assert agent.state.is_streaming is False
    assert agent.state.streaming_message is None
    assert agent.is_running is False
    assert client.calls == []  # 崩溃在首个模型调用之前


def test_crash_on_agent_end_does_not_double_emit():
    """agent_end 的归约先于监听者(state-first):监听者在 agent_end 上
    崩溃时 _terminal_seen 已置位——守卫阻止合成链二次发射 agent_end,
    原异常如实上抛。"""
    client = QueueClient(ModelResponse(text="答", usage=Usage(1, 1)))
    agent, config = _agent(client.as_stream_fn())
    seen: list = []
    agent.subscribe(seen.append)
    agent.subscribe(_boom_on(AgentEndEvent, "on agent_end"))

    with pytest.raises(RuntimeError, match="on agent_end"):
        agent.prompt("问", config)

    agent_ends = [e for e in seen if isinstance(e, AgentEndEvent)]
    assert len(agent_ends) == 1  # 没有第二次 agent_end
    assert agent.state.is_streaming is False  # finally teardown 仍执行
    assert agent.is_running is False


def test_prompt_ingestion_listener_failure_uses_failure_chain():
    client = QueueClient(ModelResponse(text="unused", usage=Usage(1, 1)))
    agent, config = _agent(client.as_stream_fn())

    def fail_user_end(event):
        if isinstance(event, MessageEndEvent) and event.source == "user":
            raise RuntimeError("ingestion failed")

    agent.subscribe(fail_user_end)
    result = agent.prompt("问", config)

    assert result.status == "failed"
    assert result.error == "RuntimeError: ingestion failed"
    assert agent.state.error_message == "RuntimeError: ingestion failed"
    assert agent.is_running is False
    assert client.calls == []


def test_stream_listener_failure_is_not_mislabeled_as_context_conversion():
    client = QueueClient(ModelResponse(text="answer", usage=Usage(1, 1)))
    agent, config = _agent(client.as_stream_fn())
    agent.subscribe(_boom_on(MessageUpdateEvent, "stream listener failed"))

    result = agent.prompt("问", config)

    assert result.status == "failed"
    assert result.error == "RuntimeError: stream listener failed"
    assert "模型上下文转换失败" not in result.error


def test_harness_respond_returns_failed_on_loop_crash(tmp_path):
    """Harness 零改动:loop 内崩溃 → respond 走既有 failed 分支(不抛),
    树里留下 user 问题与合成 assistant 条目(recorder listener 正常
    记录合成链)。"""
    settings = Settings(api_key="scripted", home=tmp_path)
    conn = connect(tmp_path, check_same_thread=False)
    client = QueueClient(ModelResponse(text="不应出现", usage=Usage(1, 1)))
    app = Harness(
        settings=settings, client=client, conn=conn,
        stream_fn=client.as_stream_fn(),
    )
    try:
        app.agent.subscribe(_boom_on(TurnStartEvent, "mid-loop crash"))
        result = app.respond("会被记录的问题", source="test")
        assert result.status == "failed"
        assert "mid-loop crash" in (result.error or "")
        assert app.agent.state.error_message is not None
        assert app.is_running is False

        import json

        entries = [
            json.loads(line)
            for line in app.session.jsonl_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        roles = [
            e["message"]["role"]
            for e in entries
            if e.get("type") == "message"
        ]
        assert "user" in roles  # 问题仍在树里
        assert roles[-1] == "assistant"  # 合成失败消息也落了树
        failure_entry = next(
            e for e in reversed(entries)
            if e.get("type") == "message"
            and e.get("message", {}).get("role") == "assistant"
        )
        assert failure_entry["message"]["stop_reason"] == "error"
        assert "mid-loop crash" in failure_entry["message"]["error_message"]
    finally:
        app.close()
