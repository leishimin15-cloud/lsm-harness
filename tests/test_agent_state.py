"""阶段 A 批 1:AgentState 定义与 sink 归约(Pi processEvents 语义)。

钉住:
- AgentState 默认值与四个 runtime 字段的只读性;
- messages 赋值 = rebind(有记录的 Pi 偏离,loop 运行期别名该列表);
- sink 对 ToolExecutionStart/End 的 pending_tool_calls copy-on-write 归约;
- sink 对 AgentEnd 的归约(清 streaming、写 error_message);
- AgentEventSink(messages=...) 旧构造的向后兼容。
"""

from __future__ import annotations

import pytest

from lsm_harness.agent.events import (
    AgentEndEvent,
    AgentEventSink,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from lsm_harness.agent.messages import assistant_message, user_message
from lsm_harness.agent.state import AgentState


def test_state_defaults_and_readonly_runtime_fields():
    state = AgentState()
    assert state.system_prompt == ""
    assert state.model is None
    assert state.thinking_level == "disabled"
    assert state.messages == []
    assert state.is_streaming is False
    assert state.streaming_message is None
    assert state.pending_tool_calls == frozenset()
    assert state.error_message is None
    # 四个 runtime 字段只有 sink 归约器与 Agent 生命周期可写。
    for prop in (
        "is_streaming",
        "streaming_message",
        "pending_tool_calls",
        "error_message",
    ):
        with pytest.raises(AttributeError):
            setattr(state, prop, None)


def test_state_messages_assignment_rebinds():
    """Pi set messages 会复制顶层数组;我们选 rebind——loop 在 run 期间
    别名该列表,截断/prepare_next_turn 的重建必须替换同一引用。此测试
    钉住 rebind 语义,防止未来被"修"成 copy。"""
    state = AgentState()
    xs = [user_message("hi")]
    state.messages = xs
    assert state.messages is xs


def test_sink_reduces_pending_tool_calls_copy_on_write():
    sink = AgentEventSink()
    before = sink.state.pending_tool_calls
    sink.process_event(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash", label="Bash")
    )
    mid = sink.state.pending_tool_calls
    assert mid == frozenset({"c1"})
    assert mid is not before  # copy-on-write(镜 Pi ReadonlySet)
    sink.process_event(
        ToolExecutionEndEvent(
            tool_call_id="c1", tool_name="bash", label="Bash", is_error=False
        )
    )
    assert sink.state.pending_tool_calls == frozenset()
    assert sink.state.pending_tool_calls is not mid


def test_sink_reduces_error_message_from_agent_end():
    sink = AgentEventSink()
    sink.process_event(MessageStartEvent(message=assistant_message("partial")))
    assert sink.state.streaming_message is not None

    sink.process_event(
        AgentEndEvent(status="failed", stop_reason="error", error="boom")
    )
    assert sink.state.error_message == "boom"
    assert sink.state.streaming_message is None  # Pi: agent_end 清 streaming

    # 成功收尾(error="")不动 error_message。
    ok = AgentEventSink()
    ok.process_event(AgentEndEvent(status="completed", stop_reason="stop"))
    assert ok.state.error_message is None


def test_sink_backcompat_messages_kwarg():
    """旧构造 AgentEventSink(messages=xs) 继续成立:sink.messages 就是
    xs 本体,message_end 追加进 xs(test_chapter7 的语义回归)。"""
    xs = [user_message("hi")]
    sink = AgentEventSink(messages=xs)
    assert sink.messages is xs
    reply = assistant_message("hello")
    sink.process_event(MessageEndEvent(message=reply))
    assert xs[-1] is reply
    assert sink.streaming_message is None


# ── 批 2:持久 sink + prompt/continue_ 的 state-first 钉测 ──────────


def _bare_agent(*messages, stream_fn, tools=None):
    """裸 Agent:无 CLI/TUI/SQLite,initial_state + stream_fn 即可跑。"""
    from lsm_harness.agent import Agent, AgentLoopConfig
    from lsm_harness.agent.tools import ToolRegistry
    from lsm_harness.ai.types import Model

    model = Model(id="test-model", api="legacy-client", provider="legacy")
    state = AgentState(
        system_prompt="system",
        model=model,
        tools=tools or ToolRegistry(),
        messages=list(messages),
    )
    agent = Agent(initial_state=state, stream_fn=stream_fn)
    config = AgentLoopConfig(model=model, max_iterations=10, max_tokens=100)
    return agent, config


def test_prompt_ingests_user_message_via_events():
    """prompt(text) 的 user 消息走 kernel 事件摄入:持久监听者收到
    MessageEnd(source="user") 时,state.messages 里已经有它
    (state-before-listeners);字符串通道保持沉默。"""
    from helpers import QueueClient
    from lsm_harness.ai.types import ModelResponse, Usage

    client = QueueClient(ModelResponse(text="好", usage=Usage(1, 1)))
    agent, config = _bare_agent(stream_fn=client.as_stream_fn())

    observed: list[tuple[str, bool]] = []
    strings: list[str] = []

    def listener(event):
        if isinstance(event, MessageEndEvent) and event.source == "user":
            # 事件到达监听者时,state 必须已更新(先归约,后通知)。
            observed.append(("user_end", event.message in agent.state.messages))

    agent.subscribe(listener)
    result = agent.prompt(
        "你好", config, emit=lambda kind, _data: strings.append(kind)
    )

    assert result.status == "completed"
    assert observed == [("user_end", True)]
    assert agent.state.messages[0].role == "user"
    assert agent.state.messages[0].content == "你好"
    # legacy adapter 只转发 steering/follow_up:source="user" 零字符串事件。
    assert "message.started" not in strings
    assert "message.completed" not in strings


def test_state_observable_mid_run():
    """运行中:agent_end 到达监听者时 is_streaming 仍为 True(Pi
    finishRun 在监听者之后才清理);run 返回后投影全部清空。"""
    from helpers import QueueClient
    from lsm_harness.agent.events import AgentEndEvent
    from lsm_harness.ai.types import ModelResponse, Usage

    client = QueueClient(ModelResponse(text="答", usage=Usage(1, 1)))
    agent, config = _bare_agent(stream_fn=client.as_stream_fn())

    at_agent_end: list[tuple[bool, object]] = []

    def listener(event):
        if isinstance(event, AgentEndEvent):
            at_agent_end.append(
                (agent.state.is_streaming, agent.state.streaming_message)
            )

    agent.subscribe(listener)
    agent.prompt("问", config)

    assert at_agent_end == [(True, None)]  # 监听者先看到仍 streaming 的 state
    assert agent.state.is_streaming is False  # teardown 之后
    assert agent.state.streaming_message is None
    assert agent.state.pending_tool_calls == frozenset()
    assert agent.state.error_message is None


def test_pending_tool_calls_visible_during_tool_execution():
    """工具执行窗口:ToolExecutionStart 归约后、End 归约前,
    state.pending_tool_calls 对监听者可见(copy-on-write frozenset)。"""
    from helpers import QueueClient
    from lsm_harness.agent.tools import Tool, ToolRegistry
    from lsm_harness.ai.types import ModelResponse, ToolCall, Usage

    tools = ToolRegistry()
    tools.register(
        Tool(
            "echo", "echo",
            {"type": "object", "properties": {"value": {"type": "string"}}},
            lambda value="": f"ok:{value}",
            "local_write",
        )
    )
    client = QueueClient(
        ModelResponse(
            text="", tool_calls=[ToolCall("t1", "echo", {"value": "x"})],
            stop_reason="tool_calls", usage=Usage(2, 2),
        ),
        ModelResponse(text="完", usage=Usage(2, 2)),
    )
    agent, config = _bare_agent(stream_fn=client.as_stream_fn(), tools=tools)

    seen: list[frozenset] = []

    def listener(event):
        if isinstance(event, ToolExecutionStartEvent):
            seen.append(agent.state.pending_tool_calls)
        elif isinstance(event, ToolExecutionEndEvent):
            seen.append(agent.state.pending_tool_calls)

    agent.subscribe(listener)
    result = agent.prompt("用工具", config)

    assert result.status == "completed"
    assert seen == [frozenset({"t1"}), frozenset()]
    assert agent.state.pending_tool_calls == frozenset()


def test_replace_state_messages_then_continue():
    """验收形态:Session 切换/分支 = 整体替换 agent.state.messages;
    user 结尾的新 transcript 上 continue() 直接回答,不需 AgentContext。"""
    from helpers import QueueClient
    from lsm_harness.ai.types import ModelResponse, Usage

    client = QueueClient(ModelResponse(text="接着答", usage=Usage(1, 1)))
    agent, config = _bare_agent(stream_fn=client.as_stream_fn())

    # 会话层换入另一段 transcript(Pi: buildSessionContext 整体替换)。
    agent.state.messages = [user_message("被中断的问题")]
    result = agent.continue_(config)

    assert result.status == "completed"
    assert result.reply == "接着答"
    assert [m.role for m in agent.state.messages] == ["user", "assistant"]


# ── 批 4:config 切分(model/thinking 入 state)───────────────────────


def test_state_model_drives_run_without_config():
    """config=None 时,run 的模型来自 state.model(Pi:config 是 per-run
    覆盖,state 是身份)。"""
    from helpers import QueueClient
    from lsm_harness.ai.types import ModelResponse, Usage

    client = QueueClient(ModelResponse(text="答", usage=Usage(1, 1)))
    agent, _config = _bare_agent(stream_fn=client.as_stream_fn())

    result = agent.prompt("问")  # 无 config:预算走模块默认,模型走 state
    assert result.status == "completed"
    assert client.calls[0]["model"] == "test-model"  # QueueClient 记的是 model.id


def test_explicit_config_overrides_state_model():
    """显式 config.model 覆盖 state.model(per-run 覆盖);state 不被改写。"""
    from helpers import QueueClient
    from lsm_harness.agent import AgentLoopConfig
    from lsm_harness.ai.types import Model, ModelResponse, Usage

    client = QueueClient(ModelResponse(text="答", usage=Usage(1, 1)))
    agent, _ = _bare_agent(stream_fn=client.as_stream_fn())
    override = Model(id="override-model", api="legacy-client", provider="legacy")
    config = AgentLoopConfig(model=override, max_iterations=10, max_tokens=100)

    result = agent.prompt("问", config)
    assert result.status == "completed"
    assert client.calls[0]["model"] == "override-model"
    assert agent.state.model.id == "test-model"  # 覆盖不污染身份


def test_no_model_anywhere_raises_clear_error():
    """state.model 与 config.model 都缺失:清晰的 ValueError,不是
    深处的 AttributeError。"""
    import pytest as _pytest

    from helpers import QueueClient
    from lsm_harness.agent import Agent
    from lsm_harness.ai.types import ModelResponse, Usage

    client = QueueClient(ModelResponse(text="不应出现", usage=Usage(1, 1)))
    agent = Agent(stream_fn=client.as_stream_fn())  # state.model 为 None
    with _pytest.raises(ValueError, match="no model"):
        agent.prompt("问")
    assert client.calls == []
    assert agent.is_running is False  # 失败不 wedge


def test_reset_signal_and_queue_modes_match_pi_surface():
    from helpers import QueueClient
    from lsm_harness.ai.types import ModelResponse, Usage

    client = QueueClient(ModelResponse(text="ok", usage=Usage(1, 1)))
    agent, _config = _bare_agent(
        user_message("old"), stream_fn=client.as_stream_fn()
    )
    assert agent.signal is None
    active = agent.begin()
    assert agent.signal is active.interrupt
    agent.finish(active)

    agent.steering_mode = "all"
    agent.follow_up_mode = "all"
    assert agent.steering_mode == "all"
    assert agent.follow_up_mode == "all"
    agent.steering_queue.enqueue("queued")
    agent.state._error_message = "old error"
    agent.reset()
    assert agent.state.messages == []
    assert agent.state.error_message is None
    assert agent.has_queued_messages() is False
