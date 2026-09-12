"""阶段一:TUI state/adapter 纯测(typed events,不导入 Textual)。

`TuiState` + `TuiEventAdapter` 脱离 App 直接喂 CodingSession typed
events 断言状态;折叠开关的全量重渲染(render_lines)也在这里钉住,
App 层只做增量追加与清屏重放。

覆盖(方案 §19):assistant 流式合并、message_end 落定、thinking 独立
累积、同名工具按 tool_call_id 区分、tool start/update/end 顺序、
error 默认展开、duration、queue、retry、compaction、abort、settled、
usage 累积、未知事件不崩。
"""

from __future__ import annotations

import sqlite3

from lsm_harness.agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
)
from lsm_harness.ai.messages import AssistantMessage, ToolResultMessage
from lsm_harness.ai.types import AssistantMessageEvent, ModelResponse
from lsm_harness.coding_agent.events import (
    AgentSettledEvent,
    AutoRetryEndEvent,
    AutoRetryStartEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    EntryAppendedEvent,
    ModelChangedEvent,
    QueueUpdateEvent,
    ThinkingLevelChangedEvent,
)
from lsm_harness.gateway.tui.adapter import TuiEventAdapter
from lsm_harness.gateway.tui.state import MessageView, ToolView, TuiState

RENDERERS = {
    "exec": (
        lambda args: f"RUN {args.get('command', '')}",
        lambda output, _details: f"DONE({len(output)})",
    ),
}


def _assistant_update(text: str, thinking: str = "") -> MessageUpdateEvent:
    partial = ModelResponse(text=text, thinking=thinking)
    return MessageUpdateEvent(
        message=AssistantMessage(text=text, thinking=thinking),
        assistant_message_event=AssistantMessageEvent(
            "text_delta", partial, text_delta=text
        ),
    )


def _run_sequence():
    """一个完整 typed 序列:assistant 流式 + 两个工具(成功/失败)
    + usage + 结束。"""
    return [
        AgentStartEvent(model="m"),
        MessageStartEvent(message=AssistantMessage(), source="assistant"),
        _assistant_update("你好,"),
        _assistant_update("你好,世界"),
        MessageEndEvent(
            message=AssistantMessage(text="你好,世界"), source="assistant"
        ),
        ToolExecutionStartEvent(
            tool_call_id="c1", tool_name="exec", label="Run command",
            args={"command": "ls"},
        ),
        ToolExecutionUpdateEvent(
            tool_call_id="c1", tool_name="exec", label="Run command",
            partial="file1",
        ),
        ToolExecutionEndEvent(
            tool_call_id="c1", tool_name="exec", label="Run command",
            is_error=False,
        ),
        MessageEndEvent(
            message=ToolResultMessage(
                tool_call_id="c1", tool_name="exec", content="file1\nfile2",
            ),
            source="tool",
        ),
        ToolExecutionStartEvent(
            tool_call_id="c2", tool_name="exec", label="Run command",
            args={"command": "pytest"},
        ),
        ToolExecutionEndEvent(
            tool_call_id="c2", tool_name="exec", label="Run command",
            is_error=True,
        ),
        MessageEndEvent(
            message=ToolResultMessage(
                tool_call_id="c2", tool_name="exec", content="boom",
                is_error=True,
            ),
            source="tool",
        ),
        TurnEndEvent(
            turn_index=1, model="m", stop_reason="stop", status="completed",
            usage={"input_tokens": 10, "output_tokens": 5},
            tool_count=2, tool_error_count=1,
        ),
        AgentEndEvent(status="completed", stop_reason="stop"),
        AgentSettledEvent(status="completed"),
    ]


def _feed_all(adapter, events):
    lines = []
    for event in events:
        lines.extend(adapter.feed(event))
    return lines


def test_streaming_updates_merge_into_one_assistant_message():
    """多次 message_update 合并为一条 assistant 消息(不碎片化),
    message_end 以最终快照落定并结束 streaming。"""
    state = TuiState()
    adapter = TuiEventAdapter(state, RENDERERS)

    lines = _feed_all(adapter, _run_sequence())

    assistants = [
        i for i in state.transcript
        if isinstance(i, MessageView) and i.role == "assistant"
    ]
    assert len(assistants) == 1
    assert assistants[0].text == "你好,世界"
    assert assistants[0].is_streaming is False
    assert state._streaming is None
    assert state.is_running is False  # AgentEndEvent 落定
    # 增量行与全量渲染一致(展开态)
    assert lines == state.render_lines()


def test_thinking_accumulates_separately():
    """thinking 与正文分开累积;show_thinking 开关控制渲染。"""
    state = TuiState()
    adapter = TuiEventAdapter(state, {})

    _feed_all(adapter, [
        MessageStartEvent(message=AssistantMessage(), source="assistant"),
        _assistant_update("答", thinking="先想"),
        _assistant_update("答", thinking="先想再想"),
        MessageEndEvent(
            message=AssistantMessage(text="答", thinking="先想再想"),
            source="assistant",
        ),
    ])

    view = state.transcript[0]
    assert isinstance(view, MessageView)
    assert view.thinking == "先想再想"
    assert view.text == "答"
    assert any("💭" in line and "先想再想" in line for line in state.render_lines())

    state.toggle_thinking()
    assert not any("💭" in line for line in state.render_lines())


def test_same_name_tools_pair_by_call_id():
    """一轮两次调用同一工具:按 tool_call_id 区分,output 不串。"""
    state = TuiState()
    adapter = TuiEventAdapter(state, RENDERERS)

    _feed_all(adapter, _run_sequence())

    tools = [i for i in state.transcript if isinstance(i, ToolView)]
    assert len(tools) == 2
    assert tools[0].tool_call_id == "c1" and tools[0].status == "ok"
    assert tools[1].tool_call_id == "c2" and tools[1].status == "error"
    assert tools[0].output == "file1\nfile2"
    assert tools[1].output == "boom"
    assert tools[1].is_error if hasattr(tools[1], "is_error") else True
    # 自定义 renderer 生效(start 行在 start 时生成;exec 的 result
    # renderer 输出 DONE(len),不帶 output 原文)
    assert "RUN ls" in tools[0].call_line
    assert "DONE(11)" in tools[0].result_line
    assert "✗" in tools[1].result_line and "DONE(4)" in tools[1].result_line


def test_tool_progress_and_duration():
    """progress 按 update 累积;end 后 duration_ms 可算。"""
    state = TuiState()
    adapter = TuiEventAdapter(state, {})

    lines = _feed_all(adapter, [
        ToolExecutionStartEvent(
            tool_call_id="c1", tool_name="exec", label="Run command",
            args={"command": "ls"},
        ),
        ToolExecutionUpdateEvent(
            tool_call_id="c1", tool_name="exec", label="Run command",
            partial="line1",
        ),
        ToolExecutionUpdateEvent(
            tool_call_id="c1", tool_name="exec", label="Run command",
            partial="line2",
        ),
        ToolExecutionEndEvent(
            tool_call_id="c1", tool_name="exec", label="Run command",
            is_error=False,
        ),
        MessageEndEvent(
            message=ToolResultMessage(
                tool_call_id="c1", tool_name="exec", content="out",
            ),
            source="tool",
        ),
    ])

    view = state.tools["c1"]
    assert view.progress == ["line1", "line2"]
    assert view.duration_ms is not None and view.duration_ms >= 0
    assert "· " in view.result_line and "ms" in view.result_line
    assert any("line1" in line for line in lines)  # 进度行增量可见


def test_fold_toggle_renders_history_too():
    """折叠开关对已渲染历史同样生效;错误行永不折叠。"""
    state = TuiState()
    adapter = TuiEventAdapter(state, RENDERERS)
    _feed_all(adapter, _run_sequence())

    expanded = state.render_lines()
    state.toggle_tool_results()
    folded = state.render_lines()

    assert len(folded) < len(expanded)
    assert any("⚙ RUN ls" in line for line in folded)
    assert not any("DONE(11)" in line for line in folded)  # 成功预览被折叠
    assert any("✗" in line and "DONE(4)" in line for line in folded)  # 错误保留
    assert "你好,世界" in folded
    state.toggle_tool_results()
    assert state.render_lines() == expanded


def test_collapsed_incremental_writes_match_replay():
    """折叠态下增量写入与全量重放一致(成功不写结果行,错误照写)。"""
    state = TuiState(show_tool_results=False)
    adapter = TuiEventAdapter(state, RENDERERS)

    lines = _feed_all(adapter, _run_sequence())

    assert not any("DONE(11)" in line for line in lines)
    assert any("✗" in line and "DONE(4)" in line for line in lines)
    assert lines == state.render_lines()


def test_aborted_agent_end_adds_note_line():
    state = TuiState()
    adapter = TuiEventAdapter(state, {})

    lines = _feed_all(adapter, [
        AgentStartEvent(model="m"),
        AgentEndEvent(status="aborted", stop_reason="aborted"),
    ])

    assert lines == ["[yellow]⏎ Interrupted[/yellow]"]
    note = state.transcript[-1]
    assert isinstance(note, MessageView) and note.role == "note"
    assert state.is_running is False


def test_failed_agent_end_records_error():
    state = TuiState()
    adapter = TuiEventAdapter(state, {})

    lines = _feed_all(adapter, [
        AgentStartEvent(model="m"),
        AgentEndEvent(status="failed", stop_reason="error", error="boom"),
    ])

    assert state.error == "boom"
    assert any("boom" in line for line in lines)


def test_queue_retry_compaction_and_identity_events():
    """产品层事件驱动状态栏字段:queue/retry/compaction/model/thinking。"""
    state = TuiState()
    adapter = TuiEventAdapter(state, {})

    lines = _feed_all(adapter, [
        QueueUpdateEvent(steering=("改 X",), follow_up=("之后做 Y",)),
        AutoRetryStartEvent(attempt=2, category="transient",
                            message="rate limited", max_attempts=3),
        AutoRetryEndEvent(success=True, attempt=2),
        CompactionStartEvent(reason="threshold"),
        CompactionEndEvent(reason="threshold", result=True),
        ModelChangedEvent(provider="deepseek", model="deepseek-chat"),
        ThinkingLevelChangedEvent(level="enabled"),
    ])

    assert state.queued_steering == ["改 X"]
    assert state.queued_follow_ups == ["之后做 Y"]
    assert state.is_retrying is False and state.retry_attempt == 2
    assert state.is_compacting is False
    assert state.model == "deepseek-chat"
    assert state.thinking == "enabled"
    assert any("重试" in line for line in lines)
    assert any("Compacting" in line for line in lines)
    assert any("Compaction complete" in line for line in lines)


def test_usage_accumulates_across_turns():
    """record_usage:最近值给状态栏,总量累积(/usage 用)。"""
    state = TuiState()
    adapter = TuiEventAdapter(state, {})

    _feed_all(adapter, [
        TurnEndEvent(turn_index=1, model="m", stop_reason="stop",
                     status="completed",
                     usage={"input_tokens": 10, "output_tokens": 5,
                            "cache_read_tokens": 3},
                     tool_count=0, tool_error_count=0),
        TurnEndEvent(turn_index=2, model="m", stop_reason="stop",
                     status="completed",
                     usage={"input_tokens": 20, "output_tokens": 7},
                     tool_count=0, tool_error_count=0),
    ])

    assert state.tokens == "↑20 ↓7"  # 最近一次
    assert state.total_input_tokens == 30  # 累积
    assert state.total_output_tokens == 12
    assert state.total_cache_read_tokens == 3


def test_unknown_event_does_not_crash():
    """未知/未消费事件(如 EntryAppendedEvent)静默放行。"""
    state = TuiState()
    adapter = TuiEventAdapter(state, {})

    lines = adapter.feed(EntryAppendedEvent(entry=None))

    assert lines == []
    assert state.transcript == []


def test_sqlite_threadsafety_is_serialized():
    """TUI 在 worker 线程共享 SQLite 连接(check_same_thread=False)
    的前提:底层 sqlite3 编译为序列化模式(连接可跨线程使用)。"""
    assert sqlite3.threadsafety == 3


def test_rebuild_restores_tool_calls_from_assistant_message():
    """历史重建:assistant 的 tool_calls 恢复工具卡片(name/args/label
    完整),tool result 按 tool_call_id 配对补齐——重启/切换会话后工具
    卡片不再降级为空参数。"""
    from lsm_harness.ai.messages import (
        AssistantMessage,
        ToolCallContent,
        ToolResultMessage,
        UserMessage,
    )

    state = TuiState()
    adapter = TuiEventAdapter(state, RENDERERS)

    history = [
        UserMessage(content="列个目录"),
        AssistantMessage(
            text="",
            tool_calls=(
                ToolCallContent(id="c1", name="exec", arguments={"command": "ls"}),
                ToolCallContent(id="c2", name="exec", arguments={"command": "pwd"}),
            ),
        ),
        ToolResultMessage(tool_call_id="c1", tool_name="exec", content="a.txt"),
        ToolResultMessage(tool_call_id="c2", tool_name="exec", content="/tmp"),
    ]
    adapter.rebuild_from_messages(history)

    tools = [i for i in state.transcript if isinstance(i, ToolView)]
    assert len(tools) == 2
    # 参数从 assistant.tool_calls 恢复(不再是 {})
    assert tools[0].args == {"command": "ls"}
    assert tools[1].args == {"command": "pwd"}
    # tool_call_id 配对正确,output 不串
    assert tools[0].output == "a.txt" and tools[1].output == "/tmp"
    assert tools[0].status == "ok" and tools[1].status == "ok"
    # 自定义 renderer 生效(RUN <command>)
    assert "RUN ls" in tools[0].call_line
    assert "RUN pwd" in tools[1].call_line
    assert "DONE(5)" in tools[0].result_line


def test_rebuild_tool_result_without_tool_calls_falls_back():
    """老会话的 tool result 没有前置 tool_calls(或跨分支缺失)时,
    回退到最小重建(空参数卡片),不崩。"""
    from lsm_harness.ai.messages import ToolResultMessage

    state = TuiState()
    adapter = TuiEventAdapter(state, RENDERERS)
    adapter.rebuild_from_messages([
        ToolResultMessage(tool_call_id="orphan", tool_name="exec", content="out"),
    ])
    tools = [i for i in state.transcript if isinstance(i, ToolView)]
    assert len(tools) == 1
    assert tools[0].args == {}
    assert tools[0].output == "out"
    assert tools[0].status == "ok"


def test_rebuild_restores_tool_label_from_registry():
    """历史重建:label 从工具注册表查(不是退化成 tool_name)。"""
    from lsm_harness.ai.messages import (
        AssistantMessage,
        ToolCallContent,
        ToolResultMessage,
    )

    state = TuiState()
    adapter = TuiEventAdapter(state, {}, tool_labels={"exec": "运行命令"})
    adapter.rebuild_from_messages([
        AssistantMessage(
            tool_calls=(
                ToolCallContent(id="c1", name="exec", arguments={"command": "ls"}),
            ),
        ),
        ToolResultMessage(tool_call_id="c1", tool_name="exec", content="out"),
    ])
    tool = next(i for i in state.transcript if isinstance(i, ToolView))
    assert tool.label == "运行命令"
    assert "运行命令" in tool.call_line  # 无自定义 renderer 时用 label
