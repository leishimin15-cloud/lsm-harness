"""阶段 5 批 1:TUI state/adapter 纯测(不导入 Textual)。

对齐 Tau 的分层测试:`TuiState` + `TuiEventAdapter` 可以脱离 App
直接喂事件断言状态;折叠开关的全量重渲染(render_lines)也在这里
钉住,App 层只做增量追加与清屏重放。
"""

from __future__ import annotations

import sqlite3

from lsm_harness.events import make_event
from lsm_harness.gateway.tui.adapter import TuiEventAdapter
from lsm_harness.gateway.tui.state import TuiState

RENDERERS = {
    "exec": (
        lambda args: f"RUN {args.get('command', '')}",
        lambda output, _details: f"DONE({len(output)})",
    ),
}


def _sequence():
    """两段文本 + 两个工具(成功/失败) + usage + 结束。"""
    return [
        make_event("llm.text.delta", "t", {"text": "你好,"}),
        make_event("llm.text.delta", "t", {"text": "世界"}),
        make_event("llm.text.end", "t", {}),
        make_event("tool.requested", "t", {
            "tool": "exec", "label": "Run command", "tool_call_id": "c1",
            "args": {"command": "ls"},
        }),
        make_event("tool.completed", "t", {
            "tool": "exec", "label": "Run command", "tool_call_id": "c1",
            "status": "ok", "output": "file1\nfile2",
        }),
        make_event("tool.requested", "t", {
            "tool": "mystery", "label": "Fancy label", "tool_call_id": "c2",
            "args": {},
        }),
        make_event("tool.completed", "t", {
            "tool": "mystery", "label": "Fancy label", "tool_call_id": "c2",
            "status": "error", "output": "boom",
        }),
        make_event("llm.completed", "t", {
            "role": "main", "model": "m",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }),
    ]


def _feed_all(adapter, events):
    lines = []
    for event in events:
        lines.extend(adapter.feed(event))
    return lines


def test_adapter_projects_events_into_state():
    state = TuiState()
    adapter = TuiEventAdapter(state, RENDERERS)

    lines = _feed_all(adapter, _sequence())

    # 流式 delta 聚合成一段 assistant 文本(不碎片化)
    assistants = [i for i in state.items if i.role == "assistant"]
    assert [i.text for i in assistants] == ["你好,世界"]
    # 工具条目:requested 一行,completed 后带状态与结果行
    tools = [i for i in state.items if i.role == "tool"]
    assert [(t.tool_label, t.tool_status) for t in tools] == [
        ("Run command", "ok"),
        ("Fancy label", "error"),
    ]
    assert "RUN ls" in tools[0].text  # 自定义 renderer 生效
    assert "DONE(11)" in tools[0].result_line
    assert "✗" in tools[1].result_line and "boom" in tools[1].result_line
    # usage 只更新状态栏,不产生聊天行
    assert state.tokens == "↑10 ↓5"
    assert not any("↑" in line for line in lines)
    # 增量行与全量渲染一致(展开态)
    assert lines == state.render_lines()


def test_fold_toggle_renders_history_too():
    """折叠开关对已经渲染过的历史同样生效(重渲染语义)。

    折叠 = 工具只留调用行、隐藏结果预览;**错误永远显示**(折叠
    不能吞掉失败)。
    """
    state = TuiState()
    adapter = TuiEventAdapter(state, RENDERERS)
    _feed_all(adapter, _sequence())

    expanded = state.render_lines()
    state.toggle_tool_results()
    folded = state.render_lines()

    # 折叠:成功工具只剩调用行,预览消失;错误行保留;其余不动
    assert len(folded) < len(expanded)
    assert any("⚙ RUN ls" in line for line in folded)
    assert not any("DONE(11)" in line for line in folded)
    assert any("✗" in line and "boom" in line for line in folded)
    assert "你好,世界" in folded
    # 切回展开完全复原
    state.toggle_tool_results()
    assert state.render_lines() == expanded


def test_collapsed_incremental_writes_match_replay():
    """折叠态下增量写入与全量重放一致(成功工具不写完成行,错误照写)。"""
    state = TuiState(show_tool_results=False)
    adapter = TuiEventAdapter(state, RENDERERS)

    lines = _feed_all(adapter, _sequence())

    assert not any("DONE(11)" in line for line in lines)
    assert any("✗" in line and "boom" in line for line in lines)  # 错误不折叠
    assert lines == state.render_lines()


def test_aborted_event_adds_note_line():
    state = TuiState()
    adapter = TuiEventAdapter(state, {})

    lines = _feed_all(adapter, [make_event("loop.aborted", "t", {})])

    assert lines == ["[yellow]⏎ Interrupted[/yellow]"]
    assert state.items[-1].role == "note"


def test_sqlite_threadsafety_is_serialized():
    """TUI 在 worker 线程共享 SQLite 连接(check_same_thread=False)
    的前提:底层 sqlite3 编译为序列化模式(连接可跨线程使用)。"""
    assert sqlite3.threadsafety == 3
