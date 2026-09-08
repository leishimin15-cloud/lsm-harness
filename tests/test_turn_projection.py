"""录制事件序列的前端一致性验收(阶段 3 · 批 3)。

同一段录制的 HarnessEvent 序列,分别走 CLI 与 TUI 的投影消费路径:
- 工具名称、状态、错误和最终结果必须一致;
- 流式文本无重复、无缺失、无碎片化换行(每段恰好一条 text_message);
- label 与自定义 renderer 在两条路径上都生效。

TUI 路径在测试里按当时 tui.py observer 的投影消费方式复刻;阶段 5 起
真实 TUI 的 state/adapter 由 test_tui_state.py 直接覆盖(textual 已装,
另有 test_tui_app.py 的 Pilot 端到端)。
"""

from __future__ import annotations

from io import StringIO

from lsm_harness.coding_agent.cli import _make_observer_and_stream, console
from lsm_harness.coding_agent.turn_projection import (
    TurnProjection,
    render_tool_call,
    render_tool_result,
)
from lsm_harness.events import make_event


RENDERERS = {
    "exec": (
        lambda args: f"RUN {args.get('command', '')}",
        lambda output, _details: f"DONE({len(output)})",
    ),
}


def _recorded_sequence():
    """一段混合录制:两段文本、两个工具(成功+失败)、usage、正常结束。"""
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
            "status": "error", "output": "boom\nstack",
        }),
        make_event("llm.completed", "t", {
            "role": "main", "model": "m",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }),
        make_event("trace.completed", "t", {"reply": "你好,世界"}),
    ]


def _tui_consume(events, renderers):
    """按 tui.py observer 的投影消费方式收集会写进聊天记录的行。"""
    projection = TurnProjection()
    lines: list[str] = []
    for event in events:
        for view_event in projection.feed(event):
            if view_event.kind == "text_message":
                lines.append(view_event.text)
            elif view_event.kind == "tool_requested":
                lines.append(render_tool_call(renderers, view_event.tool))
            elif view_event.kind == "tool_completed":
                lines.append(render_tool_result(renderers, view_event.tool))
    return projection, lines


def _cli_consume(events, renderers):
    """走 CLI 真实 observer(console 输出重定向到内存)。"""
    observer, display = _make_observer_and_stream(renderers)
    capture = StringIO()
    original_file = console.file
    console._file = capture
    try:
        for event in events:
            observer(event)
    finally:
        console._file = original_file
    return display.projection, capture.getvalue()


def test_replay_cli_and_tui_agree_on_tools_text_and_usage():
    events = _recorded_sequence()
    cli_projection, cli_out = _cli_consume(events, RENDERERS)
    tui_projection, tui_lines = _tui_consume(events, RENDERERS)

    # 工具事实一致:名称、label、状态、最终结果。
    for projection in (cli_projection, tui_projection):
        assert [(t.name, t.status) for t in projection.tools] == [
            ("exec", "ok"),
            ("mystery", "error"),
        ]
        assert projection.tools[1].output == "boom\nstack"
        assert projection.usage == {"input_tokens": 10, "output_tokens": 5}
        # 流式文本无重复无缺失。
        assert projection.full_text == "你好,世界"

    # 两条路径渲染出的工具行一致(label/renderer 都生效)。
    # 请求行(⚙ 开头)必须已经带 label——用户从工具启动那一刻就看到它,
    # 不是靠完成行补救。
    tui_requested = [line for line in tui_lines if line.startswith("⚙")]
    assert any("RUN ls" in line for line in tui_requested)
    assert any("Fancy label" in line for line in tui_requested)
    assert any("DONE(11)" in line for line in tui_lines)
    assert "RUN ls" in cli_out and "DONE(11)" in cli_out
    assert "Fancy label" in cli_out
    # 投影状态里的 label 也是请求时就解析好的。
    assert cli_projection.tools[1].label == "Fancy label"
    assert tui_projection.tools[1].label == "Fancy label"
    # 错误结果行带错误状态标识。
    assert any("✗" in line for line in tui_lines)
    assert "✗" in cli_out


def test_replay_text_has_no_fragmented_lines():
    """碎片化检查:两段 delta 聚合成恰好一条 text_message。"""
    events = _recorded_sequence()
    projection = TurnProjection()
    messages = [
        ve.text for event in events for ve in projection.feed(event)
        if ve.kind == "text_message"
    ]
    assert messages == ["你好,世界"]


def test_replay_aborted_sequence_preserves_tool_state():
    """中断变体:已完成的工具状态保留,aborted 视图事件到达。"""
    events = _recorded_sequence()[:5] + [
        make_event("loop.aborted", "t", {}),
        make_event("trace.aborted", "t", {}),
    ]
    cli_projection, _ = _cli_consume(events, RENDERERS)
    tui_projection, _ = _tui_consume(events, RENDERERS)

    for projection in (cli_projection, tui_projection):
        assert [(t.name, t.status) for t in projection.tools] == [("exec", "ok")]

    aborted = [
        ve for event in events
        for ve in TurnProjection().feed(event)  # 独立投影只数 aborted
        if ve.kind == "aborted"
    ]
    assert len(aborted) == 1
