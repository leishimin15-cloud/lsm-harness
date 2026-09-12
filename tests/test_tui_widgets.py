"""阶段二:纯控件渲染测试(无 running App,直接实例化 + render)。

覆盖:
- MessageWidget:user / assistant(thinking 开关、流式占位)/ note;
- ToolCallWidget:折叠规则(单项 collapsed × 全局 show_tool_results,
  错误结果永远显示);
- StatusBar:Pi 风格单行(stats/right 对齐)与 retry/compacting 段。
"""

from __future__ import annotations

from lsm_harness.gateway.tui.widgets import (
    AssistantWidget,
    MessageWidget,
    StatusBar,
    ToolCallWidget,
)


def _message(role, text="", thinking="", is_streaming=False) -> MessageWidget:
    from lsm_harness.gateway.tui.state import MessageView

    w = MessageWidget()
    w.sync_from(MessageView(
        role=role, text=text, thinking=thinking, is_streaming=is_streaming,
    ))
    return w


def _tool(**kw) -> ToolCallWidget:
    from lsm_harness.gateway.tui.state import ToolView

    view = ToolView(tool_call_id="c1", name="exec", label="Run", args={})
    view.call_line = kw.get("call_line", "⚙ Run ls")
    view.result_line = kw.get("result_line", "✓ ok")
    view.progress = kw.get("progress", [])
    view.status = kw.get("status", "ok")
    w = ToolCallWidget()
    w.sync_from(view)
    return w


def test_message_user_escapes_markup_and_note_is_raw():
    user = _message("user", "看这个 [bold]不是标记[/bold]")
    rendered = user.render()
    assert "you ›" in rendered
    # 用户文本里的 [...] 被转义,不会被 Rich 当标记吞掉
    assert "\\[bold]" in rendered

    note = _message("note", "[red]boom[/red]")
    assert note.render() == "[red]boom[/red]"  # note 原样(markup)


def test_assistant_widget_stores_fields_unmounted():
    """AssistantWidget 未挂载时 sync_from 只存字段(Markdown.update 需 app
    上下文);挂载路径的渲染由 test_tui_app 的 Pilot 测试覆盖。"""
    from lsm_harness.gateway.tui.state import MessageView

    w = AssistantWidget()
    w.sync_from(MessageView(
        role="assistant", text="答 **粗体**", thinking="先想", is_streaming=True,
    ))
    assert w.text == "答 **粗体**"
    assert w.thinking == "先想"
    assert w.is_streaming is True

    w.sync_from(MessageView(role="assistant", text="最终", is_streaming=False))
    assert w.text == "最终"
    assert w.is_streaming is False
    # 折叠开关字段(thinking 显示/隐藏)存在
    assert w.show_thinking is True


def test_tool_fold_rules_collapsed_and_global():
    w = _tool(progress=["p1"])
    assert w.result_line in w.render()
    assert "p1" in w.render()

    # 单项折叠:结果行消失,调用行与 progress 保留
    w.collapsed = True
    assert w.result_line not in w.render()
    assert "p1" in w.render() and w.call_line in w.render()
    w.collapsed = False

    # 全局折叠:同样隐藏结果行
    w.show_tool_results = False
    assert w.result_line not in w.render()

    # 错误结果永不折叠(即使两个开关都关)
    err = _tool(status="error", result_line="✗ fail")
    err.collapsed = True
    err.show_tool_results = False
    assert "✗ fail" in err.render()


def test_status_bar_single_line_layout():
    """Footer only renders stats on the left and model state on the right."""
    bar = StatusBar()
    bar.stats = "↑30 ↓12 R5 W1 CH80.0% $0.012 3.4%/128k (auto)"
    bar.right = "[dim](deepseek)[/dim] [bold]k3[/bold] [yellow]• thinking: auto[/yellow]"
    bar.retry = "2/3"
    bar.compacting = True

    rendered = bar.render()
    assert "\n" not in rendered
    assert "↑30 ↓12" in rendered
    assert "CH80.0%" in rendered and "$0.012" in rendered
    assert "(deepseek)" in rendered and "k3" in rendered
    assert "retry 2/3" in rendered
    assert "compacting" in rendered

    # 清空后不显示这些段
    bar.retry = ""
    bar.compacting = False
    bar.stats = ""
    rendered = bar.render()
    assert "retry" not in rendered
    assert "compacting" not in rendered
    assert "↑30" not in rendered
