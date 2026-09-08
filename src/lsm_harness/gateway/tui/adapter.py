"""TUI 事件适配层:HarnessEvent →(TurnProjection)→ TuiState(阶段 5 批 1)。

对齐 Tau 的 `tui/adapter.py`:只做事件→状态的翻译,不认识任何
Textual 控件。`feed()` 同时返回**增量行**(追加到 RichLog 用);
全量渲染由 `TuiState.render_lines()` 负责(折叠切换重放用)。
"""

from __future__ import annotations

from lsm_harness.coding_agent.turn_projection import (
    RendererPair,
    TurnProjection,
    render_tool_call,
    render_tool_result,
)
from lsm_harness.events import HarnessEvent

from .state import TuiState


class TuiEventAdapter:
    """持有一次运行的 TurnProjection,把视图事件落到 TuiState。"""

    def __init__(self, state: TuiState, renderers: dict[str, RendererPair]):
        self.state = state
        self.renderers = renderers
        self.projection = TurnProjection()

    def feed(self, event: HarnessEvent) -> list[str]:
        """消费一条原始事件,返回应追加到聊天日志的渲染行。"""
        lines: list[str] = []
        for view_event in self.projection.feed(event):
            kind = view_event.kind

            if kind == "text_message":
                self.state.add("assistant", view_event.text)
                lines.append(view_event.text)

            elif kind == "tool_requested":
                tool = view_event.tool
                call_line = (
                    f"  [dim cyan]{render_tool_call(self.renderers, tool)}[/dim cyan]"
                )
                self.state.begin_tool(tool.tool_call_id, tool.label, call_line)
                lines.append(call_line)

            elif kind == "tool_completed":
                tool = view_event.tool
                result_line = f"  {render_tool_result(self.renderers, tool)}"
                item = self.state.finish_tool(
                    tool.tool_call_id, tool.status, result_line
                )
                # 折叠态只显示错误(与 render_lines 同规则,保证
                # 增量写入与清屏重放一致)。
                if self.state.show_tool_results or (
                    item is not None and item.tool_status == "error"
                ):
                    lines.append(result_line)

            elif kind == "usage":
                usage = view_event.usage or {}
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                self.state.tokens = f"↑{inp:,} ↓{out:,}"

            elif kind == "aborted":
                line = "[yellow]⏎ Interrupted[/yellow]"
                self.state.add("note", line)
                lines.append(line)

        return lines
