"""TUI 显示状态:纯 Python,零 Textual 导入(阶段 5 批 1,Tau state.py 边界)。

`TuiState` 是 transcript 的事实来源:adapter 把投影事件落到这里,
RichLog 只是它的物化视图——所以 Ctrl+O 折叠切换可以清屏后用
`render_lines()` 全量重放,历史行同样生效。

行级结构(与增量写入保持一致):

- 用户输入:``role="you"``;
- assistant 一个文本段(投影的 text_message)= 一条 ``role="assistant"``;
- 工具调用 = 一条 ``role="tool"``:requested 时写 ``call_line``,
  completed 时补上 ``result_line`` 与状态;折叠态只渲染状态行;
- 提示/错误/中断等:``role="note"``。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatItem:
    role: str  # "you" | "assistant" | "tool" | "note"
    text: str  # 首行(工具条目为调用行,其余为正文,均为 markup)
    tool_status: str = ""  # tool 条目:"running" | "ok" | "error"
    tool_label: str = ""
    tool_call_id: str = ""
    result_line: str = ""  # tool 完成行(带预览),未完成为空


@dataclass
class TuiState:
    items: list[ChatItem] = field(default_factory=list)
    running: bool = False
    queued_steering: list[str] = field(default_factory=list)
    queued_follow_ups: list[str] = field(default_factory=list)
    model: str = ""
    thinking: str = ""
    session: str = ""
    tokens: str = ""
    show_tool_results: bool = True

    def add(self, role: str, text: str) -> ChatItem:
        item = ChatItem(role=role, text=text)
        self.items.append(item)
        return item

    def begin_tool(self, call_id: str, label: str, call_line: str) -> ChatItem:
        item = ChatItem(
            role="tool",
            text=call_line,
            tool_status="running",
            tool_label=label,
            tool_call_id=call_id,
        )
        self.items.append(item)
        return item

    def finish_tool(self, call_id: str, status: str, result_line: str) -> ChatItem | None:
        """按 call id 配对最近的 running 工具条目(与投影同规则)。"""
        for item in reversed(self.items):
            if (
                item.role == "tool"
                and item.tool_status == "running"
                and (not call_id or item.tool_call_id == call_id)
            ):
                item.tool_status = status
                item.result_line = result_line
                return item
        return None

    def toggle_tool_results(self) -> bool:
        self.show_tool_results = not self.show_tool_results
        return self.show_tool_results

    def render_lines(self) -> list[str]:
        """全量渲染(折叠开关生效于此;RichLog 清屏后按此重放)。

        折叠语义:工具条目只保留调用行(无结果预览),但**错误永远
        显示**——折叠不能吞掉失败。增量写入遵守同一规则,保证
        delta 与重放一致。
        """
        lines: list[str] = []
        for item in self.items:
            if item.role == "tool":
                lines.append(item.text)
                if item.result_line and (
                    self.show_tool_results or item.tool_status == "error"
                ):
                    lines.append(item.result_line)
            else:
                lines.append(item.text)
        return lines
