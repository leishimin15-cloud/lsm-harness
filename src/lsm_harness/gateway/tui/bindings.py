"""TUI 按键绑定表(阶段二批 ⑦:从 app.py 拆出)。

绑定与动作名分离:动作实现留在 App(或命令 mixin),这里只声明
键位 → 动作名的映射。对齐 Tau 的键位分工:

- ``escape`` = 中断当前轮(独立于应用退出);
- ``ctrl+c`` = 退出应用(``run_tui`` finally → ``shutdown`` 释放资源);
- ``ctrl+o`` = 全局折叠/展开工具结果;
- ``s-tab`` = 循环 thinking 等级(off/on/auto);
- ``ctrl+l`` / ``ctrl+t`` = 模型 / 会话树选择器。
"""

from __future__ import annotations

from textual.binding import Binding

APP_BINDINGS: list[Binding] = [
    Binding("ctrl+c", "quit", "Quit", show=False),
    Binding("escape", "abort_run", "Interrupt", show=False),
    Binding("ctrl+o", "toggle_tool_results", "Tools"),
    # Textual 8 的 BackTab 事件名是 ``shift+tab``；``s-tab``
    # 不会命中。Screen 默认也把 Shift+Tab 绑到 focus_previous，
    # 因此这里需要 priority，确保它循环 thinking 而非偷走焦点。
    Binding(
        "shift+tab",
        "toggle_thinking",
        "Thinking",
        priority=True,
    ),
    Binding("f1", "show_help", "Help"),
    Binding("ctrl+l", "show_model", "Model"),
    Binding("ctrl+t", "show_tree", "Tree"),
    # 输入框常占焦点:PageUp/PageDown 在 App 级兜底,直接翻聊天区
    # (Input 不占用这两个键;Home/End 留给输入框光标移动,聊天区
    # 聚焦时走 ScrollableContainer 自带的 home/end 绑定)。
    Binding("pageup", "transcript_page_up", "PgUp", show=False),
    Binding("pagedown", "transcript_page_down", "PgDn", show=False),
]
