"""TUI 控件:纯渲染,不持有业务逻辑(阶段 5 批 1 + 阶段二批 ④⑤⑥)。

阶段二(typed-event 改造)把 RichLog 换成「条目控件」模型:

- ``MessageWidget``:一条 user / note 消息(rich markup;user 文本转义)。
- ``AssistantWidget``:一条 assistant 消息——thinking(Static,dim italic)
  + 正文(**Markdown** 渲染,流式节流重解析,落定强制一次)。增量即时可见
  (真流式),不再是「等 message_end 整段落定」(阶段二批 ④ + 阶段三 markdown)。
- ``ToolCallWidget``:一次工具调用。**原位折叠**(``collapsed``,单击切换)
  与全局折叠(``show_tool_results``)叠加;错误结果永远显示(阶段二批 ⑤)。
- ``TranscriptView``:承载条目的竖向滚动容器,**窗口化**(挂载上限
  ``max_entries``,超出聚合进顶部边界占位,防长 transcript 堆积控件)。
- ``StatusBar`` / ``QueueBar``:底部状态与队列,补 retry/compaction/
  累积 usage 显示(阶段二批 ⑥)。

渲染规则与 ``TuiState.render_lines()`` 一致(同一套折叠/前缀约定),
只是拆成每条目一个 widget,让增量更新可以只重画该条。
"""

from __future__ import annotations

import re
import time

from rich.markup import escape
from textual.containers import Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Markdown, Static

from lsm_harness.coding_agent.turn_projection import preview

from .state import PROGRESS_PREVIEW_LIMIT


class StatusBar(Static):
    """Single-line footer: usage/context left, model/thinking right.

    ``stats`` / ``right`` are preformatted markup from ``FooterSnapshot``;
    render only handles transient state and responsive alignment.
    """

    stats: reactive[str] = reactive("")
    right: reactive[str] = reactive("")
    retry: reactive[str] = reactive("")
    compacting: reactive[bool] = reactive(False)

    @staticmethod
    def _plain(markup: str) -> str:
        """去 markup 标签后的可见文本(长度估算;footer 内容全 ASCII)。"""
        return re.sub(r"\[/?[a-z_][^\]]*\]", "", markup)

    def render(self) -> str:
        left = self.stats
        if self.retry:
            left = f"{left} [yellow]⟳ retry {self.retry}[/yellow]" if left else f"[yellow]⟳ retry {self.retry}[/yellow]"
        if self.compacting:
            left = f"{left} [yellow]compacting…[/yellow]" if left else "[yellow]compacting…[/yellow]"
        right = self.right
        width = self.size.width
        if left and right:
            if width > 0:
                gap = width - len(self._plain(left)) - len(self._plain(right))
                # 宽度不够:截断右侧,至少保住统计区
                line2 = f"{left}{' ' * gap}{right}" if gap >= 2 else left
            else:
                # 尚未排版(宽度未知):先简单拼接,排版后自然右对齐
                line2 = f"{left}  {right}"
        else:
            line2 = left or right
        return line2


class QueueBar(Static):
    """排队中的 steering / follow-up 消息(运行中提交的消息)。"""

    steering: reactive[tuple[str, ...]] = reactive(())
    follow_ups: reactive[tuple[str, ...]] = reactive(())

    def render(self) -> str:
        parts = []
        if self.steering:
            parts.append(
                f"[cyan]steer ×{len(self.steering)}[/cyan]: "
                f"{self.steering[-1][:40]}"
            )
        if self.follow_ups:
            parts.append(
                f"[magenta]follow-up ×{len(self.follow_ups)}[/magenta]: "
                f"{self.follow_ups[-1][:40]}"
            )
        return " │ ".join(parts)


class MessageWidget(Static):
    """一条 user / note 消息(assistant 走 :class:`AssistantWidget`)。

    user 文本过 ``escape`` 再进 markup——否则用户输入里的 ``[...]`` 会被
    Rich 当标记解析(markup 注入);note 是本端生成的 markup,原样输出。
    """

    role: reactive[str] = reactive("")
    text: reactive[str] = reactive("")
    thinking: reactive[str] = reactive("")
    is_streaming: reactive[bool] = reactive(False)
    show_thinking: reactive[bool] = reactive(True)

    def sync_from(self, view) -> None:
        """把 ``MessageView`` 的内容拷进 reactive 字段(相等不重渲染)。"""
        self.role = view.role
        self.text = view.text
        self.thinking = view.thinking
        self.is_streaming = view.is_streaming

    def render(self) -> str:
        if self.role == "user":
            return f"[bold cyan]you ›[/bold cyan] {escape(self.text)}"
        return self.text  # note:文本已是 markup,原样输出


class AssistantWidget(Vertical):
    """一条 assistant 消息:thinking(Static) + 正文(Markdown)。

    正文是 markdown(代码块/表格/标题按 Textual Markdown 渲染),不是
    Rich markup——顺带修掉「LLM 输出含 ``[...]`` 被 Static 当标记解析」的
    注入问题。流式期间对 ``Markdown.update`` 做时间节流(全量重解析是
    O(n),按 ~80ms 节流避免 per-token O(n²));``is_streaming`` 落定 False
    时强制一次,保证最终内容精确。
    """

    _MARKDOWN_UPDATE_INTERVAL = 0.08

    thinking: reactive[str] = reactive("")
    text: reactive[str] = reactive("")  # 最新 markdown 源(快照,测试可读)
    is_streaming: reactive[bool] = reactive(False)
    show_thinking: reactive[bool] = reactive(True)

    def __init__(self) -> None:
        super().__init__()
        self._thinking_widget = Static("", classes="msg-thinking")
        self._markdown = Markdown("")
        self._placeholder = Static("[dim]…[/dim]", classes="msg-placeholder")
        self._last_update = 0.0
        self._flush_pending = False

    def compose(self):
        yield self._thinking_widget
        yield self._markdown
        yield self._placeholder

    def on_mount(self) -> None:
        self._render_children()

    def sync_from(self, view) -> None:
        """把 ``MessageView(role=\"assistant\")`` 拷进本控件。

        未挂载时只存字段(Markdown.update 需要 app 上下文);挂载后
        ``on_mount`` 会补渲染。
        """
        changed = (
            self.thinking != view.thinking
            or self.text != view.text
            or self.is_streaming != view.is_streaming
        )
        self.thinking = view.thinking
        self.text = view.text
        self.is_streaming = view.is_streaming
        # App 会在每个 agent/tool 事件同步当前窗口。旧实现
        # 对所有历史 AssistantWidget 无条件 Markdown.update，长会话
        # 会在 UI 线程重复解析整段 Markdown，导致输入丢帧/假死。
        if changed and self.is_mounted:
            self._render_children()

    def watch_show_thinking(self, show: bool) -> None:
        """全局折叠开关变化时单独刷新 thinking 区域。"""
        if self.is_mounted:
            self._render_children()

    def _render_children(self) -> None:
        # thinking 块(折叠开关生效于此)
        show_thinking = bool(self.thinking) and self.show_thinking
        self._thinking_widget.display = show_thinking
        if show_thinking:
            self._thinking_widget.update(
                f"[dim italic]💭 {escape(self.thinking)}[/dim italic]"
            )
        # 正文 markdown:流式节流 + 落定强制
        now = time.monotonic()
        if (not self.is_streaming) or (
            now - self._last_update >= self._MARKDOWN_UPDATE_INTERVAL
        ):
            self._markdown.update(self.text)
            self._last_update = now
        elif not self._flush_pending:
            # 节流跳过时排兜底定时器:突发流中途长时间静默(如下一个
            # delta 迟迟不来)也不能让画面一直停在旧内容。
            self._flush_pending = True
            self.set_timer(self._MARKDOWN_UPDATE_INTERVAL, self._flush_markdown)
        # 起步占位:流式中且无正文无 thinking 时给「…」,避免 0 高闪烁
        self._placeholder.display = (
            self.is_streaming and not self.text and not self.thinking
        )

    def _flush_markdown(self) -> None:
        """节流兜底:把最新文本补渲染一次(落定路径另有强制渲染)。"""
        self._flush_pending = False
        if self.is_mounted and self.text:
            self._markdown.update(self.text)
            self._last_update = time.monotonic()


class ToolCallWidget(Static):
    """一次工具调用。

    ``call_line`` / ``result_line`` 是 adapter 生成的 markup 缓存
    (renderer → label → name 查找只发生一次);progress 逐条累积。
    折叠规则与 ``TuiState._render_tool`` 一致:错误结果永远显示。
    """

    call_line: reactive[str] = reactive("")
    result_line: reactive[str] = reactive("")
    progress: reactive[list[str]] = reactive(list)
    status: reactive[str] = reactive("running")
    collapsed: reactive[bool] = reactive(False)
    show_tool_results: reactive[bool] = reactive(True)

    def sync_from(self, view) -> None:
        """把 ``ToolView`` 的内容拷进 reactive 字段(折叠开关除外)。"""
        self.call_line = view.call_line
        self.result_line = view.result_line
        self.progress = list(view.progress)
        self.status = view.status

    def on_click(self) -> None:
        """单击折叠/展开本工具的结果(错误结果保持显示)。"""
        self.collapsed = not self.collapsed

    def render(self) -> str:
        lines = [self.call_line] if self.call_line else []
        for p in self.progress:
            lines.append(
                f"  [dim]{preview(p, PROGRESS_PREVIEW_LIMIT)}[/dim]"
            )
        if self.result_line and (
            not self.collapsed and self.show_tool_results
            or self.status == "error"
        ):
            lines.append(self.result_line)
        return "\n".join(lines)


class TranscriptView(VerticalScroll):
    """聊天记录容器(取代 RichLog),带窗口化。

    App 持有与 ``state.transcript`` 窗口段一一对应的 widget 列表,
    在此 mount / 清空;增量同步只改 widget 的 reactive 字段。

    窗口化由 App 驱动:``drop_entry`` 移出 DOM 与挂载清单,
    ``set_hidden_count`` 维护顶部边界占位;更早的条目既不在 DOM
    也不在 App 的同步列表里——长会话的内存与每事件同步成本是
    O(窗口),不是 O(全部)。
    """

    def __init__(self, *args, max_entries: int = 300, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_entries = max_entries
        self._entries: list = []
        self._hidden_total = 0
        self._boundary: Static | None = None
        # 向上翻页:滚动到顶部时经 on_need_earlier 回调反向加载一页。
        self.on_need_earlier = None  # Callable[[], None] | None
        self._loading_earlier = False
        # follow 模式用 Textual 内建 anchor 实现(on_mount 里 anchor()):
        # anchored 且未 released 时,compositor 每次排版用 set_reactive
        # 钉底——不触发 watcher,布局抖动(如 markdown 重排短暂变矮)
        # 不会误关 follow;用户主动滚动(scroll_to 系,release_anchor=True)
        # 解除钉底,滚回底部由父类 watch_scroll_y → _check_anchor 恢复。

    def on_mount(self) -> None:
        self.anchor()

    @property
    def _follow(self) -> bool:
        """贴底跟随中(anchor 生效且未被用户滚动解除)。"""
        return self._anchored and not self._anchor_released

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """滚动接近顶部且有隐藏历史时,通知 App 反向加载一页。

        必须先调 ``super()``:父类负责同步滚动条位置、anchor 检查与
        ``_refresh_scroll()`` 重绘——缺了它滚动值变了但界面不重绘
        (鼠标滚轮"失效"的直接原因)。follow 状态由 anchor 机制维护,
        不在此推断(布局引起的 scroll_y 抖动不等于用户上滚)。
        """
        super().watch_scroll_y(old_value, new_value)
        if (
            new_value <= 2
            and self._hidden_total > 0
            and not self._loading_earlier
            and self.on_need_earlier is not None
        ):
            self._loading_earlier = True
            self.on_need_earlier()

    def prepend_entries(self, widgets: list) -> None:
        """把更早的一页挂到窗口前面,并保持视口不跳(按新增高度补偿滚动)。

        App 已把这些 widget 的内容同步好;这里只负责挂载顺序与滚动锚定。
        """
        if not widgets:
            self._loading_earlier = False
            return
        old_virtual = self.virtual_size.height
        old_y = self.scroll_offset.y
        first = self._entries[0] if self._entries else None
        for widget in widgets:
            self.mount(widget, before=first)
            self._entries.insert(0, widget)

        def anchor() -> None:
            delta = self.virtual_size.height - old_virtual
            if delta > 0:
                self.scroll_to(y=old_y + delta, animate=False, immediate=True)
            self._loading_earlier = False

        self.call_after_refresh(anchor)

    @property
    def hidden_count(self) -> int:
        return self._hidden_total

    def clear_entries(self) -> None:
        self.remove_children()
        self._entries = []
        self._hidden_total = 0
        self._boundary = None
        # rebuild(切会话/新建/分支)视为新视图:重新钉底 follow。
        self.anchor()
        self._anchor_released = False
        self._loading_earlier = False

    def append_entry(self, widget) -> None:
        self.mount(widget)
        self._entries.append(widget)

    def drop_entry(self, widget) -> None:
        """把最旧的 widget 移出 DOM 与挂载清单。"""
        if widget in self._entries:
            self._entries.remove(widget)
        widget.remove()

    def set_hidden_count(self, count: int) -> None:
        """顶部边界占位:隐藏数 0 时移除,>0 时挂载/更新。"""
        self._hidden_total = count
        if count <= 0:
            if self._boundary is not None:
                self._boundary.remove()
                self._boundary = None
            return
        text = (
            f"[dim]↑ 已隐藏 {count} 条更早条目(历史仍在会话树中)[/dim]"
        )
        if self._boundary is None:
            self._boundary = Static(text, classes="transcript-boundary")
            self.mount(
                self._boundary,
                before=self._entries[0] if self._entries else None,
            )
        else:
            self._boundary.update(text)

    def scroll_to_end(self) -> None:
        """follow(anchor 未解除)时才自动贴底;用户上滚看历史时不打断。

        anchor 生效期间 compositor 排版时已自动钉底,这里是显式兜底
        (如新条目刚挂载、排版尚未发生)。
        """
        if not self._anchor_released:
            self.scroll_end(animate=False)
