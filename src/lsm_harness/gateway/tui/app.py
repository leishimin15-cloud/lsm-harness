"""Textual TUI 应用壳。

职责仅限:Textual 控件组装、按键/命令分发、运行线程管理、生命周期。
事件解释在 `adapter.py`,显示状态在 `state.py`,渲染在 `widgets.py`,
命令/选择器在 `commands.py`,按键表在 `bindings.py`,弹窗在
`screens.py`。

事件通道(阶段一 typed-event 改造):TUI 在 on_mount 时经
`Harness.subscribe()` 订阅 CodingSession typed events,不再使用
`respond(observer=...)` 的 legacy HarnessEvent 字符串通道;
shutdown 时 unsubscribe,无 listener 泄漏。

线程模型(保留项目的同步生成器 + 线程设计,不抄 Tau 的 asyncio):
agent 运行在独立 worker 线程;**所有** UI 更新经 `call_from_thread`
编组回 UI 线程——TuiState 也只在 UI 线程变更,因此免锁。
SQLite 连接以 `check_same_thread=False` 创建(CPython 的 sqlite3
默认序列化模式,连接可跨线程共享),由 `shutdown()` 在退出时关闭。

阶段二批 ④⑤:聊天区从 RichLog 换成「条目控件」模型——每条
transcript 一个 MessageWidget / ToolCallWidget,App 做增量同步;
assistant 流式原位重渲染,工具单击折叠。
"""

from __future__ import annotations

import os
import threading
import time

from typing import Callable

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Input

from lsm_harness.agent.events import TurnEndEvent
from lsm_harness.ai.providers import ProviderConfigurationError
from lsm_harness.coding_agent.app import Harness, RunBusyError
from lsm_harness.coding_agent.auth_storage import AuthStorageError
from lsm_harness.coding_agent.events import (
    AutoRetryEndEvent,
    AutoRetryStartEvent,
    CodingSessionEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    ModelChangedEvent,
    QueueUpdateEvent,
    SessionChangedEvent,
    ThinkingLevelChangedEvent,
)
from lsm_harness.coding_agent.footer import format_tokens
from lsm_harness.coding_agent.startup import resolve_startup_settings
from lsm_harness.coding_agent.model_config import load_model_catalog
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.tools.multimodal import parse_multimodal_message

from .adapter import TuiEventAdapter
from .autocomplete import ChatInput, SuggestionOverlay
from .bindings import APP_BINDINGS
from .commands import CommandsMixin
from .state import MessageView, ToolView, TuiState
from .widgets import (
    AssistantWidget,
    MessageWidget,
    QueueBar,
    StatusBar,
    ToolCallWidget,
    TranscriptView,
)


class LSMTui(CommandsMixin, App):
    """Main TUI application."""

    CSS = """
    Screen {
        overflow-y: hidden;
    }
    #chat {
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
    }
    #queue {
        height: auto;
        padding: 0 1;
    }
    #input-container {
        height: auto;
        margin: 0;
        overflow: hidden;
    }
    #input {
        border-top: solid $accent;
        border-bottom: solid $accent;
        border-left: none;
        border-right: none;
        scrollbar-size-horizontal: 0;
    }
    #suggestions {
        height: auto;
        max-height: 12;
        border: solid $primary;
    }
    #status {
        height: 1;
        background: $surface;
        padding: 0 1;
    }
    ToolCallWidget:hover {
        background: $surface;
    }
    AssistantWidget {
        height: auto;
    }
    AssistantWidget Markdown {
        height: auto;
        margin: 0;
        padding: 0;
    }
    .msg-thinking {
        height: auto;
    }
    .transcript-boundary {
        color: $text-muted;
        height: 1;
    }
    """

    BINDINGS = APP_BINDINGS

    def __init__(self, harness: Harness | None = None):
        super().__init__()
        # 测试注入带假模型的 harness;缺省在 on_mount 里自建。
        self.harness: Harness | None = harness
        self._startup_settings = harness.settings if harness is not None else Settings()
        self.state = TuiState()
        self._adapter: TuiEventAdapter | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._worker: threading.Thread | None = None
        self._ui_thread_id: int | None = None
        # 与 state.transcript **窗口段**一一对应的条目控件(阶段二批 ④⑤)。
        # 更早条目的 widget 已移出 DOM 与本地列表——同步成本 O(窗口)。
        self._entry_widgets: list[MessageWidget | AssistantWidget | ToolCallWidget] = []
        self._hidden_count = 0
        # @ 文件 / 斜杠命令补全弹层(on_mount 里装配)。
        self._suggestion_overlay: SuggestionOverlay | None = None
        # 最近一次已同步进 StatusBar 的 tokens 串(变化检测用——
        # StatusBar 改版后不再持有 tokens 字段)。
        self._last_tokens: str = ""

    # ── 布局与启动 ─────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield TranscriptView(id="chat")
        yield QueueBar(id="queue")
        yield Container(
            SuggestionOverlay(id="suggestions"),
            ChatInput(id="input"),
            id="input-container",
        )
        yield StatusBar(id="status")

    async def on_mount(self) -> None:
        """Start the agent harness."""
        self._ui_thread_id = threading.get_ident()
        self._suggestion_overlay = self.query_one("#suggestions", SuggestionOverlay)
        if self.harness is None:
            if not self._initialize_harness(self._startup_settings):
                self.call_after_refresh(
                    self._open_login_picker,
                    (
                        self._startup_settings.provider
                        if self._startup_settings.provider
                        in load_model_catalog(
                            self._startup_settings.home
                        ).providers
                        else None
                    ),
                )
                return
        else:
            self._activate_harness()

    def _initialize_harness(self, settings: Settings) -> bool:
        """Build and activate a Harness, keeping the TUI alive on auth errors."""
        connection = None
        try:
            notice = resolve_startup_settings(settings)
            settings.ensure_home()
            connection = connect(settings.home, check_same_thread=False)
            harness = Harness(settings=settings, conn=connection)
        except (AuthStorageError, ProviderConfigurationError, ValueError) as exc:
            if connection is not None:
                connection.close()
            self._note(f"[red]{exc}[/red]")
            self._note("Use [bold]/login[/bold] to configure a provider.")
            return False
        self.harness = harness
        self._startup_settings = settings
        self._activate_harness(notice)
        return True

    def _activate_harness(self, notice: str | None = None) -> None:
        """Attach product events and views after a Harness becomes usable."""
        harness = self.harness
        if harness is None:
            return
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

        # typed event 订阅:adapter 在 mount 时创建一次(跨运行持续——
        # compaction/retry 等产品事件不限于单轮),respond() 不再带
        # observer。订阅即开始接收,退出时由 shutdown() 取消。
        self._adapter = TuiEventAdapter(
            self.state,
            harness.tool_renderers,
            tool_labels={
                name: harness.tools.get(name).display_label
                for name in harness.tools.tool_names()
                if harness.tools.get(name) is not None
            },
        )
        self._unsubscribe = harness.subscribe(self._on_session_event)

        # 启动即恢复当前会话路径(重启后历史重进界面),再打欢迎语
        # (rebuild 会 state.clear(),欢迎语须在重建之后写入)。
        self._rebuild_from_session()
        self._note("[bold cyan]LSM Harness[/bold cyan] — Type /help for commands")
        if notice:
            self._note(f"[yellow]{notice}[/yellow]")

        # 补全弹层:workspace = harness 的 workspace_root。
        if self._suggestion_overlay is None:
            self._suggestion_overlay = self.query_one(
                "#suggestions", SuggestionOverlay
            )
        self._suggestion_overlay.set_workspace(harness.workspace_root)
        # 登录/重激活后 provider 目录可能已变(models.json 更新),
        # 下次 /login 输入时重读,而不是每个键都重读。
        self._suggestion_overlay.invalidate_providers()
        # 反向加载:滚到顶部时由 App 把更早一页挂回来。
        self.query_one("#chat", TranscriptView).on_need_earlier = self._load_earlier
        # Textual 只有聚焦到 Input 才会把可打印按键交给编辑器。
        # 启动、重连或 /login 重建 Harness 后都把焦点明确放回
        # 输入框，不依赖 DOM 的默认焦点顺序。
        self.query_one("#input", ChatInput).focus()

    # ── 状态同步(UI 线程) ─────────────────────────────────

    def _refresh_status(self) -> None:
        """state + footer_snapshot → StatusBar/QueueBar 单向同步(UI 线程)。

        事件驱动的运行态(retry/compaction/queue)来自 state;身份/环境/
        累计用量/context 由 ``CodingSession.footer_snapshot()`` 现算——
        恢复会话时历史累计立即可见,不依赖事件重放,也不存在重启清零。
        """
        state = self.state
        status = self.query_one("#status", StatusBar)
        harness = self.harness
        if harness is not None:
            snap = harness.footer_snapshot()
            stats: list[str] = []
            if snap.total_input or snap.total_output:
                stats.append(
                    f"↑{format_tokens(snap.total_input)}"
                    f" ↓{format_tokens(snap.total_output)}"
                )
            if snap.total_cache_read or snap.total_cache_write:
                stats.append(
                    f"R{format_tokens(snap.total_cache_read)}"
                    f" W{format_tokens(snap.total_cache_write)}"
                )
                hit = snap.cache_hit_rate
                if hit is not None:
                    stats.append(f"CH{hit:.1f}%")
            if snap.total_cost:
                stats.append(f"${snap.total_cost:.3f}")
            if snap.context_window > 0:
                percent = snap.context_percent
                label = (
                    f"{percent:.1f}%/{format_tokens(snap.context_window)}"
                    if percent is not None
                    else f"?/{format_tokens(snap.context_window)}"
                )
                if snap.auto_compact:
                    label += " (auto)"
                # Pi 同款:context 占用超阈值变色
                if percent is not None and percent > 90:
                    label = f"[red]{label}[/red]"
                elif percent is not None and percent > 70:
                    label = f"[yellow]{label}[/yellow]"
                stats.append(label)
            status.stats = " ".join(stats)
            # 七档原样显示;off 灰显,其余档位绿色(Pi 不区分色阶)
            thinking_color = "dim" if snap.thinking == "off" else "green"
            status.right = (
                f"[dim]({snap.provider})[/dim] "
                f"[bold]{snap.model}[/bold] "
                f"[{thinking_color}]• thinking: {snap.thinking}[/{thinking_color}]"
            )
        else:
            status.stats = ""
            status.right = ""
        status.retry = (
            f"{state.retry_attempt}/{state.retry_max}"
            if state.is_retrying and state.retry_attempt is not None
            else ""
        )
        status.compacting = state.is_compacting
        self._last_tokens = state.tokens
        queue = self.query_one("#queue", QueueBar)
        queue.steering = tuple(state.queued_steering)
        queue.follow_ups = tuple(state.queued_follow_ups)

    def _note(self, markup: str) -> None:
        """写一条提示行并登记进 state(重渲染时可重放)。UI 线程。"""
        self.state.add_note(markup)
        self._sync_transcript()

    # ── 条目控件同步(阶段二批 ④⑤) ────────────────────────

    @staticmethod
    def _make_widget(entry: MessageView | ToolView):
        if isinstance(entry, ToolView):
            return ToolCallWidget()
        if entry.role == "assistant":
            return AssistantWidget()
        return MessageWidget()

    def _sync_transcript(self) -> None:
        """state.transcript → 条目控件 增量同步(UI 线程,O(窗口))。

        transcript 只在运行时 append、在 rebuild 时整体清空。窗口化:
        超出 chat.max_entries 的最旧条目移出 DOM **和本地列表**
        (``_hidden_count`` 记录隐藏数,TranscriptView 顶部边界占位显示);
        内容同步只遍历窗口段,不随历史增长。
        """
        state = self.state
        chat = self.query_one("#chat", TranscriptView)
        total = len(state.transcript)
        # 窗口滑动(仅 follow 模式):丢掉超出上限的最旧条目。用户上滚
        # 浏览历史时不滑动——反向加载回来的页面不能被下个事件立即删掉。
        desired_hidden = max(0, total - chat.max_entries)
        if chat._follow:
            while self._hidden_count < desired_hidden and self._entry_widgets:
                widget = self._entry_widgets.pop(0)
                chat.drop_entry(widget)
                self._hidden_count += 1
            # rebuild 后本地列表为空:直接跳过隐藏段,根本不挂载它们
            if not self._entry_widgets:
                self._hidden_count = desired_hidden
        chat.set_hidden_count(self._hidden_count)
        # 追加新增条目
        while len(self._entry_widgets) < total - self._hidden_count:
            entry = state.transcript[self._hidden_count + len(self._entry_widgets)]
            widget = self._make_widget(entry)
            chat.append_entry(widget)
            self._entry_widgets.append(widget)
        # 内容同步:先设显示开关,再 sync_from(让开关当次生效)
        window = state.transcript[self._hidden_count:]
        for widget, entry in zip(self._entry_widgets, window):
            if isinstance(widget, ToolCallWidget):
                widget.show_tool_results = state.show_tool_results
            else:
                widget.show_thinking = state.show_thinking
            widget.sync_from(entry)
        chat.scroll_to_end()

    # ── 反向加载(滚到顶部,批 4) ──────────────────────────

    def _load_earlier(self) -> None:
        """TranscriptView 滚到顶部时反向加载一页(O(页),窗口向后扩展)。

        隐藏段始终在 ``state.transcript``;这里把更早的 ``page`` 条挂回
        DOM 与同步列表,TranscriptView 负责滚动锚定(视口不跳)。
        """
        chat = self.query_one("#chat", TranscriptView)
        state = self.state
        if self._hidden_count == 0:
            chat._loading_earlier = False
            return
        page = 50
        new_hidden = max(0, self._hidden_count - page)
        count = self._hidden_count - new_hidden
        widgets = []
        for entry in state.transcript[new_hidden : new_hidden + count]:
            widget = self._make_widget(entry)
            if isinstance(widget, ToolCallWidget):
                widget.show_tool_results = state.show_tool_results
            else:
                widget.show_thinking = state.show_thinking
            widget.sync_from(entry)
            widgets.append(widget)
        self._entry_widgets = widgets + self._entry_widgets
        self._hidden_count = new_hidden
        # 先挂载(滚动锚定要用旧高度),再更新边界占位文本。
        chat.prepend_entries(widgets)
        chat.set_hidden_count(new_hidden)

    def on_input_changed(self, event: Input.Changed) -> None:
        """输入变化 → 刷新补全弹层(@文件 / /命令)。"""
        if self._suggestion_overlay is not None:
            self._suggestion_overlay.update_for(event.value)

    # ── 事件应用(仅 UI 线程;worker 经 call_from_thread 进来) ──

    def _on_session_event(self, event: CodingSessionEvent) -> None:
        """typed event 监听入口。

        发布线程分两类:agent worker(模型/工具事件)与 **UI 线程本
        身**(steer/follow_up/switch_model 等 UI 动作同步发布)。
        `call_from_thread` 在 UI 线程调用会抛 RuntimeError——以前被
        静默吞掉导致这类事件全丢;现在 UI 线程直接应用,worker 线程
        才编组。
        """
        if threading.get_ident() == self._ui_thread_id:
            self._apply_typed_event(event)
        else:
            self._call_ui(self._apply_typed_event, event)

    def _apply_typed_event(self, event: CodingSessionEvent) -> None:
        """UI 线程:typed event → adapter → state + 条目控件增量。"""
        adapter = self._adapter
        if adapter is None:
            return
        adapter.feed(event)  # 归约进 state;返回的增量行在 widget 模型下不用
        if isinstance(event, SessionChangedEvent):
            # 切会话/新建:界面跟随当前路径重建(唯一事实来源是树)。
            self._rebuild_from_session()
            return
        self._sync_transcript()
        if self._is_status_event(event) or (
            adapter.state.tokens != self._last_tokens
        ):
            self._refresh_status()

    def _rebuild_from_session(self) -> None:
        """从 CodingSession 当前路径重建界面(启动/切会话/新建/分支)。

        状态来源是 CodingSession 的公开查询 API;TUI 不读 JSONL。
        """
        harness = self.harness
        adapter = self._adapter
        if harness is None or adapter is None:
            return
        # 身份同步(restore 已在 session open/switch 时作用于 settings);
        # thinking 是七档原样值(off/minimal/…/max),不再映射三档。
        self.state.model = harness.settings.model
        self.state.thinking = harness.settings.thinking
        self.state.session = harness.session.session_id[:8]
        adapter.rebuild_from_messages(harness.current_path_messages())
        self._entry_widgets.clear()
        self._hidden_count = 0
        self.query_one("#chat", TranscriptView).clear_entries()
        self._sync_transcript()
        self._refresh_status()

    @staticmethod
    def _is_status_event(event: CodingSessionEvent) -> bool:
        """状态栏相关事件(队列/模型/thinking/会话/retry/compaction)。"""
        return isinstance(event, (
            QueueUpdateEvent,
            ModelChangedEvent,
            ThinkingLevelChangedEvent,
            SessionChangedEvent,
            AutoRetryStartEvent,
            AutoRetryEndEvent,
            CompactionStartEvent,
            CompactionEndEvent,
            TurnEndEvent,
        ))

    def _call_ui(self, fn, *args) -> None:
        """worker → UI 线程编组;应用已退出/退出中则静默丢弃。

        编组异常不再全静默:写 ``<home>/logs/tui-ui.log``,便于排查
        「事件丢了但没报错」这类问题(运行中才记,退出竞态不记)。
        """
        if not self.is_running:
            return
        try:
            self.call_from_thread(fn, *args)
        except Exception as exc:
            self._log_ui_error(exc)

    def _log_ui_error(self, exc: Exception) -> None:
        """UI 编组失败写 debug 日志(尽力而为,绝不反向弄崩 worker)。"""
        try:
            harness = self.harness
            home = harness.settings.home if harness is not None else None
            if home is None:
                return
            log_dir = home / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_dir / "tui-ui.log", "a", encoding="utf-8") as fh:
                fh.write(f"{stamp} {type(exc).__name__}: {exc}\n")
        except Exception:
            pass

    # ── 输入与运行 ─────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input."""
        text = event.value.strip()
        event.input.clear()

        if not text:
            return

        # ── slash commands ──
        if text.startswith("/"):
            await self._handle_command(text)
            return

        # A credential pasted into the chat box must never enter session
        # history or be sent to a model. Provider login has its own masked
        # modal; this is a final guard against focus mistakes and event leaks.
        if len(text) >= 20 and not any(char.isspace() for char in text) and (
            text.startswith(("sk-", "sk_", "AIza", "xai-"))
        ):
            self._note(
                "[red]Blocked a possible API key from chat. "
                "Use /login to configure credentials.[/red]"
            )
            return

        if self.harness is None:
            self._note("[yellow]No model is configured. Use /login first.[/yellow]")
            self._open_login_picker(self._startup_settings.provider or None)
            return

        # ── 运行中:Enter = steer 即时注入(Tau 同款);队列由事件驱动 ──
        if self.state.running:
            if not self.harness or not self.harness.steer(text):
                self._note("[dim]本轮已结束,消息未排队——请重新发送[/dim]")
                return
            self._note(f"[cyan]steer ›[/cyan] {text}")
            return

        # ── multimodal image ──
        multimodal = parse_multimodal_message(text)
        if multimodal:
            self._note("[dim]📷 Image attached[/dim]")
            message = multimodal
        else:
            self.state.add_message("user", text)
            self._sync_transcript()
            message = text

        self._start_run(message)

    def _start_run(self, message) -> None:
        """先在 UI 线程 begin_run()(接受即 is_running),再启动 worker
        执行——RPC _start_worker 同款:消灭"已接受但未 running"的启动
        窗口,Escape/steer 从提交瞬间起就有效。worker 只通过
        call_from_thread 回传 UI 更新。"""
        assert self.harness is not None
        try:
            active = self.harness.begin_run()
        except RunBusyError:
            # 防御:_start_run 只在非运行态被调,理论上不可达。
            self._note("[yellow]上一轮仍在运行（等本轮结束）[/yellow]")
            return
        self.state.running = True

        def run_turn():
            try:
                # typed events 经 subscribe 通道到达;不再有 observer。
                self.harness.respond(
                    message, source="tui", active_run=active
                )
            except Exception as exc:
                self._call_ui(self._note, f"[red]Error: {type(exc).__name__}: {exc}[/red]")
            finally:
                self._call_ui(self._finish_run)

        self._worker = threading.Thread(target=run_turn, daemon=True, name="lsm-tui-run")
        self._worker.start()

    def _finish_run(self) -> None:
        """一轮结束的 UI 收尾(UI 线程):清队列显示、焦点回输入框、
        刷状态。

        ``is_running`` 由 typed events 驱动(AgentStart/AgentEnd/
        AgentSettled),这里不再设置——迟到的收尾不能误清新一轮
        已开始运行的状态(竞态:旧 worker 的 finally 可能晚于新一轮
        的 agent_start 到达)。
        """
        self.state.queued_steering.clear()
        self.state.queued_follow_ups.clear()
        self.query_one("#input", Input).focus()
        self._refresh_status()

    # ── 生命周期:退出释放资源 ─────────────────────────────

    def shutdown(self) -> None:
        """退出路径的统一清理:取消事件订阅、中断在跑的运行并关闭
        harness。

        `run_tui()` 的 finally 调用;测试在 run_test 退出后调用。
        幂等,可在 harness 缺失/已关闭时安全调用。
        """
        unsubscribe, self._unsubscribe = self._unsubscribe, None
        if unsubscribe is not None:
            unsubscribe()
        harness, self.harness = self.harness, None
        if harness is None:
            return
        try:
            if harness.is_running:
                harness.abort()
                harness.wait_for_idle(timeout=5)
        finally:
            harness.close()

    # ── 按键动作 ───────────────────────────────────────────

    def action_abort_run(self) -> None:
        """Escape = 中断当前轮(独立于此应用退出,Ctrl+C 才是退出)。

        对齐 Tau 的键位分工:escape → session.cancel();空闲时无操作。
        中断标记行由 loop.aborted 事件经投影落 transcript。
        """
        if not self.harness or not self.state.running:
            return
        self.harness.abort()

    def action_toggle_tool_results(self) -> None:
        """Ctrl+O = 折叠/展开工具结果(全局;对历史同样生效)。

        条目控件模型下不再需要"清屏 + 全量重放":切换 state 的
        全局开关,再把 show_tool_results 同步进每个 ToolCallWidget
        reactive 字段,各自原位重渲染。错误结果永不折叠。
        """
        show = self.state.toggle_tool_results()
        self._sync_transcript()
        self._note(f"[dim]工具结果: {'展开' if show else '折叠'}(Ctrl+O 切换)[/dim]")

    def action_toggle_thinking(self) -> None:
        if not self.harness:
            return
        try:
            level = self.harness.cycle_thinking()
        except RunBusyError:
            self._note("[dim]运行中不可切换 thinking（等本轮结束）[/dim]")
            return
        self._refresh_status()
        self._note(f"[dim]thinking → {level}[/dim]")

    def action_show_help(self) -> None:
        self._note("[dim]Type /help for command list[/dim]")

    def action_transcript_page_up(self) -> None:
        """PageUp = 聊天区向上翻页(输入框持焦点时由 App 级绑定兜底)。

        上翻经 watch_scroll_y 自动关 follow:翻页后流式输出不会把
        用户拽回底部。
        """
        self.query_one("#chat", TranscriptView).scroll_page_up(animate=False)

    def action_transcript_page_down(self) -> None:
        """PageDown = 聊天区向下翻页;翻到底部自动恢复 follow。"""
        self.query_one("#chat", TranscriptView).scroll_page_down(animate=False)

    def action_show_model(self) -> None:
        self._open_model_picker()

    def action_show_tree(self) -> None:
        self._open_tree_picker()


def run_tui(mode: str | None = None) -> None:
    """`lsm tui` 入口:保证任何退出路径都释放 harness 资源。

    当前 Textual 壳只有稳定的 fullscreen renderer。``regular``
    参数暂作兼容别名，也进 fullscreen：Textual 的 ``inline=True``
    会在主缓冲区连续提交整帧界面，造成重复 Header/输入框，
    并不等价于 Pi 的 ``TuiMainScreen``。真正 regular 模式需要
    独立的 main-screen renderer，不能在这里通过 run 参数伪装。
    """
    requested = (mode or os.getenv("LSM_TUI_MODE", "")).strip() or "fullscreen"
    if requested not in {"regular", "fullscreen"}:
        raise ValueError("TUI mode must be 'regular' or 'fullscreen'")
    app = LSMTui()
    try:
        app.run()
    finally:
        app.shutdown()
