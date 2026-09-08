"""Textual TUI 应用壳(阶段 5 批 1,Tau app.py 边界)。

职责仅限:Textual 控件组装、按键/命令分发、运行线程管理、生命周期。
事件解释在 `adapter.py`,显示状态在 `state.py`,渲染在 `widgets.py`。

线程模型(保留项目的同步生成器 + 线程设计,不抄 Tau 的 asyncio):
agent 运行在独立 worker 线程;**所有** UI 更新经 `call_from_thread`
编组回 UI 线程——TuiState 也只在 UI 线程变更,因此免锁。
SQLite 连接以 `check_same_thread=False` 创建(CPython 的 sqlite3
默认序列化模式,连接可跨线程共享),由 `shutdown()` 在退出时关闭。
"""

from __future__ import annotations

import threading

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    RichLog,
)
from textual.widgets.option_list import Option

from lsm_harness.agent.messages import message_preview
from lsm_harness.coding_agent.app import Harness, RunBusyError
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.events import HarnessEvent
from lsm_harness.ops.session_store import path_to_leaf, read_session_entries
from lsm_harness.tools.multimodal import parse_multimodal_message

from .adapter import TuiEventAdapter
from .state import TuiState
from .widgets import QueueBar, StatusBar


class PickerScreen(ModalScreen[str | None]):
    """通用单选弹窗(对齐 Tau 的 *PickerScreen 模式,但只留最小骨架)。

    Enter 选中 → dismiss(value);Escape → dismiss(None)。
    """

    CSS = """
    PickerScreen {
        align: center middle;
    }
    #picker-box {
        width: 72;
        max-height: 70%;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }
    #picker-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str, options: list[tuple[str, str]]):
        super().__init__()
        self._title = title
        self._values = [value for _label, value in options]
        self._labels = [label for label, _value in options]

    def compose(self) -> ComposeResult:
        with Container(id="picker-box"):
            yield Label(self._title, id="picker-title")
            yield OptionList(*[Option(label) for label in self._labels])

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self._values[event.option_index])

    def action_cancel(self) -> None:
        self.dismiss(None)


class LSMTui(App):
    """Main TUI application."""

    CSS = """
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
        margin: 1 0;
    }
    #input {
        border: solid $accent;
    }
    #status {
        height: 1;
        background: $surface;
        padding: 0 1;
    }
    .tool-running { color: $text-muted; }
    .tool-ok { color: $success; }
    .tool-error { color: $error; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("escape", "abort_run", "Interrupt", show=False),
        Binding("ctrl+o", "toggle_tool_results", "Tools"),
        Binding("s-tab", "toggle_thinking", "Thinking"),
        Binding("f1", "show_help", "Help"),
        Binding("ctrl+l", "show_model", "Model"),
        Binding("ctrl+t", "show_tree", "Tree"),
    ]

    def __init__(self, harness: Harness | None = None):
        super().__init__()
        # 测试注入带假模型的 harness;缺省在 on_mount 里自建。
        self.harness: Harness | None = harness
        self.state = TuiState()
        self._adapter: TuiEventAdapter | None = None
        self._worker: threading.Thread | None = None

    # ── 布局与启动 ─────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="chat", markup=True, wrap=True, highlight=True)
        yield QueueBar(id="queue")
        yield Container(
            Input(
                placeholder="you ›  (/ 命令;运行中 Enter=steer,/follow 排队,Esc 中断)",
                id="input",
            ),
            id="input-container",
        )
        yield StatusBar(id="status")

    async def on_mount(self) -> None:
        """Start the agent harness."""
        if self.harness is None:
            try:
                settings = Settings()
                settings.ensure_home()
                self.harness = Harness(
                    settings=settings,
                    conn=connect(settings.home, check_same_thread=False),
                )
            except ValueError as exc:
                chat = self.query_one("#chat", RichLog)
                chat.write(f"[red]{exc}[/red]")
                chat.write("Run [bold]lsm doctor[/bold] to fix.")
                return

        chat = self.query_one("#chat", RichLog)
        chat.write("[bold cyan]LSM Harness[/bold cyan] — Type /help for commands")
        chat.write(f"[dim]Model: {self.harness.settings.model}  Session: {self.harness.session.session_id[:8]}[/dim]")
        chat.write("")

        self._refresh_status()

    # ── 状态同步(UI 线程) ─────────────────────────────────

    def _refresh_status(self) -> None:
        """harness → state → StatusBar/QueueBar 单向同步(UI 线程)。"""
        if not self.harness:
            return
        state = self.state
        state.model = self.harness.settings.model
        state.thinking = {"disabled": "off", "enabled": "on", "auto": "auto"}.get(
            self.harness.settings.thinking, "off"
        )
        state.session = self.harness.session.session_id[:8]
        status = self.query_one("#status", StatusBar)
        status.model = state.model
        status.thinking = state.thinking
        status.session = state.session
        status.tokens = state.tokens
        queue = self.query_one("#queue", QueueBar)
        queue.steering = tuple(state.queued_steering)
        queue.follow_ups = tuple(state.queued_follow_ups)

    def _note(self, markup: str) -> None:
        """写一条提示行并登记进 state(重渲染时可重放)。UI 线程。"""
        self.state.add("note", markup)
        chat = self.query_one("#chat", RichLog)
        chat.write(markup)
        chat.scroll_end()

    # ── 事件应用(仅 UI 线程;worker 经 call_from_thread 进来) ──

    def _apply_event(self, event: HarnessEvent) -> None:
        adapter = self._adapter
        if adapter is None:
            return
        lines = adapter.feed(event)
        if lines:
            chat = self.query_one("#chat", RichLog)
            for line in lines:
                chat.write(line)
            chat.scroll_end()
        if adapter.state.tokens != self.query_one("#status", StatusBar).tokens:
            self._refresh_status()

    def _call_ui(self, fn, *args) -> None:
        """worker → UI 线程编组;应用已退出/退出中则静默丢弃。"""
        if not self.is_running:
            return
        try:
            self.call_from_thread(fn, *args)
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

        # ── 运行中:Enter = steer 即时注入(Tau 同款);排队显示 ──
        if self.state.running:
            if not self.harness or not self.harness.steer(text):
                self._note("[dim]本轮已结束,消息未排队——请重新发送[/dim]")
                return
            self.state.queued_steering.append(text)
            self._note(f"[cyan]steer ›[/cyan] {text}")
            self._refresh_status()
            return

        # ── multimodal image ──
        multimodal = parse_multimodal_message(text)
        if multimodal:
            self._note("[dim]📷 Image attached[/dim]")
            message = multimodal
        else:
            self.state.add("you", f"[bold cyan]you ›[/bold cyan] {text}")
            chat = self.query_one("#chat", RichLog)
            chat.write(f"[bold cyan]you ›[/bold cyan] {text}")
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
        # 输入框运行中保持可用:Enter=steer,/follow 排队(批 2)。
        self._adapter = TuiEventAdapter(self.state, self.harness.tool_renderers)

        def observer(event: HarnessEvent):
            self._call_ui(self._apply_event, event)

        def run_turn():
            try:
                self.harness.respond(
                    message, observer=observer, source="tui", active_run=active
                )
            except Exception as exc:
                self._call_ui(self._note, f"[red]Error: {type(exc).__name__}: {exc}[/red]")
            finally:
                self._call_ui(self._finish_run)

        self._worker = threading.Thread(target=run_turn, daemon=True, name="lsm-tui-run")
        self._worker.start()

    def _finish_run(self) -> None:
        """一轮结束(UI 线程):清队列显示、焦点回输入框、刷状态。"""
        self.state.running = False
        self.state.queued_steering.clear()
        self.state.queued_follow_ups.clear()
        self.query_one("#input", Input).focus()
        self._refresh_status()

    # ── 生命周期:退出释放资源 ─────────────────────────────

    def shutdown(self) -> None:
        """退出路径的统一清理:中断在跑的运行并关闭 harness。

        `run_tui()` 的 finally 调用;测试在 run_test 退出后调用。
        幂等,可在 harness 缺失/已关闭时安全调用。
        """
        harness, self.harness = self.harness, None
        if harness is None:
            return
        try:
            if harness.is_running:
                harness.abort()
                harness.wait_for_idle(timeout=5)
        finally:
            harness.close()

    # ── slash 命令 ─────────────────────────────────────────

    async def _handle_command(self, text: str) -> None:
        h = self.harness
        if not h:
            return

        cmd = text.strip()

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()

        elif cmd == "/help":
            self._note("[bold]Commands:[/bold]")
            for c, desc in [
                ("/model", "Switch model"),
                ("/tree", "Session history tree"),
                ("/sessions", "List all sessions"),
                ("/summary", "View context summary"),
                ("/new", "Start new session"),
                ("/usage", "Token usage stats"),
                ("/follow <文本>", "运行中排队 follow-up"),
                ("/quit", "Exit"),
                ("Esc", "中断当前轮"),
                ("Shift+Tab", "Cycle thinking level"),
                ("@filename", "Fuzzy file search"),
            ]:
                self._note(f"  {c:15s} {desc}")

        elif cmd == "/model":
            self._open_model_picker()

        elif cmd.startswith("/model:"):
            try:
                idx = int(cmd.split(":")[1]) - 1
                from lsm_harness.ai.providers import PROVIDERS
                names = list(PROVIDERS.keys())
                if 0 <= idx < len(names):
                    name = names[idx]
                    p = PROVIDERS[name]
                    try:
                        h.switch_model(name, model=p.model, small_model=p.small_model)
                    except RunBusyError:
                        self._note("[yellow]运行中不可切换模型（等本轮结束）[/yellow]")
                        return
                    self._refresh_status()
                    self._note(f"[green]→ {name}/{p.model}[/green]")
            except (ValueError, IndexError):
                self._note("[yellow]Invalid model number[/yellow]")

        elif cmd == "/tree":
            self._open_tree_picker()

        elif cmd == "/sessions":
            rows = h.session.list_sessions(15)
            if not rows:
                self._note("[dim]No sessions[/dim]")
                return
            options = [
                (f"{'*' if item['id'] == h.session.session_id else ' '} "
                 f"{item['id'][:8]}  {item['message_count']}msgs  "
                 f"{item['title'] or 'untitled'}", item["id"])
                for item in rows
            ]
            self.push_screen(
                PickerScreen("选择会话(Enter 切换,Esc 取消)", options),
                self._on_session_picked,
            )

        elif cmd == "/summary":
            info = h.session.summary_info()
            if info:
                self._note(f"[bold]Summary v{info['version']}:[/bold]")
                self._note(info['summary'])
            else:
                self._note("[dim]No summary yet[/dim]")

        elif cmd == "/new":
            try:
                sid = h.new_session()
            except RunBusyError:
                self._note("[yellow]运行中不可新建会话（等本轮结束）[/yellow]")
                return
            self._refresh_status()
            self._note(f"[dim]New session: {sid[:8]}[/dim]")

        elif cmd.startswith("/follow"):
            # 运行中排队 follow-up:本轮结束后接着跑(Tau Alt+Enter 的
            # 单行输入替代)。
            payload = cmd[len("/follow"):].strip()
            if not payload:
                self._note("[dim]用法: /follow <文本>[/dim]")
                return
            if not self.state.running:
                self._note("[dim]当前没有运行中的轮次,直接发送即可[/dim]")
                return
            if not h.follow_up(payload):
                self._note("[dim]本轮已结束,消息未排队——请重新发送[/dim]")
                return
            self.state.queued_follow_ups.append(payload)
            self._note(f"[magenta]follow-up ›[/magenta] {payload}")
            self._refresh_status()

        elif cmd == "/usage":
            s = h.tracer.usage_summary()
            if s["total_input"] == 0:
                self._note("[dim]No usage recorded[/dim]")
            else:
                self._note(f"Total: ↑{s['total_input']:,} ↓{s['total_output']:,} tokens")
                for model, stats in s.get("by_model", {}).items():
                    self._note(f"  {model}: {stats['calls']} calls, ↑{stats['input']:,} ↓{stats['output']:,}")

        else:
            self._note(f"[yellow]Unknown: {cmd}[/yellow]")

    # ── 选择器回调(批 3;均运行在选择器 dismiss 后的 UI 线程) ──

    def _open_model_picker(self) -> None:
        h = self.harness
        if not h:
            return
        from lsm_harness.ai.providers import PROVIDERS
        options = [
            (f"{'*' if p.model == h.settings.model else ' '} {name:15s} {p.model}",
             name)
            for name, p in PROVIDERS.items()
        ]
        self.push_screen(
            PickerScreen(f"选择模型(当前: {h.settings.model})", options),
            self._on_model_picked,
        )

    def _open_tree_picker(self) -> None:
        if self.state.running:
            self._note("[yellow]运行中不可切换节点（先 Esc 中断或等本轮结束）[/yellow]")
            return
        options = self._tree_options()
        if not options:
            self._note("[dim]No history[/dim]")
            return
        self.push_screen(
            PickerScreen("会话树——选节点分支(Enter 选中,Esc 取消)", options),
            self._on_tree_node_picked,
        )

    def _on_model_picked(self, name: str | None) -> None:
        if name is None or not self.harness:
            return
        from lsm_harness.ai.providers import PROVIDERS
        p = PROVIDERS[name]
        try:
            self.harness.switch_model(name, model=p.model, small_model=p.small_model)
        except RunBusyError:
            self._note("[yellow]运行中不可切换模型（等本轮结束）[/yellow]")
            return
        self._refresh_status()
        self._note(f"[green]→ {name}/{p.model}[/green]")

    def _on_session_picked(self, session_id: str | None) -> None:
        if session_id is None or not self.harness:
            return
        try:
            switched = self.harness.switch_session(session_id)
        except RunBusyError:
            self._note("[yellow]运行中不可切换会话（等本轮结束）[/yellow]")
            return
        if switched is None:
            self._note("[yellow]会话匹配失败[/yellow]")
            return
        self._refresh_status()
        self._note(f"[dim]→ session {switched[:8]}[/dim]")

    def _tree_options(self) -> list[tuple[str, str]]:
        """当前路径(到 live leaf)的条目列表:序号 + 预览。"""
        h = self.harness
        if h is None or h.session.recorder is None:
            return []
        entries = read_session_entries(h.session.jsonl_path)
        if not entries:
            return []
        path = path_to_leaf(entries, h.session.recorder.last_entry_id)
        leaf_id = h.session.recorder.last_entry_id
        options = []
        for i, entry in enumerate(path, 1):
            marker = "▸" if entry.id == leaf_id else " "
            if entry.type == "message":
                role = getattr(entry.message, "role", "?")
                text = message_preview(entry.message, limit=60).replace("\n", " ")
                label = f"{marker} #{i} [{role}] {text}"
            else:
                label = f"{marker} #{i} ({entry.type})"
            options.append((label, entry.id))
        return options

    def _on_tree_node_picked(self, entry_id: str | None) -> None:
        if entry_id is None or not self.harness:
            return
        if self.state.running:
            self._note("[yellow]运行中不可切换节点（等本轮结束）[/yellow]")
            return
        branched = self.harness.session.branch(entry_id, lambda *_: None)
        if branched is None:
            self._note("[yellow]分支失败:节点不存在[/yellow]")
            return
        self._note(f"[dim]↩ 已回到节点 {branched[:8]},后续消息将创建新分支[/dim]")

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
        """Ctrl+O = 折叠/展开工具结果(Tau 同款,全局且对历史生效)。

        RichLog 写过的行不可改,所以折叠靠"清屏 + 从 TuiState 全量
        重放"——这正是把 state 做成事实来源的回报(增量写入与
        render_lines 遵守同一折叠规则,两种路径输出一致)。
        """
        show = self.state.toggle_tool_results()
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        for line in self.state.render_lines():
            chat.write(line)
        chat.scroll_end()
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
        labels = {"disabled": "off", "enabled": "on", "auto": "auto"}
        self._note(f"[dim]thinking → {labels[level]}[/dim]")

    def action_show_help(self) -> None:
        self._note("[dim]Type /help for command list[/dim]")

    def action_show_model(self) -> None:
        self._open_model_picker()

    def action_show_tree(self) -> None:
        self._open_tree_picker()


def run_tui() -> None:
    """`lsm tui` 入口:保证任何退出路径都释放 harness 资源。"""
    app = LSMTui()
    try:
        app.run()
    finally:
        app.shutdown()
