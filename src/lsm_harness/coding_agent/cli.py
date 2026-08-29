"""Interactive coding-agent terminal — Rich streaming + tool cards + usage.

Features:
  - @ file fuzzy search (type @ then a query)
  - Tab path completion
  - /model to switch models on the fly
  - /tree to browse session branching points
"""

from __future__ import annotations

import os
import signal
import threading
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter, merge_completers
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from lsm_harness.coding_agent.app import Harness
from lsm_harness.events import HarnessEvent
from lsm_harness.gateway.file_completer import AtFileCompleter, PathCompleter
from lsm_harness.ai.providers import PROVIDERS
from lsm_harness.tools.multimodal import has_image, parse_multimodal_message
from lsm_harness.agent.types import TraceResult


console = Console()

# ── prompt_toolkit style ─────────────────────────────────────────

PROMPT_STYLE = Style.from_dict({
    "prompt": "bold green",
    "toolbar": "dim italic",
})

COMMANDS = [
    "/memory", "/sessions", "/summary", "/new", "/quit", "/exit",
    "/resume", "/usage", "/model", "/tree",
    "/branch", "/branch-summary", "/label",
]

# ── completer: commands + @file fuzzy + Tab path ────────────────

_at_completer: AtFileCompleter | None = None

def _make_completer(app: Harness):
    global _at_completer
    root = os.getcwd()
    _at_completer = AtFileCompleter(root=root)
    path_comp = PathCompleter(root=root)
    cmd_comp = WordCompleter(COMMANDS, ignore_case=True, sentence=True)
    return merge_completers([cmd_comp, _at_completer, path_comp])

# ── signal handler ───────────────────────────────────────────────

_current_app: Harness | None = None


def _on_sigint(signum, frame):
    if _current_app and _current_app.is_running:
        console.print("\n[yellow]⏎ 正在中断…[/yellow]")
        _current_app.abort()
    else:
        raise KeyboardInterrupt


# ── Rich-based observer (streaming) ─────────────────────────────


class TurnDisplay:
    """Collects streaming output and tool calls for a single turn."""

    def __init__(self):
        self.text_parts: list[str] = []
        self.tools: list[dict[str, Any]] = []
        self.usage: dict[str, int] | None = None
        self.infos: list[str] = []
        self._text_printed = 0  # chars already printed

    @property
    def full_text(self) -> str:
        return "".join(self.text_parts)

    @property
    def done(self) -> bool:
        return bool(self.full_text or self.tools)


def _trace_status_markup(result: TraceResult) -> str | None:
    if result.status == "aborted":
        return "[yellow]本轮已中断。[/yellow]"
    if result.status == "failed":
        message = result.error or result.reply or "未知错误"
        return f"[red]本轮失败：{message}[/red]"
    return None


def _make_observer_and_stream(renderers: dict | None = None):
    """Factory: returns (observer, display) with real-time streaming.

    ``renderers`` maps tool name → (render_call, render_result) collected
    from ToolDefinitions at registry build (plan §9.3).  Display priority:
    custom renderer → label → tool name.
    """
    renderers = renderers or {}
    display = TurnDisplay()

    def observe(event: HarnessEvent) -> None:
        t = event.type
        d = event.data

        if t == "llm.text.delta":
            text = d.get("text", "")
            display.text_parts.append(text)
            # Stream immediately — print each character as it arrives
            console.print(text, end="", style="", highlight=False)

        elif t == "llm.text.end":
            console.print()  # newline after streaming text

        elif t == "tool.requested":
            tool_name = d.get("tool", "?")
            tool_label = d.get("label") or tool_name
            tool_call_id = d.get("tool_call_id", "")
            args = d.get("args", {})
            render_call = (renderers.get(tool_name) or (None, None))[0]
            console.print()
            if render_call is not None:
                console.print(f"  [dim cyan]⚙ {render_call(args)}[/dim cyan]")
            else:
                args_preview = str(args)[:80]
                console.print(f"  [dim cyan]⚙ {tool_label}[/dim cyan] [dim]{args_preview}[/dim]")
            display.tools.append({
                "tool": tool_name,
                "tool_call_id": tool_call_id,
                "args": args,
                "status": "running",
                "output": "",
            })

        elif t == "tool.completed":
            name = d.get("tool", "")
            label = d.get("label") or name
            tool_call_id = d.get("tool_call_id", "")
            status = d.get("status", "ok")
            output = d.get("output", "")
            icon = "✓" if status == "ok" else "✗"
            color = "green" if status == "ok" else "red"
            render_result = (renderers.get(name) or (None, None))[1]
            if render_result is not None:
                console.print(
                    f"  [{color}]{icon} "
                    f"{render_result(output, d.get('details'))}[/{color}]"
                )
            else:
                preview = output[:150].replace("\n", " ")
                console.print(f"  [{color}]{icon} {label}[/{color}] [dim]{preview}[/dim]")
            for tool in reversed(display.tools):
                if (
                    tool["tool"] == name
                    and tool["status"] == "running"
                    and (not tool_call_id or tool["tool_call_id"] == tool_call_id)
                ):
                    tool["status"] = status
                    tool["output"] = output[:200]
                    break

        elif t == "llm.completed" and d.get("role") == "main":
            display.usage = d.get("usage")

        elif t == "loop.aborted":
            console.print("  [yellow]⏎ interrupted[/yellow]")

    return observe, display


# ── slash command handlers ───────────────────────────────────────


def _cmd_memory(app: Harness) -> None:
    facts = app.memory.facts.list(8)
    episodes = app.memory.episodes.list(5)
    lines = [f"[bold]Facts[/bold] ({len(app.memory.facts.list())})"]
    lines.extend(f"  #{item['id']} [{item['subject']}] {item['content']}" for item in facts)
    if not facts:
        lines.append("  （暂无）")
    lines.extend(["", f"[bold]Episodes[/bold] ({len(app.memory.episodes.list())})"])
    lines.extend(f"  #{item['id']} {item['happened_at']} {item['summary']}" for item in episodes)
    if not episodes:
        lines.append("  （暂无）")
    console.print(Panel("\n".join(lines), title="Local Memory", border_style="blue"))


def _cmd_sessions(app: Harness) -> None:
    rows = app.session.list_sessions()
    table = Table(title="Sessions", border_style="blue")
    table.add_column("", width=1)
    table.add_column("ID", width=10)
    table.add_column("Messages", width=10)
    table.add_column("Summary", width=8)
    table.add_column("Title", max_width=50)
    for item in rows:
        marker = "*" if item["id"] == app.session.session_id else " "
        table.add_row(
            marker, str(item["id"])[:8], str(item["message_count"]),
            f"v{item['summary_version']}", item["title"] or "未命名",
        )
    console.print(table)


def _cmd_summary(app: Harness) -> None:
    info = app.session.summary_info()
    if not info:
        console.print("[dim]当前会话还没有滚动摘要。[/dim]")
        return
    console.print(Panel(
        f"version [bold]{info['version']}[/bold] · "
        f"{info['source_message_count']} source messages\n\n"
        f"{info['summary']}",
        title="Context Summary",
        border_style="blue",
    ))


def _cmd_usage(app: Harness) -> None:
    summary = app.tracer.usage_summary()
    if summary["total_input"] == 0:
        console.print("[dim]No usage recorded yet.[/dim]")
        return
    table = Table(title="Token Usage", border_style="blue")
    table.add_column("Model", max_width=30)
    table.add_column("Calls", justify="right")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    for model, stats in summary.get("by_model", {}).items():
        table.add_row(
            model, str(stats["calls"]),
            f"{stats['input']:,}", f"{stats['output']:,}",
        )
    total = Text(
        f"\nTotal: ↑{summary['total_input']:,} ↓{summary['total_output']:,} tokens",
        style="dim",
    )
    console.print(table)
    console.print(total)


# ── /model command ────────────────────────────────────────────────


def _cmd_model(app: Harness) -> None:
    """Switch the active model without restarting."""
    current = app.settings.model
    console.print(f"[dim]Current: {current}[/dim]\n")

    table = Table(title="Available Models", border_style="cyan")
    table.add_column("#", width=3)
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Small")

    models: list[tuple[str, str, str]] = []
    for name, provider in PROVIDERS.items():
        marker = "*" if provider.model == current else " "
        models.append((name, provider.model, provider.small_model))

    for i, (name, main_model, small_model) in enumerate(models, 1):
        table.add_row(str(i), name, main_model, small_model)

    console.print(table)
    console.print("\n[dim]输入编号切换，Enter 取消[/dim]")

    try:
        choice = input("model # ").strip()
        if choice and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                provider_name = models[idx][0]
                new_model = models[idx][1]
                new_small = models[idx][2]
                app.switch_model(
                    provider_name,
                    model=new_model,
                    small_model=new_small,
                )
                console.print(f"[green]→ {provider_name}/{new_model}[/green]")
            else:
                console.print("[yellow]Invalid[/yellow]")
    except (EOFError, KeyboardInterrupt):
        pass


# ── /tree command ─────────────────────────────────────────────────


def _cmd_tree(app: Harness) -> None:
    """Display session history as a tree."""
    history = app.session.history
    if not history:
        console.print("[dim]No history yet.[/dim]")
        return

    tree = Tree(f"[bold]Session {app.session.session_id[:8]}[/bold]")
    turn = 0
    for msg in history:
        role = msg.get("role", "?")
        content = str(msg.get("content", ""))[:80].replace("\n", " ")
        if role == "user":
            turn += 1
            tree.add(f"[cyan]#{turn}[/cyan] {content}")
        elif role == "assistant" and content.strip():
            tree.add(f"[green]  ↳ {content[:60]}[/green]")

    console.print(tree)
    console.print(f"\n[dim]{turn} turns[/dim]")


# ── session tree commands (plan §9.4) ────────────────────────────


def _cli_emit(kind: str, data: dict) -> None:
    """Minimal emit for session-tree commands run outside a trace."""
    if kind == "session.branch_summary.failed":
        console.print(f"[yellow]分支摘要失败：{data.get('error', '?')}[/yellow]")


def _cmd_branch(app: Harness, entry_ref: str, *, with_summary: bool) -> None:
    if not entry_ref:
        console.print("[yellow]用法：/branch <entry-id>（见 /tree 或 /sessions）[/yellow]")
        return
    session = app.session
    if with_summary:
        target = session.branch_with_summary(entry_ref, _cli_emit)
    else:
        target = session.branch(entry_ref, _cli_emit)
    if target is None:
        console.print("[yellow]找不到唯一匹配的 entry（或已在该位置）。[/yellow]")
    else:
        console.print(f"[dim]branched → {target[:12]}[/dim]")


def _cmd_label(app: Harness, argument: str) -> None:
    entry_ref, _, label = argument.partition(" ")
    if not entry_ref or not label.strip():
        console.print("[yellow]用法：/label <entry-id> <名称>[/yellow]")
        return
    target = app.session.record_label(entry_ref, label)
    if target is None:
        console.print("[yellow]找不到唯一匹配的 entry。[/yellow]")
    else:
        console.print(f"[dim]label「{label.strip()}」→ {target[:12]}[/dim]")


def _prepare_cli_message(message: str) -> str | dict:
    multimodal = parse_multimodal_message(message)
    if multimodal and has_image(multimodal.get("content")):
        console.print("[dim]📷 图片已附加[/dim]")
        return multimodal
    return message


def _queue_running_input(
    app: Harness,
    message: str | dict,
    *,
    follow_up: bool,
) -> bool:
    if follow_up:
        return app.follow_up(message)
    return app.steer(message)


def _run_cli_trace(app: Harness, message: str | dict) -> None:
    try:
        observer, display = _make_observer_and_stream(app.tool_renderers)
        result = app.respond(message, observer=observer, source="cli")
        status_markup = _trace_status_markup(result)
        if status_markup:
            console.print(status_markup + "\n")
        elif display.usage:
            inp = display.usage.get("input_tokens", 0)
            out = display.usage.get("output_tokens", 0)
            console.print(f"[dim]↑{inp:,} ↓{out:,} tokens[/dim]\n")
    except Exception as exc:
        console.print(f"[red]本轮失败：{type(exc).__name__}: {exc}[/red]")


# ── main loop ────────────────────────────────────────────────────


def run_chat() -> int:
    global _current_app

    try:
        app = Harness()
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("先运行 [bold]lsm doctor[/bold] 检查配置。")
        return 1

    _current_app = app
    signal.signal(signal.SIGINT, _on_sigint)

    # ── header ──
    thinking_label = {"disabled": "off", "enabled": "on", "auto": "auto"}.get(
        app.settings.thinking, app.settings.thinking
    )
    console.print(Panel.fit(
        f"[bold]LSM Harness[/bold]   "
        f"model: {app.settings.model}   "
        f"thinking: {thinking_label}   "
        f"session: {app.session.session_id[:8]}",
        border_style="cyan",
    ))
    console.print(
        "[dim]Ctrl+C 中断 · 工作中 Enter 插队 · Alt+Enter 排队后续 · "
        "Shift+Tab 切换 reasoning[/dim]\n"
    )

    # ── key bindings ──
    kb = KeyBindings()
    submission = {"mode": "steering"}

    @kb.add("escape", "enter")
    def _(event):
        """Pi-compatible Alt+Enter queues a follow-up while running."""
        submission["mode"] = "follow_up"
        event.current_buffer.validate_and_handle()

    @kb.add("s-tab")
    def _(event):
        """Shift+Tab cycles thinking level."""
        levels = ["disabled", "auto", "enabled"]
        current = app.settings.thinking
        idx = levels.index(current) if current in levels else 0
        next_level = levels[(idx + 1) % len(levels)]
        app.settings.thinking = next_level
        app.session.record_thinking_change(next_level)
        labels = {"disabled": "off", "enabled": "on", "auto": "auto"}
        console.print(f"[dim]thinking → {labels[next_level]}[/dim]")

    input_session: PromptSession[str] = PromptSession(
        completer=_make_completer(app),
        style=PROMPT_STYLE,
        key_bindings=kb,
        multiline=False,
    )

    worker: threading.Thread | None = None

    try:
        with patch_stdout(raw=True):
            while True:
                if worker is not None and not worker.is_alive():
                    worker.join()
                    worker = None

                try:
                    prompt = "steer › " if app.is_running else "you › "
                    message = input_session.prompt(
                        [("class:prompt", prompt)],
                    ).strip()
                    submission_mode = submission["mode"]
                    submission["mode"] = "steering"
                except (EOFError, KeyboardInterrupt):
                    break

                if not message:
                    continue
                if message in {"/quit", "/exit"}:
                    break

                prepared_message = _prepare_cli_message(message)
                if app.is_running:
                    is_follow_up = submission_mode == "follow_up"
                    queued = _queue_running_input(
                        app,
                        prepared_message,
                        follow_up=is_follow_up,
                    )
                    if queued:
                        label = "followUp" if is_follow_up else "steering"
                        console.print(f"[dim]已加入 {label} 队列[/dim]")
                    else:
                        console.print("[yellow]当前 Trace 已结束，请重新提交。[/yellow]")
                    continue

                if message == "/help":
                    console.print(Panel(
                        "/memory     查看本地记忆\n"
                        "/sessions   列出会话\n"
                        "/summary    查看上下文摘要\n"
                        "/new        新建会话\n"
                        "/resume <id> 恢复会话\n"
                        "/usage      查看 token 用量\n"
                        "/model      切换模型\n"
                        "/tree       浏览会话分支\n"
                        "/branch <entry-id>          回退到历史节点（分叉）\n"
                        "/branch-summary <entry-id>  分叉并留下被弃探索的摘要\n"
                        "/label <entry-id> <名称>    给历史节点打标签\n"
                        "/quit       退出\n"
                        "@文件名     模糊搜索文件\n"
                        "工作中 Enter     steering 紧急插队\n"
                        "工作中 Alt+Enter followUp 完成后执行",
                        title="Commands", border_style="blue",
                    ))
                    continue
                if message == "/memory":
                    _cmd_memory(app)
                    continue
                if message == "/sessions":
                    _cmd_sessions(app)
                    continue
                if message == "/summary":
                    _cmd_summary(app)
                    continue
                if message == "/usage":
                    _cmd_usage(app)
                    continue
                if message == "/model":
                    _cmd_model(app)
                    continue
                if message == "/tree":
                    _cmd_tree(app)
                    continue
                if message.startswith("/branch-summary"):
                    _, _, ref = message.partition(" ")
                    _cmd_branch(app, ref.strip(), with_summary=True)
                    continue
                if message.startswith("/branch"):
                    _, _, ref = message.partition(" ")
                    _cmd_branch(app, ref.strip(), with_summary=False)
                    continue
                if message.startswith("/label"):
                    _, _, argument = message.partition(" ")
                    _cmd_label(app, argument.strip())
                    continue
                if message.startswith("/resume"):
                    _, _, session_ref = message.partition(" ")
                    resumed = app.session.resume(session_ref)
                    if resumed:
                        console.print(f"[dim]resumed session · {resumed}[/dim]")
                    else:
                        console.print("[yellow]找不到唯一匹配的会话。[/yellow]")
                    continue
                if message == "/new":
                    sid = app.session.start_new()
                    console.print(f"[dim]new session · {sid[:8]}[/dim]")
                    continue

                worker = threading.Thread(
                    target=_run_cli_trace,
                    args=(app, prepared_message),
                    name="lsm-cli-trace",
                    daemon=True,
                )
                worker.start()
    finally:
        if worker is not None and worker.is_alive():
            app.abort()
            worker.join(timeout=5.0)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        _current_app = None
        app.close()

    console.print("[dim]bye — memory remains local in .lsm/state.db[/dim]")
    return 0
