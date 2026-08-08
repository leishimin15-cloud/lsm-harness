"""Interactive terminal gateway — Rich streaming + tool cards + usage.

Uses prompt_toolkit for input (with tab completion for commands) and
Rich for live-updating output during streaming.
"""

from __future__ import annotations

import signal
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lsm_harness.app import Harness
from lsm_harness.events import HarnessEvent


console = Console()

# ── prompt_toolkit style ─────────────────────────────────────────

PROMPT_STYLE = Style.from_dict({
    "prompt": "bold green",
    "toolbar": "dim italic",
})

COMMANDS = [
    "/memory", "/sessions", "/summary", "/new", "/quit", "/exit",
    "/resume", "/usage",
]
COMMAND_COMPLETER = WordCompleter(COMMANDS, ignore_case=True, sentence=True)

# ── signal handler ───────────────────────────────────────────────

_current_app: Harness | None = None


def _on_sigint(signum, frame):
    if _current_app and _current_app.is_running:
        console.print("\n[yellow]⏎ 正在中断…[/yellow]")
        _current_app.abort()
    else:
        raise KeyboardInterrupt


# ── Rich-based observer ──────────────────────────────────────────


class TurnDisplay:
    """Collects streaming output and tool calls for a single turn."""

    def __init__(self):
        self.text_parts: list[str] = []
        self.tools: list[dict[str, Any]] = []
        self.usage: dict[str, int] | None = None
        self.infos: list[str] = []

    @property
    def full_text(self) -> str:
        return "".join(self.text_parts)

    @property
    def done(self) -> bool:
        return bool(self.full_text or self.tools)


def _make_observer():
    """Factory: returns (observer, get_display)."""
    display = TurnDisplay()

    def observe(event: HarnessEvent) -> None:
        t = event.type
        d = event.data

        if t == "llm.text.delta":
            display.text_parts.append(d.get("text", ""))
        elif t == "tool.requested":
            display.tools.append({
                "tool": d.get("tool", "?"),
                "args": d.get("args", {}),
                "status": "running",
                "output": "",
            })
        elif t == "tool.completed":
            # Update the last matching tool
            name = d.get("tool", "")
            for tool in reversed(display.tools):
                if tool["tool"] == name and tool["status"] == "running":
                    tool["status"] = d.get("status", "ok")
                    tool["output"] = d.get("output", "")[:200]
                    break
        elif t == "llm.completed" and d.get("role") == "main":
            display.usage = d.get("usage")
        elif t == "context.compression.started":
            display.infos.append(
                f"compressing {d.get('source_messages', '?')} messages"
            )
        elif t == "loop.truncation_rejected":
            display.infos.append(
                f"⚠ token limit — rejected {d.get('rejected_tool_calls', '?')} call(s)"
            )
        elif t == "loop.aborted":
            display.infos.append("aborted")
        elif t == "loop.steered":
            display.infos.append(f"steering: {d.get('message', '')[:40]}…")

    return observe, display


def _render_turn(display: TurnDisplay) -> None:
    """Render the completed turn with Markdown text + tool cards + usage."""
    # ── response text ──
    if display.full_text.strip():
        console.print(
            Markdown(display.full_text.strip()),
            style="",
        )

    # ── tool cards ──
    if display.tools:
        tool_table = Table(show_header=False, box=None, padding=(0, 1))
        tool_table.add_column("icon", width=2)
        tool_table.add_column("detail", max_width=80)
        for tc in display.tools:
            icon = "✓" if tc["status"] == "ok" else "✗"
            style = "green" if tc["status"] == "ok" else "red"
            detail = tc["tool"]
            if tc["output"]:
                detail += f" → {tc['output']}"
            tool_table.add_row(f"[{style}]{icon}[/]", f"[dim]{detail}[/]")
        console.print(tool_table)

    # ── info line ──
    if display.infos:
        console.print(Text(" · ").join(Text(i, style="dim") for i in display.infos))

    # ── usage ──
    if display.usage:
        inp = display.usage.get("input_tokens", 0)
        out = display.usage.get("output_tokens", 0)
        console.print(
            f"  [dim]↑{inp:,} ↓{out:,} tokens[/dim]"
        )

    console.print()  # blank line after turn


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
    console.print(Panel.fit(
        f"[bold]LSM Harness[/bold]   "
        f"model: {app.settings.model}   "
        f"session: {app.session.session_id[:8]}",
        border_style="cyan",
    ))
    console.print(
        "[dim]Ctrl+C 中断当前回答 · /help 查看命令[/dim]\n"
    )

    # ── prompt_toolkit with command completion ──
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _(event):
        """Alt+Enter inserts a newline for multi-line input."""
        event.current_buffer.insert_text("\n")

    input_session: PromptSession[str] = PromptSession(
        completer=COMMAND_COMPLETER,
        style=PROMPT_STYLE,
        key_bindings=kb,
        multiline=False,
    )

    try:
        while True:
            try:
                message = input_session.prompt(
                    [("class:prompt", "you › ")],
                ).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not message:
                continue
            if message in {"/quit", "/exit"}:
                break
            if message == "/help":
                console.print(Panel(
                    "/memory   查看本地记忆\n"
                    "/sessions 列出会话\n"
                    "/summary  查看上下文摘要\n"
                    "/new      新建会话\n"
                    "/resume <id> 恢复会话\n"
                    "/usage    查看 token 用量\n"
                    "/quit     退出\n"
                    "Alt+Enter 多行输入",
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

            # ── run turn ──
            try:
                observer, display = _make_observer()
                result = app.respond(message, observer=observer, source="cli")

                if result.aborted:
                    console.print("[yellow]本轮已中断。[/yellow]\n")
                else:
                    _render_turn(display)
            except Exception as exc:
                console.print(f"[red]本轮失败：{type(exc).__name__}: {exc}[/red]")
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        _current_app = None
        app.close()

    console.print("[dim]bye — memory remains local in .lsm/state.db[/dim]")
    return 0
