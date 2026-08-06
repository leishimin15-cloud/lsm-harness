"""Interactive terminal gateway."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel

from lsm_harness.app import Harness
from lsm_harness.events import HarnessEvent


console = Console()


def _observe(event: HarnessEvent) -> None:
    if event.type == "memory.gate.decided":
        console.print(
            f"  [dim]memory · {event.data['decision']} — {event.data.get('reason', '')}[/dim]"
        )
    elif event.type == "tool.completed":
        console.print(
            f"  [dim]tool · {event.data['tool']} → {event.data['status']}[/dim]"
        )
    elif event.type == "memory.consolidated":
        console.print(
            f"  [dim]memory · consolidated {event.data['facts']} fact(s)[/dim]"
        )
    elif event.type == "context.compression.started":
        console.print(
            f"  [dim]context · compressing {event.data['source_messages']} message(s)[/dim]"
        )
    elif event.type == "context.compression.completed":
        console.print(
            f"  [dim]context · summary v{event.data['version']} ready[/dim]"
        )
    elif event.type == "context.compression.failed":
        console.print("  [yellow]context · compression failed; raw chat retained[/yellow]")


def _memory_snapshot(app: Harness) -> str:
    facts = app.memory.facts.list(8)
    episodes = app.memory.episodes.list(5)
    lines = [f"Facts ({len(app.memory.facts.list())})"]
    lines.extend(f"- #{item['id']} [{item['subject']}] {item['content']}" for item in facts)
    if not facts:
        lines.append("- 暂无")
    lines.extend(["", f"Episodes ({len(app.memory.episodes.list())})"])
    lines.extend(
        f"- #{item['id']} {item['happened_at']} {item['summary']}" for item in episodes
    )
    if not episodes:
        lines.append("- 暂无")
    return "\n".join(lines)


def _sessions_snapshot(app: Harness) -> str:
    lines = []
    for item in app.session.list_sessions():
        marker = "*" if item["id"] == app.session.session_id else " "
        title = item["title"] or "未命名会话"
        lines.append(
            f"{marker} {str(item['id'])[:8]}  {item['message_count']} messages  "
            f"summary v{item['summary_version']}  {title}"
        )
    return "\n".join(lines) if lines else "暂无会话"


def _summary_snapshot(app: Harness) -> str:
    info = app.session.summary_info()
    if not info:
        return "当前会话还没有滚动摘要。"
    return (
        f"version: {info['version']}\n"
        f"through chat: #{info['through_chat_id']}\n"
        f"source messages: {info['source_message_count']}\n\n"
        f"{info['summary']}"
    )


def run_chat() -> int:
    try:
        app = Harness()
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("先运行 [bold]lsm doctor[/bold] 检查配置。")
        return 1
    console.print(
        Panel.fit(
            "[bold]LSM 的个人 Harness[/bold]\n"
            f"model: {app.settings.model}   home: {app.settings.home.resolve()}\n"
            f"session: {app.session.session_id[:8]}\n"
            "命令：/memory · /sessions · /resume <id> · /summary · /new · /quit",
            border_style="cyan",
        )
    )
    # prompt_toolkit owns interactive editing because terminal line discipline
    # and basic input() can miscalculate CJK wide characters next to an ANSI
    # styled prompt, leaving the first Chinese character visually undeletable.
    input_session: PromptSession[str] = PromptSession()
    try:
        while True:
            try:
                message = input_session.prompt("you › ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not message:
                continue
            if message in {"/quit", "/exit"}:
                break
            if message == "/memory":
                console.print(Panel(_memory_snapshot(app), title="Local memory"))
                continue
            if message == "/sessions":
                console.print(Panel(_sessions_snapshot(app), title="Sessions"))
                continue
            if message == "/summary":
                console.print(Panel(_summary_snapshot(app), title="Context summary"))
                continue
            if message.startswith("/resume"):
                _, _, session_ref = message.partition(" ")
                resumed = app.session.resume(session_ref)
                if resumed:
                    console.print(f"[dim]resumed session · {resumed}[/dim]")
                else:
                    console.print("[yellow]找不到唯一匹配的会话，请先使用 /sessions。[/yellow]")
                continue
            if message == "/new":
                session_id = app.session.start_new()
                console.print(f"[dim]new session · {session_id}[/dim]")
                continue
            try:
                result = app.respond(message, observer=_observe, source="cli")
                console.print(f"[bold green]lsm ›[/bold green] {result.reply}\n")
            except Exception as exc:
                console.print(f"[red]本轮失败：{type(exc).__name__}: {exc}[/red]")
    finally:
        app.close()
    console.print("[dim]bye — memory remains local in .lsm/state.db[/dim]")
    return 0
