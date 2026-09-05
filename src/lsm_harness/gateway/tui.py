"""Textual TUI for lsm-harness — polished terminal interface.

Run with:
    lsm tui
"""

from __future__ import annotations

import threading
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import (
    Footer,
    Header,
    Input,
    RichLog,
    Static,
)
from textual.reactive import reactive

from lsm_harness.app import Harness
from lsm_harness.events import HarnessEvent
from lsm_harness.tools.multimodal import parse_multimodal_message


class StatusBar(Static):
    """Bottom status line showing model, thinking, session, tokens."""
    model: reactive[str] = reactive("")
    thinking: reactive[str] = reactive("off")
    session: reactive[str] = reactive("")
    tokens: reactive[str] = reactive("")

    def render(self) -> str:
        parts = [
            f"[bold]{self.model}[/bold]" if self.model else "",
            f"thinking: [{ 'green' if self.thinking == 'on' else 'yellow' if self.thinking == 'auto' else 'dim' }]{self.thinking}[/]",
            f"session: {self.session}" if self.session else "",
            self.tokens if self.tokens else "",
        ]
        return " │ ".join(p for p in parts if p)


class LSMTui(App):
    """Main TUI application."""

    CSS = """
    #chat {
        height: 1fr;
        border: solid $primary;
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
        Binding("s-tab", "toggle_thinking", "Thinking"),
        Binding("f1", "show_help", "Help"),
        Binding("ctrl+l", "show_model", "Model"),
        Binding("ctrl+t", "show_tree", "Tree"),
    ]

    def __init__(self):
        super().__init__()
        self.harness: Harness | None = None
        self._streaming = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="chat", markup=True, wrap=True, highlight=True)
        yield Container(
            Input(placeholder="you ›  (type @ to reference files, / for commands)", id="input"),
            id="input-container",
        )
        yield StatusBar(id="status")

    async def on_mount(self) -> None:
        """Start the agent harness."""
        try:
            self.harness = Harness()
        except ValueError as exc:
            self.query_one("#chat", RichLog).write(f"[red]{exc}[/red]")
            self.query_one("#chat", RichLog).write("Run [bold]lsm doctor[/bold] to fix.")
            return

        chat = self.query_one("#chat", RichLog)
        chat.write("[bold cyan]LSM Harness[/bold cyan] — Type /help for commands")
        chat.write(f"[dim]Model: {self.harness.settings.model}  Session: {self.harness.session.session_id[:8]}[/dim]")
        chat.write("")

        self._update_status()

    def _update_status(self) -> None:
        if not self.harness:
            return
        status = self.query_one("#status", StatusBar)
        status.model = self.harness.settings.model
        status.thinking = {"disabled": "off", "enabled": "on", "auto": "auto"}.get(
            self.harness.settings.thinking, "off"
        )
        status.session = self.harness.session.session_id[:8]

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

        # ── multimodal image ──
        multimodal = parse_multimodal_message(text)
        if multimodal:
            self._append_chat("[dim]📷 Image attached[/dim]")
            message = multimodal
        else:
            self._append_chat(f"[bold cyan]you ›[/bold cyan] {text}")
            message = text

        # ── run agent ──
        self._streaming = True
        self.query_one("#input", Input).disabled = True

        chat = self.query_one("#chat", RichLog)
        thinking_block: list[str] = []
        tool_count = 0

        def observer(event: HarnessEvent):
            nonlocal thinking_block, tool_count
            t = event.type
            d = event.data

            if t == "llm.text.delta":
                chat.write(d.get("text", ""), animate=False)
                self.call_from_thread(chat.scroll_end)

            elif t == "tool.requested":
                tool_count += 1
                name = d.get("tool", "?")
                args = str(d.get("args", {}))[:100]
                chat.write(f"  [dim]⚙ {name}[/dim] [dim italic]{args}[/dim italic]")

            elif t == "tool.completed":
                name = d.get("tool", "?")
                status = d.get("status", "ok")
                output = (d.get("output", "") or "")[:120].replace("\n", " ")
                icon = "✓" if status == "ok" else "✗"
                color = "green" if status == "ok" else "red"
                chat.write(f"  [{color}]{icon} {name}[/{color}] [dim]{output}[/dim]")

            elif t == "trace.completed":
                # Show usage
                pass

            elif t == "llm.completed" and d.get("role") == "main":
                usage = d.get("usage", {})
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                self.call_from_thread(
                    lambda: setattr(
                        self.query_one("#status", StatusBar),
                        "tokens",
                        f"↑{inp:,} ↓{out:,}",
                    )
                )

        def run_turn():
            try:
                result = self.harness.respond(message, observer=observer, source="tui")
                if result.aborted:
                    self.call_from_thread(
                        lambda: chat.write("[yellow]⏎ Interrupted[/yellow]")
                    )
            except Exception as exc:
                self.call_from_thread(
                    lambda: chat.write(f"[red]Error: {type(exc).__name__}: {exc}[/red]")
                )
            finally:
                self._streaming = False
                self.call_from_thread(lambda: self.query_one("#input", Input).focus())
                self.call_from_thread(lambda: setattr(self.query_one("#input", Input), "disabled", False))
                self._update_status()

        thread = threading.Thread(target=run_turn, daemon=True)
        thread.start()

    def _append_chat(self, text: str) -> None:
        self.query_one("#chat", RichLog).write(text)

    async def _handle_command(self, text: str) -> None:
        chat = self.query_one("#chat", RichLog)
        h = self.harness
        if not h:
            return

        cmd = text.strip()

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()

        elif cmd == "/help":
            chat.write("[bold]Commands:[/bold]")
            for c, desc in [
                ("/model", "Switch model"),
                ("/tree", "Session history tree"),
                ("/sessions", "List all sessions"),
                ("/summary", "View context summary"),
                ("/new", "Start new session"),
                ("/usage", "Token usage stats"),
                ("/quit", "Exit"),
                ("Shift+Tab", "Cycle thinking level"),
                ("@filename", "Fuzzy file search"),
            ]:
                chat.write(f"  {c:15s} {desc}")

        elif cmd == "/model":
            from lsm_harness.models import PROVIDERS
            chat.write(f"[dim]Current: {h.settings.model}[/dim]")
            for i, (name, p) in enumerate(PROVIDERS.items(), 1):
                marker = "*" if p.model == h.settings.model else " "
                chat.write(f"  {marker} {i}. {name:15s} {p.model}")

            chat.write("[dim]Use /model:N to switch (e.g. /model:3)[/dim]")

        elif cmd.startswith("/model:"):
            try:
                idx = int(cmd.split(":")[1]) - 1
                names = list(PROVIDERS.keys())
                if 0 <= idx < len(names):
                    name = names[idx]
                    p = PROVIDERS[name]
                    h.switch_model(name, model=p.model, small_model=p.small_model)
                    self._update_status()
                    chat.write(f"[green]→ {name}/{p.model}[/green]")
            except (ValueError, IndexError):
                chat.write("[yellow]Invalid model number[/yellow]")

        elif cmd == "/tree":
            history = h.session.history
            if not history:
                chat.write("[dim]No history[/dim]")
                return
            turn = 0
            for msg in history:
                role = msg.get("role", "?")
                content = str(msg.get("content", ""))[:80].replace("\n", " ")
                if role == "user":
                    turn += 1
                    chat.write(f"[cyan]#{turn}[/cyan] {content}")
                elif role == "assistant" and content.strip():
                    chat.write(f"  [green]↳ {content[:60]}[/green]")
            chat.write(f"[dim]{turn} turns[/dim]")

        elif cmd == "/sessions":
            rows = h.session.list_sessions(15)
            for item in rows:
                marker = "*" if item["id"] == h.session.session_id else " "
                chat.write(f"  {marker} {item['id'][:8]}  {item['message_count']}msgs  {item['title'] or 'untitled'}")

        elif cmd == "/summary":
            info = h.session.summary_info()
            if info:
                chat.write(f"[bold]Summary v{info['version']}:[/bold]")
                chat.write(info['summary'])
            else:
                chat.write("[dim]No summary yet[/dim]")

        elif cmd == "/new":
            sid = h.session.start_new()
            self._update_status()
            chat.write(f"[dim]New session: {sid[:8]}[/dim]")

        elif cmd == "/usage":
            s = h.tracer.usage_summary()
            if s["total_input"] == 0:
                chat.write("[dim]No usage recorded[/dim]")
            else:
                chat.write(f"Total: ↑{s['total_input']:,} ↓{s['total_output']:,} tokens")
                for model, stats in s.get("by_model", {}).items():
                    chat.write(f"  {model}: {stats['calls']} calls, ↑{stats['input']:,} ↓{stats['output']:,}")

        else:
            chat.write(f"[yellow]Unknown: {cmd}[/yellow]")

    def action_toggle_thinking(self) -> None:
        if not self.harness:
            return
        levels = ["disabled", "auto", "enabled"]
        current = self.harness.settings.thinking
        idx = levels.index(current) if current in levels else 0
        self.harness.settings.thinking = levels[(idx + 1) % len(levels)]
        self.harness.session.record_thinking_change(self.harness.settings.thinking)
        self._update_status()
        labels = {"disabled": "off", "enabled": "on", "auto": "auto"}
        self._append_chat(f"[dim]thinking → {labels[self.harness.settings.thinking]}[/dim]")

    def action_show_help(self) -> None:
        self._append_chat("[dim]Type /help for command list[/dim]")

    def action_show_model(self) -> None:
        self._append_chat("[dim]Type /model to list models[/dim]")

    def action_show_tree(self) -> None:
        self._append_chat("[dim]Type /tree to view session tree[/dim]")
