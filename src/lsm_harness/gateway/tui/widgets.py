"""TUI 控件:纯渲染,不持有业务逻辑(阶段 5 批 1,Tau widgets.py 边界)。"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Bottom status line showing model, thinking, session, tokens."""

    model: reactive[str] = reactive("")
    thinking: reactive[str] = reactive("off")
    session: reactive[str] = reactive("")
    tokens: reactive[str] = reactive("")

    def render(self) -> str:
        parts = [
            f"[bold]{self.model}[/bold]" if self.model else "",
            f"thinking: [{'green' if self.thinking == 'on' else 'yellow' if self.thinking == 'auto' else 'dim'}]{self.thinking}[/]",
            f"session: {self.session}" if self.session else "",
            self.tokens if self.tokens else "",
        ]
        return " │ ".join(p for p in parts if p)


class QueueBar(Static):
    """排队中的 steering / follow-up 消息(运行中提交的消息)。"""

    steering: reactive[tuple[str, ...]] = reactive(())
    follow_ups: reactive[tuple[str, ...]] = reactive(())

    def render(self) -> str:
        parts = []
        if self.steering:
            parts.append(f"[cyan]steer ×{len(self.steering)}[/cyan]: {self.steering[-1][:40]}")
        if self.follow_ups:
            parts.append(f"[magenta]follow-up ×{len(self.follow_ups)}[/magenta]: {self.follow_ups[-1][:40]}")
        return " │ ".join(parts)
