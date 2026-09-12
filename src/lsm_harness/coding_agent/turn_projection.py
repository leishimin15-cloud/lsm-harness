"""集中事件投影:把 HarnessEvent 流转成前端共用的 Turn 视图(阶段 3)。

三个前端不再各自解释字符串事件,而是共用这一条转换:

- CLI 逐 token 打印(消费 ``text_delta``),保持流式体验;
- TUI 聚合段落(消费 ``text_message``),不再把每个 token 写成一行;
- RPC 原样转发原始 HarnessEvent,机器客户端可用同一投影重放。

投影区分**增量更新**(text_delta,属于当前消息)与**新增/完成消息**
(text_message / tool_*),工具的 label 解析、状态机、输出预览只在
这里定义一次。自定义 renderer 仍由前端注入(它们是表现层产物),但
查找顺序统一为:自定义 renderer → label → 工具名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from lsm_harness.agent.events import (
    AgentEndEvent,
    MessageEndEvent,
    MessageUpdateEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)
from lsm_harness.agent.messages import ToolResultMessage

PREVIEW_LIMIT = 150
ARGS_PREVIEW_LIMIT = 80


def preview(text: str, limit: int = PREVIEW_LIMIT) -> str:
    """单行预览:折叠换行并截断。三前端共用同一截断规则。"""
    collapsed = (text or "").replace("\n", " ")
    return collapsed[:limit]


@dataclass(frozen=True)
class ToolView:
    """一次工具调用的投影状态。"""

    tool_call_id: str
    name: str
    label: str
    args: dict[str, Any]
    status: str = "running"  # running | ok | error
    output: str = ""
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class ViewEvent:
    """投影输出。kind:

    - ``text_delta``:当前 assistant 消息的增量(不是新消息);
    - ``text_message``:一段文本消息完成,text 为该段全文;
    - ``tool_requested`` / ``tool_completed``:工具状态变化,tool 为最新 ToolView;
    - ``usage``:主模型一次调用的 token 用量;
    - ``aborted``:本轮被中断。
    """

    kind: str
    text: str = ""
    tool: ToolView | None = None
    usage: dict[str, int | float] | None = None


@dataclass
class TurnProjection:
    """一个 Trace 的滚动投影:feed 原始事件,产出视图事件。"""

    text_parts: list[str] = field(default_factory=list)
    tools: list[ToolView] = field(default_factory=list)
    usage: dict[str, int | float] | None = None
    _paragraph: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "".join(self.text_parts)

    def feed(self, event) -> list[ViewEvent]:
        if hasattr(event, "kind"):
            return self._feed_typed(event)
        kind = event.type
        data = event.data

        if kind == "llm.text.delta":
            text = data.get("text", "")
            if not text:
                return []
            self.text_parts.append(text)
            self._paragraph.append(text)
            return [ViewEvent("text_delta", text=text)]

        if kind == "llm.text.end":
            if not self._paragraph:
                return []
            paragraph = "".join(self._paragraph)
            self._paragraph = []
            return [ViewEvent("text_message", text=paragraph)]

        if kind == "tool.requested":
            name = data.get("tool", "?")
            view = ToolView(
                tool_call_id=data.get("tool_call_id", ""),
                name=name,
                label=data.get("label") or name,
                args=data.get("args", {}) or {},
            )
            self.tools.append(view)
            return [ViewEvent("tool_requested", tool=view)]

        if kind == "tool.completed":
            name = data.get("tool", "?")
            view = self._match(data.get("tool_call_id", ""), name)
            if view is None:
                view = ToolView(
                    tool_call_id=data.get("tool_call_id", ""),
                    name=name,
                    label=data.get("label") or name,
                    args=data.get("args", {}) or {},
                )
                self.tools.append(view)
            updated = ToolView(
                tool_call_id=view.tool_call_id,
                name=view.name,
                label=data.get("label") or view.label,
                args=view.args,
                status=("error" if data.get("status") == "error"
                        or data.get("is_error") else "ok"),
                output=data.get("output", "") or "",
                details=data.get("details"),
            )
            self.tools[self.tools.index(view)] = updated
            return [ViewEvent("tool_completed", tool=updated)]

        if kind == "llm.completed" and data.get("role") == "main":
            usage = data.get("usage") or {}
            self.usage = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
            return [ViewEvent("usage", usage=self.usage)]

        if kind == "loop.aborted":
            return [ViewEvent("aborted")]

        return []

    def _feed_typed(self, event) -> list[ViewEvent]:
        if isinstance(event, MessageUpdateEvent):
            update = event.assistant_message_event
            if update.kind == "text_delta" and update.text_delta:
                self.text_parts.append(update.text_delta)
                self._paragraph.append(update.text_delta)
                return [ViewEvent("text_delta", text=update.text_delta)]
            if update.kind == "text_end" and self._paragraph:
                paragraph = "".join(self._paragraph)
                self._paragraph = []
                return [ViewEvent("text_message", text=paragraph)]
            return []

        if isinstance(event, ToolExecutionStartEvent):
            view = ToolView(
                tool_call_id=event.tool_call_id,
                name=event.tool_name,
                label=event.label or event.tool_name,
                args=event.args or {},
            )
            self.tools.append(view)
            return [ViewEvent("tool_requested", tool=view)]

        if (
            isinstance(event, MessageEndEvent)
            and event.source == "tool"
            and isinstance(event.message, ToolResultMessage)
        ):
            message = event.message
            view = self._match(message.tool_call_id, message.tool_name)
            if view is None:
                view = ToolView(
                    tool_call_id=message.tool_call_id,
                    name=message.tool_name,
                    label=message.tool_name,
                    args={},
                )
                self.tools.append(view)
            updated = ToolView(
                tool_call_id=view.tool_call_id,
                name=view.name,
                label=view.label,
                args=view.args,
                status="error" if message.is_error else "ok",
                output=message.content,
                details=getattr(message, "details", None),
            )
            self.tools[self.tools.index(view)] = updated
            return [ViewEvent("tool_completed", tool=updated)]

        if isinstance(event, TurnEndEvent):
            self.usage = dict(event.usage)
            return [ViewEvent("usage", usage=self.usage)]

        if isinstance(event, AgentEndEvent) and event.status == "aborted":
            return [ViewEvent("aborted")]
        return []

    def _match(self, tool_call_id: str, name: str) -> ToolView | None:
        """配对规则与 CLI 旧实现一致:优先按 call id,回退到同名 running。"""
        for view in reversed(self.tools):
            if (
                view.name == name
                and view.status == "running"
                and (not tool_call_id or view.tool_call_id == tool_call_id)
            ):
                return view
        return None


# ── 共享渲染辅助:renderer → label → name ─────────────────────────

RendererPair = tuple[Callable | None, Callable | None]


def render_tool_call(renderers: dict[str, RendererPair], view: ToolView) -> str:
    """工具调用行(markup)。自定义 renderer 优先,其次 label+参数预览。"""
    render_call = (renderers.get(view.name) or (None, None))[0]
    if render_call is not None:
        return f"⚙ {render_call(view.args)}"
    line = f"⚙ {view.label}"
    if view.args:
        line += f" {preview(str(view.args), ARGS_PREVIEW_LIMIT)}"
    return line


def render_tool_result(renderers: dict[str, RendererPair], view: ToolView) -> str:
    """工具结果行(markup),带状态图标与颜色。"""
    icon = "✓" if view.status == "ok" else "✗"
    color = "green" if view.status == "ok" else "red"
    render_result = (renderers.get(view.name) or (None, None))[1]
    if render_result is not None:
        body = render_result(view.output, view.details)
        return f"[{color}]{icon} {body}[/{color}]"
    return f"[{color}]{icon} {view.label}[/{color}] [dim]{preview(view.output)}[/dim]"
