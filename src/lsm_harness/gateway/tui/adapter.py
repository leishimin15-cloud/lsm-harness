"""TUI 事件适配层:CodingSessionEvent → TuiState(阶段一 typed-event 改造)。

订阅 `CodingSession.subscribe()` 的 typed stream(agent 内核事件 +
coding-session 产品事件),不再消费 legacy `HarnessEvent` 字符串通道。

只做事件→状态的翻译,不认识任何 Textual 控件。`feed()` 返回应追加到
RichLog 的**增量行**;全量渲染由 `TuiState.render_lines()` 负责
(折叠切换重放用),两者遵守同一渲染规则。

渲染器(renderer → label → name 查找)只在工具行生成时用一次,
生成的 markup 缓存进 `ToolView.call_line` / `result_line`,此后
state 的折叠重放不再需要 renderers。
"""

from __future__ import annotations

import time

from lsm_harness.agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
)
from lsm_harness.agent.messages import message_preview
from lsm_harness.ai.messages import AssistantMessage, ToolResultMessage
from lsm_harness.coding_agent.events import (
    AgentSettledEvent,
    AutoRetryEndEvent,
    AutoRetryStartEvent,
    CodingSessionEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    ErrorEvent,
    ModelChangedEvent,
    QueueUpdateEvent,
    SessionChangedEvent,
    ThinkingLevelChangedEvent,
    ToolHistoryRepairedEvent,
)
from lsm_harness.coding_agent.turn_projection import (
    RendererPair,
    preview,
    render_tool_call,
    render_tool_result,
)

from .state import PROGRESS_PREVIEW_LIMIT, TuiState


class TuiEventAdapter:
    """把 CodingSession typed events 归约进 TuiState。"""

    def __init__(
        self,
        state: TuiState,
        renderers: dict[str, RendererPair],
        tool_labels: dict[str, str] | None = None,
    ):
        self.state = state
        self.renderers = renderers
        # name → 产品 label(历史重建用:AssistantMessage.tool_calls 只持
        # id/name/arguments,label 要从工具注册表查)。
        self._tool_labels = tool_labels or {}

    def feed(self, event: CodingSessionEvent) -> list[str]:
        """消费一条 typed event,返回应追加到聊天日志的渲染行。"""
        state = self.state
        lines: list[str] = []

        # ── Agent 生命周期 ────────────────────────────────
        if isinstance(event, AgentStartEvent):
            state.is_running = True
            state.error = None
            state.is_retrying = False
            state.retry_attempt = None
            return lines

        if isinstance(event, MessageStartEvent):
            # user / steering / follow_up 由 App 提交时即时回显;
            # 只有 assistant 消息走流式生命周期。
            if event.source == "assistant":
                state._streaming = state.add_message(
                    "assistant", is_streaming=True
                )
            return lines

        if isinstance(event, MessageUpdateEvent):
            # message 是完整快照:直接覆盖,不做 delta 拼接(单一事实)。
            view = state._streaming
            if view is not None and isinstance(event.message, AssistantMessage):
                view.text = event.message.text
                view.thinking = event.message.thinking
            # 流式增量不落 RichLog——段落完成时整行写(不碎片化)。
            return lines

        if isinstance(event, MessageEndEvent):
            if event.source == "assistant":
                lines.extend(self._finish_assistant(event.message))
            elif event.source == "tool" and isinstance(
                event.message, ToolResultMessage
            ):
                lines.extend(self._complete_tool(event.message))
            return lines

        if isinstance(event, ToolExecutionStartEvent):
            view = state.begin_tool(
                event.tool_call_id,
                event.tool_name,
                event.label or event.tool_name,
                event.args or {},
                started_at=time.monotonic(),
            )
            view.call_line = (
                f"  [dim cyan]{render_tool_call(self.renderers, view)}[/dim cyan]"
            )
            lines.append(view.call_line)
            return lines

        if isinstance(event, ToolExecutionUpdateEvent):
            view = state.find_tool(event.tool_call_id, event.tool_name)
            if view is not None and event.partial:
                view.progress.append(event.partial)
                lines.append(
                    f"  [dim]{preview(event.partial, PROGRESS_PREVIEW_LIMIT)}[/dim]"
                )
            return lines

        if isinstance(event, ToolExecutionEndEvent):
            view = state.find_tool(event.tool_call_id, event.tool_name)
            if view is not None:
                view.ended_at = time.monotonic()
                view.status = "error" if event.is_error else "ok"
            # output 由随后的 message_end(source="tool") 补齐;
            # 结果行在那里生成(需要 output 才有意义)。
            return lines

        if isinstance(event, TurnEndEvent):
            if event.usage:
                state.record_usage(event.usage)
            return lines

        if isinstance(event, AgentEndEvent):
            state.is_running = False
            if event.status == "aborted":
                line = "[yellow]⏎ Interrupted[/yellow]"
                state.add_note(line)
                lines.append(line)
            elif event.status == "failed" and event.error:
                state.error = event.error
                line = f"[red]✗ {event.error}[/red]"
                state.add_note(line)
                lines.append(line)
            return lines

        # ── 产品层事件 ────────────────────────────────────
        if isinstance(event, QueueUpdateEvent):
            state.queued_steering = list(event.steering)
            state.queued_follow_ups = list(event.follow_up)
            return lines

        if isinstance(event, AutoRetryStartEvent):
            state.is_retrying = True
            state.retry_attempt = event.attempt
            state.retry_max = event.max_attempts or None
            state.retry_message = event.message
            line = (
                f"[yellow]⟳ 请求失败,第 {event.attempt}"
                f"/{event.max_attempts or '?'} 次重试: {event.message}[/yellow]"
            )
            state.add_note(line)
            lines.append(line)
            return lines

        if isinstance(event, AutoRetryEndEvent):
            state.is_retrying = False
            if not event.success and event.final_error:
                line = f"[red]重试耗尽: {event.final_error}[/red]"
                state.add_note(line)
                lines.append(line)
            return lines

        if isinstance(event, CompactionStartEvent):
            state.is_compacting = True
            line = "[dim]⟳ Compacting context...[/dim]"
            state.add_note(line)
            lines.append(line)
            return lines

        if isinstance(event, CompactionEndEvent):
            state.is_compacting = False
            if event.error_message:
                line = f"[red]压缩失败: {event.error_message}[/red]"
            elif event.aborted:
                line = "[dim]压缩已取消[/dim]"
            else:
                line = "[dim]✓ Compaction complete[/dim]"
            state.add_note(line)
            lines.append(line)
            return lines

        if isinstance(event, ModelChangedEvent):
            state.model = event.model
            return lines

        if isinstance(event, ThinkingLevelChangedEvent):
            state.thinking = event.level
            return lines

        if isinstance(event, SessionChangedEvent):
            state.session = event.session_id[:8]
            return lines

        if isinstance(event, ErrorEvent):
            state.error = event.message
            line = f"[red]✗ {event.message}[/red]"
            state.add_note(line)
            lines.append(line)
            return lines

        if isinstance(event, ToolHistoryRepairedEvent):
            line = (
                f"[dim]工具历史已修复(补结果 {event.synthesized_results},"
                f" 弃孤立 {event.dropped_orphan_results})[/dim]"
            )
            state.add_note(line)
            lines.append(line)
            return lines

        if isinstance(event, AgentSettledEvent):
            # 一轮彻底结束(含重试/压缩/队列全部处理完)。
            # worker 的完成回调仍负责焦点等 UI 收尾;这里只钉状态。
            state.is_running = False
            state.is_retrying = False
            state.is_compacting = False
            return lines

        # EntryAppendedEvent / SessionInfoChangedEvent 等:
        # 阶段一不消费(会话恢复是 rebuild 的事),静默放行。
        return lines

    # ── 界面重建(切会话/新建/分支/启动恢复) ─────────────

    def rebuild_from_messages(self, messages: list) -> list[str]:
        """从 typed 消息历史重建 TuiState,返回全量渲染行。

        数据源是 `CodingSession.current_path_messages()`(当前分支
        路径的 message entries);重建后增量事件接在历史后面。

        工具卡片由 assistant 消息的 ``tool_calls`` 先行恢复(name/args/
        label 完整),随后的 ToolResultMessage 按 ``tool_call_id`` 补齐
        output 与状态——重启/切换会话后历史工具不再降级为空参数卡片。
        """
        state = self.state
        state.clear()
        for message in messages:
            role = getattr(message, "role", "")
            if role == "user":
                state.add_message("user", message_preview(message, limit=1_000_000))
            elif role == "assistant":
                state.add_message(
                    "assistant",
                    getattr(message, "text", ""),
                    thinking=getattr(message, "thinking", ""),
                    is_error=bool(getattr(message, "error_message", "")),
                )
                # 恢复该 assistant 消息携带的工具调用(参数/label 完整)。
                for call in getattr(message, "tool_calls", ()) or ():
                    call_id = getattr(call, "id", "")
                    name = getattr(call, "name", "")
                    args = dict(getattr(call, "arguments", None) or {})
                    label = self._tool_labels.get(name) or name
                    view = state.begin_tool(call_id, name, label, args)
                    view.call_line = (
                        f"  [dim cyan]{render_tool_call(self.renderers, view)}[/dim cyan]"
                    )
            elif role == "tool":
                call_id = getattr(message, "tool_call_id", "")
                name = getattr(message, "tool_name", "")
                # 优先配对 assistant 已恢复的卡片;老会话没有 tool_calls
                # 时(或跨分支缺失)回退到最小重建。
                view = state.find_tool(call_id, name)
                if view is None:
                    label = self._tool_labels.get(name) or name
                    view = state.begin_tool(call_id, name, label, {})
                    view.call_line = (
                        f"  [dim cyan]{render_tool_call(self.renderers, view)}[/dim cyan]"
                    )
                view.output = getattr(message, "content", "") or ""
                view.details = getattr(message, "details", None)
                view.status = (
                    "error" if getattr(message, "is_error", False) else "ok"
                )
                view.result_line = (
                    f"  {render_tool_result(self.renderers, view)}"
                )
        return state.render_lines()

    # ── 内部:消息与工具完成 ──────────────────────────────

    def _finish_assistant(self, message) -> list[str]:
        state = self.state
        view = state._streaming
        if isinstance(message, AssistantMessage):
            if view is None:
                view = state.add_message("assistant")
            view.text = message.text
            view.thinking = message.thinking
            view.is_error = bool(getattr(message, "error_message", ""))
        if view is None:
            return []
        view.is_streaming = False
        state._streaming = None
        # 完成时整行写(RichLog 不可改历史行;碎片化是反模式)。
        return state._render_assistant(view)

    def _complete_tool(self, message: ToolResultMessage) -> list[str]:
        state = self.state
        view = state.find_tool(message.tool_call_id, message.tool_name)
        if view is None:
            view = state.begin_tool(
                message.tool_call_id, message.tool_name, message.tool_name, {}
            )
            view.call_line = (
                f"  [dim cyan]{render_tool_call(self.renderers, view)}[/dim cyan]"
            )
        view.output = message.content or ""
        view.details = getattr(message, "details", None)
        if view.status == "running":
            view.status = "error" if message.is_error else "ok"
            view.ended_at = view.ended_at or time.monotonic()
        result = render_tool_result(self.renderers, view)
        duration = view.duration_ms
        if duration is not None:
            result += f" [dim]· {duration:.0f}ms[/dim]"
        view.result_line = f"  {result}"
        # 折叠态只显示错误(与 render_lines 同规则,保证
        # 增量写入与清屏重放一致)。
        if state.show_tool_results or view.status == "error":
            return [view.result_line]
        return []
