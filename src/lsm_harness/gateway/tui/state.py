"""TUI 显示状态:纯 Python,零 Textual 导入。

阶段一(typed-event 改造):`TuiState` 是 transcript 的事实来源——
结构化的 `MessageView` / `ToolView` 条目,由 `TuiEventAdapter` 从
CodingSession typed events 归约而来;RichLog 只是它的物化视图
(增量写入与 `render_lines()` 重放遵守同一渲染规则,所以 Ctrl+O
折叠切换可以清屏后全量重放,历史行同样生效)。

条目结构:

- 用户输入 = ``MessageView(role="user")``;
- assistant 回答 = ``MessageView(role="assistant")``,流式期间
  ``is_streaming=True``,text/thinking 随 message_update 整快照更新;
- 工具调用 = ``ToolView``:start 建行、update 累积 progress、
  end 标状态与耗时,output 由随后的 tool-result 消息补上;
  按 ``tool_call_id`` 配对(一轮可能多次调用同一工具);
- 提示/错误/中断 = ``MessageView(role="note")``,text 为 markup。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lsm_harness.coding_agent.turn_projection import preview

PROGRESS_PREVIEW_LIMIT = 100


@dataclass
class MessageView:
    """一条对话消息(user / assistant / note)。"""

    role: str  # "user" | "assistant" | "note"
    text: str = ""
    thinking: str = ""
    is_streaming: bool = False
    is_error: bool = False


@dataclass
class ToolView:
    """一次工具调用。

    status: ``running`` | ``ok`` | ``error``。``output`` 在
    tool_execution_end 之后由 tool-result 消息补齐(内核事件顺序)。

    ``call_line`` / ``result_line`` 是 adapter 在事件到达时生成的
    markup 渲染缓存——渲染器(产品层产物)只在 adapter 手里,state
    不反向依赖它们;折叠规则(read_lines)只按 status 取舍这两行。
    """

    tool_call_id: str
    name: str
    label: str
    args: dict[str, Any] = field(default_factory=dict)
    progress: list[str] = field(default_factory=list)
    output: str = ""
    details: Any = None
    status: str = "running"
    started_at: float | None = None
    ended_at: float | None = None
    call_line: str = ""    # 调用行 markup(start 时生成)
    result_line: str = ""  # 结果行 markup(result 补齐时生成)

    @property
    def duration_ms(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000


@dataclass
class TuiState:
    """TUI 的全部显示状态(事件归约产物,可脱离 Textual 单测)。"""

    # transcript:消息与工具按到达顺序交织,是渲染顺序的事实来源;
    # tools 字典按 tool_call_id 索引同一份 ToolView(配对用)。
    transcript: list[MessageView | ToolView] = field(default_factory=list)
    tools: dict[str, ToolView] = field(default_factory=dict)

    # ── 运行状态 ──
    is_running: bool = False
    is_retrying: bool = False
    retry_attempt: int | None = None
    retry_max: int | None = None
    retry_message: str = ""
    is_compacting: bool = False

    # ── 队列(运行中提交的 steering / follow-up) ──
    queued_steering: list[str] = field(default_factory=list)
    queued_follow_ups: list[str] = field(default_factory=list)

    # ── 身份 ──
    model: str = ""
    thinking: str = ""
    session: str = ""
    error: str | None = None

    # ── usage:最近一次主模型 turn + 全程累积 ──
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0

    # ── 显示开关 ──
    show_tool_results: bool = True
    show_thinking: bool = True

    # 当前流式 assistant 消息(message_start 创建,message_end 落定)。
    _streaming: MessageView | None = None

    # ── 兼容别名(既有 App/测试词汇) ──

    @property
    def running(self) -> bool:
        return self.is_running

    @running.setter
    def running(self, value: bool) -> None:
        self.is_running = value

    @property
    def tokens(self) -> str:
        """状态栏 token 显示(最近一次主模型 turn)。"""
        if not self.input_tokens and not self.output_tokens:
            return ""
        return f"↑{self.input_tokens:,} ↓{self.output_tokens:,}"

    # ── 条目操作 ──

    def add_message(
        self,
        role: str,
        text: str = "",
        *,
        thinking: str = "",
        is_streaming: bool = False,
        is_error: bool = False,
    ) -> MessageView:
        view = MessageView(
            role=role,
            text=text,
            thinking=thinking,
            is_streaming=is_streaming,
            is_error=is_error,
        )
        self.transcript.append(view)
        return view

    def add_note(self, markup: str) -> MessageView:
        return self.add_message("note", markup)

    def begin_tool(
        self,
        tool_call_id: str,
        name: str,
        label: str,
        args: dict[str, Any],
        *,
        started_at: float | None = None,
    ) -> ToolView:
        view = ToolView(
            tool_call_id=tool_call_id,
            name=name,
            label=label,
            args=args,
            started_at=started_at,
        )
        self.tools[tool_call_id] = view
        self.transcript.append(view)
        return view

    def find_tool(self, tool_call_id: str, name: str = "") -> ToolView | None:
        """按 call id 配对;空 id 回退到最近的同名 running 工具。"""
        if tool_call_id and tool_call_id in self.tools:
            return self.tools[tool_call_id]
        for item in reversed(self.transcript):
            if (
                isinstance(item, ToolView)
                and item.status == "running"
                and (not name or item.name == name)
                and (not tool_call_id or item.tool_call_id == tool_call_id)
            ):
                return item
        return None

    def record_usage(self, usage: dict[str, Any]) -> None:
        """记录一次主模型 turn 的用量:更新最近值并累积总量。"""
        inp = int(usage.get("input_tokens", 0) or 0)
        out = int(usage.get("output_tokens", 0) or 0)
        cr = int(usage.get("cache_read_tokens", 0) or 0)
        cw = int(usage.get("cache_write_tokens", 0) or 0)
        self.input_tokens = inp
        self.output_tokens = out
        self.cache_read_tokens = cr
        self.cache_write_tokens = cw
        self.total_input_tokens += inp
        self.total_output_tokens += out
        self.total_cache_read_tokens += cr
        self.total_cache_write_tokens += cw

    def toggle_tool_results(self) -> bool:
        self.show_tool_results = not self.show_tool_results
        return self.show_tool_results

    def toggle_thinking(self) -> bool:
        self.show_thinking = not self.show_thinking
        return self.show_thinking

    # ── 重建(切会话/新建/分支/启动恢复) ──

    def clear(self) -> None:
        """清空 transcript 与运行状态,保留显示开关。

        用于 rebuild_from_messages 之前;身份字段由调用方重建。
        usage 是本会话显示值,随会话一起清零(harness 级总量在
        tracer,经 /usage 读取)。
        """
        self.transcript.clear()
        self.tools.clear()
        self._streaming = None
        self.is_running = False
        self.is_retrying = False
        self.retry_attempt = None
        self.retry_max = None
        self.retry_message = ""
        self.is_compacting = False
        self.queued_steering.clear()
        self.queued_follow_ups.clear()
        self.error = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0

    # ── 渲染 ──

    def render_lines(self) -> list[str]:
        """全量渲染(折叠开关生效于此;RichLog 清屏后按此重放)。

        渲染规则与 adapter 的增量写入一致:

        - user:``you ›`` 前缀;
        - assistant:thinking 块(dim,受 show_thinking 开关)+ 正文;
        - tool:调用行;结果行按折叠规则——**错误永远显示**,
          折叠不能吞掉失败;
        - note:原样(text 已是 markup)。
        """
        lines: list[str] = []
        for item in self.transcript:
            if isinstance(item, ToolView):
                lines.extend(self._render_tool(item))
            elif item.role == "user":
                lines.append(f"[bold cyan]you ›[/bold cyan] {item.text}")
            elif item.role == "assistant":
                lines.extend(self._render_assistant(item))
            else:  # note
                lines.append(item.text)
        return lines

    def _render_assistant(self, view: MessageView) -> list[str]:
        lines: list[str] = []
        if view.thinking and self.show_thinking:
            lines.append(f"[dim italic]💭 {view.thinking}[/dim italic]")
        if view.text:
            lines.append(view.text)
        return lines

    def _render_tool(self, view: ToolView) -> list[str]:
        """工具条目:调用行 + progress 行 + 结果行(按折叠规则取舍)。

        progress 行与 adapter 的增量写入逐条一致(过程记录,重放
        时同样可见);折叠只作用于结果预览——**错误永远显示**,折叠
        不能吞掉失败。
        """
        lines = [view.call_line]
        lines.extend(
            f"  [dim]{preview(p, PROGRESS_PREVIEW_LIMIT)}[/dim]"
            for p in view.progress
        )
        if view.result_line and (
            self.show_tool_results or view.status == "error"
        ):
            lines.append(view.result_line)
        return lines
