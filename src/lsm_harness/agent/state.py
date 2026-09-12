"""AgentState — Pi ``AgentState`` (types.ts:327-352) 的 Python 版。

Agent 的完整可观察状态。会话层(Session/Harness)通过**整体赋值**
驱动它——``agent.state.messages = ...`` / ``agent.state.model = ...``,
与 Pi agent-session.ts 的 wholesale assignment 同款;Agent 不提供
setter 方法。

有记录的 Pi 偏离:

- ``messages``/``tools`` 赋值 = **rebind(引用替换),不复制**。
  Pi 的 setter 会复制顶层数组;我们的 loop 在 run 期间**别名**该列表
  (agent_loop.py 的 ``messages = context.messages``),截断恢复与
  prepare_next_turn 的重建必须替换同一引用,copy 会让 sink 的追加
  落到副本上。session 层每次构建新列表,Pi 要防的 hazard 不存在。
- ``tools`` 是 ``ToolRegistry``(含 schemas()/get()),不是 Pi 的
  ``AgentTool[]``。
- 四个 runtime 字段(``is_streaming`` / ``streaming_message`` /
  ``pending_tool_calls`` / ``error_message``)是只读属性——只有
  sink 归约器(``AgentEventSink._update_state``)与 Agent 生命周期
  (begin/run teardown/finish)可写,写法是直接动下划线字段。
- ``_terminal_seen`` 是 handleRunFailure 守卫:agent_end 已归约过
  就不再驱动合成失败链(防 agent_end 监听者抛错导致二次发射)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lsm_harness.agent.messages import AgentMessage
from lsm_harness.ai.types import Model

if TYPE_CHECKING:
    # agent.tools imports agent.events (execution events), which imports
    # this module — the registry type is needed only for annotations, and
    # the default is constructed lazily to keep the import acyclic.
    from lsm_harness.agent.tools import ToolRegistry


class AgentState:
    """Agent 的完整状态:持久身份(system/model/thinking/tools)+
    transcript(messages)+ 运行期投影(四个只读字段)。"""

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: Model | None = None,
        thinking_level: str = "disabled",
        tools: ToolRegistry | None = None,
        messages: list[AgentMessage] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.model = model  # None = 未设置;run 时仍无法解析则 ValueError
        self.thinking_level = thinking_level
        if tools is None:
            from lsm_harness.agent.tools import ToolRegistry

            tools = ToolRegistry()
        self.tools = tools
        self._messages: list[AgentMessage] = messages if messages is not None else []
        self._is_streaming = False
        self._streaming_message: AgentMessage | None = None
        self._pending_tool_calls: frozenset[str] = frozenset()
        self._error_message: str | None = None
        self._terminal_seen = False

    # ── transcript ─────────────────────────────────────────────

    @property
    def messages(self) -> list[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, messages: list[AgentMessage]) -> None:
        # REBIND,不是 copy —— 见模块 docstring 的偏离记录。
        self._messages = messages

    # ── runtime projections (read-only) ────────────────────────

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def streaming_message(self) -> AgentMessage | None:
        return self._streaming_message

    @property
    def pending_tool_calls(self) -> frozenset[str]:
        return self._pending_tool_calls

    @property
    def error_message(self) -> str | None:
        return self._error_message
