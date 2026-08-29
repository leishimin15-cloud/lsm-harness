"""Subagent spawn tool — replaces delegate_code for internal sub-tasks."""

from __future__ import annotations

from typing import Any

from lsm_harness.coding_agent.tools import ToolDefinition


def make_tool(manager: Any) -> ToolDefinition:
    """Build the ``spawn`` tool bound to a SubagentManager.

    The tool supports two modes:
      - ``wait=False`` (default): background spawn, returns task_id
      - ``wait=True``: inline execution, returns subagent reply
    """

    def spawn(
        task: str,
        label: str = "",
        wait: bool = False,
        _ctx=None,
    ) -> str:
        """Spawn a subagent to handle a task independently.

        Args:
            task: Clear description of what to do. Include expected outputs.
            label: Short display label (optional).
            wait: If True, block until done and return the result directly.
                  Use this when the result must inform the current turn.
        """
        harness = getattr(manager, '_harness', None)
        if wait:
            return manager.run_inline(
                task,
                harness=harness,
                tools_whitelist=_readonly_tools(harness),
                emit=getattr(_ctx, "emit", None),
                parent_turn_id=getattr(_ctx, "turn_id", ""),
            )
        else:
            return manager.spawn(
                task,
                harness=harness,
                label=label,
                tools_whitelist=_readonly_tools(harness),
                emit=getattr(_ctx, "emit", None),
                parent_turn_id=getattr(_ctx, "turn_id", ""),
            )

    return ToolDefinition(
        name="spawn",
        label="启动子 Agent",
        description=(
            "启动一个子 Agent 在后台独立执行任务。"
            "子 Agent 拥有独立的上下文窗口，不会污染当前对话。"
            "适用于：多步骤研究、代码探索、可并行的子任务。"
            "设置 wait=true 可同步等待结果，适用于需要将结果用于当前回答的场景。"
            "替代原有的 delegate_code 工具用于 lsm-harness 内部的子任务。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "子 Agent 的任务描述，包含期望输出和约束。",
                },
                "label": {
                    "type": "string",
                    "description": "简短标签（可选，用于状态显示）。",
                },
                "wait": {
                    "type": "boolean",
                    "description": "是否同步等待结果。默认 false（后台执行）。",
                    "default": False,
                },
            },
            "required": ["task"],
        },
        execute=spawn,
        effect="local_write",
        execution_mode="sequential",
        timeout=600.0,  # subagents can run up to 10 minutes
    )


def _readonly_tools(harness: Any) -> list[str] | None:
    """Return the subset of tool names that are read-only.

    Used as default whitelist for inline subagents.
    """
    if harness is None:
        return None
    names = []
    for name in harness.tools.tool_names():
        tool = harness.tools._tools.get(name)
        if tool and tool.effect == "read":
            names.append(name)
    # Always allow save_note so the subagent can persist findings
    extra = [n for n in harness.tools.tool_names()
             if n in ("save_note", "create_event")]
    return names + extra if names else None
