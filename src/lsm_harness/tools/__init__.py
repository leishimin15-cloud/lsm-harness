"""Build the native tool surface with optional subagent."""

from pathlib import Path

from lsm_harness.agent.tools import AgentTool
from lsm_harness.coding_agent.tools import ToolDefinition, wrap_tool_definition
from lsm_harness.tools import (
    filesystem,
    memory,
    shell,
    subagent_tool,
    web,
)
from lsm_harness.agent.tools import ToolRegistry


def build_registry(
    conn, settings, subagent_manager=None,
    file_state=None,
    workspace_root: Path | None = None,
    prompt_snippets: list[str] | None = None,
    renderers: dict | None = None,
    readable_roots: list[Path] | None = None,
) -> ToolRegistry:
    allowed = {"read", "local_write"}

    # Shell is always external_write
    allowed.add("external_write")

    registry = ToolRegistry(allowed)

    def register(tool: ToolDefinition | AgentTool) -> None:
        if isinstance(tool, ToolDefinition):
            registry.register(wrap_tool_definition(tool))
            if prompt_snippets is not None and tool.prompt_snippet.strip():
                prompt_snippets.append(tool.prompt_snippet.strip())
            # §9.3: keep product renderers reachable after the AgentTool
            # wrap erases them; the CLI/TUI listener looks them up by name.
            if renderers is not None and (
                tool.render_call is not None or tool.render_result is not None
            ):
                renderers[tool.name] = (tool.render_call, tool.render_result)
            return
        registry.register(tool)

    # ── filesystem tools ────────────────────────────────────
    workspace = (workspace_root or Path.cwd()).resolve()
    for tool in filesystem.make_tools(
        workspace, file_state=file_state, readable_roots=readable_roots
    ):
        register(tool)

    # ── shell ───────────────────────────────────────────────
    register(shell.make_tool(
        workspace,
        default_timeout=settings.shell_timeout,
        allow=settings.shell_allow,
        deny=settings.shell_deny,
    ))

    # ── web ─────────────────────────────────────────────────
    for tool in web.make_tools(default_max_chars=settings.web_fetch_max_chars):
        register(tool)

    # ── project memory ──────────────────────────────────────
    memory_store = memory.ProjectMemoryStore(settings.home / "projects")
    for tool in memory.make_tools(memory_store, workspace):
        register(tool)

    # ── subagent spawn ──────────────────────────────────────
    if subagent_manager is not None:
        register(
            subagent_tool.make_tool(subagent_manager)
        )

    return registry
