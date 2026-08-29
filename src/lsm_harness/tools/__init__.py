"""Build the native tool surface with optional RAG, subagent, and MCP."""

from pathlib import Path

from lsm_harness.agent.tools import AgentTool
from lsm_harness.coding_agent.tools import ToolDefinition, wrap_tool_definition
from lsm_harness.tools import (
    calendar,
    delegate,
    filesystem,
    memory_admin,
    notes,
    rag_tools,
    shell,
    subagent_tool,
    web,
)
from lsm_harness.tools.registry import ToolRegistry


def build_registry(
    conn, settings, memory, subagent_manager=None, mcp_client=None,
    rag_engine=None, sandbox=None, file_state=None,
    workspace_root: Path | None = None,
    prompt_snippets: list[str] | None = None,
    renderers: dict | None = None,
) -> ToolRegistry:
    allowed = {"read", "local_write"}

    # Enable pi delegation if pi is installed
    from lsm_harness.tools.delegate import _find_pi
    pi_available = _find_pi() is not None
    if pi_available:
        allowed.add("external_write")

    # Shell + MCP are always external_write
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

    # ── core tools ──────────────────────────────────────────
    for tool in calendar.make_tools(conn, settings.home):
        register(tool)
    register(notes.make_tool(memory))
    for tool in memory_admin.make_tools(settings, memory):
        register(tool)

    # ── filesystem tools ────────────────────────────────────
    workspace = (workspace_root or Path.cwd()).resolve()
    for tool in filesystem.make_tools(workspace, file_state=file_state):
        register(tool)

    # ── shell ───────────────────────────────────────────────
    register(shell.make_tool(
        workspace,
        sandbox=sandbox,
        sandbox_required=settings.sandbox_enabled,
        default_timeout=settings.shell_timeout,
        allow=settings.shell_allow,
        deny=settings.shell_deny,
    ))

    # ── web ─────────────────────────────────────────────────
    for tool in web.make_tools(default_max_chars=settings.web_fetch_max_chars):
        register(tool)

    # ── RAG tools ──────────────────────────────────────────
    if rag_engine is not None:
        for tool in rag_tools.make_tools(rag_engine):
            register(tool)

    # ── subagent spawn ──────────────────────────────────────
    if subagent_manager is not None:
        register(
            subagent_tool.make_tool(subagent_manager)
        )

    # ── pi delegation (optional) ────────────────────────────
    if pi_available:
        register(delegate.make_tool(settings.home))

    # ── MCP tools (optional) ────────────────────────────────
    if mcp_client is not None and hasattr(mcp_client, '_loop') and mcp_client._loop is not None:
        _connect_mcp_servers(settings, mcp_client, registry)

    return registry


def _connect_mcp_servers(settings, mcp_client, registry) -> None:
    """Discover and connect configured MCP servers, registering their tools."""
    from lsm_harness.mcp import MCPServerConfig
    import os

    # Load MCP config from environment or TOML
    config_path = settings.home / "mcp_servers.toml"
    servers: dict[str, dict] = {}

    if config_path.exists():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None
        if tomllib:
            raw = tomllib.loads(config_path.read_text())
            servers = raw.get("servers", {})

    for name, cfg in servers.items():
        try:
            mcp_cfg = MCPServerConfig(
                name=name or "unnamed",
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=cfg.get("env"),
                cwd=cfg.get("cwd"),
                tool_timeout=float(cfg.get("tool_timeout", 30)),
                enabled_tools=cfg.get("enabled_tools"),
            )
            if not mcp_cfg.command:
                continue
            tools = mcp_client.connect_server(mcp_cfg)
            for tool in tools:
                registry.register(tool)
        except Exception as exc:
            import sys
            print(f"[MCP] Server '{name}' failed: {exc}", file=sys.stderr)
