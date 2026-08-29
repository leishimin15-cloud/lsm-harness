"""Lightweight MCP client — stdio transport, sync API.

Connects to local MCP servers via stdio, discovers their tools/resources/prompts,
and adapts them directly into Agent-layer ``AgentTool`` objects.

Architecture::
    Main thread (sync)          Background thread (asyncio)
    ────────────────            ──────────────────────────
    MCPClient.start()   ──→    asyncio event loop
    connect_server(cfg)  ──→    stdio_client → ClientSession → list_tools()
      future.result()    ←──    return wrapped Tool list
    mcp_fn(**kwargs)    ──→    session.call_tool(...)
      future.result()    ←──    return result text
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsm_harness.agent.tools import AgentTool, ToolResult


# ── config ─────────────────────────────────────────────────────────


@dataclass
class MCPServerConfig:
    """Configuration for one MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    tool_timeout: float = 30.0
    enabled_tools: list[str] | None = None  # None = all, ["*"] = all


# ── name / schema helpers ──────────────────────────────────────────

_CHAR_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_name(name: str) -> str:
    """Sanitize an MCP tool name for model API compatibility."""
    cleaned = _CHAR_RE.sub("_", name)
    collapsed = re.sub(r"_+", "_", cleaned).strip("_")[:64]
    return collapsed or "unnamed"


def _normalize_schema(schema: Any) -> dict[str, Any]:
    """Normalize MCP JSON Schema for OpenAI/Anthropic tool definitions.

    Handles common MCP schema patterns: nullable types, missing properties,
    and local $ref references (to a limited extent).
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    result = dict(schema)

    # Normalize nullable: ["string", "null"] → "string" + nullable: true
    raw_type = result.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        if "null" in raw_type and len(non_null) == 1:
            result["type"] = non_null[0]
            result["nullable"] = True

    # Normalize oneOf [{"type": "string"}, {"type": "null"}] → string + nullable
    for key in ("oneOf", "anyOf"):
        branches = result.get(key)
        if isinstance(branches, list) and len(branches) == 2:
            types = []
            saw_null = False
            for b in branches:
                if isinstance(b, dict) and b.get("type") == "null":
                    saw_null = True
                elif isinstance(b, dict):
                    types.append(b)
            if saw_null and len(types) == 1:
                result = {k: v for k, v in result.items() if k != key}
                result.update(types[0])
                result["nullable"] = True
                break

    # Ensure object type has properties + required fields
    if result.get("type") == "object":
        result.setdefault("properties", {})
        result.setdefault("required", [])

    return result


# ── client ─────────────────────────────────────────────────────────


class MCPClient:
    """Synchronous MCP client backed by a background asyncio event loop.

    Usage::

        client = MCPClient()
        client.start()

        cfg = MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        tools = client.connect_server(cfg)
        for tool in tools:
            registry.register(tool)

        client.stop()
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, Any] = {}  # server_name → ClientSession
        self._connect_stacks: dict[str, Any] = {}  # server_name → AsyncExitStack
        self._lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the background asyncio event loop."""
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="mcp-event-loop",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut down all MCP connections and the event loop."""
        if self._loop is None:
            return
        # Close all connections
        async def _close_all():
            for name, stack in list(self._connect_stacks.items()):
                try:
                    await stack.aclose()
                except Exception:
                    pass
            self._connect_stacks.clear()
            self._sessions.clear()

        try:
            future = asyncio.run_coroutine_threadsafe(_close_all(), self._loop)
            future.result(timeout=5)
        except Exception:
            pass

        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None

    # ── connect ─────────────────────────────────────────────────

    def connect_server(self, config: MCPServerConfig) -> list[AgentTool]:
        """Connect to one MCP server and return wrapped Tool objects.

        Blocks until the server is initialized and tools are discovered.
        Returns an empty list on failure (errors are logged to stderr).
        """
        if self._loop is None:
            raise RuntimeError("MCPClient not started. Call start() first.")

        async def _connect():
            try:
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
            except ImportError:
                raise ImportError(
                    "MCP support requires the 'mcp' package. "
                    "Install it with: pip install mcp"
                )

            # Build server parameters
            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env,
                cwd=config.cwd,
            )

            # Connect
            stack = contextlib_AsyncExitStack()
            await stack.__aenter__()

            try:
                read, write = await stack.enter_async_context(
                    stdio_client(params)
                )
                session = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()

                # Discover tools
                tools_result = await session.list_tools()
                tools: list[AgentTool] = []

                allow_all = (
                    config.enabled_tools is None
                    or "*" in (config.enabled_tools or [])
                )
                enabled_set = set(config.enabled_tools or [])

                for tool_def in tools_result.tools:
                    if not allow_all and tool_def.name not in enabled_set:
                        continue

                    tool = self._wrap_tool(
                        session=session,
                        server_name=config.name,
                        tool_def=tool_def,
                        timeout=config.tool_timeout,
                    )
                    tools.append(tool)

                # Save session + stack for later cleanup
                with self._lock:
                    self._sessions[config.name] = session
                    self._connect_stacks[config.name] = stack

                return tools

            except Exception:
                await stack.aclose()
                raise

        try:
            future = asyncio.run_coroutine_threadsafe(_connect(), self._loop)
            return future.result(timeout=30)
        except ImportError:
            raise
        except Exception as exc:
            import sys
            print(f"[MCP] Failed to connect to '{config.name}': {exc}", file=sys.stderr)
            return []

    # ── tool wrapping ───────────────────────────────────────────

    def _wrap_tool(
        self,
        session: Any,
        server_name: str,
        tool_def: Any,
        timeout: float,
    ) -> AgentTool:
        """Adapt one dynamic MCP tool directly into the Agent layer."""

        tool_name = tool_def.name
        wrapped_name = _sanitize_name(f"mcp_{server_name}_{tool_name}")
        description = (
            f"[MCP/{server_name}] {tool_def.description or tool_def.name}"
        )
        raw_schema = tool_def.inputSchema or {"type": "object", "properties": {}}
        input_schema = _normalize_schema(raw_schema)

        def mcp_fn(**kwargs: Any) -> str | ToolResult:
            """Call the MCP tool from the background event loop."""
            if self._loop is None:
                return ToolResult(
                    output="MCP client not running.",
                    is_error=True,
                )

            async def _call():
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments=kwargs),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    return f"MCP 工具超时 ({timeout}s)。"
                except asyncio.CancelledError:
                    return "MCP 工具调用被取消。"

                parts: list[str] = []
                for block in result.content:
                    # TextContent
                    text = getattr(block, "text", None)
                    if text is not None:
                        parts.append(text)
                        continue
                    # EmbeddedResource / ImageContent → skip binary
                    mime = getattr(block, "mimeType", None)
                    if mime and mime.startswith("image/"):
                        parts.append("[image — not displayed]")
                        continue
                    parts.append(str(block))

                output = "\n".join(parts) or "(no output)"
                if getattr(result, "isError", False):
                    return ToolResult(output=output, is_error=True)
                return output

            try:
                future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
                return future.result(timeout=timeout + 5)
            except Exception as exc:
                return ToolResult(
                    output=f"MCP 工具调用失败: {type(exc).__name__}: {exc}",
                    is_error=True,
                )

        return AgentTool(
            name=wrapped_name,
            label=f"MCP · {tool_name}",
            description=description,
            parameters=input_schema,
            execute=mcp_fn,
            effect="external_write",  # MCP tools may have side effects
            execution_mode="sequential",
            timeout=timeout,
        )


# ── helper ────────────────────────────────────────────────────────

# Import here to avoid top-level mcp dependency
try:
    from contextlib import AsyncExitStack as _AsyncExitStack
    contextlib_AsyncExitStack = _AsyncExitStack
except ImportError:
    # Fallback for Python < 3.11 (shouldn't happen)
    import contextlib

    class contextlib_AsyncExitStack(contextlib.AsyncExitStack):  # type: ignore[no-redef]
        pass
