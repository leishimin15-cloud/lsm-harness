"""Single source of truth for the live Agent architecture and event flow."""

from __future__ import annotations

from typing import Any


FLOW_SPEC: dict[str, Any] = {
    "groups": [
        {"id": "runtime", "label": "RUNTIME — one observable trace"},
        {"id": "memory", "label": "MEMORY & RETRIEVAL"},
        {"id": "actions", "label": "ACTION SURFACE"},
        {"id": "persistence", "label": "PERSISTENCE & OPS"},
    ],
    "nodes": [
        {"id": "gateway", "label": "Gateway", "sub": "CLI · TUI · Web", "group": "runtime", "x": 30, "y": 70, "w": 130, "h": 58, "route": "overview"},
        {"id": "session", "label": "Session", "sub": "history · identity", "group": "runtime", "x": 200, "y": 70, "w": 130, "h": 58, "route": "sessions"},
        {"id": "context", "label": "Context", "sub": "budget · governor", "group": "runtime", "x": 370, "y": 70, "w": 150, "h": 58, "route": "loop"},
        {"id": "memory_gate", "label": "Memory gate", "sub": "retrieve only if needed", "group": "runtime", "x": 560, "y": 70, "w": 160, "h": 58, "route": "memory"},
        {"id": "llm", "label": "LLM agent", "sub": "reason · decide", "group": "runtime", "x": 760, "y": 70, "w": 150, "h": 58, "route": "loop"},
        {"id": "reply", "label": "Reply", "sub": "stream back", "group": "runtime", "x": 950, "y": 70, "w": 130, "h": 58, "route": "loop"},

        {"id": "procedural", "label": "Procedural", "sub": "SOUL · Skills", "group": "memory", "x": 240, "y": 215, "w": 150, "h": 58, "route": "memory"},
        {"id": "semantic", "label": "Semantic", "sub": "facts · FTS5", "group": "memory", "x": 430, "y": 215, "w": 150, "h": 58, "route": "memory"},
        {"id": "episodic", "label": "Episodic", "sub": "dated episodes", "group": "memory", "x": 620, "y": 215, "w": 150, "h": 58, "route": "memory"},
        {"id": "rag", "label": "Document RAG", "sub": "embed · FTS · rerank", "group": "memory", "x": 810, "y": 215, "w": 170, "h": 58, "route": "rag", "capability": "rag"},

        {"id": "tool_call", "label": "Tool call", "sub": "schema · args", "group": "actions", "x": 360, "y": 365, "w": 140, "h": 58, "route": "tools"},
        {"id": "approval", "label": "Approval", "sub": "read auto · writes ask", "group": "actions", "x": 540, "y": 365, "w": 160, "h": 58, "route": "tools"},
        {"id": "native_tools", "label": "Native tools", "sub": "files · web · memory", "group": "actions", "x": 740, "y": 345, "w": 170, "h": 58, "route": "tools"},
        {"id": "mcp", "label": "MCP", "sub": "external servers", "group": "actions", "x": 950, "y": 345, "w": 130, "h": 58, "route": "tools", "capability": "mcp"},
        {"id": "sandbox", "label": "Docker sandbox", "sub": "isolated exec", "group": "actions", "x": 740, "y": 440, "w": 170, "h": 58, "route": "tools", "capability": "sandbox"},
        {"id": "subagents", "label": "Subagents", "sub": "parallel contexts", "group": "actions", "x": 950, "y": 440, "w": 150, "h": 58, "route": "subagents"},

        {"id": "chat_store", "label": "Chat persistence", "sub": "SQLite · JSONL", "group": "persistence", "x": 230, "y": 600, "w": 170, "h": 58, "route": "database"},
        {"id": "summary", "label": "Rolling summary", "sub": "context compression", "group": "persistence", "x": 440, "y": 600, "w": 170, "h": 58, "route": "memory"},
        {"id": "consolidation", "label": "Consolidation", "sub": "facts · episodes", "group": "persistence", "x": 650, "y": 600, "w": 170, "h": 58, "route": "memory"},
        {"id": "file_state", "label": "FileState", "sub": "diff · undo", "group": "persistence", "x": 860, "y": 600, "w": 150, "h": 58, "route": "files"},
        {"id": "trace", "label": "Trace & usage", "sub": "always on", "group": "persistence", "x": 1050, "y": 580, "w": 160, "h": 58, "route": "ops"},
        {"id": "eval", "label": "Eval", "sub": "regression gate", "group": "persistence", "x": 1050, "y": 665, "w": 160, "h": 58, "route": "ops"},
    ],
    "edges": [
        {"id": "gateway-session", "source": "gateway", "target": "session"},
        {"id": "session-context", "source": "session", "target": "context"},
        {"id": "context-gate", "source": "context", "target": "memory_gate"},
        {"id": "gate-llm", "source": "memory_gate", "target": "llm"},
        {"id": "llm-reply", "source": "llm", "target": "reply"},
        {"id": "procedural-gate", "source": "procedural", "target": "memory_gate", "optional": True},
        {"id": "semantic-gate", "source": "semantic", "target": "memory_gate", "optional": True},
        {"id": "episodic-gate", "source": "episodic", "target": "memory_gate", "optional": True},
        {"id": "rag-gate", "source": "rag", "target": "memory_gate", "optional": True},
        {"id": "llm-tool", "source": "llm", "target": "tool_call"},
        {"id": "tool-approval", "source": "tool_call", "target": "approval"},
        {"id": "approval-native", "source": "approval", "target": "native_tools"},
        {"id": "approval-mcp", "source": "approval", "target": "mcp"},
        {"id": "approval-sandbox", "source": "approval", "target": "sandbox"},
        {"id": "approval-subagent", "source": "approval", "target": "subagents"},
        {"id": "tools-llm", "source": "native_tools", "target": "llm"},
        {"id": "mcp-llm", "source": "mcp", "target": "llm"},
        {"id": "sandbox-llm", "source": "sandbox", "target": "llm"},
        {"id": "subagent-llm", "source": "subagents", "target": "llm"},
        {"id": "reply-chat", "source": "reply", "target": "chat_store"},
        {"id": "chat-summary", "source": "chat_store", "target": "summary"},
        {"id": "chat-consolidation", "source": "chat_store", "target": "consolidation"},
        {"id": "reply-files", "source": "reply", "target": "file_state"},
        {"id": "reply-trace", "source": "reply", "target": "trace"},
        {"id": "trace-eval", "source": "trace", "target": "eval"},
    ],
}


_STATIC_BINDINGS: dict[str, tuple[str, list[str], list[str]]] = {
    "trace.accepted": ("gateway", ["gateway"], []),
    "trace.started": ("gateway", ["gateway", "session"], ["gateway-session"]),
    "turn.started": ("reason", ["llm"], ["gate-llm"]),
    "turn.completed": ("reason", ["llm"], []),
    "context.build.started": ("context", ["context"], ["session-context"]),
    "context.build.completed": ("context", ["context", "memory_gate"], ["session-context", "context-gate"]),
    "context.measured": ("context", ["context"], ["session-context"]),
    "context.built": ("context", ["context"], ["session-context", "context-gate"]),
    "memory.gate.started": ("retrieval", ["memory_gate"], ["context-gate"]),
    "memory.gate.decided": ("retrieval", ["memory_gate"], []),
    "memory.retrieved": ("retrieval", ["semantic", "episodic", "procedural", "memory_gate"], ["semantic-gate", "episodic-gate", "procedural-gate"]),
    "rag.gate.decided": ("rag", ["rag", "memory_gate"], ["rag-gate"]),
    "rag.retrieved": ("rag", ["rag", "memory_gate"], ["rag-gate"]),
    "llm.started": ("reason", ["llm"], ["gate-llm"]),
    "llm.text.start": ("reply", ["llm", "reply"], ["llm-reply"]),
    "llm.text.delta": ("reply", ["reply"], ["llm-reply"]),
    "llm.completed": ("reason", ["llm"], []),
    "tool.requested": ("tool", ["tool_call"], ["llm-tool"]),
    "tool.approval.required": ("approval", ["approval"], ["tool-approval"]),
    "tool.approval.resolved": ("approval", ["approval"], ["tool-approval"]),
    "subagent.started": ("subagent", ["subagents"], ["approval-subagent"]),
    "subagent.progress": ("subagent", ["subagents"], []),
    "subagent.completed": ("subagent", ["subagents", "llm"], ["subagent-llm"]),
    "persistence.started": ("persistence", ["chat_store"], ["reply-chat"]),
    "persistence.completed": ("persistence", ["chat_store", "trace"], ["reply-chat", "reply-trace"]),
    "memory.consolidation.started": ("consolidation", ["consolidation"], ["chat-consolidation"]),
    "memory.consolidation.completed": ("consolidation", ["consolidation", "semantic", "episodic"], ["chat-consolidation"]),
    "trace.file_changes": ("files", ["file_state"], ["reply-files"]),
    "trace.done": ("done", ["reply", "trace"], ["llm-reply", "reply-trace"]),
    "trace.completed": ("done", ["reply", "trace"], ["llm-reply", "reply-trace"]),
    "trace.aborted": ("aborted", ["reply", "trace"], ["reply-trace"]),
    "trace.error": ("error", ["reply", "trace"], ["reply-trace"]),
    "trace.failed": ("error", ["reply", "trace"], ["reply-trace"]),
}


def _tool_target(name: str) -> tuple[str, str]:
    if name == "exec":
        return "sandbox", "approval-sandbox"
    if name == "spawn":
        return "subagents", "approval-subagent"
    if name.startswith("mcp_"):
        return "mcp", "approval-mcp"
    return "native_tools", "approval-native"


def flow_for_event(event_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map one event to diagram nodes and edges from the shared topology."""
    payload = data or {}
    if event_type in {"tool.started", "tool.progress", "tool.completed"}:
        target, edge = _tool_target(str(payload.get("tool", "")))
        state = "error" if payload.get("status") == "error" else "running"
        if event_type == "tool.completed":
            state = "error" if payload.get("status") == "error" else "success"
        return {
            "phase": "tool",
            "active_nodes": ["tool_call", target],
            "active_edges": [edge] + ([f"{target.replace('_tools', '')}-llm"] if target in {"mcp", "sandbox", "subagents"} else []),
            "state": state,
        }
    phase, nodes, edges = _STATIC_BINDINGS.get(event_type, ("trace", ["trace"], []))
    state = "running"
    if event_type.endswith((".completed", ".done")):
        state = "success"
    elif event_type.endswith((".failed", ".error")):
        state = "error"
    elif event_type.endswith(".aborted"):
        state = "aborted"
    elif event_type == "memory.gate.decided" and payload.get("decision") == "skip":
        state = "skipped"
    elif event_type == "tool.approval.required":
        state = "waiting"
    if event_type == "turn.completed" and payload.get("status") in {"error", "length_exhausted"}:
        state = "error"
    elif event_type == "turn.completed" and payload.get("status") == "aborted":
        state = "aborted"
    return {"phase": phase, "active_nodes": nodes, "active_edges": edges, "state": state}


def topology_snapshot(app: Any | None = None) -> dict[str, Any]:
    capabilities = {
        "rag": bool(app and getattr(app, "rag", None)),
        "mcp": bool(app and getattr(app, "mcp", None)),
        "sandbox": bool(app and getattr(app, "sandbox", None)),
        "subagents": bool(app and getattr(app, "subagents", None)),
    }
    nodes = []
    for raw in FLOW_SPEC["nodes"]:
        node = dict(raw)
        capability = node.get("capability")
        node["enabled"] = capabilities.get(capability, True) if capability else True
        node["health"] = "healthy" if node["enabled"] else "disabled"
        nodes.append(node)
    return {
        "groups": FLOW_SPEC["groups"],
        "nodes": nodes,
        "edges": FLOW_SPEC["edges"],
        "capabilities": capabilities,
    }
