"""Single source of truth for the live Agent architecture and event flow."""

from __future__ import annotations

from typing import Any


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
