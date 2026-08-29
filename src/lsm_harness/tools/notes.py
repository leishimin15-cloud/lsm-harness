"""Explicit semantic-memory write tool."""

from lsm_harness.coding_agent.tools import ToolDefinition


def make_tool(memory) -> ToolDefinition:
    def save_note(subject: str, content: str) -> str:
        fact_id = memory.facts.add(subject, content, source="user")
        return f"Saved fact #{fact_id} under '{subject}': {content}"

    return ToolDefinition(
        name="save_note",
        label="保存长期记忆",
        description="保存一个一个月后仍值得记住的用户、人物、项目或偏好事实。",
        parameters={
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["subject", "content"],
        },
        execute=save_note,
        effect="local_write",
        execution_mode="sequential",
    )
