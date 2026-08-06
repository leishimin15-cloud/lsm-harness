"""Explicit semantic-memory write tool."""

from lsm_harness.tools.registry import Tool


def make_tool(memory) -> Tool:
    def save_note(subject: str, content: str) -> str:
        fact_id = memory.facts.add(subject, content, source="user")
        return f"Saved fact #{fact_id} under '{subject}': {content}"

    return Tool(
        name="save_note",
        description="保存一个一个月后仍值得记住的用户、人物、项目或偏好事实。",
        input_schema={
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["subject", "content"],
        },
        fn=save_note,
        effect="local_write",
    )

