"""Local CRUD, persona, and procedural-memory tools."""

from __future__ import annotations

import re

from lsm_harness.memory.skills import parse_skill
from lsm_harness.runtime import load_soul
from lsm_harness.coding_agent.tools import ToolDefinition


SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")


def make_tools(settings, memory) -> list[ToolDefinition]:
    def manage_memory(
        action: str,
        kind: str = "fact",
        id: int = 0,
        query: str = "",
        content: str = "",
        subject: str = "",
    ) -> str:
        action, kind = action.lower(), kind.lower()
        if action == "search":
            if kind == "episode":
                rows = memory.episodes.search_with_ids(query, 8)
                return "\n".join(
                    f"#{row['id']} ({row['happened_at']}) {row['summary']}" for row in rows
                ) or "no episodes"
            rows = memory.facts.search_with_ids(query, 8)
            return "\n".join(
                f"#{row['id']} [{row['subject']}] {row['content']}" for row in rows
            ) or "no matching facts"
        if action == "update":
            if kind != "fact":
                return "Only facts can be updated."
            return (
                f"Updated fact #{id}."
                if memory.facts.update(int(id), content, subject or None)
                else f"No fact with id {id}."
            )
        if action == "delete":
            store = memory.episodes if kind == "episode" else memory.facts
            return (
                f"Deleted {kind} #{id}." if store.delete(int(id)) else f"No {kind} with id {id}."
            )
        return "action must be search, update, or delete"

    def update_soul(rule: str) -> str:
        clean = rule.strip().lstrip("-").strip()
        if not clean:
            return "Nothing to add."
        path = settings.home / "SOUL.md"
        text = load_soul(settings)
        if len(text) > 8000:
            return "SOUL.md is at its size limit."
        if "## Learned rules" not in text:
            text = text.rstrip() + "\n\n## Learned rules\n"
        path.write_text(text.rstrip() + f"\n- {clean}\n", encoding="utf-8")
        return f"Saved behavior rule: {clean}"

    def create_skill(name: str, description: str, body: str) -> str:
        slug = name.strip().lower().replace(" ", "-")
        if not SLUG.match(slug):
            return "Skill name must be a lowercase slug such as weekly-review."
        path = settings.home / "skills" / slug / "SKILL.md"
        if path.exists():
            return f"A skill named '{slug}' already exists."
        text = f"---\nname: {slug}\ndescription: {description.strip()}\n---\n\n{body.strip()}\n"
        if parse_skill(text, path) is None:
            return "Invalid skill: description and body are required."
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        memory.skills.refresh()
        return f"Created local skill '{slug}'."

    return [
        ToolDefinition(
            "manage_memory",
            "搜索、纠正或删除本地 facts 与 episodes；更新或删除前必须先搜索取得 id。",
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "update", "delete"]},
                    "kind": {"type": "string", "enum": ["fact", "episode"]},
                    "id": {"type": "integer"},
                    "query": {"type": "string"},
                    "content": {"type": "string"},
                    "subject": {"type": "string"},
                },
                "required": ["action"],
            },
            manage_memory,
            "local_write",
            label="管理记忆",
            execution_mode="sequential",
        ),
        ToolDefinition(
            "update_soul",
            "保存用户对 Agent 的长期行为偏好。",
            {
                "type": "object",
                "properties": {"rule": {"type": "string"}},
                "required": ["rule"],
            },
            update_soul,
            "local_write",
            label="更新人格规则",
            execution_mode="sequential",
        ),
        ToolDefinition(
            "create_skill",
            "在用户明确同意后，将可复用工作流保存为本地 SKILL.md。",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["name", "description", "body"],
            },
            create_skill,
            "local_write",
            label="创建 Skill",
            execution_mode="sequential",
        ),
    ]
