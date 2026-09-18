"""Project Memory:结构化、schema 校验的跨 session 项目记忆。

Pi(参考实现)只有 CLAUDE.md/AGENTS.md 这类用户手写上下文;本模块是
lsm-harness 的差异化:agent 自己可读写的持久记忆。

存储:`~/.lsm/projects/<slug>-<hash8>/memory.json`(按 workspace 分目录,
键是 resolve 后的 workspace 路径;slug 只是给人看的,hash 防同名冲突)。

三条职责:
- ``ProjectMemoryStore`` — 纯存储:load/save/delete + schema 校验。
  加载失败(文件损坏/格式错)不抛,返回空或错误文案——记忆永不能拖崩
  prompt 组装。
- ``render_for_prompt`` — 把条目渲染成 ``<project_memory>`` XML 段,
  由 ``Session.build_system`` 每 run 注入(天然免疫压缩)。
- ``make_tools`` — memory_save(local_write,过审批门)/ memory_recall
  (read,不过门)两个工具。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lsm_harness.coding_agent.tools import ToolDefinition

MEMORY_TYPES = ("decision", "gotcha", "preference", "fact")

_MAX_NAME = 64
_MAX_CONTENT = 2000
_MAX_TAGS = 8
_MAX_TAG = 32
_MAX_ENTRIES = 100

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProjectMemoryStore:
    """Per-project memory persistence under ``<projects_dir>/<slug>-<hash>``."""

    def __init__(self, projects_dir: Path) -> None:
        self.projects_dir = Path(projects_dir)

    def path_for(self, workspace_root: Path) -> Path:
        key = str(Path(workspace_root).resolve())
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(key).name).strip("-") or "root"
        return self.projects_dir / f"{slug}-{digest}" / "memory.json"

    def load(self, workspace_root: Path) -> list[dict[str, Any]]:
        """Load entries; a missing or corrupt file yields an empty list."""
        path = self.path_for(workspace_root)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, dict) or data.get("version") != 1:
            return []
        entries = data.get("entries")
        if not isinstance(entries, list):
            return []
        return [e for e in entries if isinstance(e, dict) and "name" in e]

    def save_entry(
        self,
        workspace_root: Path,
        *,
        name: str,
        type: str,
        content: str,
        tags: list[str] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Validate and upsert one entry. Returns ``(entry, error)``."""
        error = validate_entry(name=name, type=type, content=content, tags=tags)
        if error is not None:
            return None, error
        entries = self.load(workspace_root)
        existing = next((e for e in entries if e.get("name") == name), None)
        now = _now()
        if existing is None and len(entries) >= _MAX_ENTRIES:
            return None, (
                f"条目数已达上限 {_MAX_ENTRIES};"
                "请先让 agent 用新 name 覆盖旧条目"
            )
        entry = {
            "name": name,
            "type": type,
            "content": content,
            "tags": list(tags or []),
            "created_at": (existing or {}).get("created_at", now),
            "updated_at": now,
        }
        if existing is not None:
            entries[entries.index(existing)] = entry
        else:
            entries.append(entry)
        path = self.path_for(workspace_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"version": 1, "entries": entries},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        return entry, None

    def delete(self, workspace_root: Path, name: str) -> bool:
        entries = self.load(workspace_root)
        remaining = [e for e in entries if e.get("name") != name]
        if len(remaining) == len(entries):
            return False
        path = self.path_for(workspace_root)
        path.write_text(
            json.dumps(
                {"version": 1, "entries": remaining},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        return True


def validate_entry(
    *, name: str, type: str, content: str, tags: list[str] | None
) -> str | None:
    """Return an error message when the entry violates the schema."""
    if not isinstance(name, str) or not (1 <= len(name) <= _MAX_NAME):
        return f"name 长度须在 1-{_MAX_NAME} 之间"
    if not _NAME_RE.match(name):
        return "name 只能含字母/数字/连字符/下划线,且以字母或数字开头"
    if type not in MEMORY_TYPES:
        return f"type 必须是 {'/'.join(MEMORY_TYPES)} 之一,收到 {type!r}"
    if not isinstance(content, str) or not (1 <= len(content) <= _MAX_CONTENT):
        return f"content 长度须在 1-{_MAX_CONTENT} 之间"
    for tag in tags or []:
        if not isinstance(tag, str) or not (1 <= len(tag) <= _MAX_TAG):
            return f"tag 长度须在 1-{_MAX_TAG} 之间"
    if tags and len(tags) > _MAX_TAGS:
        return f"tags 最多 {_MAX_TAGS} 个"
    return None


def render_for_prompt(
    entries: list[dict[str, Any]],
    *,
    max_entries: int = 50,
    max_chars: int = 4000,
) -> str:
    """Render entries as a ``<project_memory>`` block for the system prompt."""
    if not entries:
        return ""
    lines = [
        "<project_memory>",
        "",
        "本项目的历史记忆(由 memory_save 写入,跨会话持久):",
        "",
    ]
    shown = entries[:max_entries]
    for entry in shown:
        tags = entry.get("tags") or []
        suffix = " " + " ".join(f"#{t}" for t in tags) if tags else ""
        lines.append(
            f"- [{entry.get('type', '?')}] {entry.get('name', '?')}: "
            f"{entry.get('content', '')}{suffix}"
        )
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    hidden = len(entries) - len(shown)
    if hidden > 0:
        text += f"\n(还有 {hidden} 条未展示,用 memory_recall 按需查看)"
    return text + "\n\n</project_memory>"


def make_tools(
    store: ProjectMemoryStore, workspace_root: Path
) -> list[ToolDefinition]:
    """Build memory_save / memory_recall bound to one workspace."""
    workspace = Path(workspace_root).resolve()

    def _save(
        name: str, type: str, content: str, tags: list[str] | None = None
    ) -> str:
        try:
            entry, error = store.save_entry(
                workspace, name=name, type=type, content=content, tags=tags
            )
        except OSError as exc:
            return f"Error: 写入项目记忆失败: {exc}"
        if error is not None:
            return f"Error: 记忆条目未通过校验: {error}"
        total = len(store.load(workspace))
        return (
            f"已保存项目记忆 {entry['name']!r}(type={entry['type']}),"
            f"当前共 {total} 条。它会在后续会话自动注入 system prompt。"
        )

    def _recall(
        name: str = "", type: str = "", tag: str = ""
    ) -> str:
        entries = store.load(workspace)
        if name:
            entry = next((e for e in entries if e.get("name") == name), None)
            if entry is None:
                return f"Error: 没有名为 {name!r} 的项目记忆"
            return json.dumps(entry, ensure_ascii=False, indent=2)
        if type:
            entries = [e for e in entries if e.get("type") == type]
        if tag:
            entries = [e for e in entries if tag in (e.get("tags") or [])]
        if not entries:
            return "本项目还没有任何记忆条目。"
        lines = []
        for e in entries:
            tags = e.get("tags") or []
            suffix = " " + " ".join(f"#{t}" for t in tags) if tags else ""
            lines.append(
                f"- [{e.get('type')}] {e.get('name')}"
                f"(更新于 {e.get('updated_at', '?')}){suffix}: "
                f"{str(e.get('content', ''))[:80]}"
            )
        return "\n".join(lines)

    return [
        ToolDefinition(
            name="memory_save",
            label="保存项目记忆",
            description=(
                "把一条值得跨会话保留的项目事实写入持久记忆。"
                "适合记录:技术决策(decision)、踩过的坑(gotcha)、"
                "用户偏好(preference)、稳定事实(fact)。"
                "同名条目会被覆盖更新。保存后会在后续会话自动注入上下文。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "条目名(字母/数字/连字符/下划线,≤64)",
                    },
                    "type": {
                        "type": "string",
                        "enum": list(MEMORY_TYPES),
                        "description": "decision=决策, gotcha=坑, preference=偏好, fact=事实",
                    },
                    "content": {
                        "type": "string",
                        "description": "记忆内容(≤2000 字,写清 why)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选标签(≤8 个)",
                    },
                },
                "required": ["name", "type", "content"],
            },
            execute=_save,
            effect="local_write",
            execution_mode="sequential",
            prompt_snippet=(
                "遇到值得跨会话保留的决策/坑/偏好/事实时,主动用 memory_save "
                "记录;用 memory_recall 查看已有记忆,避免重复保存。"
            ),
        ),
        ToolDefinition(
            name="memory_recall",
            label="读取项目记忆",
            description=(
                "读取本项目的持久记忆。不带参数返回紧凑列表;"
                "传 name 返回该条目全文;可按 type/tag 过滤。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "条目名,精确匹配"},
                    "type": {
                        "type": "string",
                        "enum": list(MEMORY_TYPES),
                        "description": "按类型过滤",
                    },
                    "tag": {"type": "string", "description": "按标签过滤"},
                },
            },
            execute=_recall,
            effect="read",
            execution_mode="parallel",
        ),
    ]


__all__ = [
    "MEMORY_TYPES",
    "ProjectMemoryStore",
    "make_tools",
    "render_for_prompt",
    "validate_entry",
]
