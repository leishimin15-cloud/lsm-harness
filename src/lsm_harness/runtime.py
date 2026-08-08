"""Token-aware context assembly, rolling summaries, and durable sessions."""

from __future__ import annotations

import json
import math
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from lsm_harness.config import Settings
from lsm_harness.memory import Memory
from lsm_harness.ops.session_store import (
    CompactionEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionHeader,
    append_session_entry,
    read_session_entries,
    replay_session,
    write_session_header,
)
from lsm_harness.security import redact_text


DEFAULT_SOUL = """你是 LSM，一个运行在用户本地的个人 Agent Harness。
你的回答应简洁、诚实、清楚，并优先使用工具完成实际任务。

规则：
- 用户要求创建日历事件时使用 create_event；当前时间会在下方提供。
- 用户询问日历时使用 list_events。
- 用户明确要求记住长期事实时使用 save_note。
- 用户要求纠正或忘记记忆时，先用 manage_memory 搜索，再更新或删除。
- 只有在用户明确同意后才能使用 create_skill。
- 工具结果会说明数据保存位置；不得声称写入了未连接的外部系统。
"""


# ── structured summary (pi-style) ────────────────────────────────
#
# Instead of a free-form Markdown prompt, we ask the model to fill in
# specific sections.  This makes incremental merging more reliable:
# the model knows exactly which field to update when new context arrives.

SUMMARY_SECTIONS = [
    ("Goal", "当前目标和项目背景"),
    ("Progress", "已完成事项（含具体工具结果和文件路径）"),
    ("In Progress", "正在进行中的工作"),
    ("Blocked", "当前阻塞项和依赖"),
    ("Key Decisions", "用户已做出的决定与偏好"),
    ("Next Steps", "未完成的下一步行动"),
    ("Critical Context", "必须保留的关键上下文（约束、警告、截止日期）"),
]

SUMMARY_SECTION_NAMES = [name for name, _ in SUMMARY_SECTIONS]
SUMMARY_SECTION_DESC = "\n".join(f"- **{name}**: {desc}" for name, desc in SUMMARY_SECTIONS)

SUMMARY_PROMPT = f"""你负责压缩个人 Agent 当前会话的较早上下文。
生成一份供后续模型继续工作的滚动摘要。

必须包含以下章节（使用 Markdown 标题 ## 分隔）：
{SUMMARY_SECTION_DESC}

规则：
- 每个章节用 2-5 条简洁的要点（每条一行）
- 只陈述对话中已有的信息，不得添加推测
- 忽略寒暄、重复表达和临时措辞
- 脱敏：API Key、Token、密码一律写为 [REDACTED]
- 如果某章节无内容，写"（无）"

已有摘要（上一版，你需要将其与新对话合并）：
{{previous}}

本次需要并入摘要的较早对话：
{{log}}
"""

UPDATE_SUMMARY_PROMPT = f"""你正在更新一份已有的会话摘要。
新对话已经发生，你需要将新信息合并到已有摘要中。

必须保持以下章节结构：
{SUMMARY_SECTION_DESC}

规则：
- 已有摘要中的信息如果仍然相关，必须保留
- 新对话可能覆盖旧信息（如用户改变了决定），此时用新信息替换
- 每个章节 2-5 条要点
- 脱敏：API Key、Token、密码一律写为 [REDACTED]

已有摘要：
{{previous}}

新对话：
{{log}}
"""


def load_soul(settings: Settings) -> str:
    path = settings.home / "SOUL.md"
    if not path.exists():
        path.write_text(DEFAULT_SOUL, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def estimate_tokens(value: str) -> int:
    """Conservative provider-neutral estimate for mixed Chinese/ASCII text."""
    if not value:
        return 0
    cjk = len(re.findall(r"[\u3400-\u9fff]", value))
    non_cjk = len(value) - cjk
    return cjk + math.ceil(non_cjk / 4)


def estimate_context_tokens(system: str, messages: list[dict[str, Any]]) -> int:
    total = estimate_tokens(system) + 4
    for message in messages:
        total += 4 + estimate_tokens(str(message.get("content") or ""))
    return total


class Session:
    def __init__(self, settings: Settings, memory: Memory, session_id: str | None = None):
        self.settings = settings
        self.memory = memory
        self.conn = memory.conn
        self._backfill_sessions()
        self.session_id = self._select_or_create(session_id)
        self.history = self._load_history()
        self._last_compaction_usage: dict[str, int] | None = None
        self._jsonl_lock = threading.Lock()
        self._next_entry_id = 1
        self._ensure_session_jsonl()

    def _backfill_sessions(self) -> None:
        rows = self.conn.execute(
            "SELECT session_id,MIN(created_at) AS created_at,MAX(created_at) AS updated_at "
            "FROM chat_log GROUP BY session_id"
        ).fetchall()
        for row in rows:
            session_id = str(row["session_id"])
            first = self.conn.execute(
                "SELECT content FROM chat_log WHERE session_id=? AND role='user' ORDER BY id LIMIT 1",
                (session_id,),
            ).fetchone()
            title = str(first["content"])[:80] if first else ""
            self.conn.execute(
                "INSERT OR IGNORE INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                (session_id, title, row["created_at"], row["updated_at"]),
            )
        self.conn.commit()

    def _select_or_create(self, requested: str | None) -> str:
        if requested:
            self._ensure_session(requested)
            return requested
        row = self.conn.execute(
            "SELECT id FROM sessions ORDER BY updated_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        if row:
            return str(row["id"])
        legacy = self.conn.execute(
            "SELECT session_id FROM chat_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        session_id = str(legacy["session_id"]) if legacy else str(uuid4())
        self._ensure_session(session_id)
        return session_id

    def _ensure_session(self, session_id: str) -> None:
        first = self.conn.execute(
            "SELECT content FROM chat_log WHERE session_id=? AND role='user' ORDER BY id LIMIT 1",
            (session_id,),
        ).fetchone()
        title = str(first["content"])[:80] if first else ""
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions(id,title) VALUES(?,?)", (session_id, title)
        )
        self.conn.commit()

    def _load_history(self) -> list[dict[str, str]]:
        rows = self.conn.execute(
            "SELECT role,content FROM chat_log WHERE session_id=? ORDER BY id",
            (self.session_id,),
        ).fetchall()
        return [{"role": str(row["role"]), "content": str(row["content"])} for row in rows]

    def _latest_summary(self):
        return self.conn.execute(
            "SELECT version,summary,through_chat_id,source_message_count,created_at "
            "FROM session_summaries WHERE session_id=? ORDER BY version DESC LIMIT 1",
            (self.session_id,),
        ).fetchone()

    def summary(self) -> str:
        row = self._latest_summary()
        return str(row["summary"]) if row else ""

    def summary_info(self) -> dict[str, Any] | None:
        row = self._latest_summary()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT s.id,s.title,s.created_at,s.updated_at,COUNT(DISTINCT c.id) AS message_count,"
            "COALESCE(MAX(ss.version),0) AS summary_version "
            "FROM sessions s LEFT JOIN chat_log c ON c.session_id=s.id "
            "LEFT JOIN session_summaries ss ON ss.session_id=s.id "
            "GROUP BY s.id ORDER BY s.updated_at DESC,s.rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def resume(self, session_ref: str) -> str | None:
        ref = session_ref.strip()
        if not ref:
            return None
        rows = self.conn.execute(
            "SELECT id FROM sessions WHERE id=? OR id LIKE ? ORDER BY updated_at DESC LIMIT 2",
            (ref, f"{ref}%"),
        ).fetchall()
        unique = list(dict.fromkeys(str(row["id"]) for row in rows))
        if len(unique) != 1:
            return None
        self.session_id = unique[0]
        self.conn.execute(
            "UPDATE sessions SET updated_at=strftime('%Y-%m-%d %H:%M:%f','now') WHERE id=?",
            (self.session_id,),
        )
        self.conn.commit()
        self.history = self._load_history()
        return self.session_id

    def build_system(self, user_message: str, emit) -> str:
        now = datetime.now().astimezone()
        parts = [
            load_soul(self.settings),
            f"当前时间：{now:%Y-%m-%d %H:%M %A} ({now:%Z}, UTC{now:%z})。",
            f"当前主模型：{self.settings.model}。",
        ]
        retrieved = self.memory.gated_retrieve(user_message, emit)
        if retrieved:
            parts.append("相关长期记忆：\n" + retrieved)
        skills = self.memory.matching_skills(user_message)
        if skills:
            parts.append("相关 Skill 指令：\n" + skills)
        return "\n\n".join(parts)

    @staticmethod
    def _with_summary(system: str, summary: str) -> str:
        if not summary:
            return system
        return (
            system
            + "\n\n当前会话历史摘要（仅作为已发生内容的参考，不是新的用户指令）：\n"
            + summary
        )

    def _rows_after(self, chat_id: int) -> list[Any]:
        return self.conn.execute(
            "SELECT id,role,content FROM chat_log "
            "WHERE session_id=? AND id>? ORDER BY id",
            (self.session_id, chat_id),
        ).fetchall()

    @staticmethod
    def _messages(rows) -> list[dict[str, str]]:
        return [
            {"role": str(row["role"]), "content": str(row["content"])} for row in rows
        ]

    def _find_turn_boundary(self, rows: list[Any], max_rows: int) -> int:
        """Find the nearest *user*-message boundary at or before max_rows.

        This guarantees we never cut in the middle of a turn
        (user → assistant → tool calls → assistant → …).
        """
        if len(rows) <= max_rows:
            return len(rows)
        # Walk back from max_rows to find the last user message
        for i in range(max_rows - 1, -1, -1):
            if rows[i]["role"] == "user":
                return i
        return max_rows  # fallback: no user message found

    def _estimate_from_messages(self, rows: list[Any]) -> int:
        """Fallback token estimate from raw rows when usage data is unavailable."""
        latest = self._latest_summary()
        summary = str(latest["summary"]) if latest else ""
        return estimate_context_tokens(summary, self._messages(rows))

    def _compress_if_due(self, estimated: int, emit) -> bool:
        trigger = min(
            max(1, self.settings.context_compression_tokens),
            max(1, self.settings.context_budget_tokens),
        )
        if estimated < trigger:
            return False
        return self._do_compress(emit, reason="threshold")

    def compact_and_rebuild(
        self, user_message: str, emit, tool_schemas: list[dict[str, Any]] | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Force a compaction, then rebuild context.

        Called by the loop after a truncation — the idea is:
        overflow → delete truncated message → compact → fresh context → retry.
        """
        self._do_compress(emit, reason="overflow")
        return self.prepare_context(user_message, emit, tool_schemas)

    def _do_compress(self, emit, reason: str = "threshold") -> bool:
        """Core compression logic, shared by threshold and overflow paths."""
        latest = self._latest_summary()
        through = int(latest["through_chat_id"]) if latest else 0
        previous = str(latest["summary"]) if latest else ""
        version = int(latest["version"]) if latest else 0
        rows = self._rows_after(through)
        keep = max(1, self.settings.context_recent_turns) * 2

        if len(rows) <= keep:
            return False

        # ── turn-aware cutting ──────────────────────────────
        cut_at = self._find_turn_boundary(rows, len(rows) - keep)
        eligible_rows = rows[:cut_at]
        if not eligible_rows:
            return False

        # Ensure even number (user+assistant pairs)
        if len(eligible_rows) > 1 and len(eligible_rows) % 2:
            eligible_rows = eligible_rows[:-1]
        if not eligible_rows:
            return False

        # ── source budget ───────────────────────────────────
        source_budget = max(
            256,
            self.settings.context_budget_tokens
            - self.settings.summary_max_tokens
            - estimate_tokens(SUMMARY_PROMPT)
            - estimate_tokens(previous),
        )
        eligible = []
        source_tokens = 0
        for row in eligible_rows:
            row_tokens = 4 + estimate_tokens(str(row["content"]))
            if eligible and source_tokens + row_tokens > source_budget:
                break
            eligible.append(row)
            source_tokens += row_tokens
        if len(eligible) > 1 and len(eligible) % 2:
            eligible.pop()
        if not eligible:
            return False

        # ── pick prompt: incremental merge vs fresh ─────────
        prompt_template = UPDATE_SUMMARY_PROMPT if previous else SUMMARY_PROMPT
        log = "\n".join(
            f"{row['role']}: {redact_text(str(row['content']), (self.settings.api_key,))}"
            for row in eligible
        )

        emit(
            "context.compression.started",
            {
                "session_id": self.session_id,
                "previous_version": version,
                "source_messages": len(eligible),
                "through_chat_id": int(eligible[-1]["id"]),
                "reason": reason,
            },
        )

        try:
            response = self.memory.client.complete(
                model=self.settings.small_model,
                system="",
                messages=[
                    {
                        "role": "user",
                        "content": prompt_template.format(
                            previous=redact_text(previous, (self.settings.api_key,)) or "（无）",
                            log=log,
                        ),
                    }
                ],
                tools=[],
                max_tokens=self.settings.summary_max_tokens,
            )
            emit(
                "llm.completed",
                {
                    "role": "context_compression",
                    "model": self.settings.small_model,
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                },
            )
            summary = redact_text(response.text.strip(), (self.settings.api_key,))
            if not summary:
                raise ValueError("empty summary")
            next_version = version + 1
            through_chat_id = int(eligible[-1]["id"])
            self.conn.execute(
                "INSERT INTO session_summaries"
                "(session_id,version,summary,through_chat_id,source_message_count) "
                "VALUES(?,?,?,?,?)",
                (
                    self.session_id,
                    next_version,
                    summary,
                    through_chat_id,
                    len(eligible),
                ),
            )
            self.conn.commit()

            # ── write compaction entry to session JSONL ─────
            comp_entry = CompactionEntry(
                id=self._make_id(),
                parent_id=None,
                timestamp=datetime.now().isoformat(),
                version=next_version,
                summary=summary,
                through_chat_id=through_chat_id,
                source_message_count=len(eligible),
            )
            self._write_entry(comp_entry)

            # ── record usage for fallback estimation ────────
            self._last_compaction_usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            emit(
                "context.compression.completed",
                {
                    "session_id": self.session_id,
                    "version": next_version,
                    "source_messages": len(eligible),
                    "through_chat_id": through_chat_id,
                    "summary_tokens": estimate_tokens(summary),
                    "reason": reason,
                },
            )
            return True
        except Exception as exc:
            self.conn.rollback()
            emit(
                "llm.failed",
                {
                    "role": "context_compression",
                    "model": self.settings.small_model,
                    "error": type(exc).__name__,
                },
            )
            emit(
                "context.compression.failed",
                {
                    "session_id": self.session_id,
                    "previous_version": version,
                    "source_messages": len(eligible),
                    "error": type(exc).__name__,
                    "reason": reason,
                },
            )
            return False

    def prepare_context(
        self,
        user_message: str,
        emit,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        base_system = self.build_system(user_message, emit)
        tool_tokens = estimate_tokens(
            json.dumps(tool_schemas or [], ensure_ascii=False, default=str)
        )
        latest = self._latest_summary()
        through = int(latest["through_chat_id"]) if latest else 0
        summary = str(latest["summary"]) if latest else ""
        rows = self._rows_after(through)
        candidate_messages = [
            *self._messages(rows),
            {"role": "user", "content": user_message},
        ]
        system = self._with_summary(base_system, summary)
        estimated = estimate_context_tokens(system, candidate_messages) + tool_tokens
        emit(
            "context.measured",
            {
                "session_id": self.session_id,
                "estimated_tokens": estimated,
                "budget_tokens": self.settings.context_budget_tokens,
                "compression_tokens": self.settings.context_compression_tokens,
                "history_messages": len(rows),
                "summary_version": int(latest["version"]) if latest else 0,
                "tool_schema_tokens": tool_tokens,
            },
        )

        self._compress_if_due(estimated, emit)
        latest = self._latest_summary()
        through = int(latest["through_chat_id"]) if latest else 0
        summary = str(latest["summary"]) if latest else ""
        system = self._with_summary(base_system, summary)
        selected = self._messages(self._rows_after(through))
        current = {"role": "user", "content": user_message}
        budget = max(256, self.settings.context_budget_tokens)
        dropped = 0
        while (
            selected
            and estimate_context_tokens(system, [*selected, current]) + tool_tokens > budget
        ):
            remove = 2 if len(selected) >= 2 else 1
            del selected[:remove]
            dropped += remove
        messages: list[dict[str, Any]] = [*selected, current]
        final_tokens = estimate_context_tokens(system, messages) + tool_tokens
        emit(
            "context.built",
            {
                "session_id": self.session_id,
                "estimated_tokens": final_tokens,
                "budget_tokens": budget,
                "history_messages": len(selected),
                "dropped_messages": dropped,
                "summary_version": int(latest["version"]) if latest else 0,
                "within_budget": final_tokens <= budget,
                "tool_schema_tokens": tool_tokens,
            },
        )
        return system, messages

    def window(self) -> list[dict]:
        return self.history[-self.settings.history_turns * 2 :]

    def add_exchange(self, user_message: str, result, source: str) -> None:
        record = result.reply
        tool_calls_for_jsonl: list[dict] = []
        if result.tool_calls:
            tool_summary = "; ".join(
                f"{item['tool']}({item['args']}) -> {item['output']}"
                for item in result.tool_calls
            )
            record += f"\n[tools used: {tool_summary}]"
            tool_calls_for_jsonl = [
                {"tool": item["tool"], "args": item["args"], "output": item["output"]}
                for item in result.tool_calls
            ]
        self.history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": record},
            ]
        )

        meta = {
            "iterations": result.iterations,
            "tools": [item["tool"] for item in result.tool_calls],
            "model": self.settings.model,
        }

        self.memory.log_chat(
            user_message,
            record,
            session_id=self.session_id,
            source=source,
            meta=meta,
        )

        # ── write to session JSONL ──────────────────────────
        parent_id = self._make_id()
        user_entry = MessageEntry.from_exchange(
            self._make_id(), parent_id, "user", user_message, source,
            meta=meta, tool_calls=tool_calls_for_jsonl if result.tool_calls else None,
        )
        assistant_entry = MessageEntry.from_exchange(
            self._make_id(), user_entry.id, "assistant", record, source,
            meta=meta,
        )
        self._write_entry(user_entry)
        self._write_entry(assistant_entry)

        self.conn.execute(
            "UPDATE sessions SET title=CASE WHEN title='' THEN ? ELSE title END,"
            "updated_at=strftime('%Y-%m-%d %H:%M:%f','now') WHERE id=?",
            (user_message.strip()[:80], self.session_id),
        )
        self.conn.commit()

    def start_new(self) -> str:
        self.session_id = str(uuid4())
        self._ensure_session(self.session_id)
        self.history = []
        self._ensure_session_jsonl()
        return self.session_id

    # ── JSONL session storage ────────────────────────────────

    @property
    def jsonl_path(self) -> Path:
        return self.settings.home / "sessions" / f"{self.session_id}.jsonl"

    def _ensure_session_jsonl(self) -> None:
        path = self.jsonl_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            header = SessionHeader.create(
                self.session_id, str(self.settings.home)
            )
            write_session_header(path, header)

    def _make_id(self) -> str:
        eid = f"{self.session_id[:8]}-{self._next_entry_id:04d}"
        self._next_entry_id += 1
        return eid

    def _write_entry(self, entry) -> None:
        append_session_entry(self.jsonl_path, entry, self._jsonl_lock)

    # ── export / import / replay ─────────────────────────────

    def export_jsonl(self, output_path: str | Path) -> Path:
        """Copy the session JSONL to an external file."""
        import shutil
        dest = Path(output_path)
        shutil.copy2(self.jsonl_path, dest)
        return dest

    def import_jsonl(self, input_path: str | Path, *, emit=None) -> int:
        """Import entries from an external JSONL into this session.

        Returns the number of messages imported.
        """
        notify = emit or (lambda k, d: None)
        entries = read_session_entries(Path(input_path))
        if not entries:
            return 0

        # Collect messages in pairs (user → assistant)
        pending_user: str = ""
        count = 0
        for entry in entries:
            if entry.type == "message":
                me = entry  # type: ignore[assignment]
                if me.role == "user":
                    pending_user = me.content
                elif me.role == "assistant" and pending_user:
                    self.memory.log_chat(
                        pending_user,
                        me.content,
                        session_id=self.session_id,
                        source=me.source or "import",
                        meta=me.meta,
                    )
                    count += 1
                    pending_user = ""
            elif entry.type == "compaction":
                ce = entry  # type: ignore[assignment]
                self.conn.execute(
                    "INSERT OR IGNORE INTO session_summaries"
                    "(session_id,version,summary,through_chat_id,source_message_count) "
                    "VALUES(?,?,?,?,?)",
                    (self.session_id, ce.version, ce.summary,
                     ce.through_chat_id, ce.source_message_count),
                )
                self.conn.commit()
            elif entry.type == "model_change":
                mce = entry  # type: ignore[assignment]
                notify("session.replay.model_change",
                       {"provider": mce.provider, "model": mce.model})

        self.history = self._load_history()
        notify("session.imported", {"source": str(input_path), "messages": count})
        return count

    def replay(self, emit) -> list[dict[str, str]]:
        """Replay the session from JSONL, emitting events as we go.

        Returns the reconstructed message list.
        """
        entries = read_session_entries(self.jsonl_path)
        return replay_session(entries, emit, self.settings.api_key)
