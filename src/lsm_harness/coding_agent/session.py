"""Coding-agent context assembly, rolling summaries, and durable sessions."""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from lsm_harness.agent.messages import (
    AgentMessage,
    AssistantMessage,
    CustomMessage,
    UserMessage,
    message_preview,
)
from lsm_harness.agent.messages import user_message as _build_user_message
from lsm_harness.ai.messages import ToolResultMessage
from lsm_harness.coding_agent import compaction as compaction_algo
from lsm_harness.coding_agent.messages import (
    BRANCH_SUMMARY,
    COMPACTION_SUMMARY,
    branch_summary_message,
    compaction_summary_message,
    register_coding_agent_messages,
)
from lsm_harness.coding_agent.resources import (
    format_project_context,
    load_project_context_files,
)
from lsm_harness.coding_agent.session_recorder import SessionRecorder
from lsm_harness.config import Settings
from lsm_harness.coding_agent.skills import SkillLoader
from lsm_harness.ops.session_store import (
    BranchSummaryEntry,
    CompactionEntry,
    LabelEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionHeader,
    ThinkingLevelChange,
    append_session_entry,
    build_by_id,
    collect_abandoned_branch,
    path_to_leaf,
    read_session_entries,
    replay_session,
    write_session_header,
)
from lsm_harness.security import redact_text


SYSTEM_PERSONA = """你是 LSM，一个运行在用户本地的 coding agent。
你的回答应简洁、诚实、清楚，并优先使用工具完成实际任务。
工具结果会说明数据保存位置；不得声称写入了未连接的外部系统。
"""

RESPONSE_STYLE = """回复规范：
- 工具调用前不要输出“让我先看看”“我先探索一下”等过程性承诺；工具完成后直接给结论。
- 最终回答先回答用户真正的问题，再给必要证据或细节；不要把探索过程拼进最终答案。
- 避免复述用户问题、空泛总结和在窄栏中难以阅读的大型 ASCII 图。
- 可以使用简洁的 Markdown 标题、列表和代码块；不要展示隐藏思维链。
"""


# ── structured summary (pi-style, chapter 9) ─────────────────────
#
# Instead of a free-form Markdown prompt, we ask the model to fill in
# specific sections.  This makes incremental merging more reliable:
# the model knows exactly which field to update when new context arrives.
# Sections follow Pi's 6-section template; Progress carries the three
# sub-items Done / In Progress / Blocked.

SUMMARY_SECTIONS = [
    ("Goal", "当前目标和项目背景"),
    ("Constraints & Preferences", "约束条件与用户偏好"),
    ("Progress", "进展，分三个子标题：### Done（已完成，含具体工具结果和文件路径）、### In Progress（进行中）、### Blocked（阻塞项和依赖）"),
    ("Key Decisions", "用户已做出的关键决策"),
    ("Next Steps", "未完成的下一步行动"),
    ("Critical Context", "必须保留的关键上下文（约束、警告、截止日期）"),
]

SUMMARY_SECTION_NAMES = [name for name, _ in SUMMARY_SECTIONS]
SUMMARY_SECTION_DESC = "\n".join(f"- **{name}**: {desc}" for name, desc in SUMMARY_SECTIONS)

# Branch summaries (chapter 10) use Pi's lighter 5-section template —
# no Critical Context: a branch summary is auxiliary context and must not
# compete with the mainline for tokens (Pi caps it at 2048 maxTokens).
BRANCH_SUMMARY_SECTIONS = [s for s in SUMMARY_SECTIONS if s[0] != "Critical Context"]
BRANCH_SUMMARY_SECTION_DESC = "\n".join(
    f"- **{name}**: {desc}" for name, desc in BRANCH_SUMMARY_SECTIONS
)

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

BRANCH_SUMMARY_PROMPT = f"""用户在此会话中探索过一条不同的对话分支，随后回到了分叉点。
你负责把那条被放弃的分支压缩成一份简报，让当前分支的模型知道"之前试过什么"。

必须包含以下章节（使用 Markdown 标题 ## 分隔）：
{BRANCH_SUMMARY_SECTION_DESC}

规则：
- 每个章节用 1-3 条简洁要点，整体务必精简（这是辅助上下文，不是主线历史）
- 重点保留：尝试过的方案、结论（尤其是"为什么放弃"）、关键发现
- 只陈述对话中已有的信息，不得添加推测
- 脱敏：API Key、Token、密码一律写为 [REDACTED]

被放弃分支的对话：
{{log}}
"""

# Split-turn prefix (pi ch9 §5): when the cut point is an assistant
# message, the turn's user went into the main summary while its
# assistant/toolResult prefix is dropped from view — this lighter
# 3-section summary covers exactly that half-turn.
TURN_PREFIX_SUMMARY_PROMPT = """一个对话 Turn 被上下文压缩从中间切断：用户的原始请求已并入主摘要，
该 Turn 的前半部分（若干 assistant 回复与工具结果）需要压缩成一份简报，
让模型理解保留区里这个 Turn 的后半截在接续什么。

必须包含以下章节（使用 Markdown 标题 ## 分隔）：
- **Original Request**: 用户的原始请求（一句话）
- **Early Progress**: 这个 Turn 前半已完成的工作（工具调用、中间结论）
- **Context for Suffix**: 理解后半截所必需的上下文（中间结果、路径、变量）

规则：
- 每个章节 1-3 条简洁要点，整体务必精简
- 只陈述对话中已有的信息，不得添加推测
- 脱敏：API Key、Token、密码一律写为 [REDACTED]

被切断的 Turn 前缀：
{log}
"""


class ContextOverflowError(RuntimeError):
    """The pinned summary plus the current input alone exceed the context
    budget — trimming further would fabricate context, so the run must
    fail loudly instead of sending an illegal oversized request."""


def _trim_block_size(body: list) -> int:
    """Size of the leading droppable block (reliability batch 1).

    Budget trimming may only cut at legal message boundaries: a lone
    message is one block; an assistant message carrying tool_calls forms
    one block with EVERY tool result answering it.  Dropping any other
    way would leave an orphaned tool_result or a call missing its
    result — an invalid request, not "less context".
    """
    first = body[0]
    if isinstance(first, AssistantMessage) and first.tool_calls:
        call_ids = {call.id for call in first.tool_calls}
        end = 1
        while (
            end < len(body)
            and isinstance(body[end], ToolResultMessage)
            and body[end].tool_call_id in call_ids
        ):
            end += 1
        return end
    return 1


def estimate_tokens(value: str) -> int:
    """Conservative provider-neutral estimate for mixed Chinese/ASCII text."""
    if not value:
        return 0
    cjk = len(re.findall(r"[\u3400-\u9fff]", value))
    non_cjk = len(value) - cjk
    return cjk + math.ceil(non_cjk / 4)


def estimate_context_tokens(system: str, messages: list[AgentMessage]) -> int:
    """Estimate token count for a system prompt plus typed messages."""
    total = estimate_tokens(system) + 4
    for message in messages:
        total += 4 + estimate_tokens(message_preview(message, limit=1_000_000))
    return total


@dataclass
class SessionContext:
    """The tree's current view: messages + state recovered from the path.

    ``model`` / ``small_model`` / ``thinking_level`` come from the header
    baseline plus model_change / thinking_level_change entries on the
    path.  They are applied to the live runtime at session OPEN/SWITCH
    time (Pi createAgentSession parity, \u9636\u6bb5 4 \u6279 2) \u2014 in-session
    branching changes which path is current but does NOT itself apply
    the state (Pi navigateTree only replaces messages).
    """

    messages: list[AgentMessage]
    model: str | None = None
    provider: str | None = None
    thinking_level: str | None = None
    small_model: str | None = None


class Session:
    def __init__(
        self,
        settings: Settings,
        *,
        conn,
        client,
        skills: SkillLoader | None = None,
        session_id: str | None = None,
    ):
        # A Session can place compaction/branch summaries into context, so
        # it must guarantee their translators are registered even when no
        # Harness was constructed (strict LLM boundary since batch A).
        register_coding_agent_messages()
        self.settings = settings
        self.conn = conn
        self.client = client
        self.skills = skills or SkillLoader([settings.home / "skills"])
        self._backfill_sessions()
        self.session_id = self._select_or_create(session_id)
        self._last_compaction_usage: dict[str, int] | None = None
        self._jsonl_lock = threading.Lock()
        self.recorder: SessionRecorder | None = None
        # Bind the recorder FIRST: _load_history reads the tree through
        # the recorder's leaf pointer (file's last line, Pi _buildIndex).
        self._ensure_session_jsonl()
        self.history = self._load_history()

    def _log_chat(
        self,
        user_message: str,
        reply: str,
        *,
        session_id: str,
        source: str,
        meta: dict | None = None,
        commit: bool = True,
    ) -> tuple[int, int]:
        """Append one user/assistant exchange; returns both chat_log ids."""
        user_id = self.conn.execute(
            "INSERT INTO chat_log(role,content,session_id,source) VALUES('user',?,?,?)",
            (user_message, session_id, source),
        ).lastrowid
        assistant_id = self.conn.execute(
            "INSERT INTO chat_log(role,content,session_id,source,meta) "
            "VALUES('assistant',?,?,?,?)",
            (reply, session_id, source, json.dumps(meta, ensure_ascii=False) if meta else None),
        ).lastrowid
        if commit:
            self.conn.commit()
        return int(user_id), int(assistant_id)

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

    def _history_from_tree(self) -> list[dict[str, str]] | None:
        """Display history rebuilt from the current tree path, or None.

        The JSONL tree is the fact source (阶段 4 批 1, Pi parity:
        createAgentSession rebuilds messages via buildSessionContext at
        open, with leaf = the file's last line — Pi _buildIndex).  The
        linear chat_log projection still contains abandoned-branch rows
        after a branch+restart and must NOT be consulted as authority.

        None when the tree carries no message entries yet (brand-new or
        pre-backfill legacy session) so the caller can fall back.
        """
        if not self.jsonl_path.exists():
            return None
        entries = read_session_entries(self.jsonl_path)
        messages = [
            entry
            for entry in path_to_leaf(entries, self._last_entry_id)
            if entry.type == "message"
        ]
        if not messages:
            return None
        return [
            {
                "role": entry.message.role,
                "content": message_preview(entry.message, limit=1_000_000),
            }
            for entry in messages
        ]

    def _load_history(self) -> list[dict[str, str]]:
        """Tree-first, chat_log as the legacy fallback only."""
        tree_history = self._history_from_tree()
        if tree_history is not None:
            return tree_history
        rows = self.conn.execute(
            "SELECT role,content FROM chat_log WHERE session_id=? ORDER BY id",
            (self.session_id,),
        ).fetchall()
        return [{"role": str(row["role"]), "content": str(row["content"])} for row in rows]

    def _latest_summary(self):
        """SQLite projection lookup — legacy fallback only.

        In a branched session the globally-latest summary may belong to a
        SIBLING branch.  The authoritative "current summary" is the last
        CompactionEntry on the current tree path (see
        ``_path_last_compaction``); this SQLite row remains only for
        sessions whose JSONL tree has no messages yet (pre-ch10 legacy).
        """
        return self.conn.execute(
            "SELECT version,summary,through_chat_id,source_message_count,created_at "
            "FROM session_summaries WHERE session_id=? ORDER BY version DESC LIMIT 1",
            (self.session_id,),
        ).fetchone()

    def _path_last_compaction(self):
        """Last CompactionEntry on the CURRENT tree path, or None.

        The JSONL session tree is the authoritative store — in a branched
        session this is the only correct source for "the current summary"
        (code-review issue 一/六).
        """
        if not self.jsonl_path.exists():
            return None
        entries = read_session_entries(self.jsonl_path)
        if not entries:
            return None
        path = path_to_leaf(entries, self._last_entry_id)
        for entry in reversed(path):
            if entry.type == "compaction":
                return entry
        return None

    def _tree_has_messages(self) -> bool:
        """True when the JSONL tree holds any message entries at all.

        Distinguishes "branched session whose CURRENT path has no
        compaction" (answer: no summary — the globally-latest SQLite row
        belongs to a sibling branch and must not leak) from "legacy
        session without a tree" (answer: the SQLite fallback row).
        """
        if not self.jsonl_path.exists():
            return False
        return any(
            entry.type == "message"
            for entry in read_session_entries(self.jsonl_path)
        )

    def tree_tip_allows_continue(self) -> bool:
        """True when the current tree path's LAST message is a legal
        continuation point (Pi: a user question, or a tool result whose
        next model call never happened).

        This is continue()'s legality source: an interrupted run leaves
        its question — or its last tool result — as the tree tip, and the
        tree, not the previous process's in-memory state, survives a
        restart.  A completed run's tip is the assistant answer (illegal);
        a failed run's tip is the loop-recorded assistant error message
        (also illegal, matching the pre-tree contract).
        """
        if self.recorder is None or not self.jsonl_path.exists():
            return False
        entries = read_session_entries(self.jsonl_path)
        if not entries:
            return False
        path = path_to_leaf(entries, self._last_entry_id)
        for entry in reversed(path):
            if entry.type == "message":
                return getattr(entry.message, "role", None) in ("user", "tool")
        return False

    def summary(self) -> str:
        entry = self._path_last_compaction()
        if entry is not None:
            return str(entry.summary)
        if self._tree_has_messages():
            return ""
        row = self._latest_summary()
        return str(row["summary"]) if row else ""

    def summary_info(self) -> dict[str, Any] | None:
        entry = self._path_last_compaction()
        if entry is not None:
            return {
                "version": entry.version,
                "summary": entry.summary,
                "through_chat_id": entry.through_chat_id,
                "source_message_count": entry.source_message_count,
                "created_at": entry.timestamp,
            }
        if self._tree_has_messages():
            return None
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
        # Recorder first (its leaf = the file's last line decides the
        # current path), then the tree-derived display history.
        self._ensure_session_jsonl()
        self.history = self._load_history()
        return self.session_id

    def resync_history(self) -> None:
        """Re-derive the display history from the tree (the fact source).

        Used after aborted/failed runs: their user message reached the
        tree via kernel events but never went through add_exchange, so
        the in-memory mirror would otherwise drift from the tree until
        the next reload.
        """
        self.history = self._history_from_tree() or self.history

    def build_system(self, user_message: str | dict | None, emit) -> str:
        now = datetime.now().astimezone()
        parts = [
            SYSTEM_PERSONA,
            RESPONSE_STYLE,
        ]
        project_context = format_project_context(
            load_project_context_files(Path.cwd(), self.settings.home)
        )
        if project_context:
            parts.append(project_context)
        parts.extend([
            f"当前时间：{now:%Y-%m-%d %H:%M %A} ({now:%Z}, UTC{now:%z})。",
            f"当前主模型：{self.settings.model}。",
        ])
        # Skills (pi ch8): lazy loading injects ONLY the metadata
        # listing — the model pulls the SKILL.md via read_file when
        # one matches.
        listing = self.skills.listing(Path.cwd())
        if listing:
            parts.append(listing)
        return "\n\n".join(parts)

    def _context_messages(self) -> list[AgentMessage]:
        """History messages for the next LLM call (summary included).

        Prefers the session tree (chapter 10): walk the current path and
        dispatch by entry type.  Falls back to the linear chat_log for
        sessions whose JSONL mirror has no messages yet.
        """
        tree = self.build_session_context()
        if tree is not None:
            return tree.messages
        latest = self._latest_summary()
        through = int(latest["through_chat_id"]) if latest else 0
        messages: list[AgentMessage] = []
        if latest:
            messages.append(
                compaction_summary_message(
                    str(latest["summary"]),
                    through_chat_id=through,
                    version=int(latest["version"]),
                )
            )
        messages.extend(self._messages(self._rows_after(through)))
        return messages

    def build_session_context(self) -> SessionContext | None:
        """buildSessionContext: flatten the current tree path (plan §5.8).

        Dispatch by entry type:
        - ``message`` / ``custom_message`` → the typed message joins context;
        - ``compaction`` → positional coverage: entries before
          ``first_kept_entry_id`` on the path are dropped, the summary is
          injected as a compaction-summary custom message.  Legacy entries
          (no first_kept_entry_id) fall back to the chat_id skip;
        - ``branch_summary`` → a branch-summary custom message is injected;
        - header → the session's INITIAL runtime state is the baseline
          (code-review issue 四);
        - ``model_change`` / ``thinking_level_change`` → override the
          model / small_model / thinking reported in the SessionContext;
        - ``label`` / ``session_info`` / ``custom`` → no message.

        Returns ``None`` when the tree holds no messages at all, so the
        caller falls back to the linear chat_log.
        """
        entries = read_session_entries(self.jsonl_path)
        if not any(entry.type in ("message", "custom_message") for entry in entries):
            return None
        path = path_to_leaf(entries, self._last_entry_id)
        collected: list[tuple[Any, AgentMessage]] = []  # (entry, message)
        model: str | None = None
        provider: str | None = None
        small_model: str | None = None
        thinking: str | None = None
        for entry in path:
            if entry.type == "message":
                collected.append((entry, entry.message))  # type: ignore[attr-defined]
            elif entry.type == "custom_message":
                collected.append((entry, entry.message))  # type: ignore[attr-defined]
            elif entry.type == "compaction":
                ce = entry  # type: ignore[assignment]
                collected = self._apply_compaction(collected, ce)
                collected.insert(
                    0,
                    (
                        entry,
                        compaction_summary_message(
                            ce.summary,
                            through_chat_id=ce.through_chat_id,
                            version=ce.version,
                        ),
                    ),
                )
            elif entry.type == "branch_summary":
                bse = entry  # type: ignore[assignment]
                collected.append(
                    (entry, branch_summary_message(bse.summary, from_id=bse.from_id))
                )
            elif entry.type == "session":
                # Header: initial runtime state baseline (old headers
                # simply lack these fields → getattr default "").
                provider = getattr(entry, "provider", "") or provider
                model = getattr(entry, "model", "") or model
                small_model = getattr(entry, "small_model", "") or small_model
                thinking = getattr(entry, "thinking", "") or thinking
            elif entry.type == "model_change":
                mce = entry  # type: ignore[assignment]
                provider = mce.provider or provider
                model = mce.model or model
                small_model = getattr(mce, "small_model", "") or small_model
            elif entry.type == "thinking_level_change":
                thinking = entry.level  # type: ignore[attr-defined]
            # label / session_info / custom: state only, no message
        return SessionContext(
            messages=[message for _, message in collected],
            model=model,
            provider=provider,
            thinking_level=thinking,
            small_model=small_model,
        )

    @staticmethod
    def _apply_compaction(
        collected: list[tuple[Any, AgentMessage]],
        entry: Any,
    ) -> list[tuple[Any, AgentMessage]]:
        """Positional coverage with a chat_id fallback for legacy entries."""
        if entry.first_kept_entry_id:
            ids = [e.id for e, _ in collected]
            if entry.first_kept_entry_id in ids:
                return collected[ids.index(entry.first_kept_entry_id):]
            return []
        # Legacy (pre-batch-B) compaction: skip messages whose chat_id the
        # summary covers; branch summaries always survive.
        return [
            (e, m)
            for e, m in collected
            if (
                isinstance(m, CustomMessage)
                and m.custom_type == BRANCH_SUMMARY
            )
            or (
                (getattr(e, "meta", None) or {}).get("chat_id") is not None
                and int((getattr(e, "meta", None) or {})["chat_id"])
                > entry.through_chat_id
            )
        ]

    def _rows_after(self, chat_id: int) -> list[Any]:
        return self.conn.execute(
            "SELECT id,role,content FROM chat_log "
            "WHERE session_id=? AND id>? ORDER BY id",
            (self.session_id, chat_id),
        ).fetchall()

    @staticmethod
    def _messages(rows) -> list[AgentMessage]:
        """chat_log rows carry only role/content — rebuild as typed."""
        result: list[AgentMessage] = []
        for row in rows:
            content = str(row["content"])
            if str(row["role"]) == "assistant":
                result.append(AssistantMessage(text=content))
            else:
                result.append(UserMessage(content=content))
        return result

    def _estimate_from_messages(self, rows: list[Any]) -> int:
        """Fallback token estimate from raw rows when usage data is unavailable."""
        entry = self._path_last_compaction()
        if entry is not None:
            summary = str(entry.summary)
        else:
            latest = self._latest_summary()
            summary = str(latest["summary"]) if latest else ""
        return estimate_context_tokens(summary, self._messages(rows))

    def _compress_if_due(self, estimated: int, emit) -> bool:
        # context_compression_tokens plays the role of Pi's
        # "contextWindow - reserveTokens" red line.
        budget = max(1, self.settings.context_budget_tokens)
        trigger = min(max(1, self.settings.context_compression_tokens), budget)
        if not compaction_algo.should_compact(estimated, budget, budget - trigger):
            return False
        return self._do_compress(emit, reason="threshold", tokens_before=estimated)

    def compact_if_due(self, context_tokens: int, emit) -> bool:
        """Agent-end auto-compaction hook (pi ch9): compact the current
        tree path when the last MEASURED context size crosses the red line."""
        if context_tokens <= 0:
            return False
        return self._compress_if_due(context_tokens, emit)

    def compact(self, emit) -> bool:
        """Manual compaction entry point (RPC ``compact`` command)."""
        return self._do_compress(emit, reason="manual")

    def compact_and_rebuild(
        self,
        user_message: str | dict | None,
        emit,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[AgentMessage]]:
        """Force a compaction, then rebuild context.

        Called by the loop after a truncation — the idea is:
        overflow → delete truncated message → compact → fresh context → retry.
        """
        self._do_compress(emit, reason="overflow")
        return self.prepare_context(user_message, emit, tool_schemas)

    @staticmethod
    def _fit_source_budget(zone: list[Any], source_budget: int) -> list[Any]:
        """Longest prefix of ``zone`` whose estimated tokens fit the budget.

        The budget only ever breaks at a USER boundary: a turn is taken
        whole or not at all, even if a single turn exceeds the budget on
        its own — otherwise compaction could silently never happen.
        """
        kept: list[Any] = []
        source_tokens = 0
        for entry in zone:
            entry_tokens = 4 + estimate_tokens(
                message_preview(entry.message, limit=1_000_000)
            )
            if (
                kept
                and source_tokens + entry_tokens > source_budget
                and entry.message.role == "user"
            ):
                break
            kept.append(entry)
            source_tokens += entry_tokens
        return kept

    def _do_compress(
        self, emit, reason: str = "threshold", tokens_before: int | None = None
    ) -> bool:
        """Core compression logic, shared by threshold and overflow paths.

        Batch C: operates on TREE entries, not chat_log rows.  The segment
        eligible for summarisation is every message after the LAST
        compaction entry on the current path (earlier ones are already
        covered by previous summaries).  The cut point is the FIRST KEPT
        entry; valid cut points are user AND assistant rows, never a tool
        result (pi ch9: an assistant cut splits a turn — the turn's user
        goes into the main summary, the assistant/toolResult prefix gets
        its own light turnPrefix summary, both merged into ONE
        CompactionEntry whose coverage is recorded POSITIONALLY via
        ``first_kept_entry_id``).
        """
        entries = read_session_entries(self.jsonl_path)
        path = path_to_leaf(entries, self._last_entry_id)
        last_compaction = max(
            (i for i, entry in enumerate(path) if entry.type == "compaction"),
            default=-1,
        )
        # Previous summary: the last compaction ON THIS PATH (code-review
        # issue 一).  The globally-latest SQLite row may belong to a
        # sibling branch and must never leak into this branch's prompt.
        prev_entry = path[last_compaction] if last_compaction >= 0 else None
        previous = str(prev_entry.summary) if prev_entry is not None else ""
        # Version stays session-global and monotonic: the SQLite
        # session_summaries projection has UNIQUE(session_id, version),
        # and two branches may each carry a compaction "v1" on their own
        # path.  Version is a display/projection counter; CORRECTNESS
        # comes from the path-scoped `previous` above.
        latest = self._latest_summary()
        version = int(latest["version"]) if latest else 0
        # The compaction scope is the currently VISIBLE history, not a
        # physical log slice (reliability batch 1): messages the previous
        # compaction chose to KEEP sit BEFORE its entry on the path but
        # are still in context.  If they are neither re-kept nor fed into
        # the new summary's input, applying this round's positional
        # coverage would evict them without any summary ever seeing them
        # — silent history loss.  So the segment starts at the previous
        # compaction's first_kept_entry_id, and the previous summary text
        # itself enters the prompt as `previous`.
        start = last_compaction + 1
        prev_first_kept = (
            str(prev_entry.first_kept_entry_id) if prev_entry is not None else ""
        )
        if prev_first_kept:
            path_ids = [entry.id for entry in path]
            if prev_first_kept in path_ids:
                start = path_ids.index(prev_first_kept)
        segment = [
            entry
            for entry in path[start:]
            if entry.type in ("message", "custom_message")
        ]
        if not segment:
            return False

        # ── token-based cut point (pi ch9: walk backwards from the
        #    newest entry accumulating the keep budget; the cut lands on
        #    the first valid cut point — user or assistant — at/after
        #    the stop position; a tool result is never a cut point) ───
        rows_view = [
            {
                "role": entry.message.role,
                "content": message_preview(entry.message, limit=1_000_000),
            }
            for entry in segment
        ]
        cut_at = compaction_algo.find_cut_point(
            rows_view,
            max(1, self.settings.context_keep_recent_tokens),
            estimate_tokens,
        )
        if not cut_at:
            return False

        # ── source budget over the main zone; split-turn detection ───
        source_budget = max(
            256,
            self.settings.context_budget_tokens
            - self.settings.summary_max_tokens
            - estimate_tokens(SUMMARY_PROMPT)
            - estimate_tokens(previous),
        )
        turn_start = compaction_algo.find_turn_start(rows_view, cut_at)
        prefix_zone: list[Any] = []
        if turn_start != -1:
            # Assistant cut → split turn: the turn's user belongs in the
            # main summary, its assistant/toolResult prefix gets the
            # turnPrefix summary.  The split only stands if the WHOLE
            # main zone (incl. the turn-opening user) fits the source
            # budget — the prefix summary is meaningless without it.
            main_zone = segment[: turn_start + 1]
            main_kept = self._fit_source_budget(main_zone, source_budget)
            if len(main_kept) == len(main_zone):
                eligible = main_kept
                prefix_zone = segment[turn_start + 1 : cut_at]
            else:
                # Conservative fallback: shrink to an earlier user
                # boundary, leave the whole split turn visible.
                turn_start = -1
        if turn_start == -1:
            eligible = self._fit_source_budget(segment[:cut_at], source_budget)
        if not eligible:
            return False
        compressed = eligible + prefix_zone
        first_kept_entry_id = (
            segment[cut_at].id if prefix_zone else segment[len(eligible)].id
        )

        # ── pick prompt: incremental merge vs fresh ─────────
        prompt_template = UPDATE_SUMMARY_PROMPT if previous else SUMMARY_PROMPT
        redact = self.settings.api_key

        def _zone_log(zone: list[Any]) -> str:
            return "\n".join(
                f"{entry.message.role}: "
                f"{redact_text(message_preview(entry.message, limit=1_000_000), (redact,))}"
                for entry in zone
            )

        # Legacy chat_log projection: the highest chat_id this round covers
        # (recorder-written entries carry none yet — keep the previous mark).
        chat_ids = [
            int((entry.meta or {})["chat_id"])
            for entry in compressed
            if isinstance(entry, MessageEntry)
            and (entry.meta or {}).get("chat_id") is not None
        ]
        through_chat_id = (
            max(chat_ids)
            if chat_ids
            else (
                int(prev_entry.through_chat_id)
                if prev_entry is not None
                else (int(latest["through_chat_id"]) if latest else 0)
            )
        )

        emit(
            "context.compression.started",
            {
                "session_id": self.session_id,
                "previous_version": version,
                "source_messages": len(compressed),
                "through_chat_id": through_chat_id,
                "split_turn": bool(prefix_zone),
                "reason": reason,
            },
        )

        try:
            # ── main summary (and, on a split turn, the turnPrefix
            #    summary).  Pi fires both with Promise.all; this project
            #    is deliberately synchronous, so they run sequentially —
            #    BOTH must succeed before anything is persisted. ───────
            usage_records: list[dict[str, int]] = []

            def _summarize(prompt_text: str) -> str:
                response = self.client.complete(
                    model=self.settings.small_model,
                    system="",
                    messages=[{"role": "user", "content": prompt_text}],
                    tools=[],
                    max_tokens=self.settings.summary_max_tokens,
                )
                usage_records.append(
                    {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
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
                text = redact_text(response.text.strip(), (redact,))
                if not text:
                    raise ValueError("empty summary")
                return text

            summary = _summarize(
                prompt_template.format(
                    previous=redact_text(previous, (redact,)) or "（无）",
                    log=_zone_log(eligible),
                )
            )
            if prefix_zone:
                turn_prefix = _summarize(
                    TURN_PREFIX_SUMMARY_PROMPT.format(log=_zone_log(prefix_zone))
                )
                summary = f"{summary}\n\n<turn-prefix>\n{turn_prefix}\n</turn-prefix>"
            next_version = version + 1

            # ── file tracking: merge this round's tool-call file ops
            #    with the lists accumulated in the previous summary ──
            new_file_ops = compaction_algo.extract_file_operations(
                compressed, -1, 1 << 62
            )
            read_files, modified_files = compaction_algo.merge_file_lists(
                compaction_algo.parse_file_tags(previous), new_file_ops
            )
            file_tags = compaction_algo.format_file_operations(
                read_files, modified_files
            )
            if file_tags:
                summary = f"{summary}\n\n{file_tags}"

            if tokens_before is None:
                # No usage measurement was provided (overflow path) —
                # estimate the segment size before the cut.
                tokens_before = sum(
                    4 + estimate_tokens(row["content"]) for row in rows_view
                )

            # ── JSONL first, SQLite projection second (code-review
            #    issue 五): the tree is authoritative.  A failed JSONL
            #    write must NOT leave a ghost summary in SQLite that
            #    _latest_summary() would still serve. ────────────────
            comp_entry = CompactionEntry(
                id=self._make_id(),
                parent_id=self._last_entry_id,
                timestamp=datetime.now().isoformat(),
                version=next_version,
                summary=summary,
                first_kept_entry_id=first_kept_entry_id,
                through_chat_id=through_chat_id,
                source_message_count=len(compressed),
                tokens_before=tokens_before,
                read_files=read_files,
                modified_files=modified_files,
            )
            prev_error = self.recorder.error if self.recorder is not None else None
            self._write_entry(comp_entry)
            new_error = self.recorder.error if self.recorder is not None else None
            if new_error is not None and new_error != prev_error:
                raise IOError(f"session JSONL write failed: {new_error}")

            self.conn.execute(
                "INSERT INTO session_summaries"
                "(session_id,version,summary,through_chat_id,source_message_count) "
                "VALUES(?,?,?,?,?)",
                (
                    self.session_id,
                    next_version,
                    summary,
                    through_chat_id,
                    len(compressed),
                ),
            )
            self.conn.commit()

            # ── record usage for fallback estimation ────────
            self._last_compaction_usage = {
                "input_tokens": sum(u["input_tokens"] for u in usage_records),
                "output_tokens": sum(u["output_tokens"] for u in usage_records),
            }
            emit(
                "context.compression.completed",
                {
                    "session_id": self.session_id,
                    "version": next_version,
                    "source_messages": len(compressed),
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
                    "source_messages": len(compressed),
                    "error": type(exc).__name__,
                    "reason": reason,
                },
            )
            return False

    def prepare_context(
        self,
        user_message: str | dict | None,
        emit,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[AgentMessage]]:
        base_system = self.build_system(user_message, emit)
        tool_tokens = estimate_tokens(
            json.dumps(tool_schemas or [], ensure_ascii=False, default=str)
        )
        history_messages = self._context_messages()
        # user_message=None is the continue() path: re-run on the existing
        # history WITHOUT appending a fresh current user message.
        current: AgentMessage | None = (
            None
            if user_message is None
            else self._runtime_user_message(user_message)
        )
        tail: list[AgentMessage] = [current] if current is not None else []
        candidate_messages = [*history_messages, *tail]
        estimated = estimate_context_tokens(base_system, candidate_messages) + tool_tokens
        emit(
            "context.measured",
            {
                "session_id": self.session_id,
                "estimated_tokens": estimated,
                "budget_tokens": self.settings.context_budget_tokens,
                "compression_tokens": self.settings.context_compression_tokens,
                "history_messages": len(history_messages),
                "summary_version": self._summary_version(),
                "tool_schema_tokens": tool_tokens,
            },
        )

        self._compress_if_due(estimated, emit)
        history_messages = self._context_messages()
        system = base_system
        budget = max(256, self.settings.context_budget_tokens)
        # A leading summary message is the compressed memory of everything
        # already dropped — never evict it when trimming to budget.
        head = (
            history_messages[:1]
            if history_messages and isinstance(history_messages[0], CustomMessage)
            else []
        )
        body = history_messages[len(head):]
        dropped = 0
        while (
            body
            and estimate_context_tokens(system, [*head, *body, *tail]) + tool_tokens
            > budget
        ):
            remove = _trim_block_size(body)
            del body[:remove]
            dropped += remove
        messages: list[AgentMessage] = [*head, *body, *tail]
        final_tokens = estimate_context_tokens(system, messages) + tool_tokens
        if final_tokens > budget:
            # The pinned summary and the current input alone overflow the
            # budget — nothing left to legally trim.  Fail loudly instead
            # of looping forever or sending an illegal oversized request.
            raise ContextOverflowError(
                f"context overflow: summary + current input alone need "
                f"~{final_tokens} tokens, budget is {budget}; "
                f"start a new session or raise context_budget_tokens"
            )
        emit(
            "context.built",
            {
                "session_id": self.session_id,
                "estimated_tokens": final_tokens,
                "budget_tokens": budget,
                "history_messages": len(body),
                "dropped_messages": dropped,
                "summary_version": self._summary_version(),
                "within_budget": final_tokens <= budget,
                "tool_schema_tokens": tool_tokens,
            },
        )
        return system, messages

    def _summary_version(self) -> int:
        entry = self._path_last_compaction()
        if entry is not None:
            return int(entry.version)
        latest = self._latest_summary()
        return int(latest["version"]) if latest else 0

    def window(self) -> list[dict]:
        return self.history[-self.settings.history_turns * 2 :]

    @staticmethod
    def _runtime_user_message(user_message: str | dict) -> UserMessage:
        """Normalize the product input boundary into a typed UserMessage.

        A plain string passes through.  A dict payload must look like
        ``{"role": "user", "content": str | [blocks]}`` where every block
        is ``{"type": "text", "text": ...}`` or
        ``{"type": "image_url", "image_url": {"url": ...}}``.  Validation
        happens HERE, at the boundary, and construction is delegated to
        the shared ``agent.messages.user_message`` normalizer — no
        parallel message system, and translators keep receiving only
        canonical typed blocks.

        Unsupported or malformed blocks raise a clear error that names
        the block type but never embeds the payload (an image block can
        carry megabytes of base64).
        """
        if isinstance(user_message, str):
            return UserMessage(content=user_message)
        if not isinstance(user_message, dict):
            raise TypeError(
                "user_message must be a string or a "
                f"{{'role': 'user', 'content': ...}} dict, got "
                f"{type(user_message).__name__}"
            )
        content = user_message.get("content", "")
        if isinstance(content, str):
            return UserMessage(content=content)
        if not isinstance(content, (list, tuple)):
            raise ValueError(
                "user_message 'content' must be a string or a list of "
                f"content blocks, got {type(content).__name__}"
            )
        for item in content:
            if not isinstance(item, dict):
                raise ValueError(
                    "content blocks must be dicts, got "
                    f"{type(item).__name__}"
                )
            block_type = item.get("type")
            if block_type == "text":
                if not isinstance(item.get("text"), str):
                    raise ValueError("text block requires a string 'text'")
            elif block_type == "image_url":
                image_url = item.get("image_url")
                url = (
                    image_url.get("url")
                    if isinstance(image_url, dict)
                    else image_url
                )
                if not isinstance(url, str) or not url:
                    raise ValueError(
                        "image_url block requires a non-empty 'image_url.url' string"
                    )
            else:
                raise ValueError(
                    f"unsupported user content block type: {block_type!r}"
                )
        return _build_user_message(content)

    @staticmethod
    def _persistable_user_content(user_message: str | dict) -> tuple[str, dict[str, Any]]:
        """Return a compact, DB-safe representation without image payloads."""
        if isinstance(user_message, str):
            return user_message, {"multimodal": False, "image_count": 0}
        content = user_message.get("content", "")
        if isinstance(content, str):
            return content, {"multimodal": False, "image_count": 0}
        parts: list[str] = []
        image_count = 0
        for item in content if isinstance(content, list) else []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "image_url":
                image_count += 1
                parts.append("[image]")
        return "\n".join(part for part in parts if part), {
            "multimodal": image_count > 0,
            "image_count": image_count,
        }

    def persistable_user_message(self, user_message: str | dict) -> UserMessage:
        """The typed user message the RECORDER persists: image payloads are
        replaced by ``[image]`` placeholders so session files stay small."""
        content, _ = self._persistable_user_content(user_message)
        return UserMessage(content=content)

    def add_exchange(
        self,
        user_message: str | dict | None,
        result,
        source: str,
        *,
        session_id: str | None = None,
        recorder=None,
    ) -> dict[str, Any]:
        """ChatProjector: project one completed exchange into SQLite.

        The JSONL tree is NO LONGER written here — the SessionRecorder
        persisted every message (user / assistant-with-tool-calls /
        tool_result / assistant) live from kernel events during the run.
        In particular the fused ``[tools used: ...]`` string is gone:
        tool calls are their own tree entries now.

        ``session_id``/``recorder`` pin the owning session explicitly (a
        run passes the id and recorder it started with); the defaults keep
        the historical behaviour of using the session's current state.
        When the pinned session is no longer the live one, the in-memory
        ``history`` extend is skipped — the exchange must never leak into
        a DIFFERENT session's history, and the pinned session rebuilds its
        history from SQLite/JSONL on resume.
        """
        sid = session_id or self.session_id
        still_live = sid == self.session_id
        record = result.reply
        if user_message is None:
            # continue() exchange: no fresh user turn — the reply answers
            # the INTERRUPTED question already on the tree.  The chat
            # projection keeps the pair shape with an empty user half and
            # marks the assistant row; schema unchanged.
            user_content, media_meta = "", {"multimodal": False, "image_count": 0}
        else:
            user_content, media_meta = self._persistable_user_content(user_message)

        if still_live:
            if user_message is None:
                self.history.extend([
                    {"role": "assistant", "content": record},
                ])
            else:
                self.history.extend([
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": record},
                ])

        meta = {
            "iterations": result.iterations,
            "tools": [item["tool"] for item in result.tool_calls],
            "model": self.settings.model,
            **media_meta,
        }
        if user_message is None:
            meta["continued"] = True

        # SQLite is the canonical chat store.  Both messages and the session
        # metadata are committed as one transaction.
        self._log_chat(
            user_content,
            record,
            session_id=sid,
            source=source,
            meta=meta,
            commit=False,
        )
        self.conn.execute(
            "UPDATE sessions SET title=CASE WHEN title='' THEN ? ELSE title END,"
            "updated_at=strftime('%Y-%m-%d %H:%M:%f','now') WHERE id=?",
            (user_content.strip()[:80], sid),
        )
        self.conn.commit()

        # JSONL status comes from the recorder, honestly: a deferred fresh
        # session reports "deferred", a failed write reports "error" — the
        # mirror is never reported ok when it is not.  A run passes its
        # PINNED recorder; only the default path reads the live one.
        status: dict[str, Any] = {"sqlite": "ok", "multimodal": media_meta["multimodal"]}
        recorder = recorder if recorder is not None else self.recorder
        if recorder is None or recorder.error is not None:
            status["jsonl"] = "error"
            if recorder is not None and recorder.error:
                status["error"] = recorder.error
        elif recorder.deferred:
            status["jsonl"] = "deferred"
        else:
            status["jsonl"] = "ok"
        return status

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

    @property
    def _last_entry_id(self) -> str | None:
        """The tree leaf — owned by the recorder (it owns the parent chain)."""
        return self.recorder.last_entry_id if self.recorder is not None else None

    @_last_entry_id.setter
    def _last_entry_id(self, value: str | None) -> None:
        if self.recorder is not None:
            self.recorder.last_entry_id = value

    def _ensure_session_jsonl(self) -> None:
        """(Re)bind the recorder for the current session id.

        Legacy sessions (chat_log rows but no JSONL yet) are backfilled
        immediately — their file exists after this call, so the recorder
        appends line by line.  Brand-new sessions get NO header up front:
        the recorder defers the first write until the first assistant
        message completes, so a failed run never leaves half a session.
        """
        path = self.jsonl_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._backfill_jsonl_from_chat_log()
        entries = read_session_entries(path)
        last = entries[-1].id if entries else self.session_id
        self.recorder = SessionRecorder(
            path,
            self.session_id,
            cwd=str(Path.cwd()),
            start_parent_id=last,
            initial_state={
                "provider": self.settings.provider or "",
                "model": self.settings.model or "",
                "small_model": self.settings.small_model or "",
                "thinking": self.settings.thinking or "",
            },
        )

    def _backfill_jsonl_from_chat_log(self) -> None:
        """Mirror chat_log into the tree for sessions created before the
        JSONL storage existed (chapter 10 migration path).

        Runs once per session: only when the file is missing/empty while
        chat_log has rows.  Backfilled entries carry their chat_id so the
        legacy compaction fallback works on them.  Backfill writes the
        header immediately (immediate mode) — the conversation already
        happened, there is nothing to defer.
        """
        path = self.jsonl_path
        if path.exists() and path.stat().st_size > 0:
            return
        rows = self.conn.execute(
            "SELECT id,role,content,source FROM chat_log WHERE session_id=? ORDER BY id",
            (self.session_id,),
        ).fetchall()
        if not rows:
            return
        write_session_header(
            path, SessionHeader.create(self.session_id, str(Path.cwd()))
        )
        parent_id = self.session_id
        for row in rows:
            role = str(row["role"])
            content = str(row["content"])
            message: AgentMessage = (
                UserMessage(content=content)
                if role == "user"
                else AssistantMessage(text=content)
            )
            entry = MessageEntry.create(
                self._make_id(),
                parent_id,
                message,
                source=str(row["source"] or "backfill"),
                meta={"chat_id": int(row["id"]), "backfilled": True},
            )
            append_session_entry(path, entry, self._jsonl_lock)
            parent_id = entry.id
        # Existing summaries become compaction entries so the tree view
        # skips the messages they cover (legacy: chat_id fallback).
        summaries = self.conn.execute(
            "SELECT version,summary,through_chat_id,source_message_count "
            "FROM session_summaries WHERE session_id=? ORDER BY version",
            (self.session_id,),
        ).fetchall()
        for row in summaries:
            entry = CompactionEntry(
                id=self._make_id(),
                parent_id=parent_id,
                timestamp=datetime.now().isoformat(),
                version=int(row["version"]),
                summary=str(row["summary"]),
                through_chat_id=int(row["through_chat_id"]),
                source_message_count=int(row["source_message_count"]),
            )
            append_session_entry(path, entry, self._jsonl_lock)
            parent_id = entry.id

    # ── branch operations (chapter 10) ───────────────────────────

    def _resolve_entry_ref(self, entry_ref: str) -> str | None:
        entries = read_session_entries(self.jsonl_path)
        by_id = build_by_id(entries)
        ref = entry_ref.strip()
        if ref in by_id:
            return ref
        matches = [entry_id for entry_id in by_id if entry_id.startswith(ref)]
        return matches[0] if len(matches) == 1 else None

    def branch(self, entry_ref: str, emit) -> str | None:
        """Move the leaf pointer to an earlier entry — *the* branch op.

        Nothing is deleted: the abandoned nodes stay in the tree (and in
        the JSONL file), they simply leave the current path.  Returns the
        resolved entry id, or ``None`` when the ref doesn't resolve.
        """
        target = self._resolve_entry_ref(entry_ref)
        if target is None or target == self._last_entry_id:
            return None
        old_leaf = self._last_entry_id
        self._last_entry_id = target
        # Same tree-derived rebuild as _load_history (阶段 4 批 1) — one
        # helper owns "display history = current path's messages".
        self.history = self._history_from_tree() or []
        emit(
            "session.branched",
            {
                "session_id": self.session_id,
                "from_entry": old_leaf,
                "to_entry": target,
                "with_summary": False,
            },
        )
        return target

    def branch_with_summary(self, entry_ref: str, emit) -> str | None:
        """Branch, and leave a summary of the abandoned exploration behind.

        The summary is generated with the light 5-section prompt (Pi caps
        branch summaries at 2048 tokens — auxiliary context must not crowd
        the mainline), then appended as a BranchSummaryEntry whose parent
        is the fork node, so it shows up at the top of the new branch.
        """
        entries = read_session_entries(self.jsonl_path)
        target = self._resolve_entry_ref(entry_ref)
        if target is None or target == self._last_entry_id:
            return None
        old_leaf = self._last_entry_id
        abandoned = [
            entry
            for entry in collect_abandoned_branch(entries, old_leaf, target)
            if entry.type == "message"
        ]
        if not abandoned:
            return self.branch(entry_ref, emit)

        emit(
            "session.branch_summary.started",
            {
                "session_id": self.session_id,
                "from_entry": old_leaf,
                "to_entry": target,
                "abandoned_messages": len(abandoned),
            },
        )
        log = "\n".join(
            f"{entry.message.role}: "
            f"{redact_text(message_preview(entry.message, limit=1_000_000), (self.settings.api_key,))}"
            for entry in abandoned
        )
        try:
            response = self.client.complete(
                model=self.settings.small_model,
                system="",
                messages=[{"role": "user", "content": BRANCH_SUMMARY_PROMPT.format(log=log)}],
                tools=[],
                max_tokens=compaction_algo.BRANCH_SUMMARY_MAX_TOKENS,
            )
            summary = redact_text(response.text.strip(), (self.settings.api_key,))
            if not summary:
                raise ValueError("empty branch summary")
        except Exception as exc:
            emit(
                "session.branch_summary.failed",
                {
                    "session_id": self.session_id,
                    "from_entry": old_leaf,
                    "to_entry": target,
                    "error": type(exc).__name__,
                },
            )
            return None

        read_files, modified_files = compaction_algo.extract_file_operations(
            abandoned, -1, 1 << 62
        )
        file_tags = compaction_algo.format_file_operations(read_files, modified_files)
        if file_tags:
            summary = f"{summary}\n\n{file_tags}"

        # Move the leaf first, then hang the summary on the fork node —
        # the next user message parents onto the summary entry.
        self._last_entry_id = target
        summary_entry = BranchSummaryEntry(
            id=self._make_id(),
            parent_id=target,
            timestamp=datetime.now().isoformat(),
            summary=summary,
            from_id=old_leaf,
            read_files=read_files,
            modified_files=modified_files,
        )
        prev_error = self.recorder.error if self.recorder is not None else None
        self._write_entry(summary_entry)
        new_error = self.recorder.error if self.recorder is not None else None
        if new_error is not None and new_error != prev_error:
            self._last_entry_id = old_leaf
            emit(
                "session.branch_summary.failed",
                {
                    "session_id": self.session_id,
                    "from_entry": old_leaf,
                    "to_entry": target,
                    "error": new_error,
                },
            )
            return None
        self.history = self._history_from_tree() or []
        emit(
            "session.branch_summary.completed",
            {
                "session_id": self.session_id,
                "from_entry": old_leaf,
                "to_entry": target,
                "summary_tokens": estimate_tokens(summary),
            },
        )
        emit(
            "session.branched",
            {
                "session_id": self.session_id,
                "from_entry": old_leaf,
                "to_entry": target,
                "with_summary": True,
            },
        )
        return self._last_entry_id

    # ── state-change entries (plan §5.9) ─────────────────────

    def record_model_change(
        self, provider: str, model: str, small_model: str = ""
    ) -> None:
        """Append a model_change entry at the current leaf.

        The entry makes the switch part of the tree's record: opening or
        resuming the session later restores the model in effect on the
        current path (阶段 4 批 2 — Pi createAgentSession parity;
        in-session branching does NOT re-apply it).  ``small_model`` is
        recorded too — it drives compaction and branch summaries
        (code-review issue 三).
        """
        self._write_entry(
            ModelChangeEntry(
                id=self._make_id(),
                parent_id=self._last_entry_id,
                timestamp=datetime.now().isoformat(),
                provider=provider,
                model=model,
                small_model=small_model,
            )
        )

    def record_thinking_change(self, level: str) -> None:
        """Append a thinking_level_change entry at the current leaf."""
        self._write_entry(
            ThinkingLevelChange(
                id=self._make_id(),
                parent_id=self._last_entry_id,
                timestamp=datetime.now().isoformat(),
                level=level,
            )
        )

    def record_label(self, entry_ref: str, label: str) -> str | None:
        """Pin a navigation label to an earlier entry (plan §9.4).

        Labels are display metadata — they never enter model context.
        Returns the resolved target entry id, or ``None`` when the ref
        doesn't resolve.
        """
        target = self._resolve_entry_ref(entry_ref)
        if target is None or not label.strip():
            return None
        self._write_entry(
            LabelEntry(
                id=self._make_id(),
                parent_id=self._last_entry_id,
                timestamp=datetime.now().isoformat(),
                target_id=target,
                label=label.strip(),
            )
        )
        return target

    def _make_id(self) -> str:
        return f"{self.session_id[:8]}-{uuid4().hex[:12]}"

    def _write_entry(self, entry) -> None:
        """Persist a non-message tree entry (compaction / branch / state).

        Message entries during a run are written by the recorder's event
        listener, not here.  Write failures are reported via
        ``recorder.error`` + the ``session.jsonl_write_failed`` event —
        never raised into the caller.
        """
        if self.recorder is not None:
            self.recorder.append_entry(entry)

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
                text = message_preview(me.message, limit=1_000_000)
                if me.message.role == "user":
                    pending_user = text
                elif me.message.role == "assistant" and pending_user:
                    self._log_chat(
                        pending_user,
                        text,
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
