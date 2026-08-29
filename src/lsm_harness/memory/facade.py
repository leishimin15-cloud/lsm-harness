"""Facade that coordinates all durable memory paths."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable

from lsm_harness.config import Settings
from lsm_harness.memory import consolidation, retrieval
from lsm_harness.memory.skills import SkillLoader
from lsm_harness.memory.stores import EpisodeStore, FactStore
from lsm_harness.types import ModelClient


Emit = Callable[[str, dict], None]


class Memory:
    def __init__(
        self,
        conn: sqlite3.Connection,
        settings: Settings,
        client: ModelClient,
        bundled_skills: Path | None = None,
    ):
        self.conn = conn
        self.settings = settings
        self.client = client
        self.facts = FactStore(conn)
        self.episodes = EpisodeStore(conn)
        directories = [settings.home / "skills"]
        if bundled_skills:
            directories.insert(0, bundled_skills)
        self.skills = SkillLoader(directories)

    def gated_retrieve(self, message: str, emit: Emit) -> str:
        emit("memory.gate.started", {"message": message[:200]})
        retrieve, query, reason = retrieval.should_retrieve(
            self.client, self.settings.small_model, message, emit=emit
        )
        emit(
            "memory.gate.decided",
            {"decision": "retrieve" if retrieve else "skip", "reason": reason, "query": query},
        )
        if not retrieve:
            return ""
        facts = self.facts.search(query, self.settings.retrieval_top_k)
        episodes = self.episodes.search(query, 3)
        emit(
            "memory.retrieved",
            {"query": query, "facts": len(facts), "episodes": len(episodes)},
        )
        return "\n".join([*facts, *episodes])

    def matching_skills(self, message: str) -> str:
        """Legacy matched mode: inline the bodies of keyword-matched skills."""
        matches = self.skills.match(message)
        return "\n\n".join(f"### {skill.name}\n{skill.body}" for skill in matches)

    def skills_listing(self) -> str:
        """Pi-mode lazy loading: metadata-only <available_skills> listing.

        Locations are relative to the process cwd — the same root the
        read_file tool resolves against.
        """
        return self.skills.listing(Path.cwd())

    def log_chat(
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

    def consolidate(self, emit: Emit) -> tuple[int, bool]:
        facts, episode = consolidation.consolidate_if_due(
            self.conn,
            self.client,
            self.settings.small_model,
            self.settings.consolidate_every,
            self.facts,
            self.episodes,
            emit=emit,
        )
        if facts or episode:
            emit("memory.consolidated", {"facts": facts, "episode": episode})
        return facts, episode

    def export_markdown(self) -> None:
        facts = self.facts.list()
        episodes = self.episodes.list()
        lines = [
            "# LSM Memory",
            "",
            "_该文件是 `.lsm/state.db` 的可读镜像，每轮对话后自动生成。_",
            "",
            f"## Facts ({len(facts)})",
            "",
        ]
        lines.extend(f"- **{item['subject']}** — {item['content']}" for item in reversed(facts))
        if not facts:
            lines.append("_暂无_" )
        lines.extend(["", f"## Episodes ({len(episodes)})", ""])
        lines.extend(
            f"- **{item['happened_at']}** — {item['summary']}" for item in reversed(episodes)
        )
        if not episodes:
            lines.append("_暂无_")
        (self.settings.home / "MEMORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
