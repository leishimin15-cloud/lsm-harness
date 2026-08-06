"""Working-memory assembly for each turn."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from lsm_harness.config import Settings
from lsm_harness.memory import Memory


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


def load_soul(settings: Settings) -> str:
    path = settings.home / "SOUL.md"
    if not path.exists():
        path.write_text(DEFAULT_SOUL, encoding="utf-8")
    return path.read_text(encoding="utf-8")


class Session:
    def __init__(self, settings: Settings, memory: Memory, session_id: str | None = None):
        self.settings = settings
        self.memory = memory
        self.session_id = session_id or str(uuid4())
        self.history: list[dict] = []

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

    def window(self) -> list[dict]:
        return self.history[-self.settings.history_turns * 2 :]

    def add_exchange(self, user_message: str, result, source: str) -> None:
        record = result.reply
        if result.tool_calls:
            summary = "; ".join(
                f"{item['tool']}({item['args']}) -> {item['output']}"
                for item in result.tool_calls
            )
            record += f"\n[tools used: {summary}]"
        self.history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": record},
            ]
        )
        self.memory.log_chat(
            user_message,
            record,
            session_id=self.session_id,
            source=source,
            meta={
                "iterations": result.iterations,
                "tools": [item["tool"] for item in result.tool_calls],
                "model": self.settings.model,
            },
        )

    def start_new(self) -> str:
        self.session_id = str(uuid4())
        self.history = []
        return self.session_id
