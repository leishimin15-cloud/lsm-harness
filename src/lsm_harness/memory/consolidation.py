"""Batch raw chat exchanges into semantic facts and one episodic summary."""

from __future__ import annotations

import json
from datetime import date

from lsm_harness.memory.stores import EpisodeStore, FactStore
from lsm_harness.types import ModelClient


PROMPT = """你负责把个人 Agent 的近期对话提炼为长期记忆。
提取一个月后仍值得记住的稳定事实，并用一句话总结这段对话发生了什么。
忽略寒暄、临时信息和工具技术细节。只返回 JSON：
{{"facts": [{{"subject": "主题", "content": "事实"}}], "episode": "经历摘要"}}

对话：
{log}"""


def consolidate_if_due(
    conn,
    client: ModelClient,
    model: str,
    every_n: int,
    facts: FactStore,
    episodes: EpisodeStore,
    emit=None,
) -> tuple[int, bool]:
    rows = conn.execute(
        "SELECT id, role, content FROM chat_log WHERE consolidated=0 ORDER BY id"
    ).fetchall()
    if len(rows) < max(1, every_n) * 2:
        return 0, False
    log = "\n".join(f"{row['role']}: {row['content']}" for row in rows)
    try:
        response = client.complete(
            model=model,
            system="",
            messages=[{"role": "user", "content": PROMPT.format(log=log)}],
            tools=[],
            max_tokens=800,
        )
        if emit:
            emit(
                "llm.completed",
                {
                    "role": "consolidation",
                    "model": model,
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                },
            )
        distilled = json.loads(
            response.text[response.text.index("{") : response.text.rindex("}") + 1]
        )
        fact_items = distilled.get("facts", [])
        episode = str(distilled.get("episode") or "").strip()
        if not isinstance(fact_items, list):
            raise ValueError("facts must be a list")
    except Exception:
        if emit:
            emit("llm.failed", {"role": "consolidation", "model": model})
        return 0, False

    saved = 0
    for item in fact_items:
        if isinstance(item, dict) and item.get("subject") and item.get("content"):
            facts.add(str(item["subject"]), str(item["content"]), source="consolidation")
            saved += 1
    episode_saved = bool(episode)
    if episode_saved:
        episodes.add(episode, happened_at=date.today().isoformat())

    placeholders = ",".join("?" for _ in rows)
    conn.execute(
        f"UPDATE chat_log SET consolidated=1 WHERE id IN ({placeholders})",
        [row["id"] for row in rows],
    )
    conn.commit()
    return saved, episode_saved
