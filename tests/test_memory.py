from pathlib import Path

from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.memory import consolidation, retrieval
from lsm_harness.memory.skills import SkillLoader
from lsm_harness.memory.stores import EpisodeStore, FactStore
from lsm_harness.types import ModelResponse

from helpers import QueueClient


def stores(tmp_path):
    conn = connect(tmp_path)
    return conn, FactStore(conn), EpisodeStore(conn)


def test_chinese_trigram_fact_retrieval(tmp_path):
    _, facts, _ = stores(tmp_path)
    facts.add("个人项目", "我正在开发 LSM Harness")
    assert "LSM Harness" in facts.search("个人项目 LSM")[0]


def test_two_character_name_uses_like_fallback(tmp_path):
    _, facts, _ = stores(tmp_path)
    facts.add("小明", "小明喜欢上午开会")
    assert "小明" in facts.search("小明")[0]


def test_non_contiguous_chinese_project_query_uses_bigram_ranking(tmp_path):
    _, facts, _ = stores(tmp_path)
    facts.add(
        "用户正在开发的 agent 项目",
        "用户正在开发的项目名为 LSM Harness，即当前运行的本地个人 Agent 系统。",
    )
    assert "LSM Harness" in facts.search("当前开发项目")[0]
    assert facts.search_with_ids("开发项目", 8)[0]["subject"] == "用户正在开发的 agent 项目"


def test_episode_chinese_search(tmp_path):
    _, _, episodes = stores(tmp_path)
    episodes.add("讨论了个人 Harness 的记忆系统", "2026-08-06")
    assert "记忆系统" in episodes.search("Harness 记忆")[0]


def test_retrieval_gate_skip_and_retrieve():
    skip = QueueClient(ModelResponse(text='{"retrieve":false,"query":"","reason":"数学"}'))
    assert retrieval.should_retrieve(skip, "small", "2+2") == (False, "", "数学")
    yes = QueueClient(
        ModelResponse(text='{"retrieve":true,"query":"LSM 项目","reason":"个人项目"}')
    )
    assert retrieval.should_retrieve(yes, "small", "我的项目是什么") == (
        True,
        "LSM 项目",
        "个人项目",
    )


def test_retrieval_gate_fails_open():
    client = QueueClient(RuntimeError("network"))
    decision = retrieval.should_retrieve(client, "small", "我的项目")
    assert decision[0] is True
    assert decision[1] == "我的项目"


def _chat(conn, pairs=1):
    for index in range(pairs):
        conn.execute("INSERT INTO chat_log(role,content) VALUES('user',?)", (f"u{index}",))
        conn.execute("INSERT INTO chat_log(role,content) VALUES('assistant',?)", (f"a{index}",))
    conn.commit()


def test_consolidation_writes_fact_episode_and_marks_rows(tmp_path):
    conn, facts, episodes = stores(tmp_path)
    _chat(conn, 2)
    client = QueueClient(
        ModelResponse(
            text='{"facts":[{"subject":"项目","content":"用户开发 LSM Harness"}],'
            '"episode":"讨论了 Harness"}'
        )
    )
    assert consolidation.consolidate_if_due(conn, client, "small", 2, facts, episodes) == (
        1,
        True,
    )
    assert conn.execute("SELECT COUNT(*) FROM chat_log WHERE consolidated=0").fetchone()[0] == 0


def test_failed_consolidation_keeps_raw_chat(tmp_path):
    conn, facts, episodes = stores(tmp_path)
    _chat(conn, 1)
    client = QueueClient(ModelResponse(text="not-json"))
    assert consolidation.consolidate_if_due(conn, client, "small", 1, facts, episodes) == (
        0,
        False,
    )
    assert conn.execute("SELECT COUNT(*) FROM chat_log WHERE consolidated=0").fetchone()[0] == 2


def test_skill_creation_is_detected_and_matches_chinese(tmp_path):
    root = tmp_path / "skills"
    loader = SkillLoader([root])
    path = root / "weekly-review" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\nname: weekly-review\ndescription: 每周复盘项目进度\n---\n\n先总结，再计划。\n",
        encoding="utf-8",
    )
    matches = loader.match("请帮我进行每周复盘")
    assert matches and matches[0].name == "weekly-review"
