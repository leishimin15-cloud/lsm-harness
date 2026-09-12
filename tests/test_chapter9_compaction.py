"""Chapter 9: compaction — should_compact, token-based cut points, file
tracking across compactions, and summary-message injection."""

from __future__ import annotations

from lsm_harness.agent.messages import (
    AgentToolResultMessage,
    AssistantMessage,
    CustomMessage,
    ToolCallContent,
    UserMessage,
    message_preview,
)
from lsm_harness.coding_agent.compaction import (
    extract_file_operations,
    find_cut_point,
    find_turn_start,
    format_file_operations,
    merge_file_lists,
    parse_file_tags,
    should_compact,
)
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.ops.session_store import MessageEntry, read_session_entries
from lsm_harness.coding_agent.session import Session, estimate_tokens
from lsm_harness.agent.types import TurnResult
from lsm_harness.ai.types import ModelResponse

from helpers import QueueClient


def build_session(tmp_path, client, session_id="session-9", **overrides):
    settings = Settings(
        api_key=overrides.pop("api_key", "test-key"),
        home=tmp_path,
        **overrides,
    )
    conn = connect(tmp_path)
    return conn, Session(settings, conn=conn, client=client, session_id=session_id)


# ── should_compact ───────────────────────────────────────────────


def test_should_compact_red_line_is_budget_minus_reserve():
    assert not should_compact(183, 200, 16)   # 183 <= 200 - 16... boundary
    assert should_compact(185, 200, 16)       # past the red line
    assert not should_compact(100, 200, 16)   # comfortably below
    # degenerate reserve still compacts when full
    assert should_compact(201, 200, 0)


# ── find_cut_point ───────────────────────────────────────────────


class _Row(dict):
    pass


def _rows(pairs: int, width: int = 40) -> list[dict]:
    rows = []
    for i in range(pairs):
        rows.append({"id": i * 2 + 1, "role": "user", "content": f"问{i}:" + "题" * width})
        rows.append({"id": i * 2 + 2, "role": "assistant", "content": f"答{i}:" + "案" * width})
    return rows


def test_find_cut_point_keeps_recent_token_budget():
    rows = _rows(5, width=80)  # each row ≈ 165 tokens
    cut = find_cut_point(rows, keep_recent_tokens=400, estimate=estimate_tokens)
    # 400 tokens ≈ 2-3 rows; the cut is a VALID cut point (user/assistant)
    assert rows[cut]["role"] in ("user", "assistant")
    assert cut > 0  # something actually gets compressed
    # The kept region covers roughly the budget: it may slightly overshoot
    # (cut moves forward to a valid point) but never loses a whole row.
    kept_tokens = sum(4 + estimate_tokens(r["content"]) for r in rows[cut:])
    assert kept_tokens >= 400 - 2 * 170


def test_find_cut_point_may_split_a_turn_at_an_assistant_row():
    """Pi ch9: token precision beats turn integrity — when the budget stops
    inside a turn, the cut lands on the assistant row (turnPrefix case)."""
    rows = _rows(5, width=80)
    # Keep exactly ~1 row of budget: the stop falls on the LAST row
    # (assistant) — a valid cut point itself.
    one_row = 4 + estimate_tokens(rows[-1]["content"])
    cut = find_cut_point(rows, keep_recent_tokens=one_row, estimate=estimate_tokens)
    assert rows[cut]["role"] == "assistant"
    assert cut == len(rows) - 1  # only the last assistant row survives


def test_find_cut_point_tool_result_is_never_a_cut_point():
    """A tool result must stay with the assistant call that produced it —
    it is never the first kept row, no matter where the budget stops."""
    rows = [
        {"role": "user", "content": "问" * 40},
        {"role": "assistant", "content": "调" * 40},
        {"role": "tool", "content": "果" * 40},
        {"role": "assistant", "content": "答" * 40},
    ]
    for budget in (1, 50, 100, 200, 400):
        cut = find_cut_point(rows, keep_recent_tokens=budget, estimate=estimate_tokens)
        assert rows[cut]["role"] != "tool"


def test_find_turn_start_locates_the_turn_opening_user():
    rows = _rows(3)
    # user cut → no split
    assert find_turn_start(rows, 2) == -1
    # assistant cut → the user row that opened its turn
    assert find_turn_start(rows, 3) == 2
    # cut at the very first row → no split possible
    assert find_turn_start(rows, 0) == -1
    # no preceding user in range (turn start already compacted) → -1
    tail = [r for r in _rows(2) if r["role"] == "assistant"]
    assert find_turn_start(tail, 1) == -1


def test_find_cut_point_returns_zero_when_everything_fits():
    rows = _rows(2)
    assert find_cut_point(rows, keep_recent_tokens=999999, estimate=estimate_tokens) == 0


# ── file tracking ────────────────────────────────────────────────


def _entry(entry_id, chat_id, tool_calls):
    return MessageEntry.create(
        entry_id,
        None,
        AssistantMessage(
            text="x",
            tool_calls=tuple(
                ToolCallContent(
                    id="", name=str(call["tool"]), arguments=dict(call["args"])
                )
                for call in tool_calls
            ),
        ),
        meta={"chat_id": chat_id},
    )


def test_extract_file_operations_by_chat_id_range():
    entries = [
        _entry("e1", 1, [{"tool": "read_file", "args": {"path": "a.py"}, "output": ""}]),
        _entry("e2", 2, [{"tool": "write_file", "args": {"path": "b.py"}, "output": ""}]),
        _entry("e3", 5, [{"tool": "read_file", "args": {"path": "c.py"}, "output": ""}]),
    ]
    read, modified = extract_file_operations(entries, 0, 2)
    assert read == ["a.py"] and modified == ["b.py"]
    # entries outside the range are excluded
    read2, modified2 = extract_file_operations(entries, 2, 10)
    assert read2 == ["c.py"] and modified2 == []


def test_file_tags_round_trip_and_merge():
    text = format_file_operations(["a.py", "b.py"], ["c.py"])
    assert "<read-files>" in text and "<modified-files>" in text
    parsed = parse_file_tags(f"## Goal\n- x\n\n{text}")
    assert parsed == (["a.py", "b.py"], ["c.py"])
    merged = merge_file_lists(parsed, (["b.py", "d.py"], []))
    assert merged == (["a.py", "b.py", "d.py"], ["c.py"])


def test_compaction_appends_and_accumulates_file_tags(tmp_path):
    """File ops from tool calls land in the summary and survive compactions."""
    client = QueueClient(
        ModelResponse(text="## Goal\n- 第一轮"),
        ModelResponse(text="## Goal\n- 第二轮"),
    )
    _, session = build_session(
        tmp_path, client,
        context_budget_tokens=100000,
        context_reserve_tokens=99999,
        context_keep_recent_tokens=50,
    )
    big = "内容" * 400

    def exchange(user, tool_calls=None):
        # Recorder persists per-message tree entries (typed tool calls
        # on the assistant message), then the projector writes SQLite.
        recorder = session.recorder
        recorder.record(UserMessage(content=user), source="test")
        recorder.record(
            AssistantMessage(
                text=big,
                tool_calls=tuple(
                    ToolCallContent(
                        id="", name=str(call["tool"]), arguments=dict(call["args"])
                    )
                    for call in (tool_calls or [])
                ),
            ),
            source="test",
        )
        session.add_exchange(
            user,
            TurnResult(reply=big, iterations=1, tool_calls=tool_calls or []),
            "test",
        )

    # keep_recent=50 keeps only the LAST turn whole — so exchange 3 stays
    # out of the first compaction, which covers exchanges 1-2.
    exchange("读一下 a.py", [{"tool": "read_file", "args": {"path": "a.py"}, "output": "..."}])
    exchange("改一下 b.py", [{"tool": "write_file", "args": {"path": "b.py"}, "output": "..."}])
    exchange("随便聊聊")
    session.prepare_context("继续", lambda *_: None)
    summary = session.summary()
    assert "<read-files>" in summary and "a.py" in summary
    assert "<modified-files>" in summary and "b.py" in summary

    # second round: exchange 4's file op merges with the accumulated lists
    # (exchange 5 is the turn kept whole this time)
    exchange("再读 c.py", [{"tool": "read_file", "args": {"path": "c.py"}, "output": "..."}])
    exchange("再聊聊")
    session.prepare_context("再继续", lambda *_: None)
    summary_v2 = session.summary()
    assert session.summary_info()["version"] == 2
    # a.py came from the *previous* summary's tags (its messages are long
    # compacted away), c.py is new in this round's compression range
    assert "a.py" in summary_v2 and "c.py" in summary_v2
    assert "b.py" in summary_v2

    # the JSONL compaction entry carries the structured lists too
    entries = read_session_entries(session.jsonl_path)
    compactions = [e for e in entries if e.type == "compaction"]
    assert compactions[-1].read_files
    assert "b.py" in compactions[-1].modified_files


def test_summary_is_first_message_not_system_prompt(tmp_path):
    client = QueueClient(ModelResponse(text="## Goal\n- 测试"))
    _, session = build_session(
        tmp_path, client,
        context_budget_tokens=100000,
        context_reserve_tokens=99999,
        context_keep_recent_tokens=50,
    )
    # two big exchanges: the first is compacted, the second is the turn
    # kept whole under the tiny keep-recent budget
    recorder = session.recorder
    for user_text, reply_text in (("你好" * 200, "回答" * 400), (" recent ", "ok")):
        recorder.record(UserMessage(content=user_text), source="test")
        recorder.record(AssistantMessage(text=reply_text), source="test")
        session.add_exchange(user_text, TurnResult(reply=reply_text, iterations=1), "test")
    system, _hist, _cur = session.prepare_context("继续", lambda *_: None)
    messages = [*_hist, *([_cur] if _cur is not None else [])]
    assert "当前会话历史摘要" not in system
    assert isinstance(messages[0], CustomMessage)
    assert messages[0].custom_type == "compaction_summary"
    assert "## Goal" in str(messages[0].content)


# ── batch C: tree compaction — split turns, coverage, branches ────


def _seed(session, *messages):
    for message in messages:
        session.recorder.record(message, source="test")


def _compactions(session):
    return [e for e in read_session_entries(session.jsonl_path) if e.type == "compaction"]


def _previews(context):
    return [message_preview(m, limit=1_000_000) for m in context.messages]


def test_split_turn_compaction_merges_turn_prefix_into_one_entry(tmp_path):
    """§6.8: an assistant cut splits a turn — the turn's user joins the
    main summary, its assistant/toolResult prefix gets a light turnPrefix
    summary, and both merge into ONE CompactionEntry whose coverage ends
    exactly at the cut entry."""
    client = QueueClient(
        ModelResponse(text="## Goal\n- 主摘要"),
        ModelResponse(text="## Original Request\n- 前缀摘要"),
    )
    _, session = build_session(
        tmp_path, client,
        context_budget_tokens=100000,
        context_keep_recent_tokens=10,
    )
    _seed(
        session,
        UserMessage(content="任务一" + "长" * 300),
        AssistantMessage(text="回答一" + "长" * 300),
        UserMessage(content="任务二"),
        AssistantMessage(
            text="先读文件",
            tool_calls=(
                ToolCallContent(id="1", name="read_file", arguments={"path": "a.py"}),
            ),
        ),
        AgentToolResultMessage(
            tool_call_id="1", tool_name="read_file", content="文件内容", is_error=False
        ),
        AssistantMessage(text="最终答复"),
    )

    assert session._do_compress(lambda *_: None)

    compactions = _compactions(session)
    assert len(compactions) == 1
    entry = compactions[0]
    # one entry carries BOTH summaries, prefix in its own section
    assert "主摘要" in entry.summary
    assert "<turn-prefix>" in entry.summary
    assert "前缀摘要" in entry.summary
    assert entry.tokens_before > 0  # estimated when not measured

    # two LLM calls: main zone first (with the turn-opening user),
    # then the prefix zone (assistant/tool only, no 任务一)
    assert len(client.calls) == 2
    main_prompt = client.calls[0]["messages"][0]["content"]
    prefix_prompt = client.calls[1]["messages"][0]["content"]
    assert "任务二" in main_prompt
    assert "文件内容" in prefix_prompt
    assert "任务一" not in prefix_prompt

    # coverage ends exactly at the cut entry = the last assistant message
    messages = [e for e in read_session_entries(session.jsonl_path) if e.type == "message"]
    assert entry.first_kept_entry_id == messages[-1].id

    # context after: summary + only the kept assistant message
    context = session.build_session_context()
    previews = _previews(context)
    assert "主摘要" in previews[0]
    assert previews[1:] == ["最终答复"]


def test_split_turn_prefix_failure_writes_nothing(tmp_path):
    """§6.8: the prefix summary failing AFTER the main one succeeded must
    not move the leaf nor persist half a compaction entry."""
    client = QueueClient(ModelResponse(text="## Goal\n- 主摘要"), RuntimeError("boom"))
    _, session = build_session(
        tmp_path, client,
        context_budget_tokens=100000,
        context_keep_recent_tokens=10,
    )
    _seed(
        session,
        UserMessage(content="任务一" + "长" * 300),
        AssistantMessage(text="回答一" + "长" * 300),
        UserMessage(content="任务二"),
        AssistantMessage(
            text="先读文件",
            tool_calls=(
                ToolCallContent(id="1", name="read_file", arguments={"path": "a.py"}),
            ),
        ),
        AgentToolResultMessage(
            tool_call_id="1", tool_name="read_file", content="文件内容", is_error=False
        ),
        AssistantMessage(text="最终答复"),
    )
    leaf_before = session.recorder.last_entry_id

    assert not session._do_compress(lambda *_: None)
    assert session.recorder.last_entry_id == leaf_before  # leaf unmoved
    assert not _compactions(session)  # no half-written entry
    assert session.summary_info() is None  # no SQLite projection either


def test_compaction_entry_records_first_kept_and_tokens_before(tmp_path):
    """§6.8: the CompactionEntry persists first_kept_entry_id and the
    measured tokens_before (surviving the JSONL round-trip)."""
    client = QueueClient(ModelResponse(text="## Goal\n- 摘要"))
    # keep budget = one row + slack: the walk stops on 问题二 (user cut)
    row_tokens = 4 + estimate_tokens("问题二" + "长" * 300)
    _, session = build_session(
        tmp_path, client,
        context_budget_tokens=100000,
        context_keep_recent_tokens=row_tokens + 1,
    )
    _seed(
        session,
        UserMessage(content="问题一" + "长" * 300),
        AssistantMessage(text="回答一" + "长" * 300),
        UserMessage(content="问题二" + "长" * 300),
        AssistantMessage(text="回答二" + "长" * 300),
    )

    assert session._do_compress(lambda *_: None, tokens_before=12345)

    entry = _compactions(session)[-1]  # read back from JSONL: round-tripped
    assert entry.tokens_before == 12345
    message_ids = [
        e.id for e in read_session_entries(session.jsonl_path) if e.type == "message"
    ]
    # keep budget keeps the last turn whole → the first kept entry is 问题二
    assert entry.first_kept_entry_id == message_ids[2]


def test_branch_back_before_compaction_restores_old_messages(tmp_path):
    """§6.8: compaction covers the current path positionally — branching
    back to before the compaction entry makes the old messages visible
    again (and the summary disappears)."""
    client = QueueClient(ModelResponse(text="## Goal\n- 旧摘要"))
    _, session = build_session(
        tmp_path, client,
        context_budget_tokens=100000,
        context_keep_recent_tokens=50,
    )
    recorder = session.recorder
    recorder.record(UserMessage(content="旧问题" + "旧" * 300), source="test")
    recorder.record(AssistantMessage(text="旧回答" + "旧" * 300), source="test")
    old_point = recorder.last_entry_id
    recorder.record(UserMessage(content="新问题" + "新" * 300), source="test")
    recorder.record(AssistantMessage(text="新回答" + "新" * 300), source="test")

    assert session._do_compress(lambda *_: None)
    previews = _previews(session.build_session_context())
    assert any("旧摘要" in p for p in previews)  # summary injected
    assert not any("旧问题" in p for p in previews)  # covered nodes skipped
    assert any("新回答" in p for p in previews)  # kept entry survives

    session.branch(old_point, lambda *_: None)
    restored = _previews(session.build_session_context())
    assert any("旧问题" in p for p in restored)  # visible again
    assert not any("旧摘要" in p for p in restored)  # compaction off-path


def test_compaction_on_one_branch_never_sees_abandoned_sibling(tmp_path):
    """§6.8: branch A abandoned, compaction on branch B — the summary
    input contains only the current path, never the sibling's content."""
    client = QueueClient(ModelResponse(text="## Goal\n- B摘要"))
    _, session = build_session(
        tmp_path, client,
        context_budget_tokens=100000,
        context_keep_recent_tokens=50,
    )
    recorder = session.recorder
    recorder.record(UserMessage(content="共同起点" + "共" * 300), source="test")
    recorder.record(AssistantMessage(text="共同回答" + "共" * 300), source="test")
    fork = recorder.last_entry_id
    # branch A
    recorder.record(UserMessage(content="苹果方案" + "苹" * 300), source="test")
    recorder.record(AssistantMessage(text="苹果答复" + "苹" * 300), source="test")
    # back to the fork, then branch B
    session.branch(fork, lambda *_: None)
    recorder.record(UserMessage(content="香蕉方案" + "香" * 300), source="test")
    recorder.record(UserMessage(content="香蕉补充" + "香" * 300), source="test")

    assert session._do_compress(lambda *_: None)

    prompt = client.calls[0]["messages"][0]["content"]
    assert "香蕉" in prompt and "共同" in prompt
    assert "苹果" not in prompt


# ── 红线跟随模型窗口(Pi reserveTokens 对齐,2026-09-12 实机问题)─────

def _compactions(session):
    return [
        e for e in read_session_entries(session.jsonl_path)
        if e.type == "compaction"
    ]


def _seed_big_turns(session, n=3, width=50000):
    recorder = session.recorder
    for i in range(n):
        recorder.record(
            UserMessage(content=f"问题{i}:" + "背景" * width), source="test"
        )
        recorder.record(
            AssistantMessage(text=f"回答{i}:" + "结果" * width), source="test"
        )


def test_red_line_follows_model_context_window(tmp_path):
    """红线 = 有效窗口 - reserve:1M 窗口下约 150k 估算不压缩(旧固定
    18k 红线下必压——反向变异点);同一内容切到 24k 窗口立即触发。"""
    client = QueueClient(ModelResponse(text="## Goal\n- 压缩了"))
    _, session = build_session(tmp_path, client)  # budget=0:跟随模型窗口
    session.context_window_getter = lambda: 1_048_576
    _seed_big_turns(session)

    session.prepare_context("继续", lambda *_: None)
    assert not _compactions(session)  # ~150k << 1M - 16k

    # 同一内容、小窗模型:24k 窗口红线 18000,立即触发
    session.context_window_getter = lambda: 24_000
    session.prepare_context("继续", lambda *_: None)
    assert len(_compactions(session)) == 1


def test_explicit_budget_override_wins_over_model_window(tmp_path):
    """context_budget_tokens > 0 是显式 override:模型 1M 也被 50k 限住。
    红线 = 50000 - min(16384, 50000//4) = 37500。"""
    client = QueueClient(ModelResponse(text="## Goal\n- 压缩了"))
    _, session = build_session(tmp_path, client, context_budget_tokens=50_000)
    session.context_window_getter = lambda: 1_048_576
    _seed_big_turns(session)  # ~150k > 37.5k 红线
    session.prepare_context("继续", lambda *_: None)
    assert len(_compactions(session)) == 1


def test_compact_if_due_dedupes_same_measurement(tmp_path):
    """同一次测量值最多触发一次自动压缩;压缩后的新一轮会重新测量
    (不同值),不会被去重挡住的正常判断继续生效。"""
    client = QueueClient(ModelResponse(text="## Goal\n- 压缩了"))
    _, session = build_session(tmp_path, client, context_budget_tokens=50_000)
    _seed_big_turns(session, n=2)

    assert session.compact_if_due(40_000, lambda *_: None) is True
    # 同一测量值不重复触发(压无可压时防止每轮空转)
    assert session.compact_if_due(40_000, lambda *_: None) is False
    # 不同测量值照常走红线判断(低于红线不触发)
    assert session.compact_if_due(10_000, lambda *_: None) is False
    # 大窗模型下 116k 不触发(实机问题的测量路径)
    session.context_window_getter = lambda: 1_048_576
    session.settings.context_budget_tokens = 0
    assert session.compact_if_due(116_000, lambda *_: None) is False
