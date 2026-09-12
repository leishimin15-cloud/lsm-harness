"""Context integrity (reliability batch 1).

Two invariants under test:

1. Repeated compaction never loses visible history silently — every
   message evicted from context must either stay in the kept zone or
   have been part of the new summary's INPUT.
2. Budget-fallback trimming cuts at legal message boundaries — an
   assistant tool-call message and its tool results are dropped as one
   block, and when even the pinned summary + current input overflow the
   budget the session raises ContextOverflowError instead of sending an
   illegal request.
"""

from __future__ import annotations

import pytest

from lsm_harness.ai.messages import (
    AssistantMessage,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
)
from lsm_harness.ai.types import ModelResponse
from lsm_harness.coding_agent.session import ContextOverflowError, Session
from lsm_harness.config import Settings
from lsm_harness.db import connect


class Summarizer:
    """Scripted summary client that records every prompt it is shown."""

    def __init__(self, text: str = "SUMMARY_ONLY"):
        self.prompts: list[str] = []
        self.text = text

    def complete(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        return ModelResponse(text=self.text)


@pytest.fixture
def session(tmp_path):
    settings = Settings(
        home=tmp_path,
        api_key="dummy",
        provider="deepseek",
        model="dummy",
        small_model="dummy",
        base_url="",
        context_keep_recent_tokens=28,
        context_budget_tokens=10000,
        context_reserve_tokens=1000,
    )
    conn = connect(tmp_path)
    value = Session(settings, conn=conn, client=Summarizer())
    yield value
    conn.close()


def add_pair(session, user, answer):
    session.recorder.record(UserMessage(content=user.ljust(40, ".")), source="user")
    session.recorder.record(
        AssistantMessage(text=answer.ljust(40, ".")), source="assistant"
    )


def _visible(session) -> str:
    context = session.build_session_context()
    return str(context.messages if context else [])


def test_second_compaction_preserves_previously_kept_messages(session):
    add_pair(session, "old user", "old answer")
    add_pair(session, "PREVIOUSLY_KEPT_FACT", "recent answer")
    assert session.compact(lambda *_: None)
    assert "PREVIOUSLY_KEPT_FACT" not in session.client.prompts[0]
    assert "PREVIOUSLY_KEPT_FACT" in _visible(session)
    add_pair(session, "new user one", "new answer one")
    add_pair(session, "new user two", "new answer two")
    assert session.compact(lambda *_: None)
    visible = _visible(session)
    assert (
        "PREVIOUSLY_KEPT_FACT" in session.client.prompts[1]
        or "PREVIOUSLY_KEPT_FACT" in visible
    ), "Retained history vanished without ever being summarized"


def test_three_compactions_never_drop_unsummarized_facts(session):
    """Acceptance: three rounds; each round's key fact is either still in
    the kept zone or was fed into that round's summary input."""
    facts = []
    for round_index in range(3):
        fact = f"ROUND_{round_index}_KEY_FACT"
        facts.append(fact)
        add_pair(session, fact, f"answer {round_index}")
        # Pad with fresh chatter so the new round has something to cut.
        add_pair(session, f"chatter {round_index} a", f"reply {round_index} a")
        add_pair(session, f"chatter {round_index} b", f"reply {round_index} b")
        assert session.compact(lambda *_: None)
        visible = _visible(session)
        summarized = "\n".join(session.client.prompts)
        for older in facts:
            assert (
                older in visible or older in summarized
            ), f"{older} vanished after compaction round {round_index + 1}"


# ── budget fallback trimming ────────────────────────────────


def _tool_exchange(call_id: str, tool: str = "read_file") -> list:
    return [
        AssistantMessage(
            tool_calls=(ToolCallContent(id=call_id, name=tool, arguments={}),)
        ),
        ToolResultMessage(tool_call_id=call_id, content=f"{call_id} result"),
    ]


def _trimmed(session, monkeypatch, history, budget=256):
    """Run prepare_context against a scripted history with a tiny budget."""
    session.settings.context_budget_tokens = budget
    monkeypatch.setattr(session, "build_system", lambda *_: "")
    monkeypatch.setattr(session, "_compress_if_due", lambda *_: False)
    monkeypatch.setattr(session, "_context_messages", lambda: list(history))
    _sys, _hist, _cur = session.prepare_context("next", lambda *_: None, [])
    return _sys, [*_hist, *([_cur] if _cur is not None else [])]


def _assert_tool_pairs_intact(messages):
    known_calls = {
        call.id
        for message in messages
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    }
    answered_calls = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolResultMessage)
    }
    orphaned = answered_calls - known_calls
    missing = known_calls - answered_calls
    assert not orphaned, f"Unpaired tool results: {orphaned}"
    assert not missing, f"Tool calls missing their results: {missing}"


def test_budget_fallback_preserves_single_tool_pair(session, monkeypatch):
    history = [
        UserMessage(content="x" * 2000),
        *_tool_exchange("c1"),
        AssistantMessage(text="done"),
    ]
    _, messages = _trimmed(session, monkeypatch, history)
    _assert_tool_pairs_intact(messages)


def test_budget_fallback_drops_multi_tool_block_whole(session, monkeypatch):
    """The old fixed-pair trim would cut an assistant with two parallel
    calls away from its results — the block must go as a unit.

    覆盖有效性：budget 有 256 的最低限制，旧版本传 budget=64 时删掉开头
    的大 user 消息后就已装下，实际删除块大小只有 [1]，从未触发三消息
    工具批次删除。这里把工具结果本身做成超出有效预算的体积，确保工具
    批次必定进入删除分支，并用记录的块大小作证。
    """
    import lsm_harness.coding_agent.session as session_module

    block_sizes: list[int] = []
    original = session_module._trim_block_size

    def record(body):
        size = original(body)
        block_sizes.append(size)
        return size

    monkeypatch.setattr(session_module, "_trim_block_size", record)

    history = [
        UserMessage(content="question"),
        AssistantMessage(
            tool_calls=(
                ToolCallContent(id="c1", name="read_file", arguments={}),
                ToolCallContent(id="c2", name="list_dir", arguments={}),
            )
        ),
        ToolResultMessage(tool_call_id="c1", content="x" * 2000),
        ToolResultMessage(tool_call_id="c2", content="r2"),
        *_tool_exchange("c3"),
        AssistantMessage(text="done"),
    ]
    _, messages = _trimmed(session, monkeypatch, history, budget=256)
    _assert_tool_pairs_intact(messages)
    assert 3 in block_sizes, (
        "expected the three-message tool batch to be dropped whole; "
        f"actual block sizes: {block_sizes}"
    )
    # 目标 tool call 及其所有结果确实被移除……
    remaining_call_ids = {
        call.id
        for message in messages
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    }
    remaining_result_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolResultMessage)
    }
    assert not ({"c1", "c2"} & (remaining_call_ids | remaining_result_ids))
    # ……而后面的可保留消息仍在（"全部删光"不算通过）。
    assert "c3" in remaining_call_ids and "c3" in remaining_result_ids
    assert any(
        isinstance(message, AssistantMessage) and message.text == "done"
        for message in messages
    )


def test_budget_fallback_after_compaction_failure_stays_legal(session, monkeypatch):
    """Compaction fails (returns False) → the fallback trim still
    respects tool-pair boundaries."""
    history = [
        UserMessage(content="x" * 2000),
        *_tool_exchange("c1"),
        AssistantMessage(text="done"),
    ]
    session.settings.context_budget_tokens = 256
    session.settings.context_reserve_tokens = 255  # compaction gate opens(红线=256-255=1)
    monkeypatch.setattr(session, "build_system", lambda *_: "")
    # Production compaction failure surfaces as _do_compress → False
    # (it catches its own exceptions); simulate exactly that contract.
    monkeypatch.setattr(session, "_do_compress", lambda *a, **k: False)
    monkeypatch.setattr(session, "_context_messages", lambda: list(history))
    _, _hist, _cur = session.prepare_context("next", lambda *_: None, [])
    messages = [*_hist, *([_cur] if _cur is not None else [])]
    _assert_tool_pairs_intact(messages)


def test_untrimmable_overflow_raises_clear_error(session, monkeypatch):
    """Summary + current input alone exceed the budget → explicit error,
    not an infinite trim loop or an illegal oversized request."""
    monkeypatch.setattr(session, "build_system", lambda *_: "sys")
    monkeypatch.setattr(session, "_compress_if_due", lambda *_: False)
    monkeypatch.setattr(session, "_context_messages", lambda: [])
    # The budget has a 256-token floor — so the "untrimmable" case is a
    # current input that alone exceeds that floor.
    session.settings.context_budget_tokens = 1
    with pytest.raises(ContextOverflowError, match="context overflow"):
        session.prepare_context("x" * 4000, lambda *_: None, [])
