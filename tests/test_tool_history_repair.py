"""Interrupted tool calls are repaired before provider reuse."""

from __future__ import annotations

from helpers import QueueClient

from lsm_harness.agent import Agent, AgentLoopConfig, AgentState
from lsm_harness.agent.messages import assistant_message, user_message
from lsm_harness.agent.tools import ToolRegistry
from lsm_harness.agent.tool_history import (
    INTERRUPTED_TOOL_RESULT,
    repair_tool_history,
)
from lsm_harness.ai.messages import ToolCallContent, ToolResultMessage
from lsm_harness.ai.types import Model, ModelResponse, Usage
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect


MODEL = Model(id="test-model", api="legacy-client", provider="legacy")


def test_agent_continue_repairs_interrupted_tool_batch():
    client = QueueClient(ModelResponse(text="recovered", usage=Usage(1, 1)))
    state = AgentState(
        model=MODEL,
        tools=ToolRegistry(),
        messages=[
            user_message("run it"),
            assistant_message(
                tool_calls=[ToolCallContent(id="call-1", name="shell")],
                stop_reason="tool_calls",
            ),
        ],
    )
    agent = Agent(initial_state=state, stream_fn=client.as_stream_fn())

    result = agent.continue_(AgentLoopConfig(max_tokens=100))

    assert result.reply == "recovered"
    sent = client.calls[0]["messages"]
    assert sent[-1]["role"] == "tool"
    assert sent[-1]["tool_call_id"] == "call-1"
    assert sent[-1]["content"] == INTERRUPTED_TOOL_RESULT


def test_session_restart_persists_interrupted_tool_result(tmp_path):
    settings = Settings(api_key="scripted", home=tmp_path)
    first_client = QueueClient()
    first = Harness(
        settings=settings,
        client=first_client,
        conn=connect(tmp_path, check_same_thread=False),
        stream_fn=first_client.as_stream_fn(),
    )
    session_id = first.session.session_id
    assert first.session.recorder is not None
    first.session.recorder.record(user_message("run it"), source="test")
    first.session.recorder.record(
        assistant_message(
            tool_calls=[ToolCallContent(id="call-1", name="shell")],
            stop_reason="tool_calls",
        ),
        source="test",
    )
    first.close()

    second_client = QueueClient(
        ModelResponse(text="recovered", usage=Usage(1, 1))
    )
    second = Harness(
        settings=settings,
        client=second_client,
        conn=connect(tmp_path, check_same_thread=False),
        stream_fn=second_client.as_stream_fn(),
    )
    try:
        assert second.session.session_id == session_id
        context = second.session.build_session_context()
        assert context is not None
        assert isinstance(context.messages[-1], ToolResultMessage)
        assert context.messages[-1].tool_call_id == "call-1"
        assert context.messages[-1].is_error is True
        assert second.session.tree_tip_allows_continue() is True

        result = second.respond_continue(source="test")
        assert result.reply == "recovered"
        sent = second_client.calls[0]["messages"]
        assert sent[-1]["role"] == "tool"
        assert sent[-1]["tool_call_id"] == "call-1"
    finally:
        second.close()


def test_full_repair_reorders_and_drops_invalid_results():
    messages = [
        user_message("run both"),
        assistant_message(tool_calls=[
            ToolCallContent(id="call-1", name="one"),
            ToolCallContent(id="call-2", name="two"),
        ]),
        ToolResultMessage(tool_call_id="call-2", content="two"),
        user_message("interleaved"),
        ToolResultMessage(tool_call_id="orphan", content="orphan"),
        ToolResultMessage(tool_call_id="call-1", content="one"),
        ToolResultMessage(tool_call_id="call-1", content="duplicate"),
    ]

    repair = repair_tool_history(messages)

    assert repair.changed is True
    assert [message.role for message in repair.messages] == [
        "user", "assistant", "tool", "tool", "user"
    ]
    assert [
        message.tool_call_id
        for message in repair.messages
        if isinstance(message, ToolResultMessage)
    ] == ["call-1", "call-2"]
    assert repair.reordered_results == 2
    assert repair.dropped_orphan_results == 1
    assert repair.dropped_duplicate_results == 1


def test_full_repair_synthesizes_missing_result_anywhere_in_history():
    messages = [
        user_message("old"),
        assistant_message(
            tool_calls=[ToolCallContent(id="missing", name="shell")]
        ),
        user_message("later"),
        assistant_message("done"),
    ]

    repair = repair_tool_history(messages)

    assert repair.synthesized_results == 1
    result = repair.messages[2]
    assert isinstance(result, ToolResultMessage)
    assert result.tool_call_id == "missing"
    assert result.is_error is True
