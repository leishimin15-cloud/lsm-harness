"""Typed coding-session event stream."""

from __future__ import annotations

from helpers import QueueClient

from lsm_harness.agent.events import AgentEndEvent, AgentStartEvent
from lsm_harness.ai.types import AssistantMessageEvent, Model, ModelResponse, Usage
from lsm_harness.coding_agent.app import Harness
from lsm_harness.coding_agent.coding_session import CodingSession
from lsm_harness.coding_agent.events import (
    AgentSettledEvent,
    AutoRetryEndEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    EntryAppendedEvent,
    ErrorEvent,
    QueueUpdateEvent,
    RetryEvent,
    ThinkingLevelChangedEvent,
)
from lsm_harness.config import Settings
from lsm_harness.db import connect


def _session(tmp_path, *responses):
    client = QueueClient(*responses)
    client.model = Model(
        id="fake-main", api="legacy-client", provider="injected", reasoning=True
    )
    return CodingSession(
        settings=Settings(api_key="scripted", home=tmp_path),
        client=client,
        conn=connect(tmp_path, check_same_thread=False),
        stream_fn=client.as_stream_fn(),
    )


def test_coding_session_is_backward_compatible_harness(tmp_path):
    session = _session(tmp_path, ModelResponse(text="ok", usage=Usage(1, 1)))
    try:
        assert isinstance(session, Harness)
        assert session.respond("hi").reply == "ok"
    finally:
        session.close()


def test_one_typed_stream_contains_agent_and_settled_events(tmp_path):
    session = _session(tmp_path, ModelResponse(text="ok", usage=Usage(1, 1)))
    events = []
    session.subscribe(events.append)
    try:
        session.respond("hi")
        assert any(isinstance(event, AgentStartEvent) for event in events)
        assert any(isinstance(event, AgentEndEvent) for event in events)
        assert isinstance(events[-1], AgentSettledEvent)
        assert events[-1].status == "completed"
        appended = [event for event in events if isinstance(event, EntryAppendedEvent)]
        assert [event.entry.message.role for event in appended] == [
            "user", "assistant"
        ]
    finally:
        session.close()


def test_queue_and_thinking_changes_are_typed(tmp_path):
    session = _session(tmp_path)
    events = []
    session.subscribe(events.append)
    active = session.begin_run()
    try:
        assert session.steer("adjust") is True
        assert session.follow_up("after") is True
        queue = [event for event in events if isinstance(event, QueueUpdateEvent)]
        assert queue[-1].steering == ("adjust",)
        assert queue[-1].follow_up == ("after",)
    finally:
        session.agent.finish(active)

    session.set_thinking("enabled")  # 旧三档归一为 high
    assert isinstance(events[-1], ThinkingLevelChangedEvent)
    assert events[-1].level == "high"
    session.close()


def test_model_retry_is_emitted_directly_on_typed_stream(tmp_path):
    client = QueueClient()

    def retrying_stream(_model, _context, options):
        assert options.on_retry is not None
        options.on_retry(1, "transient", "try again")
        response = ModelResponse(text="ok", usage=Usage(1, 1))
        yield AssistantMessageEvent("done", response)

    session = CodingSession(
        settings=Settings(api_key="scripted", home=tmp_path),
        client=client,
        conn=connect(tmp_path, check_same_thread=False),
        stream_fn=retrying_stream,
    )
    events = []
    session.subscribe(events.append)
    try:
        assert session.respond("hi").reply == "ok"
        retries = [event for event in events if isinstance(event, RetryEvent)]
        assert len(retries) == 1
        assert retries[0].attempt == 1
        assert retries[0].max_attempts == 2
        ended = [event for event in events if isinstance(event, AutoRetryEndEvent)]
        assert ended == [AutoRetryEndEvent(True, 1)]
    finally:
        session.close()


def test_model_auth_failure_has_one_typed_terminal_error(tmp_path):
    session = _session(
        tmp_path,
        ModelResponse(
            stop_reason="error",
            error_message="Invalid Authentication",
        ),
    )
    events = []
    session.subscribe(events.append)
    try:
        result = session.respond("hi")
        ends = [event for event in events if isinstance(event, AgentEndEvent)]
        assert result.status == "failed"
        assert len(ends) == 1
        assert ends[0].error == "Invalid Authentication"
        assert not any(isinstance(event, ErrorEvent) for event in events)
    finally:
        session.close()


def test_compaction_has_typed_start_and_end_events(tmp_path):
    session = _session(tmp_path)
    events = []
    session.subscribe(events.append)
    try:
        assert session.compact() is False
        compaction = [
            event
            for event in events
            if isinstance(event, (CompactionStartEvent, CompactionEndEvent))
        ]
        assert compaction == [
            CompactionStartEvent("manual"),
            CompactionEndEvent("manual", result=False),
        ]
    finally:
        session.close()
