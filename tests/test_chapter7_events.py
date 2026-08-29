"""Chapter 7: typed kernel events, sink barrier semantics, legacy adapter."""

import pytest

from lsm_harness.agent.events import (
    AgentEndEvent,
    AgentEventSink,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
    make_legacy_adapter,
)
from lsm_harness.agent.messages import assistant_message, user_message
from lsm_harness.ai.types import AssistantMessageEvent, ModelResponse


def _ai_event(kind, **kwargs):
    kwargs.setdefault("partial", ModelResponse(text="快照"))
    return AssistantMessageEvent(kind=kind, **kwargs)


# ── sink state machine (Pi processEvents semantics) ──────────────


def test_sink_owns_message_append_on_message_end():
    messages = [user_message("hi")]
    sink = AgentEventSink(messages=messages)
    reply = assistant_message("hello")

    sink.process_event(MessageStartEvent(message={"role": "assistant"}))
    assert sink.streaming_message == {"role": "assistant"}
    assert messages == [user_message("hi")]  # not appended yet

    sink.process_event(MessageEndEvent(message=reply))
    assert sink.streaming_message is None
    # sink.messages IS the context list — appended by the sink, not the loop
    assert sink.messages is messages
    assert messages[-1] is reply


def test_sink_streaming_message_tracks_updates():
    sink = AgentEventSink()
    snapshot = assistant_message("hel")
    sink.process_event(MessageStartEvent(message={"role": "assistant"}))
    sink.process_event(MessageUpdateEvent(
        message=snapshot,
        assistant_message_event=_ai_event("text_delta", text_delta="hel"),
    ))
    assert sink.streaming_message is snapshot


def test_sink_turn_tracking_and_update_state_exemption():
    sink = AgentEventSink()
    sink.process_event(TurnStartEvent(turn_index=2, model="m"))
    assert sink.current_turn == 2
    # progress updates must not touch streaming state
    sink.process_event(ToolExecutionUpdateEvent(
        tool_call_id="c", tool_name="bash", label="Bash", partial="line",
    ))
    assert sink.streaming_message is None


def test_replace_messages_rebinds_owned_list():
    sink = AgentEventSink(messages=[user_message("old")])
    rebuilt = [user_message("summary")]
    sink.replace_messages(rebuilt)
    sink.process_event(MessageEndEvent(message=assistant_message("a")))
    assert rebuilt[-1] == assistant_message("a")
    assert sink.messages is rebuilt


# ── barrier: order, fail-fast, wrap, unsubscribe ─────────────────


def test_listeners_called_in_subscription_order():
    sink = AgentEventSink()
    calls = []
    sink.subscribe(lambda e: calls.append("first"))
    sink.subscribe(lambda e: calls.append("second"))
    sink.process_event(AgentStartEvent(model="m"))
    assert calls == ["first", "second"]


def test_kernel_listener_error_propagates():
    sink = AgentEventSink()

    def boom(_event):
        raise RuntimeError("renderer bug")

    sink.subscribe(boom)
    with pytest.raises(RuntimeError, match="renderer bug"):
        sink.process_event(AgentStartEvent(model="m"))


def test_wrapped_listener_error_is_isolated():
    sink = AgentEventSink()
    calls = []

    def boom(_event):
        raise RuntimeError("extension bug")

    sink.subscribe(boom, wrap=True)
    sink.subscribe(lambda e: calls.append("after"))
    sink.process_event(AgentStartEvent(model="m"))  # must not raise
    assert calls == ["after"]


def test_unsubscribe_stops_delivery():
    sink = AgentEventSink()
    calls = []
    unsubscribe = sink.subscribe(lambda e: calls.append(e.kind))
    sink.process_event(AgentStartEvent(model="m"))
    unsubscribe()
    sink.process_event(AgentStartEvent(model="m"))
    assert calls == ["agent_start"]


def test_state_is_current_before_listeners_run():
    sink = AgentEventSink()
    observed = []
    sink.subscribe(lambda e: observed.append(sink.streaming_message))
    message = assistant_message("hi")
    sink.process_event(MessageStartEvent(message=message))
    sink.process_event(MessageEndEvent(message=message))
    assert observed == [message, None]


# ── legacy adapter fidelity ──────────────────────────────────────


def _capture():
    events = []
    return events, make_legacy_adapter(lambda kind, data: events.append((kind, data)))


def test_adapter_turn_events():
    events, adapter = _capture()
    adapter(TurnStartEvent(turn_index=3, model="claude"))
    adapter(TurnEndEvent(
        turn_index=3, model="claude", stop_reason="stop", status="completed",
        usage={"input_tokens": 1, "output_tokens": 2}, tool_count=0, tool_error_count=0,
    ))
    assert events == [
        ("turn.started", {"turn_index": 3, "iteration": 3, "model": "claude"}),
        ("turn.completed", {
            "turn_index": 3, "iteration": 3, "model": "claude",
            "stop_reason": "stop", "status": "completed",
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "tool_count": 0, "tool_error_count": 0,
        }),
    ]


def test_adapter_folds_stream_deltas_to_llm_names():
    events, adapter = _capture()
    update = lambda ai: MessageUpdateEvent(
        message=assistant_message("x"), assistant_message_event=ai, turn_index=1,
    )
    adapter(update(_ai_event("text_start")))
    adapter(update(_ai_event("text_delta", text_delta="你")))
    adapter(update(_ai_event("text_end")))
    adapter(update(_ai_event("thinking_delta", thinking_delta="t")))
    adapter(update(_ai_event("toolcall_start", tool_index=0, tool_id="c1", tool_name="read")))
    adapter(update(_ai_event("toolcall_delta", tool_index=0, arguments_delta='{"p"')))

    kinds = [kind for kind, _ in events]
    assert kinds == [
        "llm.text.start", "llm.text.delta", "llm.text.end",
        "llm.thinking.delta", "llm.tool_call.start", "llm.tool_call.delta",
    ]
    assert events[1][1] == {"text": "你", "iteration": 1}
    assert events[2][1] == {"iteration": 1, "text": "快照"}
    assert events[4][1] == {
        "iteration": 1, "tool_index": 0, "tool_id": "c1", "tool_name": "read",
    }


def test_adapter_ignores_start_done_error_ai_events():
    events, adapter = _capture()
    for kind in ("start", "done", "error"):
        adapter(MessageUpdateEvent(
            message=assistant_message(None),
            assistant_message_event=_ai_event(kind),
            turn_index=1,
        ))
    assert events == []


def test_adapter_message_events_only_for_injected_sources():
    events, adapter = _capture()
    adapter(MessageStartEvent(message=user_message("steer"), source="steering"))
    adapter(MessageEndEvent(message=user_message("steer"), source="steering"))
    # assistant message boundaries never produced legacy message.* events
    adapter(MessageStartEvent(message=assistant_message("a"), source="assistant"))
    adapter(MessageEndEvent(message=assistant_message("a"), source="assistant"))
    assert [kind for kind, _ in events] == ["message.started", "message.completed"]
    assert events[0][1]["source"] == "steering"
    assert events[0][1]["role"] == "user"


def test_adapter_tool_execution_events():
    events, adapter = _capture()
    adapter(ToolExecutionStartEvent(
        tool_call_id="c1", tool_name="bash", label="Bash", effect="local_write",
    ))
    adapter(ToolExecutionUpdateEvent(
        tool_call_id="c1", tool_name="bash", label="Bash", partial="out",
    ))
    adapter(ToolExecutionEndEvent(
        tool_call_id="c1", tool_name="bash", label="Bash", is_error=False,
    ))
    assert events == [
        ("tool.started", {"tool": "bash", "label": "Bash", "tool_call_id": "c1", "effect": "local_write"}),
        ("tool.progress", {"tool": "bash", "label": "Bash", "tool_call_id": "c1", "delta": "out"}),
        ("tool.execution_end", {"tool": "bash", "label": "Bash", "tool_call_id": "c1", "is_error": False}),
    ]


def test_adapter_agent_boundary_events_are_typed_only():
    events, adapter = _capture()
    adapter(AgentStartEvent(model="m"))
    adapter(AgentEndEvent(status="completed", stop_reason="stop"))
    assert events == []


# ── run_loop integration: typed events through a real Turn ───────

from lsm_harness.agent.events import AgentEvent
from lsm_harness.loop.agent import run_loop
from lsm_harness.tools.registry import Tool, ToolRegistry
from lsm_harness.types import ModelResponse, ToolCall

from helpers import QueueClient


def _echo_registry():
    tools = ToolRegistry()
    tools.register(
        Tool(
            "echo",
            "echo a value",
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            lambda value="": f"ok:{value}",
            "local_write",
        )
    )
    return tools


def _run_with_listeners(client, listeners, messages=None):
    events = []
    result = run_loop(
        client=client,
        model="scripted",
        system="system",
        messages=messages if messages is not None else [{"role": "user", "content": "hello"}],
        tools=_echo_registry(),
        max_iterations=5,
        max_tokens=100,
        emit=lambda kind, data: None,
        listeners=listeners,
    )
    return result, events


def _recorder(collected: list) -> None:
    def listener(event: AgentEvent) -> None:
        collected.append(event)
    return listener


def test_run_loop_emits_full_typed_sequence_with_tool_turn():
    collected: list[AgentEvent] = []
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="done"),
    )
    result, _ = _run_with_listeners(client, [_recorder(collected)])

    assert result.status == "completed"
    kinds = [event.kind for event in collected]
    assert kinds[0] == "agent_start"
    assert kinds[-1] == "agent_end"

    # Turn 1: message lifecycle with stream updates, then turn_end.
    # Turn 2: same message lifecycle without tool calls.
    turn1 = kinds.index("turn_start")
    assert "message_start" in kinds
    assert "message_update" in kinds
    assert kinds.count("message_end") >= 2  # two assistant messages
    assert kinds.count("turn_end") == 2
    # nested ordering: turn_start < message_start < message_end < turn_end
    assert turn1 < kinds.index("message_start")
    assert kinds.index("message_start") < kinds.index("message_end")
    assert kinds.index("message_end") < kinds.index("turn_end")


def test_message_update_passes_through_the_ai_event_identity():
    collected: list[AgentEvent] = []
    client = QueueClient(ModelResponse(text="你好"))
    _run_with_listeners(client, [_recorder(collected)])

    updates = [e for e in collected if isinstance(e, MessageUpdateEvent)]
    assert updates, "stream should produce message_update events"
    for update in updates:
        # the ai-layer event is passed through, not re-built
        assert update.assistant_message_event is not None
        assert update.message.role == "assistant"
    text_deltas = [
        u for u in updates if u.assistant_message_event.kind == "text_delta"
    ]
    assert text_deltas[0].message.text == "你好"


def test_message_end_listener_observes_appended_state():
    context_messages = [{"role": "user", "content": "hello"}]
    observed = []

    def listener(event: AgentEvent) -> None:
        if isinstance(event, MessageEndEvent):
            # barrier semantics: the append already happened (the sink did
            # it) before any listener runs
            observed.append(context_messages[-1] is event.message)

    client = QueueClient(ModelResponse(text="done"))
    _run_with_listeners(client, [listener], messages=context_messages)

    assert observed == [True]
    assert context_messages[-1].role == "assistant"
    assert context_messages[-1].text == "done"


def test_kernel_listener_exception_fails_fast():
    def boom(event: AgentEvent) -> None:
        if isinstance(event, TurnStartEvent):
            raise RuntimeError("listener bug")

    client = QueueClient(ModelResponse(text="done"))
    try:
        _run_with_listeners(client, [boom])
    except RuntimeError as exc:
        assert "listener bug" in str(exc)
    else:
        raise AssertionError("kernel listener exception must propagate")


def test_hooks_and_typed_listeners_coexist():
    from lsm_harness.loop.hooks import LoopHooks

    hook_calls = []
    hooks = LoopHooks(on_turn_end=lambda ctx, emit: hook_calls.append(ctx.turn_index))
    collected: list[AgentEvent] = []
    client = QueueClient(ModelResponse(text="done"))

    run_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hello"}],
        tools=_echo_registry(),
        max_iterations=3,
        max_tokens=100,
        emit=lambda kind, data: None,
        hooks=hooks,
        listeners=[_recorder(collected)],
    )
    assert hook_calls == [1]
    assert any(isinstance(e, TurnEndEvent) for e in collected)


# ── tool-layer typed events (ExecutionContext.event_sink) ────────

import threading

from lsm_harness.agent.tools import AgentTool, ExecutionContext


def test_tool_execution_events_carry_identity_through_run_loop():
    collected: list[AgentEvent] = []
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("call-1", "echo", {"value": "x"})]),
        ModelResponse(text="done"),
    )
    _run_with_listeners(client, [_recorder(collected)])

    starts = [e for e in collected if isinstance(e, ToolExecutionStartEvent)]
    ends = [e for e in collected if isinstance(e, ToolExecutionEndEvent)]
    assert len(starts) == len(ends) == 1
    assert starts[0].tool_call_id == "call-1"
    assert starts[0].tool_name == "echo"
    assert starts[0].args == {"value": "x"}
    assert ends[0].tool_call_id == "call-1"
    assert ends[0].is_error is False

    # tool result messages arrive via the sink's message lifecycle too
    tool_messages = [
        e for e in collected
        if isinstance(e, MessageEndEvent) and e.source == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].message.content == "ok:x"
    assert tool_messages[0].message.tool_call_id == "call-1"


def test_tool_settle_gate_drops_late_typed_updates():
    release = threading.Event()
    workers: list[threading.Thread] = []
    collected: list[AgentEvent] = []
    sink = AgentEventSink()
    sink.subscribe(_recorder(collected))

    def execute(_on_update=None):
        _on_update("now")

        def late_update():
            release.wait()
            _on_update("late")

        worker = threading.Thread(target=late_update)
        worker.start()
        workers.append(worker)
        return "done"

    tools = ToolRegistry()
    tools.register(AgentTool(
        name="progress",
        label="Progress tool",
        description="",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    ))
    tools.execute_batch(
        [("call-7", "progress", {})],
        ctx=ExecutionContext(event_sink=sink),
    )
    release.set()
    for worker in workers:
        worker.join(timeout=1)

    updates = [e for e in collected if isinstance(e, ToolExecutionUpdateEvent)]
    # the acceptingUpdates gate (= _ProgressGate) dropped the late update
    assert [u.partial for u in updates] == ["now"]
    assert updates[0].tool_call_id == "call-7"
    assert updates[0].label == "Progress tool"
    # start/end still arrived through the typed channel
    kinds = [e.kind for e in collected]
    assert kinds[0] == "tool_execution_start"
    assert kinds[-1] == "tool_execution_end"


# ── Agent shell subscription (Chapter 7, Pi Agent.subscribe) ─────

from lsm_harness.agent.runtime import Agent


def test_agent_shell_subscribe_collects_listeners_in_order():
    agent = Agent()
    calls = []
    agent.subscribe(lambda e: calls.append(("a", e.kind)))
    unsubscribe = agent.subscribe(lambda e: calls.append(("b", e.kind)))

    listeners = agent.listeners
    assert len(listeners) == 2
    listeners[0](AgentStartEvent(model="m"))
    listeners[1](AgentStartEvent(model="m"))
    assert calls == [("a", "agent_start"), ("b", "agent_start")]

    unsubscribe()
    assert len(agent.listeners) == 1


def test_agent_shell_wrapped_listener_isolated_from_fail_fast():
    agent = Agent()

    def boom(_event):
        raise RuntimeError("extension bug")

    agent.subscribe(boom, wrap=True)
    agent.listeners[0](AgentStartEvent(model="m"))  # must not raise
